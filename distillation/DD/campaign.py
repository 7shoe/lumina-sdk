"""Resumable Wang-style OPF distillation, one independent experiment per MPI node.

Continuous meta-gradients average real minibatches and optionally initializations.
Discrete, solver-gated topology proposals never enter autograd. Training splits
are fixed at seed 42. Validation examples are held out; random-mode evaluation
also uses unseen model seeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

import numpy as np
import torch
from torch.func import functional_call
from torch.utils.checkpoint import checkpoint as recompute
from torch_geometric.data import Batch
from threadpoolctl import threadpool_limits

from distillation.DD.data import load_batch
from distillation.DD.mvp import (SyntheticOPF,
                                 audit_networks,
                                 mse)
from distillation.DD.topology import (batch_report,
                                      bus_limits,
                                      diversity,
                                      generate_dataset)

from distillation.conversion.matrix_form import mat2opf

from distillation.solver import ACOPF

from distillation.synthetic.learning import (model_for, model_from_export, model_options,
                                             pretrained_config, validate_initialization)
from distillation.synthetic.runtime import collectively
from distillation.synthetic.experiments import case_bus_range, common_config
from distillation.synthetic.inputs import changes_summary as physical_changes
from experiments.lumina_bench.models import SUPPORTED_MODELS
from distillation.training.T1_single_topo.common import (atomic_file,
                                                         canonical_case,
                                                         claim,
                                                         write_json)
from distillation.experiments.staging import case_dataset as CaseDataset

# path management
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets')
FORMAT = 'wang-opf-campaign-v1'

def save(path, payload):
    with atomic_file(path) as stream:
        torch.save(payload, stream)


def log(path, event):
    """Logging of written JSONs"""
    event = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
    print(json.dumps(event, allow_nan=False), flush=True)


def validate_config(config):
    """Check if input config is viable"""
    if config['model'] not in SUPPORTED_MODELS:
        raise ValueError(f'Unknown model: {config["model"]}')
    if any(type(config[k]) is not int or config[k] < 1 for k in ('batches', 'batch_size', 'passes')):
        raise ValueError('Positive integer batches, batch_size and passes required')
    if config['synthetic_count'] != config['batches'] * config['batch_size']:
        raise ValueError('synthetic_count must equal batches * batch_size')
    if config.get('split_seed', 42) != 42:
        raise ValueError('The benchmark split remains fixed at seed 42')
    validate_initialization(config)
    if config.get('target_mode', 'legacy') not in ('legacy', 'solver', 'frozen', 'learned'):
        raise ValueError('Unknown target_mode')
    if config.get('topology_every') and config.get('target_mode') in ('frozen', 'learned'):
        raise ValueError('Online topology edits need target correspondence; use topology_every=0 for frozen/learned')
    for key in ('checkpoint_inner', 'calibrate_inner_rate'):
        if type(config.get(key, False)) is not bool:
            raise ValueError(f'{key} must be boolean')
    cap = config.get('inner_rate_cap')
    if cap is not None and not config['rate_min'] < cap < config['rate_max']:
        raise ValueError('inner_rate_cap must lie strictly between the rate bounds')
    rate = config.get('outer_input_rate')
    if rate is not None and (not math.isfinite(rate) or rate <= 0):
        raise ValueError('outer_input_rate must be positive and finite')


def student_seed(config, iteration, rank, world):
    """Keep seed reproducible"""
    base = config.get('model_seed', 1_000_000)
    return base if config.get('initialization') in ('fixed', 'pretrained') else config.get('training_model_seed', base) + iteration * world + rank


class Schedule(SyntheticOPF):
    """Data distillation runner ; distinct K batches (totalling M samples), E passes "epochs", and an independent learned rate per SGD step"""
    def __init__(self, graphs, config):
        super().__init__(graphs, steps=config['batches'], strength=config['strength'],
                         target_strength=config['target_strength'], initial_rate=config['inner_rate'],
                         rate_min=config['rate_min'], rate_max=config['rate_max'],
                         target_mode=config.get('target_mode'),
                         learn_inputs=config.get('learn_inputs', True), learn_rates=config.get('learn_rates', True))
        self.passes = config['passes']
        self.checkpoint_inner = config.get('checkpoint_inner', False)
        self.rate_logits = torch.nn.Parameter(self.rate_logits.detach().repeat(self.passes),
                                             requires_grad=self.rate_logits.requires_grad)
        # These benchmark backbones do not consume branch attributes. With
        # detached OPF labels there is no differentiable path from these edits.
        self.freeze_unused_features(config['model'])


def inner(model, synthetic, *, create_graph, observe=None):
    """
    Train closed model parameters on synthetic batches using the learned step SGD size
    
    Paras:
    - create_graph=True: retains the mixed derivatives (how Wang '18 did it in the seminal paper)
    """
    
    device = next(model.parameters()).device
    theta = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    buffers = {name: p.detach().clone() for name, p in model.named_buffers()}
    # Reuse graph construction across passes, without detaching meta-gradients.
    batches = [(b.to(device), {k: t.to(device) for k, t in y.items()})
               for b, y in (synthetic.batch(k) for k in range(synthetic.steps))]
    rates = synthetic.rates.to(device)
    if observe is not None:
        observe(0, theta, buffers)
    for step in range(synthetic.steps * synthetic.passes):
        batch, target = batches[step % synthetic.steps]
        names = tuple(theta)
        # Bind each batch now: backward recomputes this exact step, including its
        # mixed derivatives. No trajectory truncation or detached data gradient.
        def update(rate, *values, batch=batch, target=target):
            parameters = dict(zip(names, values))
            loss = mse(functional_call(model, (parameters, buffers), (batch,), strict=True), target, batch)
            gradients = torch.autograd.grad(loss, values, create_graph=create_graph, allow_unused=True)
            return tuple(p if g is None else p - rate * g for p, g in zip(values, gradients))
        values = (recompute(update, rates[step], *theta.values(), use_reentrant=False)
                  if create_graph and synthetic.checkpoint_inner else update(rates[step], *theta.values()))
        theta = dict(zip(names, values))
        if not create_graph:
            theta = {key: p.detach().requires_grad_(True) for key, p in theta.items()}
        if observe is not None:
            observe(step + 1, theta, buffers)
    return theta, buffers


def outer_loss(model, synthetic, real, *, create_graph):
    """
    For a fixed model, evakl on real-data MSE after adapting the model to synthetic data in the inner loop
    """
    
    theta, buffers = inner(model, synthetic, create_graph=create_graph)
    
    # copy model
    real = real.clone().to(next(model.parameters()).device)
    
    # call, infer, and compute MSE
    return mse(functional_call(model, (theta, buffers), (real,), strict=True),
               {kind: real[kind].y for kind in ('bus', 'generator')}, real)


def graph_batch(dataset, ids):
    return Batch.from_data_list([dataset[int(i)] for i in ids])


def sampler_ids(population, iteration, rank, world, count, seed=20261010):
    """Stateless cyclic shuffled epochs; all workers sample disjoint positions.

    The iteration checkpoint is the sampler cursor. A new deterministic shuffle
    is drawn for each epoch, so the full population is accessible on resume.
    """
    start = (iteration * world + rank) * count
    result = []
    while len(result) < count:
        epoch, offset = divmod(start, len(population))
        order = np.random.default_rng(seed + epoch).permutation(len(population))
        take = min(count - len(result), len(population) - offset)
        result.extend(population[order[offset:offset + take]].tolist())
        start += take
    return result


def averaged_gradients(synthetic, comm):
    """CPU float64 reduction; only identically shaped shared synthetic data enter."""
    from mpi4py import MPI
    parameters = [p for p in synthetic.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        valid = False
    else:
        valid = True
    if not comm.allreduce(valid, op=MPI.LAND):
        raise FloatingPointError('Missing/nonfinite meta-gradient on at least one worker')
    vector = np.concatenate([p.grad.detach().double().numpy().ravel() for p in parameters])
    comm.Allreduce(MPI.IN_PLACE, vector, op=MPI.SUM)
    vector /= comm.size
    offset = 0
    for p in parameters:
        p.grad.copy_(torch.from_numpy(vector[offset:offset + p.numel()].copy()).reshape(p.shape))
        offset += p.numel()
    return float(np.linalg.norm(vector))


def training_gradient(model, synthetic, optimizer, real, comm):
    """Project rates down if a fresh initialization makes the unroll overflow.

    This leaves the exact SGD inner trajectory intact (no gradient clipping).
    The same initialization/minibatch is retried; no difficult seed is discarded.
    
    
    """
    from mpi4py import MPI
    
    # max number of retries.
    num_retries = 12
    for attempt in range(num_retries):
        optimizer.zero_grad(set_to_none=True)
        loss = outer_loss(model, synthetic, real, create_graph=True)
        finite = comm.allreduce(math.isfinite(float(loss.detach())), op=MPI.LAND)
        if finite:
            loss.backward()
            missing = any(p.grad is None for p in synthetic.parameters() if p.requires_grad)
            if comm.allreduce(missing, op=MPI.LOR):
                raise RuntimeError('Missing meta-gradient; cannot repair by reducing rates')
            finite = comm.allreduce(all(torch.isfinite(p.grad).all().item() for p in synthetic.parameters()
                                       if p.requires_grad), op=MPI.LAND)
        if finite:
            reference = comm.allreduce(float(loss.detach()), op=MPI.SUM) / comm.size
            del loss
            return reference, averaged_gradients(synthetic, comm), attempt
        del loss
        with torch.no_grad():
            reduced = (synthetic.rates * 0.5).clamp_min(synthetic.rate_min * 1.001)
            probability = (reduced - synthetic.rate_min) / (synthetic.rate_max - synthetic.rate_min)
            synthetic.rate_logits.copy_(torch.logit(probability))
        optimizer.state.pop(synthetic.rate_logits, None)
    raise FloatingPointError(f'No finite meta-gradient after {num_retries} inner-rate backtracks')


def collective_audit(synthetic, comm):
    """
    udit across MPI ranks and return overall success
    """
    local = []
    
    # eval-only mode
    with torch.no_grad():
        for i in range(comm.rank, len(synthetic.anchors), comm.size):
            matrix = synthetic.edits[i](synthetic.anchors[i], synthetic.strength)
            checks, solutions = audit_networks([matrix])
            row = checks.iloc[0].to_dict()
            row['sample'] = i
            graph = solutions[0].to_hetero(ACOPF(matrix)) if row['success'] else None
            local.append((i, row, graph))
    # joint records
    records = sorted(item for group in comm.allgather(local) for item in group)
    
    return all(row['success'] for _, row, _ in records), [g for _, _, g in records], [r for _, r, _ in records]


def optimizer_for(synthetic, config):
    # Legacy checkpoints retain their original single optimizer group.
    if 'target_mode' not in config:
        return torch.optim.Adam(synthetic.parameters(), lr=config['outer_rate'])
    groups = []
    for name, parameters in (('input', synthetic.edits.parameters()),
                             ('target', synthetic.target_offsets.parameters()),
                             ('rate', [synthetic.rate_logits])):
        params = [p for p in parameters if p.requires_grad]
        if params:
            rate = config.get(f'outer_{name}_rate')
            groups.append(dict(params=params, lr=config['outer_rate'] if rate is None else rate))
    return torch.optim.Adam(groups)


def transplant(old, graphs, config, changed, optimizer=None):
    """Keep every unchanged graph/rate and its Adam moments across topology edits."""
    new = Schedule(graphs, config)
    def replaced(name):
        pieces = name.split('.')
        return pieces[0] in ('edits', 'target_offsets') and int(pieces[1]) in changed
    state = new.state_dict()
    for name, value in old.state_dict().items():
        if not replaced(name):
            state[name] = value.clone()
    new.load_state_dict(state)
    for i in range(len(graphs)):
        if i not in changed:
            new.seed_targets[i] = deepcopy(old.seed_targets[i])
    new_optimizer = optimizer_for(new, config)
    if optimizer is not None:
        old_named = dict(old.named_parameters())
        for name, p in new.named_parameters():
            if not replaced(name) and old_named[name] in optimizer.state:
                new_optimizer.state[p] = deepcopy(optimizer.state[old_named[name]])
    return new, new_optimizer


def proposal_step(synthetic, optimizer, config, reference, score, audit):
    """Transactional line search. Solver-mode proposals are scored AFTER relabelling."""
    before = deepcopy(synthetic.state_dict())
    moments, labels = deepcopy(optimizer.state_dict()), deepcopy(synthetic.seed_targets)
    accepted, attempts, physical, checks = False, [], None, None
    refresh = synthetic.target_mode == 'solver' or (
        synthetic.target_mode == 'legacy' and config.get('fresh_solver_targets', False))
    try:
        optimizer.step()
        proposal = deepcopy(synthetic.state_dict())
        for fraction in config.get('backtrack_fractions', (1., .5, .25, .125, .0625)):
            if not 0 < fraction <= 1:
                raise ValueError('Backtracking fractions must be in (0, 1]')
            synthetic.load_state_dict({k: v + fraction * (proposal[k] - v) for k, v in before.items()})
            synthetic.seed_targets = deepcopy(labels)
            trial = dict(fraction=fraction, real_mse=None, feasible=None, accepted=False)
            # Preserve the old order only for legacy configurations. In frozen/
            # learned mode the teaching targets do not depend on this audit.
            if synthetic.target_mode != 'solver':
                value = score()
                trial['real_mse'] = value if math.isfinite(value) else None
                if not math.isfinite(value) or value > reference:
                    trial['reason'] = 'nonfinite_objective' if not math.isfinite(value) else 'objective'
                    attempts.append(trial)
                    continue
            ok, proposed, rows = audit()
            trial['feasible'] = ok
            trial['solver_failures'] = sum(not r['success'] for r in rows)
            trial['nonconverged'] = sum(not r.get('converged', r['success']) for r in rows)
            if ok and refresh:
                synthetic.seed_targets = [{k: g[k].y.detach().clone() for k in ('bus', 'generator')}
                                          for g in proposed]
            if ok and refresh:
                value = score()
                trial['real_mse'] = value if math.isfinite(value) else None
                ok = math.isfinite(value) and value <= reference
            trial['reason'] = 'accepted' if ok else ('solver' if not trial['feasible'] else 'objective')
            trial['accepted'] = ok
            attempts.append(trial)
            if ok:
                accepted, physical, checks = True, proposed, rows
                break
    finally:
        if not accepted:
            synthetic.load_state_dict(before)
            synthetic.seed_targets = labels
            optimizer.load_state_dict(moments)
    return accepted, physical, checks, attempts


def changes_summary(synthetic):
    return physical_changes(synthetic)


def evaluation(model, synthetic, dataset, ids, batch_size=128, *, trained=True,
               trace=None, trace_every=10, trace_samples=32):
    started, probe_seconds = time.monotonic(), 0.
    device = next(model.parameters()).device
    probe = graph_batch(dataset, ids[:trace_samples]).to(next(model.parameters()).device) if trace is not None else None
    def observe(step, theta, buffers):
        nonlocal probe_seconds
        if step % trace_every and step != synthetic.steps * synthetic.passes:
            return
        # Charge queued accelerator work to student training, not to the probe.
        if device.type == 'xpu':
            torch.xpu.synchronize(device)
        elif device.type == 'cuda':
            torch.cuda.synchronize(device)
        seconds = time.monotonic() - started - probe_seconds
        tick = time.monotonic()
        with torch.no_grad():
            value = float(mse(functional_call(model, (theta, buffers), (probe,), strict=True),
                              {k: probe[k].y for k in ('bus', 'generator')}, probe))
        trace.append(dict(step=step, graph_presentations=step * synthetic.batch_size,
                          real_mse=value if math.isfinite(value) else None, student_seconds=seconds))
        probe_seconds += time.monotonic() - tick
    if trained:
        theta, buffers = inner(model, synthetic, create_graph=False, observe=observe if trace is not None else None)
    else:
        theta, buffers = dict(model.named_parameters()), dict(model.named_buffers())
    value = 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = graph_batch(dataset, ids[start:start + batch_size]).to(next(model.parameters()).device)
            loss = mse(functional_call(model, (theta, buffers), (batch,), strict=True),
                       {k: batch[k].y for k in ('bus', 'generator')}, batch)
            value += float(loss) * batch.num_graphs
    return value / len(ids), {k: v.detach().cpu() for k, v in theta.items()}, buffers


def score_summary(scores):
    finite = all(math.isfinite(x) for x in scores)
    return {'mean': float(np.mean(scores)) if finite else None,
            'median': float(np.median(scores)) if finite else None,
            'std': float(np.std(scores)) if finite else None,
            'per_initialization': [x if math.isfinite(x) else None for x in scores],
            'nonfinite_initializations': sum(not math.isfinite(x) for x in scores)}


def panel(synthetic, config, dataset, ids, comm, device, seed_base=10_000_000, trained=True):
    config = pretrained_config(config)
    seed = config.get('model_seed', 1_000_000) if config.get('initialization') in ('fixed', 'pretrained') else seed_base + comm.rank
    model = model_for(mat2opf(synthetic.anchors[0]), config, seed, device)
    trace = [] if config.get('trace_every', 0) and trained else None
    value, theta, buffers = evaluation(model, synthetic, dataset, ids, config['eval_batch'], trained=trained,
                                      trace=trace, trace_every=config.get('trace_every', 10),
                                      trace_samples=config.get('trace_samples', 32))
    scores = comm.allgather(value)
    record = score_summary(scores) | {
        'model_seeds': comm.allgather(seed), 'real_examples': len(ids),
        'initialization': config.get('initialization', 'random')}
    record['unique_initializations'] = len(set(record['model_seeds']))
    if config.get('initialization') == 'pretrained':
        record['pretrained_sha256'] = config['pretrained']['sha256']
    if trace is not None:
        record['learning_curves'] = comm.allgather(trace)
        record['curve_examples'] = min(len(ids), config.get('trace_samples', 32))
    student = {'initial_state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               'trained_state': theta | {k: v.detach().cpu() for k, v in buffers.items()},
               'seed': seed, 'model': config['model'], 'model_options': config['model_options']}
    return record, student


def checkpoint(folder, synthetic, optimizer, state, config, physical, checks, comm):
    if comm.rank:
        return
    payload = {'format': FORMAT, 'config': config, 'state': state,
               'anchor_graphs': [mat2opf(m) for m in synthetic.anchors],
               'synthetic_state': synthetic.state_dict(), 'optimizer': optimizer.state_dict(),
               'seed_targets': synthetic.seed_targets, 'physical_graphs': physical, 'solver_checks': checks,
               'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state()}
    save(folder / 'latest.pt', payload)
    write_json(folder / 'status.json', {k: v for k, v in state.items() if k not in ('seen_ids',)} |
               {'distinct_real_training_examples': len(state['seen_ids']), 'checkpoint': str(folder / 'latest.pt')})


def restore(payload):
    synthetic = Schedule(payload['anchor_graphs'], payload['config'])
    synthetic.seed_targets = payload['seed_targets']
    synthetic.load_state_dict(payload['synthetic_state'])
    optimizer = optimizer_for(synthetic, payload['config'])
    optimizer.load_state_dict(payload['optimizer'])
    torch.set_rng_state(payload['rng_torch'])
    np.random.set_state(payload['rng_numpy'])
    return synthetic, optimizer


def export(folder, synthetic, config, state, physical, checks, student, name='distilled.pt'):
    with torch.no_grad():
        schedule = []
        batches = [synthetic.batch(k) for k in range(synthetic.steps)]
        for step in range(synthetic.steps * synthetic.passes):
            batch, teaching = batches[step % synthetic.steps]
            schedule.append({'step': step, 'batch': step % synthetic.steps, 'inputs': batch.detach().cpu(),
                             'teaching_targets': {k: v.detach().cpu() for k, v in teaching.items()},
                             'rate': float(synthetic.rates[step])})
    save(folder / name, {'format': FORMAT, 'config': config, 'training_state': state,
                        'schedule': schedule, 'physical_solved_graphs': physical,
                        'solver_checks': checks, 'model_checkpoint': student,
                        'anchor_graphs': [mat2opf(m) for m in synthetic.anchors],
                        'synthetic_state': synthetic.state_dict(), 'seed_targets': synthetic.seed_targets,
                        'target_mode': synthetic.target_mode,
                        'target_semantics': {'solver': 'fresh detached AC-OPF solutions',
                            'frozen': 'initial fixed teaching targets; not certified solutions of edited grids',
                            'learned': 'independent learned teaching targets; not certified AC-OPF solutions',
                            'legacy': 'legacy bounded offsets; inspect target_strength and fresh_solver_targets'}[synthetic.target_mode]})
    save(folder / 'student.pt', student)


def run(config, folder, comm, device, *, allocation_seconds=3100, qualify=False, stop_after_updates=None):
    """Train for a cumulative time budget; resume from the last atomic commit."""
    from mpi4py import MPI
    validate_config(config)
    config = collectively(comm, lambda: pretrained_config(config, refresh=True))
    error = None
    started = time.monotonic()
    folder = Path(folder)
    if comm.rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
    comm.Barrier()
    dataset = CaseDataset(config['data_root'], f"case{config['case']}")
    split = dataset.splits(42)
    payload = None
    if comm.rank == 0:
        try:
            if (folder / 'latest.pt').exists():
                payload = torch.load(folder / 'latest.pt', weights_only=False, map_location='cpu')
                if payload['config'] != config:
                    raise ValueError('Resume configuration or pretrained checkpoint differs; use a new experiment directory')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    if error:
        raise ValueError(error)
    payload = comm.bcast(payload, root=0)
    if comm.rank == 0:
        write_json(folder / 'config.json', config)
    if payload:
        synthetic, optimizer = restore(payload)
        state = payload['state']
        physical, checks = payload['physical_graphs'], payload['solver_checks']
        original = torch.load(folder / 'original.pt', weights_only=False)['graphs']
        if state['status'] == 'COMPLETE':
            return state
        # A preemption can occur after atomic best export but before latest
        # state is committed. Preserve that better artifact across the replay.
        best_path = folder / 'best.pt'
        if best_path.exists():
            saved_best = torch.load(best_path, weights_only=False, map_location='cpu')['training_state']
            if saved_best['best_validation'] is not None and (
                    state['best_validation'] is None or saved_best['best_validation'] < state['best_validation']):
                state['best_validation'] = saved_best['best_validation']
                state['best_iteration'] = saved_best['best_iteration']
    else:
        if config.get('generation_protocol'):
            from distillation.synthetic.experiments import initialize
            original, seed_ids, physical, changes = initialize(config, folder, comm)
            ordered = list(zip(range(len(physical)), physical, changes))
            synthetic = Schedule(physical, config)
            optimizer = optimizer_for(synthetic, config)
            ok, physical, checks = collective_audit(synthetic, comm)
            if not ok:
                raise RuntimeError('Generated initialization failed feasibility')
        else:
            seed_payload = None
            if comm.rank == 0:
                originals, seed_ids = load_batch(config['raw_root'], config['case'], count=config['synthetic_count'],
                                                  seed=config['data_seed'], split='train')
                seed_payload = (originals, seed_ids.tolist())
            original, seed_ids = comm.bcast(seed_payload, root=0)
            synthetic = Schedule(original, config)
            ok, physical, checks = collective_audit(synthetic, comm)
            if not ok:
                raise RuntimeError('Initial real seeds failed fresh AC-OPF solves')
            # Diverse, feasible initial topology; later discrete updates optimize the objective.
            local, local_changes = generate_dataset(1, physical[comm.rank::comm.size],
                min_buses=config.get('bus_range', bus_limits(config['case']))[0],
                max_buses=config.get('bus_range', bus_limits(config['case']))[1],
                seed=config['data_seed'] + comm.rank, max_edits=2, attempts_per_graph=4)
            parts = comm.allgather(list(zip(range(comm.rank, len(physical), comm.size), local, local_changes)))
            ordered = sorted(item for group in parts for item in group)
            physical = [g for _, g, _ in ordered]
            if min(diversity(physical, config['batches'])) < min(3, config['batch_size']):
                raise RuntimeError('Initial topology diversity requirement failed')
            synthetic = Schedule(physical, config)
            optimizer = optimizer_for(synthetic, config)
            ok, physical, checks = collective_audit(synthetic, comm)
            if not ok:
                raise RuntimeError('Diverse initialization failed feasibility')
        state = {'iteration': 0, 'accepted_updates': 0, 'rejected_updates': 0, 'topology_updates': 0,
                 'active_seconds': 0.0, 'status': 'TRAINING', 'seen_ids': set(), 'seed_ids': seed_ids,
                 'best_validation': None, 'best_iteration': None, 'resume_count': 0}
        if comm.rank == 0:
            save(folder / 'original.pt', {'graphs': original, 'ids': seed_ids})
            save(folder / 'initial.pt', {'graphs': physical, 'changes': [r for _, _, r in ordered]})
    state['resume_count'] += 1
    if synthetic.target_mode == 'solver':
        synthetic.seed_targets = [{k: g[k].y.detach().clone() for k in ('bus', 'generator')} for g in physical]
    population = split['train'][~np.isin(split['train'], state['seed_ids'])]
    if not set(state['seed_ids']) <= set(split['train'].tolist()):
        raise ValueError('Initialization references must belong to the train split')
    val_ids = split['val'][:config['validation_samples']]
    if comm.rank == 0:
        write_json(folder / 'split.json', {'split_seed': 42, 'case': canonical_case(f"case{config['case']}"),
            'train_pool_size': len(population), 'excluded_seed_ids': state['seed_ids'],
            'validation_ids': val_ids.tolist(), 'test_accessed': False,
            'train_pool_sha256': hashlib.sha256(population.tobytes()).hexdigest(),
            'sampler': 'cyclic independently shuffled epochs, iteration/rank cursor, seed 20261010'})
    if config.get('calibrate_inner_rate') and 'initial_inner_rate' not in state:
        from distillation.synthetic.evaluation import calibrate
        rate, calibration = calibrate(original, config | {'evaluation_epochs': config['passes']},
                                      dataset, population, comm, device)
        rate = min(max(rate, synthetic.rate_min * 1.001), synthetic.rate_max * .999)
        if config.get('inner_rate_cap') is not None:
            rate = min(rate, config['inner_rate_cap'])
        with torch.no_grad():
            probability = (rate - synthetic.rate_min) / (synthetic.rate_max - synthetic.rate_min)
            synthetic.rate_logits.fill_(math.log(probability / (1 - probability)))
        state['initial_inner_rate'] = rate
        if comm.rank == 0:
            write_json(folder/'inner_calibration.json', calibration | {'native_initial_rate': rate,
                'native_initial_rate_cap': config.get('inner_rate_cap')})
    if qualify:
        from distillation.DD.mvp import check_derivatives
        sample = physical[:3]
        real_ids = sampler_ids(population, 0, comm.rank, comm.size, 4)
        real = graph_batch(dataset, real_ids)
        model = model_for(sample[0], config, 500 + comm.rank, torch.device('cpu'))
        try:
            result = check_derivatives(model, sample, real, {k: real[k].y for k in ('bus', 'generator')},
                                       {'steps': 3, 'initial_rate': config['inner_rate'], 'target_mode': synthetic.target_mode,
                                        'rate_min': config['rate_min'], 'rate_max': config['rate_max']}, device)
        except Exception as exc:
            result = {'passed': False, 'loss_passed': False, 'gradients': [],
                      'error': f'{type(exc).__name__}: {exc}'}
        # FP32 backends can disagree on nearly-zero second derivatives below
        # the forward objective's floating-point resolution. Keep original
        # diagnostics; enforce an explicit absolute floor plus relative error.
        result['original_strict_passed'] = result['passed']
        result['original_loss_passed'] = result['loss_passed']
        result['loss_tolerance'] = 1e-6 + 1e-3 * abs(result.get('cpu_loss', 0.))
        if 'cpu_loss' in result:
            result['loss_passed'] = abs(result['cpu_loss'] - result['device_loss']) <= result['loss_tolerance']
        result['absolute_gradient_tolerance'] = 1e-8
        result['criterion'] = 'per-family L2 error <= 1e-8 + 0.02 * CPU gradient L2 norm'
        for row in result['gradients']:
            row['original_tolerance'] = row['tolerance']
            row['tolerance'] = max(1e-8, row['tolerance'])
            row['passed_max_abs'] = row['max_abs_error'] <= row['tolerance']
            row['tolerance_l2'] = 1e-8 + 0.02 * row['cpu_gradient_norm']
            row['passed'] = row['error_l2'] <= row['tolerance_l2']
        result['passed'] = result['loss_passed'] and all(row['passed'] for row in result['gradients'])
        qualification = comm.allgather(result | {'rank': comm.rank})
        if comm.rank == 0:
            write_json(folder / 'device_checks.json', qualification)
        if not all(row['passed'] for row in qualification):
            raise RuntimeError('CPU/XPU second-order qualification failed')
        profile = []
        for count in config.get('profile_real_batches', (16, 64, 256)):
            optimizer.zero_grad(set_to_none=True)
            torch.xpu.reset_peak_memory_stats(device)
            tick = time.monotonic()
            probe_ids = sampler_ids(population, 0, comm.rank, comm.size, count)
            probe = graph_batch(dataset, probe_ids)
            probe_model = model_for(physical[0], config, 600 + comm.rank, device)
            probe_loss = outer_loss(probe_model, synthetic, probe, create_graph=True)
            probe_loss.backward()
            gradient_norm = averaged_gradients(synthetic, comm)
            torch.xpu.synchronize(device)
            row = {'real_graphs_per_worker': count, 'real_graphs_global': count * comm.size,
                   'seconds_max': comm.allreduce(time.monotonic() - tick, op=MPI.MAX),
                   'peak_memory_bytes_max': comm.allreduce(torch.xpu.max_memory_allocated(device), op=MPI.MAX),
                   'meta_gradient_norm': gradient_norm}
            profile.append(row)
            if comm.rank == 0:
                log(folder / 'history.jsonl', {'stage': 'profile', **row})
            del probe_loss, probe_model, probe
        if comm.rank == 0:
            write_json(folder / 'profile.json', profile)
    
    # init checkpoint/baseline also makes initialization recoverable.
    if state['best_validation'] is None:
        evaluation_started = time.monotonic()
        result, student = panel(synthetic, config, dataset, val_ids, comm, device)
        if result['mean'] is None:
            raise FloatingPointError('Initial schedule has nonfinite validation loss; lower the starting rate')
        state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
        if comm.rank == 0:
            log(folder / 'history.jsonl', {'stage': 'initial_validation', 'metrics': result,
                                          'seconds': time.monotonic() - evaluation_started})
            export(folder, synthetic, config, state, physical, checks, student, name='best.pt')
    
    # checkpoint mgtm
    checkpoint(folder, synthetic, optimizer, state, config, physical, checks, comm)
    last_save = last_eval = time.monotonic()
    while state['active_seconds'] < config['training_seconds'] and state['iteration'] < config['max_updates']:
        if stop_after_updates is not None and state['iteration'] >= stop_after_updates:
            break
        remaining = allocation_seconds - (time.monotonic() - started)
        if comm.allreduce(remaining, op=MPI.MIN) < config['reserve_seconds']:
            break
        tick = time.monotonic()
        iteration = state['iteration']
        ids = sampler_ids(population, iteration, comm.rank, comm.size, config['real_batch'],
                          seed=config.get('sampler_seed', 20261010))
        real = graph_batch(dataset, ids)
        model = model_for(physical[0], config, student_seed(config, iteration, comm.rank, comm.size), device)
        reference, grad_norm, rate_backtracks = training_gradient(model, synthetic, optimizer, real, comm)
        def score():
            value = float(outer_loss(model, synthetic, real, create_graph=False).detach())
            return comm.allreduce(value, op=MPI.SUM) / comm.size
        accepted, proposed, rows, attempts = proposal_step(synthetic, optimizer, config, reference, score,
                                                         lambda: collective_audit(synthetic, comm))
        if accepted:
            physical, checks = proposed, rows
        
        # A fully rejected proposal rolls back parameters AND all Adam state.
        state['accepted_updates' if accepted else 'rejected_updates'] += 1
        state['iteration'] += 1
        for group in comm.allgather(ids):
            state['seen_ids'].update(group)
        topology_event = None
        if config['topology_every'] and state['iteration'] % config['topology_every'] == 0:
            synthetic, optimizer, physical, checks, topology_event = topology_step(
                synthetic, optimizer, physical, checks, config, real, model, comm, device, state['iteration'])
            state['topology_updates'] += int(topology_event['accepted'])
        seconds = comm.allreduce(time.monotonic() - tick, op=MPI.MAX)
        state['active_seconds'] += seconds
        if comm.rank == 0:
            log(folder / 'history.jsonl', {'stage': 'outer', 'iteration': state['iteration'],
                'real_mse_before': reference, 'accepted': accepted, 'attempts': attempts, 'meta_gradient_norm': grad_norm,
                'seconds': seconds, 'active_seconds': state['active_seconds'],
                'distinct_real_examples': len(state['seen_ids']), 'rates': synthetic.rates.detach().tolist(),
                'rate_backtracks_for_new_initialization': rate_backtracks, 'topology': topology_event,
                'grid_changes': changes_summary(synthetic)})
        do_eval = time.monotonic() - last_eval >= config['eval_seconds']
        do_eval = comm.bcast(do_eval if comm.rank == 0 else None, root=0)
        if do_eval:
            result, student = panel(synthetic, config, dataset, val_ids, comm, device)
            if result['mean'] is not None and result['mean'] < state['best_validation']:
                state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
                if comm.rank == 0:
                    export(folder, synthetic, config, state, physical, checks, student, name='best.pt')
            if comm.rank == 0:
                log(folder / 'history.jsonl', {'stage': 'validation', 'iteration': state['iteration'],
                                              'metrics': result, 'grid_changes': changes_summary(synthetic)})
            last_eval = time.monotonic()
        do_save = comm.bcast(time.monotonic() - last_save >= config['checkpoint_seconds'] if comm.rank == 0 else None, root=0)
        if do_save:
            checkpoint(folder, synthetic, optimizer, state, config, physical, checks, comm)
            last_save = time.monotonic()
    complete = state['active_seconds'] >= config['training_seconds'] or state['iteration'] >= config['max_updates']
    state['status'] = 'FINALIZING' if complete else 'RESUMABLE'
    checkpoint(folder, synthetic, optimizer, state, config, physical, checks, comm)
    result, student = panel(synthetic, config, dataset, val_ids, comm, device)
    if result['mean'] is not None and result['mean'] < state['best_validation']:
        state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
        if comm.rank == 0:
            export(folder, synthetic, config, state, physical, checks, student, name='best.pt')
    if comm.rank == 0:
        export(folder, synthetic, config, state, physical, checks, student)
        log(folder / 'history.jsonl', {'stage': 'final_validation', 'iteration': state['iteration'], 'metrics': result})
    comm.Barrier()
    if complete:
        done = final_evaluation(folder, synthetic, original, config, dataset, split, state, comm, device,
                                deadline=started + allocation_seconds)
        state['status'] = 'COMPLETE' if done else 'FINALIZING'
    checkpoint(folder, synthetic, optimizer, state, config, physical, checks, comm)
    return state


def topology_step(synthetic, optimizer, physical, checks, config, real, training_model, comm, device, iteration):
    """Each XPU screens one feasible local edit; winner faces the full seed panel.

    Common screening seeds/data and paired full-panel replays prevent differences
    in initialization or minibatch from masquerading as a topology improvement.
    Unchanged parameters and moments survive; only edited graphs are rebased.
    """
    from mpi4py import MPI
    i = (iteration + comm.rank * 17) % len(physical)
    edited, changes = generate_dataset(comm.rank + 1, [physical[i]],
        min_buses=config.get('bus_range', bus_limits(config['case']))[0],
        max_buses=config.get('bus_range', bus_limits(config['case']))[1],
        seed=83_000 + iteration, max_edits=config.get('topology_max_edits', 2),
        attempts_per_graph=config.get('topology_attempts', 3))

    candidate = None
    local_score = float('inf')
    common_real = comm.bcast(real if comm.rank == 0 else None, root=0)
    if changes[0]['changed']:
        anchors = [mat2opf(m) for m in synthetic.anchors]
        anchors[i] = edited[0]
        candidate, _ = transplant(synthetic, anchors, config, {i})
        screening_model = model_for(physical[0], config, 7_000_000 + iteration, device)
        local_score = float(outer_loss(screening_model, candidate, common_real, create_graph=False).detach())
    scores = comm.allgather(local_score)
    
    if not any(math.isfinite(s) for s in scores):
        return synthetic, optimizer, physical, checks, {'accepted': False, 'reason': 'no feasible proposal'}
    winner = min(range(comm.size), key=lambda j: scores[j])
    index, new_graph, change = comm.bcast((i, edited[0], changes[0]) if comm.rank == winner else None, root=winner)
    anchors = [mat2opf(m) for m in synthetic.anchors]
    anchors[index] = new_graph
    proposed, proposed_optimizer = transplant(synthetic, anchors, config, {index}, optimizer)
    old_loss = float(outer_loss(training_model, synthetic, real, create_graph=False).detach())
    new_loss = float(outer_loss(training_model, proposed, real, create_graph=False).detach())
    old_loss = comm.allreduce(old_loss, op=MPI.SUM) / comm.size
    new_loss = comm.allreduce(new_loss, op=MPI.SUM) / comm.size
    physical_proposal = list(physical)
    physical_proposal[index] = new_graph
    
    # check acceptance
    required = min(config.get('min_topologies', 3), config['batch_size'])
    accepted = math.isfinite(new_loss) and new_loss < old_loss and min(diversity(physical_proposal, config['batches'])) >= required
    if accepted:
        ok, new_physical, new_checks = collective_audit(proposed, comm)
        accepted = ok
        if accepted and proposed.target_mode == 'solver':
            proposed.seed_targets = [{k: g[k].y.detach().clone() for k in ('bus', 'generator')} for g in new_physical]
            new_loss = float(outer_loss(training_model, proposed, real, create_graph=False).detach())
            new_loss = comm.allreduce(new_loss, op=MPI.SUM) / comm.size
            accepted = math.isfinite(new_loss) and new_loss < old_loss
    event = {'accepted': accepted,
             'graph': index,
             'before': old_loss,
             'after': new_loss if math.isfinite(new_loss) else None,
             'change': change}
    if accepted:
        return proposed, proposed_optimizer, new_physical, new_checks, event
    
    return synthetic, optimizer, physical, checks, event


def fit_export(model, artifact):
    """Apply the saved curriculum to the supplied starting weights and buffers."""
    theta = {k: p.detach().clone().requires_grad_(True) for k, p in model.named_parameters()}
    buffers = {k: b.detach().clone() for k, b in model.named_buffers()}
    device = next(model.parameters()).device
    for step in artifact['schedule']:
        batch = step['inputs'].clone().to(device)
        loss = mse(functional_call(model, (theta, buffers), (batch,), strict=True), step['teaching_targets'], batch)
        grad = torch.autograd.grad(loss, tuple(theta.values()), allow_unused=True)
        theta = {k: (p if g is None else p - step['rate'] * g).detach().requires_grad_(True)
                 for (k, p), g in zip(theta.items(), grad)}
    return theta, buffers


def replay_export(model, artifact, dataset, ids, batch_size):
    theta, buffers = fit_export(model, artifact)
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = graph_batch(dataset, ids[start:start + batch_size]).to(device)
            total += float(mse(functional_call(model, (theta, buffers), (batch,), strict=True),
                               {k: batch[k].y for k in ('bus', 'generator')}, batch)) * batch.num_graphs
    return total / len(ids)


def final_evaluation(folder, synthetic, original, config, dataset, split, state, comm, device, deadline=float('inf')):
    """Paired held-out evaluation and train-only real-control tuning; resumable by arm."""
    ids = split['val'][config['validation_samples']:config['validation_samples'] + config['final_samples']]
    artifact = torch.load(folder / 'best.pt', weights_only=False, map_location='cpu')
    best = None
    if 'synthetic_state' in artifact:
        best = Schedule(artifact['anchor_graphs'], config)
        best.seed_targets = artifact['seed_targets']
        best.load_state_dict(artifact['synthetic_state'])
    path = folder / 'evaluation.partial.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    if report and report['best_iteration'] != artifact['training_state']['best_iteration']:
        raise ValueError('Partial evaluation belongs to a different best artifact')
    report['best_iteration'] = artifact['training_state']['best_iteration']
    def persist():
        if comm.rank == 0:
            write_json(path, report)
        comm.Barrier()
    def have_time():
        return all(comm.allgather(deadline - time.monotonic() > config.get('evaluation_reserve_seconds', 90)))

    if 'best_distilled' not in report:
        if not have_time():
            return False
        if best is not None:
            report['best_distilled'], _ = panel(best, config, dataset, ids, comm, device, seed_base=20_000_000)
        else:
            model = model_for(original[0], config, 20_000_000 + comm.rank, device)
            report['best_distilled'] = score_summary(comm.allgather(replay_export(model, artifact, dataset, ids, config['eval_batch'])))
        persist()

    # A separate real-only LR sweep prevents DD's rate schedule handicapping the control.
    population = split['train'][~np.isin(split['train'], state['seed_ids'])]
    calibration_ids = sampler_ids(population, 0, 0, 1, config.get('calibration_samples', 128), seed=314159)
    tuning = report.setdefault('real_control_tuning', {'training_ids': calibration_ids, 'candidates': []})
    for nominal_rate in config.get('real_control_rates', []):
        rate = nominal_rate * config.get('student_rate_factor', 1.)
        if any(row['rate'] == rate for row in tuning['candidates']):
            continue
        if not have_time():
            persist()
            return False
        candidate = Schedule(original, config | {'inner_rate': rate, 'target_mode': 'frozen'})
        metrics, _ = panel(candidate, config | {'trace_every': 0}, dataset, calibration_ids, comm, device, seed_base=30_000_000)
        tuning['candidates'].append(dict(rate=rate, **metrics))
        persist()
    controls = [('real_subset_initial_rates', Schedule(original, config | {
                    'inner_rate': state.get('initial_inner_rate', config['inner_rate'])})),
                ('real_subset_learned_rates', Schedule(original, config)),
                ('initial_topology_learned_rates', Schedule(torch.load(folder / 'initial.pt', weights_only=False)['graphs'], config))]
    finite = [r for r in tuning['candidates'] if r['mean'] is not None]
    if finite:
        tuning['chosen_rate'] = min(finite, key=lambda r: r['mean'])['rate']
        controls.append(('real_subset_tuned_rates', Schedule(original, config | {'inner_rate': tuning['chosen_rate'], 'target_mode': 'frozen'})))
    learned_rates = torch.tensor([step['rate'] for step in artifact['schedule']], dtype=torch.float64)
    for name, control in controls:
        if name in report:
            continue
        if not have_time():
            return False
        if 'learned_rates' in name:
            probability = (learned_rates - control.rate_min) / (control.rate_max - control.rate_min)
            with torch.no_grad():
                control.rate_logits.copy_(torch.logit(probability))
        result, _ = panel(control, config, dataset, ids, comm, device, seed_base=20_000_000)
        report[name] = result
        persist()
    if config.get('initialization') == 'fixed' and best is not None:
        # Fixed-initialization success is not a claim about fresh initializations.
        for name, control in [('best_distilled_unseen', best)] + [
                ('real_subset_tuned_unseen', c) for n, c in controls if n == 'real_subset_tuned_rates']:
            if name not in report:
                if not have_time():
                    return False
                report[name], _ = panel(control, config | {'initialization': 'random'}, dataset, ids, comm, device, seed_base=20_000_000)
                persist()
    baseline = 'pretrained_baseline' if config.get('initialization') == 'pretrained' else 'untrained'
    if baseline not in report:
        if not have_time():
            return False
        report[baseline], _ = panel(synthetic, config, dataset, ids, comm, device, seed_base=20_000_000, trained=False)
    report.update({'validation_ids': ids.tolist(), 'test_accessed': False, 'case': config['case'], 'model': config['model'],
                   'outer_updates': state['iteration'], 'initialization': config.get('initialization', 'random'),
                   'target_mode': synthetic.target_mode, 'student_updates': synthetic.steps * synthetic.passes,
                   'distinct_real_training_examples': len(state['seen_ids']),
                   'distinct_real_examples_at_best': len(artifact['training_state']['seen_ids']),
                   'synthetic_count': config['synthetic_count'],
                   'bus_limits': config.get('bus_range', bus_limits(config['case'])),
                   'batches': batch_report(artifact['physical_solved_graphs'], config['batches']).to_dict(orient='records')})
    if config.get('initialization') == 'pretrained':
        report['pretrained'] = config['pretrained']
    if config.get('common_evaluation', False):
        from distillation.synthetic.evaluation import compare_export
        common = compare_export(folder, artifact, original, config, dataset, split, comm, device, deadline)
        if common is None:
            persist()
            return False
        report['common_evaluation'] = common
    if comm.rank == 0:
        write_json(folder / 'evaluation.json', report)
    persist()
    return True


def default_config(case, model):
    return common_config(case) | dict(model=model, model_options=model_options(model, width=128, layers=3),
        passes=10, target_strength=0., inner_rate=1e-6, rate_min=1e-10, rate_max=1e-2,
        outer_rate=.03, outer_target_rate=.003, topology_every=0, common_evaluation=True,
        checkpoint_inner=False, calibrate_inner_rate=False, inner_rate_cap=None, outer_input_rate=None,
        real_control_rates=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--mode', choices=('train', 'smoke'), default='train')
    parser.add_argument('--allocation-seconds', type=float, default=3100)
    args = parser.parse_args()
    
    from mpi4py import MPI
    
    # setup world
    world = MPI.COMM_WORLD
    hosts = world.allgather(socket.gethostname())
    unique = list(dict.fromkeys(hosts))
    node = unique.index(socket.gethostname())
    comm = world.Split(node, world.rank)
    
    config_file = json.loads(args.manifest.read_text())
    if args.mode == 'train' and len(config_file['experiments']) != len(unique):
        raise ValueError('Allocate exactly one node per manifest experiment')
    if args.mode == 'smoke' and len(unique) != 1:
        raise ValueError('Sequential qualification uses exactly one node')
    if comm.size != 12:
        raise RuntimeError(f'Expected 12 XPUs per node, got {comm.size}')
    
    # setup torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    torch.xpu.set_device(comm.rank)
    device = torch.device(f'xpu:{comm.rank}')
    
    assignments = comm.allgather({'rank': comm.rank, 'host': socket.gethostname(), 'device': str(device),
                                   'uuid': str(torch.xpu.get_device_properties(device).uuid)})
    if len({x['uuid'] for x in assignments}) != 12:
        raise RuntimeError('XPUs are not distinct')
    experiments = config_file['experiments'] if args.mode == 'smoke' else [config_file['experiments'][node]]
    
    # loop experiments
    for experiment in experiments:
        config = experiment['config']
        folder = Path(experiment['output'])
        if comm.rank == 0:
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / 'assignments.json', assignments)
            sources = [Path(__file__), Path(__file__).with_name('mvp.py'), Path(__file__).with_name('data.py'),
                       Path(__file__).with_name('topology.py'),
                       ROOT / 'distillation/conversion/matrix_form.py',
                       ROOT / 'experiments/lumina_bench/models.py',
                       ROOT / 'lumina/model/opf/homo_model.py', ROOT / 'lumina/model/opf/hetero_model.py']
            sources += sorted((ROOT / 'distillation/synthetic').glob('*.py'))
            write_json(folder / f"provenance_{os.environ.get('PBS_JOBID', 'manual')}.json", {
                'torch': str(torch.__version__), 'numpy': np.__version__, 'world_size': world.size,
                'node_world_size': comm.size, 'mpi': MPI.Get_library_version(),
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                'method': 'full second-order Wang DD; mean over real minibatches; solver-gated topology search',
                'initialization': config.get('initialization', 'random')})
        
        # Only node leader locks; collective communication remains within its node.
        lock = claim(folder / 'run.lock') if comm.rank == 0 else None
        acquired = lock.__enter__() if lock else None
        if not comm.bcast(acquired, root=0):
            raise RuntimeError(f'Experiment already running: {folder}')
        try:
            result = run(config, folder, comm, device, allocation_seconds=args.allocation_seconds,
                         qualify=args.mode == 'smoke', stop_after_updates=1 if args.mode == 'smoke' else None)
            if args.mode == 'smoke' and result['status'] == 'RESUMABLE':
                result = run(config, folder, comm, device, allocation_seconds=args.allocation_seconds)
            if comm.rank == 0:
                log(folder / 'history.jsonl', {'stage': 'exit', 'status': result['status'], 'iteration': result['iteration']})
        except Exception:
            if comm.rank == 0:
                (folder / 'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            if lock:
                lock.__exit__(None, None, None)
    world.Barrier()

if __name__ == '__main__':
    main()