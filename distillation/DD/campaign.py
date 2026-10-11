"""Resumable Wang-style OPF distillation, one independent experiment per MPI node.

Continuous meta-gradients average independent initializations/real minibatches.
Discrete, solver-gated topology proposals never enter autograd. Training splits
are fixed at seed 42. Validation and model-seed panels are held out from training.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
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
from distillation.synthetic.learning import benchmark_graph
from distillation.training.T1_single_topo.common import (atomic_file,
                                                         canonical_case,
                                                         claim,
                                                         write_json)
from distillation.training.T1_single_topo.data import CaseDataset
from experiments.lumina_bench.runner import build_model

# path management
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets')
FORMAT = 'wang-opf-campaign-v1'

def save(path, payload):
    with atomic_file(path) as stream:
        torch.save(payload, stream)


def log(path, event):
    event = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
    print(json.dumps(event, allow_nan=False), flush=True)


def model_options(kind, width=32, layers=2):
    return ({'hidden_channels': width, 'num_layers': layers, 'backend': 'gat'}
            if kind == 'HeteroGNN' else
            {'hidden_dim': width, 'num_layers': layers, 'dropout': 0.0})


def model_for(graph, config, seed, device):
    kind = config['model']
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        torch.manual_seed(seed)
        model = build_model({'model_type': kind, 'models': {kind: config['model_options']}},
                            benchmark_graph(graph), torch.device('cpu'))
    return model.to(device).eval()


class Schedule(SyntheticOPF):
    """Distinct K batches, E passes, and an independent learned rate per SGD step."""
    def __init__(self, graphs, config):
        super().__init__(graphs, steps=config['batches'], strength=config['strength'],
                         target_strength=config['target_strength'], initial_rate=config['inner_rate'],
                         rate_min=config['rate_min'], rate_max=config['rate_max'])
        self.passes = config['passes']
        self.rate_logits = torch.nn.Parameter(self.rate_logits.detach().repeat(self.passes))


def inner(model, synthetic, *, create_graph):
    device = next(model.parameters()).device
    theta = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    buffers = {name: p.detach().clone() for name, p in model.named_buffers()}
    # Reuse graph construction across passes, without detaching meta-gradients.
    batches = [(b.to(device), {k: t.to(device) for k, t in y.items()})
               for b, y in (synthetic.batch(k) for k in range(synthetic.steps))]
    rates = synthetic.rates.to(device)
    for step in range(synthetic.steps * synthetic.passes):
        batch, target = batches[step % synthetic.steps]
        loss = mse(functional_call(model, (theta, buffers), (batch,), strict=True), target, batch)
        grad = torch.autograd.grad(loss, tuple(theta.values()), create_graph=create_graph, allow_unused=True)
        theta = {key: p if g is None else p - rates[step] * g
                 for (key, p), g in zip(theta.items(), grad)}
        if not create_graph:
            theta = {key: p.detach().requires_grad_(True) for key, p in theta.items()}
    return theta, buffers


def outer_loss(model, synthetic, real, *, create_graph):
    theta, buffers = inner(model, synthetic, create_graph=create_graph)
    real = real.clone().to(next(model.parameters()).device)
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
    for attempt in range(12):
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
    raise FloatingPointError('No finite meta-gradient after 12 inner-rate backtracks')


def collective_audit(synthetic, comm):
    local = []
    with torch.no_grad():
        for i in range(comm.rank, len(synthetic.anchors), comm.size):
            matrix = synthetic.edits[i](synthetic.anchors[i], synthetic.strength)
            checks, solutions = audit_networks([matrix])
            row = checks.iloc[0].to_dict()
            row['sample'] = i
            graph = solutions[0].to_hetero(ACOPF(matrix)) if row['success'] else None
            local.append((i, row, graph))
    records = sorted(item for group in comm.allgather(local) for item in group)
    return all(row['success'] for _, row, _ in records), [g for _, _, g in records], [r for _, r, _ in records]


def optimizer_for(synthetic, config):
    return torch.optim.Adam(synthetic.parameters(), lr=config['outer_rate'])


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
    new_optimizer = optimizer_for(new, config)
    if optimizer is not None:
        old_named = dict(old.named_parameters())
        for name, p in new.named_parameters():
            if not replaced(name) and old_named[name] in optimizer.state:
                new_optimizer.state[p] = deepcopy(optimizer.state[old_named[name]])
    return new, new_optimizer


def evaluation(model, synthetic, dataset, ids, batch_size=128, *, trained=True):
    if trained:
        theta, buffers = inner(model, synthetic, create_graph=False)
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
            'std': float(np.std(scores)) if finite else None,
            'per_initialization': [x if math.isfinite(x) else None for x in scores],
            'nonfinite_initializations': sum(not math.isfinite(x) for x in scores)}


def panel(synthetic, config, dataset, ids, comm, device, seed_base=10_000_000, trained=True):
    model = model_for(mat2opf(synthetic.anchors[0]), config, seed_base + comm.rank, device)
    value, theta, buffers = evaluation(model, synthetic, dataset, ids, config['eval_batch'], trained=trained)
    scores = comm.allgather(value)
    record = score_summary(scores) | {
        'model_seeds': list(range(seed_base, seed_base + comm.size)), 'real_examples': len(ids)}
    student = {'initial_state': {k: v.detach().cpu() for k, v in model.state_dict().items()},
               'trained_state': theta | {k: v.detach().cpu() for k, v in buffers.items()},
               'seed': seed_base + comm.rank, 'model': config['model'], 'model_options': config['model_options']}
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
        for step in range(synthetic.steps * synthetic.passes):
            batch, teaching = synthetic.batch(step % synthetic.steps)
            schedule.append({'step': step, 'batch': step % synthetic.steps, 'inputs': batch.detach().cpu(),
                             'teaching_targets': {k: v.detach().cpu() for k, v in teaching.items()},
                             'rate': float(synthetic.rates[step])})
    save(folder / name, {'format': FORMAT, 'config': config, 'training_state': state,
                        'schedule': schedule, 'physical_solved_graphs': physical,
                        'solver_checks': checks, 'model_checkpoint': student,
                        'target_semantics': ('fresh detached AC-OPF labels' if config.get('fresh_solver_targets') else
                                             'learned teaching targets; physical solutions stored separately')})
    save(folder / 'student.pt', student)


def run(config, folder, comm, device, *, allocation_seconds=3100, qualify=False, stop_after_updates=None):
    """Train for a cumulative time budget; resume from the last atomic commit."""
    from mpi4py import MPI
    started = time.monotonic()
    folder = Path(folder)
    if comm.rank == 0:
        folder.mkdir(parents=True, exist_ok=True)
        write_json(folder / 'config.json', config)
    comm.Barrier()
    dataset = CaseDataset(config['data_root'], f"case{config['case']}")
    split = dataset.splits(42)
    payload = None
    if comm.rank == 0 and (folder / 'latest.pt').exists():
        payload = torch.load(folder / 'latest.pt', weights_only=False, map_location='cpu')
        if payload['config'] != config:
            raise ValueError('Resume configuration differs; use a new experiment directory')
    payload = comm.bcast(payload, root=0)
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
                min_buses=bus_limits(config['case'])[0], max_buses=bus_limits(config['case'])[1],
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
    population = split['train'][~np.isin(split['train'], state['seed_ids'])]
    val_ids = split['val'][:config['validation_samples']]
    if comm.rank == 0:
        write_json(folder / 'split.json', {'split_seed': 42, 'case': canonical_case(f"case{config['case']}"),
            'train_pool_size': len(population), 'excluded_seed_ids': state['seed_ids'],
            'validation_ids': val_ids.tolist(), 'test_accessed': False,
            'train_pool_sha256': hashlib.sha256(population.tobytes()).hexdigest(),
            'sampler': 'cyclic independently shuffled epochs, iteration/rank cursor, seed 20261010'})
    if qualify:
        from distillation.DD.mvp import check_derivatives
        sample = physical[:3]
        real_ids = sampler_ids(population, 0, comm.rank, comm.size, 4)
        real = graph_batch(dataset, real_ids)
        model = model_for(sample[0], config, 500 + comm.rank, torch.device('cpu'))
        try:
            result = check_derivatives(model, sample, real, {k: real[k].y for k in ('bus', 'generator')},
                                       {'steps': 3, 'initial_rate': config['inner_rate'],
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
        for count in (16, 64, 256):
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
            raise FloatingPointError('Initial schedule fails held-out initialization panel; lower the starting rate')
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
        ids = sampler_ids(population, iteration, comm.rank, comm.size, config['real_batch'])
        real = graph_batch(dataset, ids)
        model = model_for(physical[0], config, 1_000_000 + iteration * comm.size + comm.rank, device)
        reference, grad_norm, rate_backtracks = training_gradient(model, synthetic, optimizer, real, comm)
        before_state = deepcopy(synthetic.state_dict())
        before_optimizer = deepcopy(optimizer.state_dict())
        before_targets = deepcopy(synthetic.seed_targets)
        optimizer.step()
        proposal = deepcopy(synthetic.state_dict())
        accepted, attempts = False, []
        
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            synthetic.seed_targets = deepcopy(before_targets)
            synthetic.load_state_dict({k: v + fraction * (proposal[k] - v) for k, v in before_state.items()})
            value = float(outer_loss(model, synthetic, real, create_graph=False).detach())
            value = comm.allreduce(value, op=MPI.SUM) / comm.size
            trial = {'fraction': fraction, 'real_mse': value if math.isfinite(value) else None, 'feasible': None}
            if math.isfinite(value) and value <= reference:
                ok, proposed_physical, proposed_checks = collective_audit(synthetic, comm)
                trial['feasible'] = ok
                if ok and config.get('fresh_solver_targets'):
                    synthetic.seed_targets = [{k: g[k].y.detach().clone() for k in ('bus', 'generator')}
                                              for g in proposed_physical]
                    value = float(outer_loss(model, synthetic, real, create_graph=False).detach())
                    value = comm.allreduce(value, op=MPI.SUM) / comm.size
                    trial['relabelled_mse'] = value if math.isfinite(value) else None
                    ok = math.isfinite(value) and value <= reference
                if ok:
                    physical, checks = proposed_physical, proposed_checks
                    accepted = True
            attempts.append(trial)
            if accepted:
                break
        if not accepted:
            synthetic.load_state_dict(before_state)
            optimizer.load_state_dict(before_optimizer)
            synthetic.seed_targets = before_targets
        
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
                'rate_backtracks_for_new_initialization': rate_backtracks, 'topology': topology_event})
        do_eval = time.monotonic() - last_eval >= config['eval_seconds']
        do_eval = comm.bcast(do_eval if comm.rank == 0 else None, root=0)
        if do_eval:
            result, student = panel(synthetic, config, dataset, val_ids, comm, device)
            if result['mean'] is not None and result['mean'] < state['best_validation']:
                state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
                if comm.rank == 0:
                    export(folder, synthetic, config, state, physical, checks, student, name='best.pt')
            if comm.rank == 0:
                log(folder / 'history.jsonl', {'stage': 'validation', 'iteration': state['iteration'], 'metrics': result})
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
        final_evaluation(folder, synthetic, original, config, dataset, split, state, comm, device)
        state['status'] = 'COMPLETE'
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
        seed=83_000 + iteration, max_edits=2, attempts_per_graph=3)

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
    accepted = math.isfinite(new_loss) and new_loss < old_loss and min(diversity(physical_proposal, config['batches'])) >= 3
    if accepted:
        ok, new_physical, new_checks = collective_audit(proposed, comm)
        accepted = ok
    event = {'accepted': accepted,
             'graph': index,
             'before': old_loss,
             'after': new_loss if math.isfinite(new_loss) else None,
             'change': change}
    if accepted:
        return proposed, proposed_optimizer, new_physical, new_checks, event
    
    return synthetic, optimizer, physical, checks, event


def replay_export(model, artifact, dataset, ids, batch_size):
    theta = {k: p.detach().clone().requires_grad_(True) for k, p in model.named_parameters()}
    buffers = dict(model.named_buffers())
    device = next(model.parameters()).device
    for step in artifact['schedule']:
        batch = step['inputs'].clone().to(device)
        loss = mse(functional_call(model, (theta, buffers), (batch,), strict=True), step['teaching_targets'], batch)
        grad = torch.autograd.grad(loss, tuple(theta.values()), allow_unused=True)
        theta = {k: (p if g is None else p - step['rate'] * g).detach().requires_grad_(True)
                 for (k, p), g in zip(theta.items(), grad)}
    total = 0.0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = graph_batch(dataset, ids[start:start + batch_size]).to(device)
            total += float(mse(functional_call(model, (theta, buffers), (batch,), strict=True),
                               {k: batch[k].y for k in ('bus', 'generator')}, batch)) * batch.num_graphs
    return total / len(ids)


def final_evaluation(folder, synthetic, original, config, dataset, split, state, comm, device):
    # Fresh model seeds AND a disjoint validation subset; test remains untouched.
    ids = split['val'][config['validation_samples']:config['validation_samples'] + config['final_samples']]
    artifact = torch.load(folder / 'best.pt', weights_only=False, map_location='cpu')
    model = model_for(original[0], config, 20_000_000 + comm.rank, device)
    value = replay_export(model, artifact, dataset, ids, config['eval_batch'])
    values = comm.allgather(value)
    report = {'best_distilled': score_summary(values)}
    controls = [('real_subset_initial_rates', Schedule(original, config)),
                ('real_subset_learned_rates', Schedule(original, config)),
                ('initial_topology_learned_rates', Schedule(torch.load(folder / 'initial.pt', weights_only=False)['graphs'], config))]
    learned_rates = torch.tensor([step['rate'] for step in artifact['schedule']], dtype=torch.float64)
    
    # loop
    for name, control in controls:
        if 'learned_rates' in name:
            probability = (learned_rates - control.rate_min) / (control.rate_max - control.rate_min)
            with torch.no_grad():
                control.rate_logits.copy_(torch.logit(probability))
        result, _ = panel(control, config, dataset, ids, comm, device, seed_base=20_000_000)
        report[name] = result
    
    # build report
    report['untrained'], _ = panel(synthetic, config, dataset, ids, comm, device, seed_base=20_000_000, trained=False)
    report.update({'validation_ids': ids.tolist(), 'test_accessed': False, 'case': config['case'], 'model': config['model'],
                   'best_iteration': state['best_iteration'], 'outer_updates': state['iteration'],
                   'distinct_real_training_examples': len(state['seen_ids']),
                   'distinct_real_examples_at_best': len(artifact['training_state']['seen_ids']),
                   'synthetic_count': config['synthetic_count'],
                   'bus_limits': config.get('bus_range', bus_limits(config['case'])),
                   'batches': batch_report(artifact['physical_solved_graphs'], config['batches']).to_dict(orient='records')})
    if comm.rank == 0:
        write_json(folder / 'evaluation.json', report)


def default_config(case, model):
    batches, batch_size, passes = {'GAT': (3, 30, 1), 'GIN': (4, 25, 1), 'HeteroGNN': (5, 20, 1)}[model]
    return {'case': case, 'model': model, 'model_options': model_options(model),
            'batches': batches, 'batch_size': batch_size, 'passes': passes, 'synthetic_count': batches * batch_size,
            'strength': 0.02, 'target_strength': 0.02, 'inner_rate': 1e-6, 'rate_min': 1e-8, 'rate_max': 1e-3,
            'outer_rate': 0.03, 'data_seed': 42, 'split_seed': 42, 'real_batch': 64, 'eval_batch': 128,
            'validation_samples': 256, 'final_samples': 1024, 'training_seconds': 2400,
            'max_updates': 1000, 'topology_every': 20, 'checkpoint_seconds': 120, 'eval_seconds': 300,
            'reserve_seconds': 240, 'raw_root': str(DATA_ROOT / 'raw/OPFData/dataset_release_1'),
            'data_root': str(DATA_ROOT / 'processed/T1')}


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
                'method': 'full second-order Wang DD; mean over independent initializations; solver-gated topology search'})
        
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
