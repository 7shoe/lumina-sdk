"""Resumable single-architecture DCGM experiments; MPI averages data gradients.

The small algorithm lives in dcgm.py. This file handles training-only rate
calibration, held-out evaluation, solver workers, checkpoints and provenance.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from distillation.DC.dcgm import (SyntheticInputs,
                                  data_gradients,
                                  feasible_step,
                                  train_student)
from distillation.conversion.matrix_form import mat2opf

from distillation.synthetic import describe
from distillation.synthetic.experiments import initialize, common_config, validate_condensation
from distillation.synthetic import evaluation
from distillation.synthetic.inputs import optimizer_for, changes_summary
from distillation.synthetic.learning import model_for, model_options, pretrained_config
from distillation.synthetic.runtime import (averaged_gradients,
                                            collective_audit,
                                            collectively,
                                            graph_batch,
                                           log, sampler_ids,
                                           save,
                                           )
from distillation.training.T1_single_topo.common import write_json
from distillation.experiments.staging import case_dataset as CaseDataset

FORMAT = 'dcgm-opf-v2'

# Re-export these helpers for existing callers; implementation is shared with HMDC/DD.
cpu_state, evaluate = evaluation.cpu_state, evaluation.evaluate


def panel(*args, **kwargs):
    return evaluation.panel(*args, evaluate_fn=evaluate, model_fn=model_for, **kwargs)


def calibrate(*args, **kwargs):
    return evaluation.calibrate(*args, panel_fn=panel, **kwargs)


def default_config(case, model):
    return common_config(case) | dict(model=model, model_options=model_options(model, width=128, layers=3),
        data_rate=.1, restart_rounds=11, student_epochs=1)


def qualify_device(graphs, config, real, comm, device):
    """Compare matching objective and every input-gradient family to CPU."""
    try:
        results = []
        for target in ('cpu', device):
            model = model_for(graphs[0], config, 500 + comm.rank, target)
            synthetic = SyntheticInputs(graphs[:3], config['strength'], target_mode=config.get('target_mode', 'solver'))
            value, _ = data_gradients(model, synthetic, real)
            groups = {}
            for name, p in synthetic.named_parameters():
                groups.setdefault(name.split('.')[-1], []).append(p.grad.flatten().double().cpu())
            results.append((value, {k: torch.cat(v) for k,v in groups.items()}))
        (a, cpu), (b, xpu) = results
        rows = [dict(family=k, cpu_norm=float(v.norm()), xpu_norm=float(xpu[k].norm()),
                     error=float((v-xpu[k]).norm()), tolerance=1e-7 + .02*float(v.norm())) for k,v in cpu.items()]
        result = dict(rank=comm.rank, cpu_loss=a, xpu_loss=b, gradients=rows,
                      passed=abs(a-b) <= 1e-5 + .001*abs(a) and all(r['error'] <= r['tolerance'] for r in rows),
                      criterion='loss 1e-5 + 0.1%; each gradient family L2 1e-7 + 2%')
    except Exception as exc:
        result = dict(rank=comm.rank, passed=False, error=f'{type(exc).__name__}: {exc}')
    return comm.allgather(result)


def payload_for(synthetic, optimizer, config, state, physical, checks, students):
    return dict(format=FORMAT, config=config, state=state, anchor_graphs=[mat2opf(m) for m in synthetic.anchors],
                synthetic_state=synthetic.state_dict(), labels=synthetic.labels, optimizer=optimizer.state_dict(),
                physical_graphs=physical, solver_checks=checks, students=students)


def restore(payload):
    synthetic = SyntheticInputs(payload['anchor_graphs'], payload['config']['strength'],
                                target_mode=payload['config'].get('target_mode', 'solver'))
    synthetic.load_state_dict(payload['synthetic_state'])
    synthetic.labels = deepcopy(payload['labels'])
    optimizer = optimizer_for(synthetic, payload['config'])
    optimizer.load_state_dict(payload['optimizer'])
    return synthetic, optimizer


def checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm):
    local = (dict(model=cpu_state(model), optimizer=student_optimizer.state_dict()) if model is not None else None)
    students = comm.allgather(local)
    if comm.rank == 0:
        save(folder/'latest.pt', payload_for(synthetic, optimizer, config, state, physical, checks, students))
        write_json(folder/'status.json', {k:v for k,v in state.items() if k != 'seen_ids'} |
                   dict(distinct_real_training_examples=len(state['seen_ids'])))


def export(folder, name, synthetic, config, state, physical, checks, student):
    with torch.no_grad():
        inputs, labels = synthetic.batch(detached=True)
    save(folder/name, dict(format=FORMAT, config=config, training_state=state, inputs=inputs,
                          teaching_targets=labels, physical_solved_graphs=physical, solver_checks=checks,
                          model_checkpoint=student, student_rate=state['student_rate'],
                          **synthetic.snapshot(),
                          generation_statistics=[describe(g) for g in physical]))


def run(config, folder, comm, device, *, allocation_seconds=3300, qualify=False, stop_after_updates=None):
    from mpi4py import MPI
    validate_condensation(config)
    if type(config['student_epochs']) is not int or config['student_epochs'] <= 0:
        raise ValueError('student_epochs must be a positive integer')
    config = collectively(comm, lambda: pretrained_config(config, refresh=True))
    folder, started = Path(folder), time.monotonic()
    folder.mkdir(parents=True, exist_ok=True)
    def load_existing():
        if (folder/'config.json').exists():
            saved_config = json.loads((folder/'config.json').read_text())
            if saved_config != config:
                changed = [k for k in saved_config.keys() | config.keys() if saved_config.get(k) != config.get(k)]
                raise ValueError(f'Resume config or pretrained checkpoint differs ({changed}); use a new directory')
        if (folder/'latest.pt').exists():
            payload = torch.load(folder/'latest.pt', weights_only=False, map_location='cpu')
            if payload['config'] != config or len(payload['students']) != comm.size:
                raise ValueError('Resume requires identical configuration and replica count')
            return payload
    payload = collectively(comm, lambda: load_existing() if comm.rank == 0 else None)
    payload = comm.bcast(payload, root=0)
    if comm.rank == 0:
        write_json(folder/'config.json', config)
    if payload and payload['state']['status'] == 'COMPLETE': return payload['state']
    dataset = CaseDataset(config['data_root'], f"case{config['case']}")
    split = dataset.splits(42)
    model, student_optimizer = None, None
    if payload:
        synthetic, optimizer = restore(payload)
        state, physical, checks = payload['state'], payload['physical_graphs'], payload['solver_checks']
        if (folder/'best.pt').exists():
            best = torch.load(folder/'best.pt', weights_only=False, map_location='cpu')['training_state']
            if best['best_validation'] < state['best_validation']:
                state['best_validation'], state['best_iteration'] = best['best_validation'], best['best_iteration']
        original = torch.load(folder/'original.pt', weights_only=False, map_location='cpu')['graphs']
        saved = payload['students'][comm.rank]
        if saved:
            model = model_for(original[0], config, 1, device)
            model.load_state_dict(saved['model'])
            student_optimizer = torch.optim.SGD(model.parameters(), lr=state['student_rate'])
            student_optimizer.load_state_dict(saved['optimizer'])
    else:
        original, seed_ids, physical, _ = initialize(config, folder, comm)
        synthetic = SyntheticInputs(physical, config['strength'], target_mode=config.get('target_mode', 'solver'))
        optimizer = optimizer_for(synthetic, config)
        ok, physical, checks = collective_audit(synthetic, comm)
        if not ok: raise RuntimeError('Initial synthetic inputs failed solver recheck')
        synthetic.refresh_labels(physical)
        state = dict(iteration=0, active_seconds=0., accepted_updates=0, rejected_updates=0,
                     seed_ids=seed_ids, seen_ids=set(), status='TRAINING', resume_count=0,
                     student_rate=None, best_validation=None, best_iteration=None)
    state['resume_count'] += 1
    state['status'] = 'TRAINING'
    if not set(state['seed_ids']) <= set(split['train'].tolist()):
        raise ValueError('Initialization references must belong to the training split')
    if config['validation_samples'] + config['final_samples'] > len(split['val']):
        raise ValueError('Selection and final validation panels do not fit the split')
    population = split['train'][~np.isin(split['train'], state['seed_ids'])]
    val_ids = split['val'][:config['validation_samples']]
    if comm.rank == 0:
        write_json(folder/'split.json', dict(split_seed=42, train_pool_size=len(population),
                   excluded_seed_ids=state['seed_ids'], validation_ids=val_ids.tolist(), test_accessed=False,
                   train_pool_sha256=hashlib.sha256(population.tobytes()).hexdigest()))
    if qualify:
        real = graph_batch(dataset, sampler_ids(population, 0, comm.rank, comm.size, 4))
        checks_device = qualify_device(physical, config, real, comm, device)
        if comm.rank == 0: write_json(folder/'device_checks.json', checks_device)
        if not all(r['passed'] for r in checks_device): raise RuntimeError('DCGM CPU/XPU qualification failed')
    if state['student_rate'] is None:
        state['student_rate'], calibration = calibrate(original, config, dataset, population, comm, device)
        if comm.rank == 0: write_json(folder/'calibration.json', calibration)
    rate = state['student_rate']
    if state.get('evaluation_ready', False):
        finished = final_evaluation(folder, synthetic, original, config, state, dataset, split, comm, device,
                                    deadline=started+allocation_seconds)
        state['status'] = 'COMPLETE' if finished else 'FINALIZING'
        checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
        return state
    if state['best_validation'] is None:
        result, student = panel(synthetic, config, rate, dataset, val_ids, comm, device)
        if result['mean'] is None: raise FloatingPointError('Initial synthetic evaluation nonfinite')
        state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
        if comm.rank == 0:
            export(folder, 'best.pt', synthetic, config, state, physical, checks, student)
            log(folder/'history.jsonl', dict(stage='initial_validation', metrics=result, rate=rate))
    checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
    last_save = last_eval = time.monotonic()
    this_run = 0
    while state['active_seconds'] < config['training_seconds'] and state['iteration'] < config['max_updates']:
        expired = time.monotonic()-started >= allocation_seconds-config['reserve_seconds']
        if comm.bcast(expired if comm.rank == 0 else None, root=0): break
        if stop_after_updates is not None and this_run >= stop_after_updates: break
        tick, iteration = time.monotonic(), state['iteration']
        if model is None:
            seed = config.get('training_model_seed', 1_000_000) + (iteration//config['restart_rounds'])*comm.size + comm.rank
            model = model_for(original[0], config, seed, device)
            student_optimizer = torch.optim.SGD(model.parameters(), lr=rate)
        ids = sampler_ids(population, iteration, comm.rank, comm.size, config['real_batch'], seed=config.get('sampler_seed', 20261010))
        real = graph_batch(dataset, ids)
        optimizer.zero_grad(set_to_none=True)
        failure = None
        try: value, info = data_gradients(model, synthetic, real)
        except Exception as exc: failure = f'{type(exc).__name__}: {exc}'
        failures = comm.allgather(failure)
        if any(failures): raise RuntimeError(f'Matching gradient failed: {failures}')
        norm = averaged_gradients(synthetic, comm)
        accepted, proposed, audit, attempts = feasible_step(synthetic, optimizer,
                                                          audit=lambda s: collective_audit(s, comm))
        if accepted: physical, checks = proposed, audit
        state['accepted_updates' if accepted else 'rejected_updates'] += 1
        failure = None
        if (iteration+1) % config['restart_rounds']:
            try:
                train_student(model, student_optimizer, synthetic, epochs=config['student_epochs'],
                              batch_size=config['batch_size'], seed=4_000_000+iteration*comm.size+comm.rank)
            except Exception as exc: failure = f'{type(exc).__name__}: {exc}'
        else:
            model, student_optimizer = None, None  # Reference skips student SGD after the last matching round.
        failures = comm.allgather(failure)
        if any(failures): raise RuntimeError(f'Student training failed: {failures}')
        for group in comm.allgather(ids): state['seen_ids'].update(group)
        state['iteration'] += 1
        this_run += 1
        state['active_seconds'] += comm.allreduce(time.monotonic()-tick, op=MPI.MAX)
        rows = comm.allgather(dict(matching_loss=value, **info))
        if comm.rank == 0:
            log(folder/'history.jsonl', dict(stage='matching', iteration=state['iteration'],
                matching_loss=float(np.mean([r['matching_loss'] for r in rows])), data_gradient_norm=norm,
                changes=changes_summary(synthetic), accepted=accepted, attempts=attempts, active_seconds=state['active_seconds'],
                distinct_real_examples=len(state['seen_ids']), workers=rows))
        if comm.bcast(time.monotonic()-last_eval >= config['eval_seconds'] if comm.rank == 0 else None, root=0):
            result, student = panel(synthetic, config, rate, dataset, val_ids, comm, device)
            if result['mean'] is not None and result['mean'] < state['best_validation']:
                state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
                if comm.rank == 0: export(folder, 'best.pt', synthetic, config, state, physical, checks, student)
            if comm.rank == 0: log(folder/'history.jsonl', dict(stage='validation', iteration=state['iteration'], metrics=result))
            last_eval = time.monotonic()
        if comm.bcast(time.monotonic()-last_save >= config['checkpoint_seconds'] if comm.rank == 0 else None, root=0):
            checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
            last_save = time.monotonic()
    complete = state['active_seconds'] >= config['training_seconds'] or state['iteration'] >= config['max_updates']
    state['status'] = 'FINALIZING' if complete else 'RESUMABLE'
    checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
    result, student = panel(synthetic, config, rate, dataset, val_ids, comm, device)
    if result['mean'] is not None and result['mean'] < state['best_validation']:
        state['best_validation'], state['best_iteration'] = result['mean'], state['iteration']
        if comm.rank == 0: export(folder, 'best.pt', synthetic, config, state, physical, checks, student)
    if comm.rank == 0: export(folder, 'condensed.pt', synthetic, config, state, physical, checks, student)
    comm.Barrier()
    if complete:
        state['evaluation_ready'] = True  # Freeze selected exports across evaluation-only resumptions.
        checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
        finished = final_evaluation(folder, synthetic, original, config, state, dataset, split, comm, device,
                                    deadline=started+allocation_seconds)
        state['status'] = 'COMPLETE' if finished else 'FINALIZING'
    checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
    return state


def final_evaluation(folder, synthetic, original, config, state, dataset, split, comm, device, deadline=float('inf')):
    ids = split['val'][config['validation_samples']:config['validation_samples']+config['final_samples']]
    best = torch.load(folder/'best.pt', weights_only=False, map_location='cpu')
    initial = torch.load(folder/'initial.pt', weights_only=False, map_location='cpu')['graphs']
    path = folder/'evaluation.partial.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if previous and previous['best_iteration'] != best['training_state']['best_iteration']:
        raise ValueError('Partial evaluation belongs to a different best dataset')
    report = dict(results=previous.get('results', {}), validation_ids=ids.tolist(), test_accessed=False,
        best_iteration=best['training_state']['best_iteration'], initialization=config.get('initialization', 'random'),
        pretrained=config.get('pretrained'), target_mode=synthetic.target_mode,
        synthetic_count=len(synthetic.anchors), student_updates=config['batches']*config['evaluation_epochs'],
        matching_updates=state['iteration'], active_seconds=state['active_seconds'],
        distinct_real_training_examples=len(state['seen_ids']),
        distinct_real_examples_at_best=len(best['training_state']['seen_ids']),
        note='Paired starting weights and shuffle seeds; equal ordinary SGD budgets; unseen means new random weights')
    def persist(results):
        report['results'] = results
        if comm.rank == 0: write_json(path, report)
        comm.Barrier()
    report['results'], complete = evaluation.evaluate_controls(
        evaluation.controls(best, initial, original, config, last=synthetic), config, state['student_rate'],
        dataset, ids, comm, device, results=report['results'], persist=persist,
        student_path=folder/'student.pt', deadline=deadline)
    if complete and comm.rank == 0:
        write_json(folder/'evaluation.json', report | {'complete': True})
    return complete
