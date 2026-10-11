"""Resumable single-architecture DCGM experiments; MPI averages data gradients.

The small algorithm lives in dcgm.py. This file handles training-only rate
calibration, held-out evaluation, solver workers, checkpoints and provenance.
"""
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import time

import numpy as np
import torch

from distillation.DC.dcgm import (SyntheticInputs,
                                  data_gradients,
                                  feasible_step,
                                  train_student)
from distillation.conversion.matrix_form import mat2opf

from distillation.synthetic import describe, teaching_targets
from distillation.synthetic.experiments import initialize
from distillation.synthetic.learning import model_for, mse
from distillation.synthetic.runtime import (averaged_gradients,
                                            collective_audit,
                                            graph_batch,
                                           log, sampler_ids,
                                           save,
                                           score_summary)
from distillation.training.T1_single_topo.common import write_json
from distillation.training.T1_single_topo.data import CaseDataset
from experiments.lumina_bench.metrics import sample_metrics

FORMAT = 'dcgm-opf-v1'

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def evaluate(model, dataset, ids, batch_size):
    """Graph-balanced MSE plus physical residuals on every requested real graph."""
    device = next(model.parameters()).device
    model.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for start in range(0, len(ids), batch_size):
            batch = graph_batch(dataset, ids[start:start + batch_size]).to(device)
            prediction = model(batch)
            if any(not torch.isfinite(v).all() for v in prediction.values()):
                raise FloatingPointError('Nonfinite evaluation predictions')
            values = sample_metrics(prediction, batch)
            values['graph_balanced_mse'] = (values['sdk_mse'] / 2)
            for name in ('graph_balanced_mse', 'violation', 'balance_l2', 'thermal_l2', 'bounds_l2'):
                totals[name] = totals.get(name, 0.) + float(values[name].sum())
            count += batch.num_graphs
    if not all(math.isfinite(v) for v in totals.values()):
        raise FloatingPointError('Nonfinite aggregate evaluation metric')
    return {k: v / count for k, v in totals.items()}


def panel(synthetic, config, rate, dataset, ids, comm, device, *, seed_base=10_000_000, trained=True):
    model = model_for(mat2opf(synthetic.anchors[0]), config, seed_base + comm.rank, device)
    initial = cpu_state(model)
    error, fatal = '', False
    try:
        if trained:
            train_student(model, torch.optim.SGD(model.parameters(), lr=rate), synthetic,
                          epochs=config['evaluation_epochs'], batch_size=config['batch_size'],
                          seed=seed_base + comm.rank)
        metrics = evaluate(model, dataset, ids, config['eval_batch'])
    except Exception as exc:
        metrics, error = {}, str(exc)
        fatal = not (isinstance(exc, FloatingPointError) or
                     isinstance(exc, ValueError) and str(exc).startswith('Non-finite'))
    records = comm.allgather(dict(metrics=metrics, error=error, fatal=fatal, seed=seed_base + comm.rank))
    if any(r['fatal'] for r in records):
        raise RuntimeError(f'Evaluation failed collectively: {records}')
    scores = [r['metrics'].get('graph_balanced_mse', float('inf')) for r in records]
    summary = score_summary(scores) | dict(per_initialization_metrics=records, real_examples=len(ids),
                                           student_epochs=config['evaluation_epochs'] if trained else 0)
    student = dict(initial_state=initial, trained_state=cpu_state(model), seed=seed_base + comm.rank,
                   model=config['model'], model_options=config['model_options'])
    return summary, student


def calibrate(original, config, dataset, population, comm, device):
    """Select one fixed student rate using training data and calibration-only seeds."""
    synthetic = SyntheticInputs(original, config['strength'])
    ids = sampler_ids(population, 0, 0, 1, 128, seed=91231)
    records = []
    for rate in config['student_rates']:
        tick = time.monotonic()
        samples = []
        for index in range(config.get('calibration_panels', 1)):
            result, _ = panel(synthetic, config, rate, dataset, ids, comm, device,
                              seed_base=300_000 + index * comm.size)
            samples.extend(result['per_initialization_metrics'])
        result = score_summary([r['metrics'].get('graph_balanced_mse', float('inf')) for r in samples])
        result.update(per_initialization_metrics=samples, real_examples=len(ids),
                      student_epochs=config['evaluation_epochs'])
        records.append(dict(rate=rate, seconds=time.monotonic()-tick, **result))
        if comm.rank == 0:
            print(f"Calibration rate={rate:g} mean={result['mean']} nonfinite={result['nonfinite_initializations']} seconds={records[-1]['seconds']:.2f}", flush=True)
    finite = [r for r in records if r['mean'] is not None]
    if not finite:
        raise FloatingPointError('No student rate is finite for all calibration initializations')
    best = min(finite, key=lambda r: r['mean'])['rate']
    factor = config.get('student_rate_factor', 1.)
    if not 0 < factor <= 1:
        raise ValueError('student_rate_factor must be in (0, 1]')
    chosen = best * factor
    return chosen, dict(chosen_rate=chosen, best_calibration_rate=best, safety_factor=factor,
                       calibration_initializations=comm.size * config.get('calibration_panels', 1),
                       training_ids=ids, candidates=records,
                       note='Fixed thereafter; train-only selection on real seed graphs, common across protocols')


def qualify_device(graphs, config, real, comm, device):
    """Compare matching objective and every input-gradient family to CPU."""
    try:
        results = []
        for target in ('cpu', device):
            model = model_for(graphs[0], config, 500 + comm.rank, target)
            synthetic = SyntheticInputs(graphs[:3], config['strength'])
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
    synthetic = SyntheticInputs(payload['anchor_graphs'], payload['config']['strength'])
    synthetic.load_state_dict(payload['synthetic_state'])
    synthetic.labels = deepcopy(payload['labels'])
    optimizer = torch.optim.SGD(synthetic.parameters(), lr=payload['config']['data_rate'], momentum=.5)
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
                          target_semantics='fresh detached AC-OPF labels',
                          generation_statistics=[describe(g) for g in physical]))


def run(config, folder, comm, device, *, allocation_seconds=3300, qualify=False, stop_after_updates=None):
    from mpi4py import MPI
    folder, started = Path(folder), time.monotonic()
    error, payload = None, None
    if comm.rank == 0:
        if (folder/'latest.pt').exists():
            payload = torch.load(folder/'latest.pt', weights_only=False, map_location='cpu')
            if payload['config'] != config: error = 'Resume config differs; use a new directory'
        write_json(folder/'config.json', config)
    error = comm.bcast(error, root=0)
    if error: raise ValueError(error)
    payload = comm.bcast(payload, root=0)
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
        synthetic = SyntheticInputs(physical, config['strength'])
        optimizer = torch.optim.SGD(synthetic.parameters(), lr=config['data_rate'], momentum=.5)
        ok, physical, checks = collective_audit(synthetic, comm)
        if not ok: raise RuntimeError('Initial synthetic inputs failed solver recheck')
        synthetic.labels = teaching_targets(physical)
        state = dict(iteration=0, active_seconds=0., accepted_updates=0, rejected_updates=0,
                     seed_ids=seed_ids, seen_ids=set(), status='TRAINING', resume_count=0,
                     student_rate=None, best_validation=None, best_iteration=None)
    state['resume_count'] += 1
    state['status'] = 'TRAINING'
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
            seed = 1_000_000 + (iteration//config['restart_rounds'])*comm.size + comm.rank
            model = model_for(original[0], config, seed, device)
            student_optimizer = torch.optim.SGD(model.parameters(), lr=rate)
        ids = sampler_ids(population, iteration, comm.rank, comm.size, config['real_batch'])
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
                accepted=accepted, attempts=attempts, active_seconds=state['active_seconds'],
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
        final_evaluation(folder, synthetic, original, config, state, dataset, split, comm, device)
        state['status'] = 'COMPLETE'
    checkpoint(folder, synthetic, optimizer, config, state, physical, checks, model, student_optimizer, comm)
    return state


def final_evaluation(folder, synthetic, original, config, state, dataset, split, comm, device):
    ids = split['val'][config['validation_samples']:config['validation_samples']+config['final_samples']]
    best = torch.load(folder/'best.pt', weights_only=False, map_location='cpu')
    initial = torch.load(folder/'initial.pt', weights_only=False, map_location='cpu')['graphs']
    variants = dict(best=best['physical_solved_graphs'], last=None, initial_synthetic=initial, random_real=original,
                    untrained=original)
    results = {}
    for name, graphs in variants.items():
        teaching = synthetic if graphs is None else SyntheticInputs(graphs, config['strength'])
        results[name], student = panel(teaching, config, state['student_rate'], dataset, ids, comm, device,
                                       seed_base=20_000_000, trained=name != 'untrained')
        if comm.rank == 0 and name == 'best': save(folder/'student.pt', student)
    if comm.rank == 0:
        write_json(folder/'evaluation.json', dict(results=results, validation_ids=ids.tolist(), test_accessed=False,
            note='12 fresh student initializations; one synthesized set per condition; equal ordinary SGD budgets'))
