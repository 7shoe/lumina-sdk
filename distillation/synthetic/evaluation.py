"""Ordinary-SGD evaluation and train-only calibration, shared by DD/DCGM/HMDC."""
import math
import json
import time

import numpy as np
import torch

from distillation.synthetic.learning import train_student
from distillation.conversion.matrix_form import mat2opf
from distillation.synthetic.inputs import SyntheticInputs
from distillation.synthetic.learning import initialization_seed, model_for, model_from_export
from distillation.synthetic.learning import pretrained_config
from distillation.synthetic.runtime import collectively, graph_batch, sampler_ids, score_summary, save
from experiments.lumina_bench.metrics import sample_metrics
from distillation.training.T1_single_topo.common import write_json
from distillation.training.T1_single_topo.common import file_hash
from distillation.experiments.staging import case_dataset as CaseDataset

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


def panel(synthetic, config, rate, dataset, ids, comm, device, *, seed_base=10_000_000, trained=True,
          evaluate_fn=None, model_fn=None, retention=False):
    evaluate_fn = evaluate_fn or evaluate
    model_fn = model_fn or model_for
    config = collectively(comm, lambda: pretrained_config(config))
    seed = initialization_seed(config, seed_base + comm.rank)
    model = collectively(comm, lambda: model_fn(mat2opf(synthetic.anchors[0]), config, seed, device))
    initial = cpu_state(model)
    error, fatal, trace = '', False, []
    steps = math.ceil(len(synthetic.anchors) / config['batch_size']) * config['evaluation_epochs'] if trained else 0
    every = config.get('trace_every', 0)
    started, probe_seconds = time.monotonic(), 0.
    def observe(step, presentations):
        nonlocal probe_seconds
        if step % every and step != steps:
            return
        target = next(model.parameters()).device
        if target.type == 'xpu': torch.xpu.synchronize(target)
        elif target.type == 'cuda': torch.cuda.synchronize(target)
        seconds = time.monotonic() - started - probe_seconds
        tick = time.monotonic()
        metric = evaluate_fn(model, dataset, ids[:config.get('trace_samples', 32)], config['eval_batch'])
        trace.append(dict(step=step, graph_presentations=presentations,
                          real_mse=metric['graph_balanced_mse'], student_seconds=seconds))
        probe_seconds += time.monotonic() - tick
    try:
        if trained:
            train_student(model, torch.optim.SGD(model.parameters(), lr=rate), synthetic,
                          epochs=config['evaluation_epochs'], batch_size=config['batch_size'],
                          seed=seed_base + comm.rank, observe=observe if every else None)
        metrics = evaluate_fn(model, dataset, ids, config['eval_batch'])
    except Exception as exc:
        metrics, error = {}, str(exc)
        fatal = not (isinstance(exc, FloatingPointError) or
                     isinstance(exc, ValueError) and str(exc).startswith('Non-finite'))
    sources = {}
    if retention and config.get('initialization') == 'pretrained':
        def source_metrics():
            result = {}
            for case, checksum in config.get('retention_data', {}).items():
                source = CaseDataset(config['data_root'], case)
                if file_hash(source.folder/'manifest.json') != checksum:
                    raise ValueError(f'Source-case data changed: {case}')
                offset, count = config['validation_samples'], config['retention_samples']
                selected = source.splits(config['split_seed'])['val'][offset:offset+count]
                if len(selected) != count: raise ValueError('Insufficient source-case validation examples')
                try:
                    values, failure = evaluate_fn(model, source, selected, config['eval_batch']), ''
                except FloatingPointError as exc:
                    values, failure = {}, str(exc)
                result[case] = dict(metrics=values, error=failure, validation_ids=selected.tolist(),
                                    data_sha256=checksum, test_accessed=False)
            return result
        sources = collectively(comm, source_metrics)
    records = comm.allgather(dict(metrics=metrics, error=error, fatal=fatal, seed=seed,
                                 training_seed=seed_base + comm.rank, learning_curve=trace,
                                 source_retention=sources))
    if any(r['fatal'] for r in records):
        raise RuntimeError(f'Evaluation failed collectively: {records}')
    scores = [r['metrics'].get('graph_balanced_mse', float('inf')) for r in records]
    summary = score_summary(scores) | dict(per_initialization_metrics=records, real_examples=len(ids),
                                           student_epochs=config['evaluation_epochs'] if trained else 0,
                                           initialization=config.get('initialization', 'random'),
                                           unique_initializations=len({r['seed'] for r in records}),
                                           student_updates=steps, student_rate=rate,
                                           graph_presentations=len(synthetic.anchors)*config['evaluation_epochs'] if trained else 0)
    if config.get('initialization') == 'pretrained':
        summary['pretrained_sha256'] = config['pretrained']['sha256']
    if sources:
        summary['source_retention'] = {case: score_summary([
            row['source_retention'][case]['metrics'].get('graph_balanced_mse', float('inf')) for row in records]) |
            {k: v for k, v in info.items() if k not in ('metrics', 'error')} for case, info in sources.items()}
    student = dict(initial_state=initial, trained_state=cpu_state(model), seed=seed,
                   training_seed=seed_base + comm.rank,
                   model=config['model'], model_options=config['model_options'])
    return summary, student


def calibrate(original, config, dataset, population, comm, device, *, panel_fn=None):
    """Select one fixed student rate using training data and calibration-only seeds."""
    synthetic = SyntheticInputs(original, config['strength'])
    panel_fn = panel_fn or panel
    ids = sampler_ids(population, 0, 0, 1, config.get('calibration_samples', 128),
                      seed=config.get('calibration_seed', 91231))
    records = []
    for rate in config['student_rates']:
        tick = time.monotonic()
        samples = []
        for index in range(config.get('calibration_panels', 1)):
            result, _ = panel_fn(synthetic, config | {'trace_every': 0}, rate, dataset, ids, comm, device,
                              seed_base=config.get('calibration_model_seed', 300_000) + index * comm.size)
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
                       calibration_initializations=(1 if config.get('initialization') in ('fixed', 'pretrained')
                           else comm.size * config.get('calibration_panels', 1)),
                       calibration_trajectories=comm.size * config.get('calibration_panels', 1),
                       training_ids=ids, candidates=records,
                       note='Fixed thereafter; train-only selection on real seed graphs, common across protocols')


def controls(artifact, initial, original, config, *, last=None):
    """Physical solutions certify inputs; exports provide the actual teaching targets."""
    best = SyntheticInputs.from_export(artifact)
    result = dict(best=best, initial_synthetic=SyntheticInputs(initial, config['strength'], target_mode='frozen'),
                  random_real=SyntheticInputs(original, config['strength'], target_mode='frozen'))
    if last is not None:
        result['last'] = last
    baseline = 'pretrained_baseline' if config.get('initialization') == 'pretrained' else 'untrained'
    result[baseline] = result['random_real']
    if config.get('initialization') == 'fixed':
        result.update(best_unseen=best, random_real_unseen=result['random_real'])
    return result


def evaluate_controls(variants, config, rate, dataset, ids, comm, device, *, results=None,
                      persist=None, student_path=None, deadline=float('inf'), student_arm='best',
                      stop_after_arms=None):
    """Resume by evaluation arm. Every arm shares SGD budget, rate and paired seeds."""
    results = {} if results is None else results
    completed = 0
    for name, teaching in variants.items():
        if name in results:
            continue
        if stop_after_arms is not None and completed >= stop_after_arms:
            return results, False
        if not all(comm.allgather(deadline-time.monotonic() > config.get('evaluation_reserve_seconds', 90))):
            return results, False
        settings = config | {'initialization': 'random'} if name.endswith('_unseen') else config
        metrics, student = panel(teaching, settings, rate, dataset, ids, comm, device,
            seed_base=20_000_000, trained=name not in ('untrained', 'pretrained_baseline'), retention=True)
        if comm.rank == 0 and name == student_arm and student_path is not None:
            save(student_path, student)
        results[name] = metrics
        if persist is not None:
            persist(results)
        completed += 1
    return results, True


def fit_export(artifact, device='cpu', *, model=None, seed=None):
    """Replay a DCGM/HMDC export without source checkpoints or OPF solves."""
    student = artifact['model_checkpoints'][model] if 'model_checkpoints' in artifact else artifact['model_checkpoint']
    network = model_from_export(artifact, device, seed=student['seed'] if seed is None else seed, model=model)
    rate = artifact['student_rates'][model] if 'student_rates' in artifact else artifact['student_rate']
    data = SyntheticInputs.from_export(artifact)
    train_student(network, torch.optim.SGD(network.parameters(), lr=rate), data,
                  epochs=artifact['config']['evaluation_epochs'], batch_size=artifact['config']['batch_size'],
                  seed=student.get('training_seed', student['seed']) if seed is None else seed)
    return network


def compare_export(folder, artifact, original, config, dataset, split, comm, device, deadline=float('inf')):
    """DD's additional ordinary-SGD comparison; its learned curriculum stays primary."""
    path = folder/'common_evaluation.partial.json'
    report = json.loads(path.read_text()) if path.exists() else {}
    iteration = artifact['training_state']['best_iteration']
    if report and report['best_iteration'] != iteration:
        raise ValueError('Common evaluation belongs to a different best dataset')
    report['best_iteration'] = iteration
    def persist(results=None):
        if results is not None: report['results'] = results
        if comm.rank == 0: write_json(path, report)
        comm.Barrier()
    if 'calibration' not in report:
        if not all(comm.allgather(deadline-time.monotonic() > config.get('evaluation_reserve_seconds', 90))):
            return None
        excluded = artifact['training_state']['seed_ids']
        population = split['train'][~np.isin(split['train'], excluded)]
        calibrated = folder/'inner_calibration.json'
        if config.get('calibrate_inner_rate') and config['passes'] == config['evaluation_epochs'] and calibrated.exists():
            report['calibration'] = json.loads(calibrated.read_text())
        else:
            _, report['calibration'] = calibrate(original, config, dataset, population, comm, device)
        persist()
    ids = split['val'][config['validation_samples']:config['validation_samples']+config['final_samples']]
    initial = torch.load(folder/'initial.pt', weights_only=False, map_location='cpu')['graphs']
    last = torch.load(folder/'distilled.pt', weights_only=False, map_location='cpu')
    report['results'], complete = evaluate_controls(controls(artifact, initial, original, config,
        last=SyntheticInputs.from_export(last)), config,
        report['calibration']['chosen_rate'], dataset, ids, comm, device,
        results=report.get('results'), persist=persist, student_path=folder/'student_common.pt', deadline=deadline)
    if not complete: return None
    report.update(validation_ids=ids.tolist(), test_accessed=False, complete=True,
        target_mode=artifact['target_mode'], initialization=config.get('initialization','random'),
        synthetic_count=len(artifact['physical_solved_graphs']),
        student_updates=config['batches']*config['evaluation_epochs'],
        note='Ordinary shuffled SGD; DD learned rates and ordering are not used in this additional comparison')
    if comm.rank == 0: write_json(folder/'common_evaluation.json', report)
    persist()
    return report
