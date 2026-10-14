"""Resumable Algorithm-2 generation and a matched uniform-load control."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from distillation.selection.generated import GeneratedDataset
from distillation.training.T1_single_topo.common import (ROOT,
                                                         canonical_case,
                                                         claim,
                                                         digest,
                                                         file_hash,
                                                         write_json)
from distillation.experiments.staging import case_dataset as CaseDataset
from distillation.training.T1_single_topo.state import (Stop,
                                                        save_torch)

from .backend import JuliaBackend
from .data import (attach_solution,
                   export_dataset,
                   network_record,
                   prepare_template,
                   write_matpower)
from .objective import (distance,
                        sample_subsets)


def configuration(**updates):
    # apply overrides only to known settings
    config = yaml.safe_load(Path(__file__).with_name('config.yaml').read_text())
    if set(updates) - set(config):
        raise ValueError(f'Unknown RAMBO configuration: {set(updates) - set(config)}')
    config.update(updates)
    config['case'] = canonical_case(config['case'])
    
    if not (Path(__file__).with_name('cases') / (config['case'] + '.m')).is_file():
        raise ValueError('RAMBO requires a pinned nominal case (14, 30, 57, 118, 500)')
    if config['method'] not in ('rambo', 'uniform'):
        raise ValueError('method must be rambo or uniform')
    if config['warm_start'] not in ('opf', 'feasible'):
        raise ValueError('warm_start must be opf or feasible')
    if config['max_attempts'] is None and type(config['count']) is int:
        config['max_attempts'] = max(1000, 10 * config['count'])
    for key in ('count', 'history_size', 'variable_size', 'max_attempts'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['seed']) is not int or config['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    for key in ('epsilon', 'log_delta', 'tolerance', 'duplicate_tolerance', 'stage_seconds', 'request_seconds'):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    bounds = config['load_bounds']
    if len(bounds) != 2 or not all(np.isfinite(bounds)) or not 0 < bounds[0] < bounds[1]:
        raise ValueError('Use positive ordered nominal-load multiplier bounds')
    if config['request_seconds'] <= config['stage_seconds']:
        raise ValueError('Wall deadline must allow model construction in addition to solving')
    if config['max_attempts'] < config['count']:
        raise ValueError('Need at least one attempt per requested example')
    return config


def sources():
    folder = Path(__file__).parent
    files = list(folder.glob('*.py')) + list((folder / 'julia').glob('*.jl')) + list((folder / 'julia').glob('*.toml'))
    files += [ROOT / p for p in ('distillation/selection/generated.py', 'distillation/selection/pool.py',
              'distillation/conversion/matrix_form.py', 'distillation/conversion/convert.py',
              'distillation/solver/problem.py', 'distillation/solver/solution.py', 'distillation/solver/validate.py')]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def new_attempt(config, network, state):
    index = state['attempts']
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], index]))
    stage = 'opf' if config['method'] == 'uniform' or not state['library'] else 'feasible'
    pending = dict(index=index, seed=int(rng.integers(0, 2**31)), stage=stage)
    if stage == 'opf':
        for key in ('pd', 'qd'):
            pending[key] = rng.uniform(network[key + '_lo'], network[key + '_hi']).tolist()
    else:
        pending['subsets'] = sample_subsets(state['library'], network, rng,
                                           config['history_size'], config['variable_size'])
    state['attempts'] += 1
    return pending


def duplicate(point, library, network, tolerance):
    keys = ('pd', 'qd')
    width = np.concatenate([np.asarray(network[k + '_hi']) - network[k + '_lo'] for k in keys])
    current = np.concatenate([point[k] for k in keys])
    for old in library:
        difference = np.abs(current - np.concatenate([old[k] for k in keys]))
        if float(np.max(difference / np.maximum(width, 1e-12))) <= tolerance:
            return True
    return False


def run(config, output, *, stop=None, session_stages=0, backend_factory=JuliaBackend, dataset=None):
    config, output = configuration(**config), Path(output)
    with claim(output / 'run.lock') as acquired:
        if not acquired:
            return dict(status='busy')
        return _run(config, output, stop or Stop(), session_stages, backend_factory, dataset)


def _run(config, output, stop, session_stages, backend_factory, dataset):
    started = time.monotonic()
    dataset = dataset or CaseDataset(config['data_root'], config['case'])
    case_path = Path(__file__).with_name('cases') / (config['case'] + '.m')
    template, provenance = prepare_template(dataset, case_path, config['raw_root'])
    net = network_record(template, config['load_bounds'])
    spec = dict(format='lumina-rambo-v1', config=config, provenance=provenance,
                sources=sources(), network_sha256=digest(net))
    path = output / 'config.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('RAMBO configuration, code, nominal case or data changed; use a new directory')
    write_json(path, spec)
    if (output / 'complete.json').exists():
        GeneratedDataset(output / 'dataset.json')
        completed = json.loads((output / 'complete.json').read_text())
        if completed['dataset_sha256'] != file_hash(output / 'dataset.json'):
            raise ValueError('Completed dataset manifest changed')
        export_dataset(output)
        return completed
    casefile = output / 'case.m'
    write_matpower(template, casefile)
    write_json(output / 'network.json', net)
    state_path = output / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['specification'] != digest(spec):
            raise ValueError('Checkpoint identity mismatch')
        for record in state['records']:
            if file_hash(output / record['file']) != record['sha256']:
                raise ValueError('Accepted graph changed')
    else:
        state = dict(specification=digest(spec), attempts=0, solver_calls=0, library=[], records=[],
                     pending=None, runtime=None, wall_seconds=0., rejections=0, stages={}, calls=[])
    elapsed = state['wall_seconds']
    def checkpoint(status):
        state['wall_seconds'] = elapsed + time.monotonic() - started
        write_json(state_path, state)
        manifest = dict(format='lumina-generated-v1', role='training', method=config['method'],
                        specification=digest(spec), config=config, provenance=provenance,
                        records=state['records'], accepted=len(state['records']), test_used=False)
        write_json(output / 'dataset.json', manifest)
        summary = dict(status=status, requested=config['count'], accepted=len(state['records']),
                       attempts=state['attempts'], rejections=state['rejections'], solver_calls=state['solver_calls'],
                       stages=state['stages'], wall_seconds=state['wall_seconds'], test_used=False)
        write_json(output / 'status.json', summary)
        return summary
    checkpoint('running')
    completed_stages = 0
    with backend_factory(output / 'backend.log') as backend:
        if state['runtime'] is not None and state['runtime'] != backend.versions:
            raise ValueError('Julia solver environment changed')
        state['runtime'] = backend.versions
        while len(state['records']) < config['count']:
            # Leave enough time for one atomic solver stage, including compilation.
            if (stop() or time.monotonic() + config['request_seconds'] + 5 >= stop.deadline or
                    session_stages and completed_stages >= session_stages):
                return checkpoint('paused')
            if state['pending'] is None:
                if state['attempts'] >= config['max_attempts']:
                    return checkpoint('exhausted')
                state['pending'] = new_attempt(config, net, state)
                checkpoint('running')
            pending = state['pending']
            stage = pending['stage']
            request = dict(network=net, stage=stage, seed=pending['seed'], history=state['library'],
                           seconds=config['stage_seconds'], epsilon=config['epsilon'],
                           log_delta=config['log_delta'], tolerance=config['tolerance'],
                           casefile=str(casefile.resolve()))
            request.update({k: pending[k] for k in ('pd', 'qd', 'subsets', 'warm') if k in pending})
            result_path = output / 'attempts' / f'{pending["index"]:06d}_{stage}.json'
            if result_path.exists():
                saved = json.loads(result_path.read_text())
                if saved['request_sha256'] != digest(request):
                    raise ValueError('Previously completed stage has different inputs')
                result = saved['result']
            else:
                # Count attempted calls before launch, including interrupted or
                # timed-out work. A completed result recovered after a crash is
                # processed without charging for an additional solver call.
                state['solver_calls'] += 1
                state['stages'][stage] = state['stages'].get(stage, 0) + 1
                state['calls'].append(dict(stage=stage, attempt=pending['index'], status='started'))
                checkpoint('running')
                try:
                    result = backend.request(request, timeout=config['request_seconds'])
                except (TimeoutError, RuntimeError, OSError) as error:
                    # The worker may be dead. Retain progress and permit same-command resume.
                    write_json(output / 'failure.json', dict(stage=stage, attempt=pending['index'], error=str(error)))
                    state['pending'] = None
                    state['rejections'] += 1
                    state['calls'][-1]['status'] = type(error).__name__
                    checkpoint('paused')
                    raise
                write_json(result_path, dict(request_sha256=digest(request), subsets=pending.get('subsets'), result=result))
            state['calls'][-1].update(status=result['status'], wall_seconds=result.get('wall_seconds'))
            completed_stages += 1
            if not result.get('success'):
                state['pending'] = None
                state['rejections'] += 1
                checkpoint('running')
                if result.get('status') == 'ERROR':
                    raise RuntimeError(result.get('error', 'Julia model error'))
                continue
            if stage == 'feasible':
                pending.update(stage='warm_opf' if config['warm_start'] == 'opf' else 'bilevel', warm=result)
            elif stage == 'warm_opf':
                pending.update(stage='bilevel', warm=result)
            elif stage == 'bilevel':
                pending.update(stage='opf', bilevel=result, pd=result['point']['pd'], qd=result['point']['qd'])
                pending.pop('warm', None)
            else:
                point = result['point']
                try:
                    for key in ('pd', 'qd'):
                        a = np.asarray(point[key])
                        if (a < np.asarray(net[key + '_lo']) - config['tolerance']).any() or (a > np.asarray(net[key + '_hi']) + config['tolerance']).any():
                            raise ValueError('Final loads outside configured bounds')
                    graph, checks = attach_solution(template, point, config['tolerance'])
                    input_hash = digest({k: point[k] for k in ('pd', 'qd')})
                    if duplicate(point, state['library'], net, config['duplicate_tolerance']):
                        raise ValueError('Duplicate operating point')
                except (ValueError, FloatingPointError) as error:
                    write_json(result_path.with_name(result_path.stem + '_rejected.json'), dict(error=str(error)))
                    state['pending'] = None
                    state['rejections'] += 1
                    checkpoint('running')
                    continue
                index = len(state['records'])
                identity = f'generated:{digest(spec)[:16]}:{index:06d}'
                sample_path = output / 'samples' / f'{index:06d}.pt'
                save_torch(sample_path, dict(id=identity, graph=graph, point=point, checks=checks))
                record = dict(id=identity, file=str(sample_path.relative_to(output)), sha256=file_hash(sample_path),
                              input_sha256=input_hash, attempt=pending['index'], checks=checks,
                              solver_status=result['status'])
                if 'bilevel' in pending:
                    relaxed = pending['bilevel']['point']
                    record['relaxed_verification'] = dict(cost_relative=abs(point['cost'] - relaxed['cost']) / max(1., abs(point['cost'])),
                        **{k: float(np.max(np.abs(np.asarray(point[k]) - relaxed[k]))) for k in ('pg', 'qg', 'vm', 'theta')})
                    record['distance_final'] = distance(point, state['library'], pending['subsets'], config['log_delta'])
                    record['distance_relaxed'] = pending['bilevel']['objective']
                state['records'].append(record)
                state['library'].append(point)  # Algorithm 2: only verified FINAL OPF points.
                state['pending'] = None
                print(json.dumps(dict(accepted=index + 1, attempts=state['attempts'], method=config['method'],
                                      balance=checks['balance'])), flush=True)
            checkpoint('running')
    summary = checkpoint('complete')
    summary['dataset_sha256'] = file_hash(output / 'dataset.json')
    export_dataset(output)
    write_json(output / 'complete.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--case')
    parser.add_argument('--method', choices=['rambo', 'uniform'])
    parser.add_argument('--count', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--paper-bounds', action='store_true')
    parser.add_argument('--seconds', type=float, default=3300)
    args = parser.parse_args()
    
    config = yaml.safe_load(args.config.read_text())
    config.update({k: getattr(args, k) for k in ('case', 'method', 'count', 'seed') if getattr(args, k) is not None})
    
    if args.paper_bounds:
        config['load_bounds'] = [.5, 2.]
    torch.set_num_threads(1)
    with Stop(args.seconds) as stop:
        result = run(config, args.output, stop=stop)
    print(json.dumps(result), flush=True)
    if result['status'] == 'exhausted':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
