"""RAMBO*: concurrent proposals against frozen history, deterministic batch commits."""
import argparse
from concurrent.futures import as_completed
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch

from distillation.RAMBO import run as sequential
from distillation.RAMBO.data import attach_solution, network_record, prepare_template, write_matpower
from distillation.RAMBO.objective import distance
from distillation.experiments.staging import case_dataset as CaseDataset
from distillation.selection.generated import GeneratedDataset
from distillation.training.T1_single_topo.common import ROOT, claim, digest, file_hash, write_json
from distillation.training.T1_single_topo.state import Stop, save_torch
from .artifacts import DISPLAY, METHOD, PROTOCOL, export_dataset
from .workers import JuliaBackend, ProposalPool


def configuration(*, proposal_batch_size=32, bootstrap=None, method=METHOD, **updates):
    if method != METHOD:
        raise ValueError('RAMBO_para has a separate method identity; use its own output directory')
    if type(proposal_batch_size) is not int or proposal_batch_size < 1:
        raise ValueError('proposal_batch_size must be a positive integer')
    cfg = sequential.configuration(**updates)
    return cfg | dict(method=METHOD, proposal_batch_size=proposal_batch_size,
                      bootstrap=str(Path(bootstrap).expanduser().resolve()) if bootstrap else None)


def sources():
    files = list(Path(__file__).parent.glob('*.py'))
    files += [ROOT / name for name in (
        'distillation/RAMBO/config.yaml', 'distillation/training/T1_single_topo/common.py',
        'distillation/training/T1_single_topo/state.py', 'distillation/training/T1_single_topo/data.py',
        'distillation/experiments/staging.py', 'distillation/data/split.py')]
    return sequential.sources() | {str(p.relative_to(ROOT)): file_hash(p) for p in files}


def verified(template, point, config, network, library):
    for key in ('pd', 'qd'):
        value = np.asarray(point[key])
        if (not np.isfinite(value).all() or
                (value < np.asarray(network[key + '_lo']) - config['tolerance']).any() or
                (value > np.asarray(network[key + '_hi']) + config['tolerance']).any()):
            raise ValueError('Final loads outside configured bounds')
    graph, checks = attach_solution(template, point, config['tolerance'])
    if sequential.duplicate(point, library, network, config['duplicate_tolerance']):
        raise ValueError('Duplicate operating point')
    return graph, checks


def append_sample(folder, state, spec, graph, point, checks, **details):
    index = len(state['records'])
    identity = f'generated:{METHOD}:{digest(spec)[:16]}:{index:06d}'
    path = folder / 'samples' / f'{index:06d}.pt'
    save_torch(path, dict(id=identity, graph=graph, point=point, checks=checks))
    record = dict(id=identity, file=str(path.relative_to(folder)), sha256=file_hash(path),
                  input_sha256=digest({k: point[k] for k in ('pd', 'qd')}), checks=checks, **details)
    state['records'].append(record)
    state['library'].append(point)


def bootstrap_info(path, config, network, provenance):
    """Inspect an optional old library read-only; never resume it as RAMBO*."""
    path = Path(path)
    generated = GeneratedDataset(path)
    manifest = generated.manifest
    original = json.loads(path.with_name('config.json').read_text())
    if manifest['specification'] != digest(original):
        raise ValueError('Bootstrap configuration changed')
    if (manifest['method'] != 'rambo' or original['network_sha256'] != digest(network) or
            manifest['config']['case'] != config['case'] or
            manifest['config']['load_bounds'] != config['load_bounds'] or
            any(manifest['provenance'].get(k) != provenance.get(k)
                for k in ('data_identity', 'split_hashes', 'nominal_case_sha256'))):
        raise ValueError('Bootstrap must be sequential RAMBO with the same case, physics and split')
    if not 0 < len(generated) < config['count']:
        raise ValueError('Bootstrap must contain between 1 and count-1 accepted grids')
    return dict(manifest=str(path.resolve()), sha256=file_hash(path), count=len(generated),
                source_specification=manifest['specification'], method='RAMBO')


def protect_baseline(folder):
    # Reject legacy directories BEFORE opening even their lock file.
    folder = Path(folder).expanduser().resolve()
    for parent in (folder, *folder.parents):
        path = parent / 'config.json'
        if path.exists() and json.loads(path.read_text()).get('format') == 'lumina-rambo-v1':
            raise ValueError('Sequential RAMBO outputs are read-only; choose a separate RAMBO_para directory')
        path = parent / 'artifact.json'
        if path.exists() and json.loads(path.read_text()).get('method') == 'RAMBO':
            raise ValueError('Published RAMBO artifacts are read-only; choose a separate RAMBO_para directory')


def run(config, output, *, workers=32, stop=None, session_batches=0, dataset=None,
        backend_factory=JuliaBackend, bind=True):
    cfg, folder = configuration(**config), Path(output).expanduser().resolve()
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    protect_baseline(folder)
    if folder.exists() and not (folder / 'config.json').exists():
        if any(p.name != 'run.lock' for p in folder.iterdir()):
            raise FileExistsError('Use a fresh directory or an existing RAMBO_para checkpoint')
    with claim(folder / 'run.lock') as acquired:
        if not acquired:
            return dict(status='busy')
        return _run(cfg, folder, workers, stop or Stop(), session_batches, dataset, backend_factory, bind)


def _run(cfg, folder, workers, stop, session_batches, dataset, backend_factory, bind):
    started = time.monotonic()
    dataset = dataset or CaseDataset(cfg['data_root'], cfg['case'])
    case_path = Path(sequential.__file__).with_name('cases') / (cfg['case'] + '.m')
    template, provenance = prepare_template(dataset, case_path, cfg['raw_root'])
    network = network_record(template, cfg['load_bounds'])
    spec_path, state_path = folder / 'config.json', folder / 'state.json'
    old = json.loads(spec_path.read_text()) if spec_path.exists() else None
    bootstrap = old.get('bootstrap') if old else (
        bootstrap_info(cfg['bootstrap'], cfg, network, provenance) if cfg['bootstrap'] else None)
    spec = dict(format='lumina-rambo-para-v1', method=METHOD, protocol=PROTOCOL, config=cfg,
                provenance=provenance, network_sha256=digest(network), sources=sources(), bootstrap=bootstrap)
    if old is not None and old != spec:
        raise ValueError('RAMBO_para configuration, code or data changed; use a new directory')
    if old is None:
        write_json(spec_path, spec)
    write_matpower(template, folder / 'case.m')
    write_json(folder / 'network.json', network)
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['specification'] != digest(spec):
            raise ValueError('Checkpoint identity changed')
        GeneratedDataset(folder / 'dataset.json')
        for record in state['records']:
            if file_hash(folder / record['file']) != record['sha256']:
                raise ValueError('Accepted sample changed')
    else:
        state = dict(specification=digest(spec), library=[], records=[], attempts=0, rejections=0,
                     batch=None, batches=0, wall_seconds=0.)
        if bootstrap:
            if file_hash(bootstrap['manifest']) != bootstrap['sha256']:
                raise ValueError('Bootstrap manifest changed before import')
            original = GeneratedDataset(bootstrap['manifest'])
            for record in original.records:
                value = torch.load(original.path.parent / record['file'], map_location='cpu', weights_only=False)
                point = value['point']
                graph, checks = verified(template, point, cfg, network, state['library'])
                append_sample(folder, state, spec, graph, point, checks,
                              origin=dict(manifest_sha256=bootstrap['sha256'], sample_id=record['id']),
                              solver_status=record['solver_status'], attempt=None)
    if (folder / 'complete.json').exists():
        complete = json.loads((folder / 'complete.json').read_text())
        if complete['dataset_sha256'] != file_hash(folder / 'dataset.json'):
            raise ValueError('Completed manifest changed')
        export_dataset(folder)
        return complete
    elapsed = state['wall_seconds']

    def checkpoint(status, **extra):
        state['wall_seconds'] = elapsed + time.monotonic() - started
        write_json(state_path, state)
        write_json(folder / 'dataset.json', dict(format='lumina-generated-v1', method=METHOD, display_name=DISPLAY,
            role='training', specification=digest(spec), config=cfg, provenance=provenance, bootstrap=bootstrap,
            records=state['records'], accepted=len(state['records']), test_used=False))
        summary = dict(status=status, method=METHOD, requested=cfg['count'], accepted=len(state['records']),
            attempts=state['attempts'], rejections=state['rejections'], batches=state['batches'],
            wall_seconds=state['wall_seconds'], test_used=False, **extra)
        write_json(folder / 'status.json', summary)
        return summary

    checkpoint('running')
    completed_batches = 0
    with ProposalPool(folder, cfg, network, stop, workers, backend_factory=backend_factory, bind=bind) as pool:
        while len(state['records']) < cfg['count']:
            if stop() or session_batches and completed_batches >= session_batches:
                return checkpoint('paused')
            if state['batch'] is None:
                remaining = min(cfg['count'] - len(state['records']), cfg['max_attempts'] - state['attempts'])
                if remaining <= 0:
                    return checkpoint('exhausted')
                size = min(cfg['proposal_batch_size'] if state['library'] else 1, remaining)
                proposals = [sequential.new_attempt(cfg, network, state) for _ in range(size)]
                state['batch'] = dict(index=state['batches'], history_count=len(state['library']),
                                      history_sha256=digest(state['library']), proposals=proposals)
                checkpoint('running')
            batch = state['batch']
            history = deepcopy(state['library'][:batch['history_count']])
            if digest(history) != batch['history_sha256']:
                raise ValueError('Frozen batch history changed')
            futures = {pool.executor.submit(pool.proposal, pending, history): pending['index']
                       for pending in batch['proposals']}
            results = {}
            try:
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
            except Exception as error:
                checkpoint('failed', error=repr(error))
                raise
            # Commit the whole batch only after every proposal has an outcome.
            # Paused/failed batches resume their cached stages against the SAME history.
            unfinished = [r for r in results.values() if r['status'] in ('paused', 'error')]
            if unfinished:
                return checkpoint('failed' if any(r['status'] == 'error' for r in unfinished) else 'paused',
                                  pending_results=unfinished)
            decisions = []
            for pending in batch['proposals']:
                candidate = results[pending['index']]
                decision = dict(attempt=pending['index'], status=candidate['status'])
                if candidate['status'] == 'ready':
                    point, relaxed = candidate['result']['point'], candidate['relaxed']
                    try:
                        graph, checks = verified(template, point, cfg, network, state['library'])
                    except (ValueError, FloatingPointError) as error:
                        decision.update(status='rejected', reason=str(error))
                    else:
                        details = dict(attempt=pending['index'], batch=batch['index'],
                                       history_count=batch['history_count'],
                                       solver_status=candidate['result']['status'])
                        if relaxed is not None:
                            details.update(distance_relaxed=relaxed['objective'],
                                           distance_final=distance(point, history, pending['subsets'], cfg['log_delta']))
                        append_sample(folder, state, spec, graph, point, checks, **details)
                        decision['status'] = 'accepted'
                else:
                    decision.update(reason=candidate['reason'], stage=candidate['stage'])
                state['rejections'] += int(decision['status'] == 'rejected')
                decisions.append(decision)
            write_json(folder / 'batches' / f'{batch["index"]:06d}.json', batch | dict(decisions=decisions))
            state['batch'] = None
            state['batches'] += 1
            completed_batches += 1
            summary = checkpoint('running')
            print(json.dumps(summary), flush=True)
    summary = checkpoint('complete')
    export_dataset(folder)
    summary['dataset_sha256'] = file_hash(folder / 'dataset.json')
    write_json(folder / 'complete.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, help='JSON configuration frozen by RAMBO_para.launch')
    parser.add_argument('--case')
    parser.add_argument('--count', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--proposal-batch-size', type=int)
    parser.add_argument('--bootstrap', type=Path, help='Read-only sequential RAMBO dataset.json')
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=3300)
    parser.add_argument('--session-batches', type=int, default=0)
    parser.add_argument('--no-bind', action='store_true', help='Disable physical-core pinning (tests/other platforms)')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text()) if args.config else {}
    cfg.update({k: getattr(args, k) for k in ('case', 'count', 'seed', 'proposal_batch_size', 'bootstrap')
                if getattr(args, k) is not None})
    torch.set_num_threads(1)
    with Stop(args.seconds) as stop:
        result = run(configuration(**cfg), args.output, workers=args.workers, stop=stop,
                     session_batches=args.session_batches, bind=not args.no_bind)
    print(json.dumps(result), flush=True)
    if result['status'] in ('failed', 'exhausted'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
