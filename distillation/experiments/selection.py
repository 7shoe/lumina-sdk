"""Separate BAS/uniform/demand-stratified campaigns; no synthesis registry changes."""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import yaml

from distillation.BAS import run as bas
from distillation.selection import Buckets, TrainingPool, ids_hash
from distillation.selection.data import load_descriptors
from distillation.selection.artifacts import publish, output_guard
from distillation.training.T1_single_topo.common import ROOT, canonical_case, claim, digest, file_hash, write_json
from distillation.training.T1_single_topo.data import CaseDataset
from distillation.training.T1_single_topo.state import Stop

FORMAT = 'lumina-selection-campaign-v1'
METHODS = ('bas', 'uniform', 'demand_stratified')
DEFAULT = Path(__file__).with_name('configs')/'selection.yaml'


def sources():
    """Freeze dependencies without copying historical outputs or auxiliary material."""
    paths = []
    for parent in ('lumina', 'experiments', 'distillation'):
        for folder, dirs, files in os.walk(ROOT/parent):
            dirs[:] = sorted(d for d in dirs if d not in {'aux', 'test', 'tests', 'runs', 'reports', '__pycache__', '.git'})
            paths.extend(Path(folder)/name for name in sorted(files) if name.endswith('.py'))
    paths.append(ROOT/'distillation/BAS/config.yaml')
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def prepare(path=DEFAULT):
    spec = yaml.safe_load(Path(path).read_text())
    allowed = {'schema_version', 'campaign', 'output', 'data_root', 'cases', 'methods', 'seed',
               'export_seed', 'counts', 'pool_size', 'panel_size', 'buckets', 'partition', 'bas'}
    if set(spec) != allowed or spec['schema_version'] != 1:
        raise ValueError('Use the selection.yaml schema')
    if not spec['campaign'] or Path(spec['campaign']).name != spec['campaign'] or spec['campaign'] in ('.', '..'):
        raise ValueError('Use a simple campaign name')
    if not spec['methods'] or len(set(spec['methods'])) != len(spec['methods']) or set(spec['methods'])-set(METHODS):
        raise ValueError('Choose bas, uniform and/or demand_stratified')
    counts = spec['counts']
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or n < 1 for n in counts):
        raise ValueError('Use unique positive export counts')
    if type(spec['pool_size']) is not int or spec['pool_size'] < max(counts):
        raise ValueError('Pool ceiling must accommodate every export size')
    for key in ('seed', 'export_seed'):
        if type(spec[key]) is not int or not 0 <= spec[key] < 2**32:
            raise ValueError('Seeds must be integers in [0, 2**32)')
    cases = [canonical_case(f'case{c}' if isinstance(c, int) else c) for c in spec['cases']]
    if not cases or len(set(cases)) != len(cases):
        raise ValueError('Choose distinct cases')
    root = Path(spec['output']).expanduser().resolve()/spec['campaign']
    output_guard(root)
    hashes = sources()
    rows = []
    for case in cases:
        config = bas.configuration(**(spec['bas'] | dict(data_root=spec['data_root'], case=case,
            split_seed=42, selection_seed=spec['seed'], max_selected=spec['pool_size'],
            panel_size=spec['panel_size'], buckets=spec['buckets'], partition=spec['partition'], method='bas')))
        data = Path(config['data_root'])/case/'manifest.json'
        for method in spec['methods']:
            identity = dict(method=method, config=config, counts=counts, export_seed=spec['export_seed'],
                            data_sha256=file_hash(data), sources=hashes)
            rows.append(dict(experiment_id='SELECTION-'+digest(identity)[:16], method=method,
                config=config, data_sha256=identity['data_sha256'], counts=counts, export_seed=spec['export_seed'],
                output=str(root/'runs'/case/method/f"seed{spec['seed']}")))
    manifest = dict(format=FORMAT, root=str(root), specification=spec, source_hashes=hashes, experiments=rows)
    path = root/'manifest.json'
    with claim(root/'prepare.lock') as acquired:
        if not acquired:
            raise RuntimeError('Selection preparation is already running')
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError('Selection campaign changed; use a new campaign name')
            return manifest
        for relative, checksum in hashes.items():
            destination = root/'source'/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/relative, destination)
            if file_hash(destination) != checksum:
                raise ValueError('Source changed during preparation')
        write_json(root/'source_hashes.json', hashes)
        write_json(path, manifest)
    return manifest


def run_experiment(row, *, device='cpu', stop=None, session_steps=0, dataset=None):
    """Resume BAS; controls sample directly and never train a selector."""
    folder = Path(row['output']); output_guard(folder)
    cfg = row['config']
    stop = stop or Stop()
    with claim(folder/'sidecar.lock') as acquired:
        if not acquired:
            return dict(status='BUSY')
        dataset = dataset or CaseDataset(cfg['data_root'], cfg['case'])
        if file_hash(dataset.folder/'manifest.json') != row['data_sha256']:
            raise ValueError('Source data manifest changed')
        splits = dataset.splits(42)
        if cfg['panel_size'] + cfg['max_selected'] > len(splits['train']):
            raise ValueError('Panel and pool exceed the training population')
        if row['method'] == 'bas':
            result = bas.run(cfg, folder/'selector', device=device, stop=stop,
                             session_steps=session_steps, dataset=dataset)
            if result['status'] != 'complete':
                write_json(folder/'status.json', result)
                return result
            native = json.loads((folder/'selector/selection.json').read_text())
            ids, panel = native['pool']['selected_ids'], native['pool']['reference_ids']
            initial = [i for event in native['pool']['history'] if event['reason'] == 'initial' for i in event['ids']]
            events = sum(e['reason'] != 'initial' for e in native['pool']['history'])
            label_access = dict(scoring=len(panel), acquired=len(ids), validation=min(
                cfg['validation_size'] or len(splits['val']), len(splits['val'])), test=0)
            provenance = dict(native_selection_sha256=file_hash(folder/'selector/selection.json'),
                              native_completion=result, config=cfg,
                              selection_outcome='active_acquisition' if events else 'initial_random_only')
            strata = None
        else:
            values, descriptor = load_descriptors(dataset, splits['train'], folder.parent/f'{cfg["case"]}_train_inputs.pt')
            partition = Buckets.fit(values, cfg['buckets'], cfg['partition'])
            membership = np.full(len(dataset), -1, dtype=np.int64)
            membership[splits['train']] = partition.assign(values)
            panel, _ = partition.sample(splits['train'], membership[splits['train']], np.ones(partition.count),
                                        cfg['panel_size'], np.random.default_rng(cfg['selection_seed']))
            eligible = TrainingPool(dataset, splits, panel).remaining_ids
            rng = np.random.default_rng(cfg['selection_seed'] + 1000)
            if row['method'] == 'uniform':
                ids = rng.choice(eligible, cfg['max_selected'], replace=False).tolist()
                strata = None
            elif row['method'] == 'demand_stratified':
                ids, _ = partition.sample(eligible, membership[eligible], np.ones(partition.count),
                                          cfg['max_selected'], rng)
                ids = ids.tolist(); strata = membership[ids]
            else:
                raise ValueError('Unknown selection method')
            panel = panel.tolist(); initial = []; events = 0
            label_access = dict(scoring=0, acquired=0, validation=0, test=0)
            provenance = dict(descriptor=descriptor, buckets=partition.state_dict(),
                selection_outcome='direct input-only selection',
                note='Reserved panel matches BAS eligibility; its labels are not read. Only exported labels are materialized.')
        selection = dict(complete=True, method=row['method'], selection_seed=cfg['selection_seed'],
            selector=dict(model=cfg['model'], seed=cfg['model_seed'], metric=cfg['metric'], loss=cfg['loss'],
                          model_options=cfg['model_options']) if row['method'] == 'bas' else None,
            selected_ids=ids, reference_ids=panel, initial_ids=initial, acquisition_events=events,
            label_access=label_access, data_identity=digest(dataset.manifest),
            split_hashes={k: ids_hash(v) for k,v in splits.items()},
            provenance=provenance | dict(experiment_id=row['experiment_id'], training_population=len(splits['train'])))
        summary = publish(dataset, selection, folder/'exports', counts=row['counts'], seed=row['export_seed'], strata=strata)
        write_json(folder/'status.json', summary)
        return summary


def catalog(manifest):
    rows = []
    for experiment in manifest['experiments']:
        path = Path(experiment['output'])/'status.json'
        status = json.loads(path.read_text()) if path.exists() else dict(status='PENDING')
        rows.append({k: experiment[k] for k in ('experiment_id','method','output')} |
                    dict(case=experiment['config']['case'], **status))
    write_json(Path(manifest['root'])/'catalog.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'catalog'))
    parser.add_argument('--config', type=Path, default=DEFAULT)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--index', type=int, help='One independent experiment, suitable for an external PBS launcher')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seconds', type=float, default=3300)
    args = parser.parse_args()
    if args.action == 'prepare':
        value = prepare(args.config)
        print(json.dumps(dict(root=value['root'], experiments=len(value['experiments']))))
        return
    if args.root is None:
        parser.error('--root is required')
    output_guard(args.root.resolve())
    manifest = json.loads((args.root/'manifest.json').read_text())
    if manifest.get('format') != FORMAT:
        raise ValueError('Expected a selection sidecar manifest')
    if args.action == 'run':
        for relative, expected in manifest['source_hashes'].items():
            if file_hash(ROOT/relative) != expected:
                raise ValueError('Source changed; run from this campaign\'s frozen source directory')
        if args.index is not None and not 0 <= args.index < len(manifest['experiments']):
            parser.error('Experiment index is outside the manifest')
        import torch
        torch.set_num_threads(1)
        rows = manifest['experiments'] if args.index is None else [manifest['experiments'][args.index]]
        with Stop(args.seconds) as stop:
            for row in rows:
                if stop(): break
                try:
                    result = run_experiment(row, device=args.device, stop=stop)
                except Exception as exc:
                    write_json(Path(row['output'])/'status.json', dict(status='FAILED', error=repr(exc)))
                    raise
                print(json.dumps(dict(experiment_id=row['experiment_id'], **result)), flush=True)
    catalog(manifest)


if __name__ == '__main__':
    main()
