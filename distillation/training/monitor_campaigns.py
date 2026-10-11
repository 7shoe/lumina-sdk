"""Monitor T1, resume unfinished jobs, verify it, then run T2."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import torch

from distillation.data.split import split_indices
from distillation.training.T1_single_topo.common import (
    atomic_file, claim, digest, read_json, write_json)
from distillation.training.T1_single_topo.data import split_record

BASE = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models')


def modules(name):
    package = 'T1_single_topo' if name == 'T1' else 'T2_multi_topo'
    prefix = f'distillation.training.{package}'
    return (importlib.import_module(prefix + '.common'),
            importlib.import_module(prefix + '.campaign'))


def scheduler(job_id):
    try:
        output = subprocess.check_output(['qstat', '-xf', '-F', 'json', job_id], text=True, timeout=30)
        job = next(iter(json.loads(output)['Jobs'].values()))
    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        return {'id': job_id, 'state': 'unknown', 'exit': None, 'warning': str(exc)}
    return {'id': job_id, 'state': job['job_state'], 'exit': job.get('Exit_status'),
            'walltime': job.get('resources_used', {}).get('walltime')}


def snapshot(common, config, tasks):
    statuses, groups = Counter(), {}
    for task in tasks:
        checkpoints, _ = common.run_paths(config, *task)
        path = checkpoints / 'status.json'
        row = read_json(path) if path.exists() else {'status': 'pending', 'samples_seen': 0}
        statuses[row['status']] += 1
        groups.setdefault(task[1], []).append(row['samples_seen'])
    return {'statuses': dict(statuses), 'progress': {
        group: {'min': min(values), 'max': max(values), 'mean': round(sum(values) / len(values))}
        for group, values in groups.items()}}


def audit(name, common, campaign, config, tasks):
    expected_splits = {}
    for group in dict.fromkeys(task[1] for task in tasks):
        if name == 'T1':
            record = split_record(group, split_indices(group, 300000, 42), 42)
        else:
            from distillation.training.T2_multi_topo.data import CollectionDataset, split_record as mixed_split
            dataset = CollectionDataset(config, group)
            assert all(len(part) == 300000 for part in dataset.datasets)
            record = mixed_split(dataset)
        expected_splits[group] = digest(record)
    rows = []
    for task in tasks:
        assert campaign.is_complete(config, task), f'Incomplete artifacts: {task}'
        checkpoints, results = common.run_paths(config, *task)
        assert digest(read_json(checkpoints / 'split.json')) == expected_splits[task[1]], task
        final = torch.load(checkpoints / 'final.pt', map_location='cpu', weights_only=False)
        assert final['state']['samples'] == config['training']['max_samples'], task
        assert final['specification']['split_seed'] == 42, task
        assert final['specification']['seed'] == task[2], task
        if name == 'T2':
            assert sum(final['samples_per_case'].values()) == 1000000, task
            assert sorted(final['samples_per_case'].values()) == [333333, 333333, 333334], task
        for split in ('val', 'test'):
            paths = [('all', results / f'{split}.json')]
            if name == 'T2':
                paths += [(case, results / split / f'{case}.json')
                          for case in config['collections'][task[1]]]
            for case, path in paths:
                report = read_json(path)
                metrics = report['metrics']
                assert metrics['n_samples'] == (90000 if name == 'T2' and case == 'all' else 30000), path
                assert report['trained_samples'] == config['training']['max_samples'], path
                assert all(math.isfinite(value) for value in metrics.values()), path
                rows.append({'model': task[0], 'group': task[1], 'seed': task[2], 'split': split,
                             'case': case, 'trained_samples': report['trained_samples'],
                             'checkpoint_samples': report['checkpoint_samples'], **metrics})
    results = BASE / name / 'results'
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with atomic_file(results / 'summary.csv', 'w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {'campaign': name, 'verified_at': datetime.now(timezone.utc).isoformat(),
              'runs': len(tasks), 'samples_per_run': config['training']['max_samples'],
              'split_seed': 42, 'model_seeds': config['seeds'], 'rows': rows}
    write_json(results / 'summary.json', report)
    write_json(BASE / name / 'jobs' / 'verified.json', {key: value for key, value in report.items() if key != 'rows'})
    print(f'{name}: verified all {len(tasks)} runs and holdouts', flush=True)


def smoke_passed(name, submissions):
    job = next(row for row in reversed(submissions) if row['kind'].startswith('smoke'))
    status = scheduler(job['id'])
    if status['state'] != 'F':
        return False
    assert status['exit'] == 0, f'{name} smoke failed: {status}'
    report = read_json(BASE / name / 'smoke' / job['id'] / 'smoke-results.json')
    expected = 120 if name == 'T1' else 72
    assert report['success'] and len(report['runs']) == expected
    assert all(row['status'] == 'PASS' for row in report['runs'])
    common, campaign = modules(name)
    config = common.load_config()
    group = 'case' if name == 'T1' else 'collection'
    qualified = {(row['model'], row[group], row['seed']) for row in report['runs']}
    assert qualified == set(campaign.task_list(config)), 'Incomplete smoke coverage'
    task = next(iter(qualified))
    smoke_config = dict(config, paths=dict(config['paths'], output=str(BASE / name / 'smoke' / job['id'])))
    folder, _ = common.run_paths(smoke_config, *task)
    tested = read_json(folder / 'config.json')
    current = common.specification(config, *task, {})
    assert tested['source_identity'] == current['source_identity'], 'Code changed since smoke test'
    return True


def submit(name, common, config, config_path, pending, submissions):
    jobs = BASE / name / 'jobs'
    nodes = min(12, math.ceil(len(pending) / 6))
    attempt = 1 + sum(row['kind'] == 'train' for row in submissions)
    log = jobs / f'train-{datetime.now(timezone.utc):%Y%m%d}-{attempt:02d}.log'
    script = common.DEFAULT_CONFIG.with_name(f'run_{name}.pbs')
    job_id = subprocess.check_output([
        'qsub', '-k', 'doe', '-l', f'select={nodes}', '-v',
        f'{name}_PPN=6,{name}_CONFIG={config_path}', '-o', str(log), str(script)], text=True).strip().split('.')[0]
    row = {'id': job_id, 'kind': 'train', 'queue': 'debug-scaling', 'nodes': nodes,
           'workers_per_node': 6, 'config': str(config_path), 'log': str(log)}
    submissions.append(row)
    write_json(jobs / 'submissions.json', submissions)
    print(f'{name}: submitted {job_id}, {len(pending)} unfinished runs on {nodes} nodes', flush=True)


def monitor(name, interval):
    common, campaign = modules(name)
    jobs = BASE / name / 'jobs'
    jobs.mkdir(parents=True, exist_ok=True)
    config_path = jobs / f'{name.lower()}-config.yaml'
    if not config_path.exists():
        shutil.copyfile(common.DEFAULT_CONFIG, config_path)
    config = common.load_config(config_path)
    tasks = campaign.task_list(config)
    while True:
        submissions = read_json(jobs / 'submissions.json')
        training = [row for row in submissions if row['kind'] == 'train']
        latest = scheduler(training[-1]['id']) if training else None
        report = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
                  'campaign': name, 'job': latest, **snapshot(common, config, tasks)}
        write_json(jobs / 'training-status.json', report)
        print(json.dumps(report), flush=True)
        if latest and latest['state'] != 'F':
            time.sleep(interval)
            continue
        pending = [task for task in tasks if not campaign.is_complete(config, task)]
        if not pending:
            audit(name, common, campaign, config, tasks)
            return
        if latest and latest['exit'] != 0:
            raise RuntimeError(f'{name} job failed; inspect logs before resubmitting: {latest}')
        if smoke_passed(name, submissions):
            submit(name, common, config, config_path, pending, submissions)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=float, default=30)
    args = parser.parse_args()
    with claim(BASE / 'campaign-monitor.lock') as acquired:
        if not acquired:
            raise SystemExit('Another campaign monitor is active')
        for name in ('T1', 'T2'):
            monitor(name, args.interval)


if __name__ == '__main__':
    main()
