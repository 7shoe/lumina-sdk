"""Independent one-node selection/RAMBO workers; scientific runners stay unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def assignments(manifest, kind):
    rows = manifest['experiments']
    if kind == 'rambo':
        return [[i] for i in range(len(rows))]
    cases = dict.fromkeys(r['config']['case'] for r in rows)
    return [sorted([i for i, row in enumerate(rows) if row['config']['case'] == case],
                   key=lambda i: rows[i]['method'] == 'bas') for case in cases]


def verify(root, manifest, kind):
    expected = ('lumina-selection-campaign-v1' if kind == 'selection' else
                'RAMBO operating-point generation')
    if manifest.get('format' if kind == 'selection' else 'method') != expected:
        raise ValueError('Expected a separate selection/RAMBO campaign, never a synthesis campaign')
    hashes = manifest.get('source_hashes') or json.loads((root/'source_hashes.json').read_text())
    for relative, checksum in hashes.items():
        if hashlib.sha256((root/'source'/relative).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Frozen source changed: {relative}')


def worker(args, manifest):
    # The launcher lives outside the scientific snapshot; imports use that snapshot.
    sys.path.insert(0, str(args.root/'source'))
    import torch
    from distillation.training.T1_single_topo.state import Stop
    torch.set_num_threads(1)
    outcome = []
    indices = assignments(manifest, args.kind)[args.worker]
    device = f'xpu:{2*args.worker}' if args.kind == 'selection' else 'cpu'
    if args.kind == 'selection':
        from distillation.experiments.selection import run_experiment
        from distillation.training.T1_single_topo.data import CaseDataset
        config = manifest['experiments'][indices[0]]['config']
        dataset = CaseDataset(config['data_root'], config['case'])
    else:
        from distillation.RAMBO.run import run
    with Stop(max(0, args.deadline-time.time())) as stop:
        for index in indices:
            if stop(): break
            row = manifest['experiments'][index]
            try:
                result = (run_experiment(row, device=device, stop=stop, dataset=dataset)
                          if args.kind == 'selection' else run(row['config'], row['output'], stop=stop))
            except Exception as error:
                result = dict(status='FAILED', error=repr(error), traceback=traceback.format_exc())
            outcome.append(dict(index=index, output=row['output'], result=result))
            write(args.session/f'worker{args.worker}.json', outcome)
            print(json.dumps(dict(worker=args.worker, index=index, **result)), flush=True)
    return int(any(r['result']['status'].lower() == 'failed' for r in outcome))


def warm_selection(manifest):
    """Sequential reads populate this node's page cache before random mmap access."""
    roots = dict.fromkeys(str(Path(r['config']['data_root'])/r['config']['case'])
                         for r in manifest['experiments'])
    for folder in roots:
        started = time.monotonic(); size = 0
        for path in sorted(Path(folder).glob('group_*.pt')):
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8*1024*1024), b''): size += len(block)
        print(json.dumps(dict(stage='warm_data', case=Path(folder).name,
                              bytes=size, seconds=time.monotonic()-started)), flush=True)


def supervise(args, manifest):
    verify(args.root, manifest, args.kind)
    groups = assignments(manifest, args.kind)
    if args.check:
        print(json.dumps(dict(kind=args.kind, workers=len(groups), assignments=groups)))
        return 0
    args.session.mkdir(parents=True, exist_ok=True)
    write(args.session/'launch.json', dict(kind=args.kind, root=str(args.root), deadline=args.deadline,
        assignments=groups, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Independent processes; no cross-condition MPI collectives. Same manifest resumes native checkpoints.'))
    if args.kind == 'selection': warm_selection(manifest)
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, stop)
    cpus = sorted(os.sched_getaffinity(0))
    active, results, retries = {}, {}, {}
    def launch(index):
        log = (args.session/f'worker{index}.log').open('a')
        affinity = cpus[index::len(groups)] or cpus
        command = [sys.executable, '-u', str(Path(__file__).resolve()), args.kind, str(args.root),
                   '--worker', str(index), '--deadline', str(args.deadline), '--session', str(args.session)]
        process = subprocess.Popen(command, cwd=args.root/'source', stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, preexec_fn=lambda: os.sched_setaffinity(0, affinity))
        log.close(); active[index] = process
        print(json.dumps(dict(worker=index, pid=process.pid, indices=groups[index])), flush=True)
    for index in range(len(groups)):
        if time.time()+300 >= args.deadline: break
        launch(index)
        time.sleep(1)
    signalled = None
    while active:
        for index, process in list(active.items()):
            code = process.poll()
            if code is None: continue
            del active[index]
            results[str(index)] = dict(exit_code=code, retries=retries.get(index, 0))
            write(args.session/'processes.json', results)
            # One infrastructure retry, with the unchanged native configuration/checkpoint.
            if code and retries.get(index, 0) < 1 and time.time()+600 < args.deadline and not stopping:
                retries[index] = retries.get(index, 0)+1
                launch(index)
        if (stopping or time.time() > args.deadline+15) and signalled is None:
            signalled = time.time()
            for process in active.values():
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
        if signalled and time.time()-signalled > 60:
            for process in active.values():
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
        time.sleep(2)
    rows = []
    for index, row in enumerate(manifest['experiments']):
        folder = Path(row['output']); path = folder/'status.json'
        status = json.loads(path.read_text()) if path.exists() else dict(status='PENDING')
        rows.append(dict(index=index, case=row['config']['case'], method=row.get('method', row['config'].get('method')),
                         output=str(folder), **status))
    write(args.root/'catalog.json', rows)
    write(args.session/'session.json', dict(processes=results, conditions=rows))
    return int(any(v['exit_code'] for v in results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('selection', 'rambo'))
    parser.add_argument('root', type=Path)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--deadline', type=float, default=time.time()+3300)
    parser.add_argument('--session', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); args.root = args.root.resolve()
    args.session = args.session or args.root/'jobs'/os.environ.get('PBS_JOBID', 'manual')
    manifest = json.loads((args.root/'manifest.json').read_text())
    raise SystemExit(supervise(args, manifest) if args.worker is None else worker(args, manifest))


if __name__ == '__main__':
    main()
