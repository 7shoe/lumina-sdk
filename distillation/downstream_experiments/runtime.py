"""Independent node/student processes; operational fixes leave frozen trainers unchanged."""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import sys
import tempfile
import threading
import time

from distillation.experiments.launch import supervise
from distillation.experiments.staging import local_path, stage
from distillation.training.T1_single_topo.campaign import run_child
from distillation.training.T1_single_topo.common import claim, file_hash, read_json, write_json
from distillation.training.T1_single_topo.state import Stop
from .scripts.data import data_identity
from .scripts.train import completed, folder_for, source_identity, train_run

MODULE = 'distillation.downstream_experiments.runtime'


def student(manifest_path, identifier, device, deadline):
    manifest = read_json(manifest_path)
    run = next(r for r in manifest['runs'] if r['run_id'] == identifier)
    folder = folder_for(manifest, run)
    try:
        with Stop(max(0, deadline-time.time())) as stop:
            if stop():
                return 0
            result = train_run(manifest, run, device=device, stop=stop)
        print(result, flush=True)
        return 75 if result == 'busy' else 0
    except BlockingIOError:
        return 75
    except Exception as error:
        write_json(folder/'failure.json', dict(error=repr(error), recover_from='last.pt', time=time.time()))
        raise


def worker(manifest_path, bundle, slot, device, deadline):
    manifest = read_json(manifest_path)
    by_id = {r['run_id']: r for r in manifest['runs']}
    tasks = bundle['run_ids'][slot::len(bundle['devices'])]
    report = Path(bundle['output'])/f'worker-{slot}.json'
    previous = read_json(report) if report.exists() else {}
    outcomes = {identifier: previous[identifier] for identifier in tasks if identifier in previous}
    minimum = manifest['config']['execution'].get('minimum_student_seconds', 90)
    with Stop(max(0, deadline-time.time())) as stop:
        for identifier in tasks:
            if stop() or deadline-time.time() < minimum:
                break
            run = by_id[identifier]; folder = folder_for(manifest, run)
            with claim(folder/'worker.lock') as acquired:
                if not acquired:
                    outcomes[identifier] = 'busy'; continue
                try:
                    if completed(folder, run['specification']):
                        outcomes[identifier] = 'complete'; continue
                    command = [sys.executable, '-u', '-m', MODULE, 'student',
                        '--manifest', str(manifest_path), '--run-id', identifier,
                        '--device', device, '--deadline', str(deadline-30)]
                    with (folder/'run.log').open('a', buffering=1) as log:
                        log.write(f'\nJob {os.environ.get("PBS_JOBID", "local")} host={socket.gethostname()} {device}\n')
                        while True:
                            code = run_child(command, log, stop)
                            if code != 75 or stop() or deadline-time.time() < minimum:
                                break
                            time.sleep(2)
                    # A signal can arrive after completion was committed. Trust the artifact.
                    outcomes[identifier] = ('complete' if completed(folder, run['specification']) else
                        'paused' if code in (0, 75) or (stop() and not (folder/'failure.json').exists()) else 'failed')
                except Exception as error:
                    outcomes[identifier] = 'failed'
                    write_json(folder/'failure.json', dict(error=repr(error), time=time.time()))
            write_json(report, outcomes)
            print(json.dumps(dict(run_id=identifier, device=device, status=outcomes[identifier])), flush=True)
            if outcomes[identifier] == 'paused':
                break
    write_json(report, outcomes)
    return outcomes


@contextmanager
def heartbeat(output, state):
    """Cover startup/staging too; the supervisor records observed heartbeat changes."""
    stopped = threading.Event()
    write_json(output/'status.json', dict(state, updated=time.time()))
    def beat():
        while not stopped.wait(5):
            write_json(output/'status.json', dict(state, updated=time.time()))
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set(); thread.join(timeout=10)
        if not thread.is_alive():
            write_json(output/'status.json', dict(state, updated=time.time()))


def node_workers(manifest_path, bundle, deadline, result_path, state):
    output = Path(bundle['output'])
    with Stop(max(0, deadline-time.time())) as stop, ThreadPoolExecutor(max_workers=len(bundle['devices'])) as pool:
        def launch(slot, device):
            command = [sys.executable, '-u', '-m', MODULE, 'worker', '--manifest', str(manifest_path),
                '--bundle', str(output/'bundle.json'), '--slot', str(slot), '--device', f'xpu:{device}',
                '--deadline', str(deadline-90)]
            with (output/f'worker-{slot}.log').open('a', buffering=1) as log:
                code = run_child(command, log, stop)
            return code, bool(code and not stop())
        futures = [pool.submit(launch, slot, device) for slot, device in enumerate(bundle['devices'])]
        while not all(f.done() for f in futures):
            state['workers'] = sum(not f.done() for f in futures)
            wait(futures, timeout=5)
        outcomes = [f.result() for f in futures]
    manifest = read_json(manifest_path)
    done = all(completed(folder_for(manifest, r), r['specification']) for r in manifest['runs'])
    reports = [read_json(output/f'worker-{slot}.json') for slot in range(len(bundle['devices']))
               if (output/f'worker-{slot}.json').exists()]
    failed = any(bad for _, bad in outcomes) or any(v == 'failed' for report in reports for v in report.values())
    status = 'COMPLETE' if done else 'FAILED' if failed else 'RESUMABLE'
    write_json(output/'artifact.json', dict(complete=done, status=status))
    write_json(result_path, dict(status=status, run_ids=bundle['run_ids'], worker_codes=[c for c, _ in outcomes]))
    state.update(status=status.lower(), phase='finished', workers=0)
    return int(failed and not done)


def node(manifest_path, bundle, deadline, result_path):
    output = Path(bundle['output']); output.mkdir(parents=True, exist_ok=True)
    state = dict(status='running', phase='startup', host=socket.gethostname())
    with heartbeat(output, state):
        try:
            manifest = read_json(manifest_path)
            if not manifest['config']['execution'].get('stage_inputs', False):
                state['phase'] = 'workers'
                return node_workers(manifest_path, bundle, deadline, result_path, state)
            cases = sorted({c for r in manifest['runs'] for c in r['specification']['evaluation_cases']})
            checkpoints = {r['specification']['initialization']['path']: r['specification']['initialization']
                           for r in manifest['runs'] if r['specification']['initialization']['kind'] != 'fresh'}
            checksum = file_hash(manifest_path)
            rows = [dict(method='downstream', dataset_id=f'case{c}', config=dict(case=c, data_root=manifest['paths']['data'],
                artifact_identity=dict(data_sha256=data_identity(manifest['paths']['data'], c)['sha256'],
                    reference_sha256=checksum), reference_file=str(manifest_path),
                pretrained_checkpoints={k: k for k in checkpoints}, pretrained=checkpoints)) for c in cases]
            with tempfile.TemporaryDirectory(prefix='lumina-downstream-', dir=os.environ.get('TMPDIR', '/tmp')) as temporary:
                state['phase'] = 'staging'
                report = stage(rows, temporary)
                write_json(output/'staging.json', {k:v for k,v in report.items() if k != 'paths'})
                os.environ['STUDY_INPUT_MAP'] = str(Path(temporary)/'inputs.json')
                state['phase'] = 'workers'
                return node_workers(local_path(manifest_path), bundle, deadline, result_path, state)
        except BaseException:
            state.update(status='failed', phase='finished')
            raise


def mpi_command(hostfile, subset, remaining, result):
    bundle = read_json(subset)['experiments'][0]
    return ['mpiexec', '--pmi=pmix', '--envall', '--hostfile', str(hostfile), '-n', '1', '--ppn', '1',
        '--cpu-bind', 'none', '/usr/bin/env', sys.executable, '-u', '-m', MODULE, 'node',
        '--manifest', bundle['manifest'], '--bundle', str(Path(bundle['output'])/'bundle.json'),
        '--deadline', str(time.time()+remaining), '--result', str(result)]


def pending_runs(manifest):
    # Keep full artifact verification, but overlap Lustre reads on resumed campaigns.
    with ThreadPoolExecutor(max_workers=8) as pool:
        done = pool.map(lambda run: completed(folder_for(manifest, run), run['specification']), manifest['runs'])
        return [run for run, finished in zip(manifest['runs'], done) if not finished]


def launch(manifest_path, seconds, *, command=mpi_command):
    deadline = float(os.environ.get('DOWNSTREAM_STARTED', time.time()))+seconds
    manifest = read_json(manifest_path)
    if source_identity() != manifest['sources']:
        raise ValueError('Frozen training/data sources changed; cannot resume this campaign')
    hosts = list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    root = Path(manifest['paths']['models'])
    directory = root/'jobs'/str(os.environ.get('PBS_JOBID', time.time_ns()))
    with claim(root/'launcher.lock') as acquired:
        if not acquired:
            raise RuntimeError('This campaign already has an active launcher')
        pending = pending_runs(manifest)
        print(json.dumps(dict(completed=len(manifest['runs'])-len(pending), pending=len(pending))), flush=True)
        pending.sort(key=lambda r: (r['specification']['evaluation_cases'], r['specification']['corpus_id']))
        count = max(1, (len(pending)+len(hosts)-1)//len(hosts))
        bundles = []
        for index, start in enumerate(range(0, len(pending), count)):
            folder = directory/f'node-{index}'; assigned = pending[start:start+count]
            # Identical run specifications, with only this node's catalog entries.
            compact = dict(manifest, runs=assigned, corpora={r['specification']['corpus_id']:
                manifest['corpora'][r['specification']['corpus_id']] for r in assigned})
            path = folder/'manifest.json'; write_json(path, compact)
            bundle = dict(dataset_id=f'node-{index}', method='downstream', config={}, output=str(folder),
                manifest=str(path), run_ids=[r['run_id'] for r in assigned], devices=manifest['config']['execution']['devices'])
            write_json(folder/'bundle.json', bundle); bundles.append(bundle)
        execution = dict(manifest['config']['execution'], shutdown_margin_seconds=180, termination_grace_seconds=150)
        wrapper = dict(root=str(directory), experiments=bundles, specification=dict(execution=execution))
        files = [Path(__file__), Path(__file__).parents[1]/'experiments/launch.py', Path(__file__).with_name('run_downstream.pbs')]
        write_json(directory/'runtime.json', dict(manifest=str(manifest_path), manifest_sha256=file_hash(manifest_path),
            sources={str(p):file_hash(p) for p in files}, execution=execution, nodes=len(hosts), pending=len(pending)))
        async def supervise_nodes():
            stop = asyncio.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                asyncio.get_running_loop().add_signal_handler(sig, stop.set)
            return await supervise(wrapper, hosts, directory, deadline, command=command, stop=stop)
        summary = asyncio.run(supervise_nodes())
        # Full report aggregation can outlast the allocation; run it separately.
        collection = dict(status='deferred', manifest=str(manifest_path), command=[sys.executable, '-m',
            'distillation.downstream_experiments.scripts.campaign', 'collect', '--manifest', str(manifest_path)])
        write_json(directory/'collection.json', collection)
        print(json.dumps(dict(counts=summary['counts'], collection=str(directory/'collection.json'))), flush=True)
        return int(any(summary['counts'].get(k, 0) for k in ('FAILED', 'TIMED_OUT')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'node', 'worker', 'student'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--seconds', type=float, default=3300)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'launch':
        return launch(args.manifest.resolve(), args.seconds)
    if args.deadline is None:
        parser.error('--deadline is required for child processes')
    if args.action == 'student':
        return student(args.manifest, args.run_id, args.device, args.deadline)
    bundle = read_json(args.bundle)
    if args.action == 'node':
        return node(args.manifest, bundle, args.deadline, args.result)
    outcomes = worker(args.manifest, bundle, args.slot, args.device, args.deadline)
    return int(any(v == 'failed' for v in outcomes.values()))


if __name__ == '__main__':
    raise SystemExit(main())
