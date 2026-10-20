"""Supervise independent one-node MPI jobs; no MPI or torch in the supervisor."""
import argparse
import asyncio
from collections import Counter, deque
import json
import os
from pathlib import Path
import signal
import sys
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return default


def complete(experiment):
    folder = Path(experiment['output'])
    try:
        artifact = read_json(folder/'artifact.json', {})
        if not artifact.get('complete'):
            return False
        return not (experiment.get('smoke') or experiment.get('verify_run')) or (
            read_json(folder/'smoke.json', {}).get('status') == 'PASS')
    except (OSError, ValueError):
        # A broken condition is retried/reported in isolation, never skipped.
        return False


def mpi_command(hostfile, manifest, seconds, result):
    return ['mpiexec', '--pmi=pmix', '--envall', '--hostfile', str(hostfile),
            '-n', '12', '--ppn', '12', '--cpu-bind', 'list:4:56:9:61:14:66:19:71:20:74:25:79',
            # PALS may resolve an executable symlink on remote nodes. Let env
            # exec the original venv path so Python retains its pyvenv.cfg.
            '/usr/bin/env', sys.executable, '-u', '-m', 'distillation.synthetic.campaign',
            '--manifest', str(manifest), '--allocation-seconds', str(seconds),
            '--result', str(result)]


async def terminate(child, grace=20):
    """Forward termination through mpiexec, then bound even a stuck launcher."""
    if child.returncode is not None:
        return
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 10)):
        try:
            os.killpg(child.pid, sig)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(child.wait(), wait)
            return
        except asyncio.TimeoutError:
            continue


async def supervise(manifest, hosts, directory, deadline, *, command=mpi_command,
                    poll_seconds=5, stop=None):
    """One active condition per host; failed attempts never cancel healthy siblings.

    Resume state belongs to the experiment, not its current node. A subsequent
    PBS submission can assign it anywhere. Only complete, replay-verified runs skip.
    """
    if not hosts or len(hosts) != len(set(hosts)):
        raise ValueError('Expected distinct allocated hosts')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    execution = manifest.get('specification', {}).get('execution', {})
    retries = execution.get('max_retries', 1)
    timeout = execution.get('stall_seconds', 1200)
    shutdown_margin = execution.get('shutdown_margin_seconds', 30)
    termination_grace = execution.get('termination_grace_seconds', 20)
    minimum = execution.get('minimum_attempt_seconds', 180)
    pending = deque((row, 0, None) for row in manifest['experiments'] if not complete(row))
    active, records, quarantined, owners = {}, {}, set(), {}
    stop = stop or asyncio.Event()
    next_launch = 0.

    async def attempt(row, host, number):
        identifier = row['dataset_id']
        attempt_dir = directory/identifier/f'attempt{number+1}'
        attempt_dir.mkdir(parents=True, exist_ok=True)
        hostfile = attempt_dir/'hostfile'; hostfile.write_text(host+'\n')
        subset = dict(manifest, experiments=[row], catalog_manifest=str(Path(manifest['root'])/'manifest.json'))
        write_json(attempt_dir/'manifest.json', subset)
        result = attempt_dir/'result.json'
        started = time.time()
        # Keep time for mpiexec cleanup before the supervisor's allocation deadline.
        seconds = max(1, deadline-started-shutdown_margin)
        env = {k:v for k,v in os.environ.items()
               if not k.startswith(('PMI', 'PMIX', 'OMPI_', 'PALS_'))
               and k not in {'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MPI_LOCALRANKID', 'STUDY_INPUT_MAP'}}
        env['STUDY_STARTED'] = str(started)
        env['STUDY_ATTEMPT'] = str(number+1)
        env['STUDY_ATTEMPT_DIR'] = str(attempt_dir)
        if row['method'] == 'HMDC' and row['config']['case'] >= 500:
            env['STUDY_HMDC_MICROBATCH'] = str(execution.get('hmdc_microbatch', 8))
        record = dict(dataset_id=identifier, host=host, attempt=number+1, started=started,
                      status='RUNNING', log=str(attempt_dir/'run.log'),
                      checkpoint=str(Path(row['output'])/'latest.pt'),
                      runtime_source=str(Path.cwd()),
                      hmdc_microbatch=int(env.get('STUDY_HMDC_MICROBATCH', '0')))
        write_json(Path(row['output'])/'launch_status.json', record)
        write_json(attempt_dir/'status.json', record)
        child = None
        try:
            with (attempt_dir/'run.log').open('w') as stream:
                argv = command(hostfile, attempt_dir/'manifest.json', seconds, result)
                record['command'] = argv
                child = await asyncio.create_subprocess_exec(*argv, stdout=stream,
                    stderr=asyncio.subprocess.STDOUT, env=env, start_new_session=True)
                wait = asyncio.create_task(child.wait())
                previous, last_progress = None, time.monotonic()
                while child.returncode is None:
                    # History records committed work and calibration/validation stages.
                    paths = [Path(row['output'])/name for name in
                             ('history.jsonl', 'latest.pt', 'status.json', 'evaluation.partial.json')]
                    # Observe changes locally: remote mtimes can be stale or clock-skewed.
                    stamps = []
                    for path in paths:
                        try:
                            stat = path.stat()
                            stamps.append((path.name, stat.st_ino, stat.st_mtime_ns, stat.st_size,
                                           path.read_text() if path.name == 'status.json' else None))
                        except FileNotFoundError:
                            pass
                    if stamps != previous:
                        previous, last_progress = stamps, time.monotonic()
                    if stop.is_set() or time.time() >= deadline-30:
                        record['status'] = 'INTERRUPTED'
                        await terminate(child, termination_grace); break
                    if timeout and time.monotonic()-last_progress > timeout:
                        record['status'] = 'TIMED_OUT'
                        record['watchdog'] = dict(idle_seconds=time.monotonic()-last_progress,
                                                  observed_files=[s[0] for s in stamps],
                                                  heartbeat=read_json(Path(row['output'])/'status.json', {}))
                        await terminate(child, termination_grace); break
                    await asyncio.wait({wait}, timeout=poll_seconds)
                code = await child.wait()
            record['returncode'] = code
            report = read_json(result, {})
            record['runner'] = report
            if record['status'] in ('RUNNING', 'INTERRUPTED'):
                record['status'] = ('COMPLETE' if code == 0 and complete(row) else
                                    'RESUMABLE' if code == 0 and report.get('status') in
                                    ('RESUMABLE', 'FINALIZING') else
                                    'INTERRUPTED' if record['status'] == 'INTERRUPTED' else 'FAILED')
        except Exception as exc:
            record.update(status='FAILED', error=f'{type(exc).__name__}: {exc}')
        finally:
            if child is not None and child.returncode is None:
                await terminate(child, termination_grace)
            record['finished'] = time.time()
            write_json(attempt_dir/'status.json', record)
            write_json(Path(row['output'])/'launch_status.json', record)
        print(json.dumps({k:record[k] for k in ('dataset_id', 'host', 'attempt', 'status')}), flush=True)
        return record

    while (pending or active) and not (stop.is_set() and not active):
        now = time.time()
        available = [host for host in hosts if host not in active and host not in quarantined]
        if pending and available and now < deadline-minimum and not stop.is_set():
            # Stagger PALS startup rather than creating hundreds of launchers at once.
            if now >= next_launch:
                for _ in range(len(pending)):
                    row, number, previous = pending.popleft()
                    alternatives = [h for h in hosts if h != previous and h not in quarantined]
                    if number and alternatives and not any(h in available for h in alternatives):
                        pending.append((row,number,previous)); continue
                    library = row['config'].get('generation_id') if row['method']=='RAMBO' else None
                    owner = owners.get(library) if library else None
                    if owner and owner not in available and owner not in quarantined:
                        pending.append((row,number,previous)); continue
                    host = owner if owner in available else next((h for h in available if h != previous),available[0])
                    if library: owners[library] = host
                    active[host] = (asyncio.create_task(attempt(row, host, number)), row, number)
                    next_launch = now + execution.get('launch_interval', .2)
                    break
        for host, (task, row, number) in list(active.items()):
            if not task.done():
                continue
            try:
                record = await task
            except Exception as exc:
                # Even per-run metadata/permission errors must not cancel siblings.
                record = dict(dataset_id=row['dataset_id'],host=host,attempt=number+1,
                              status='FAILED',error=f'{type(exc).__name__}: {exc}',finished=time.time())
                print(json.dumps(record),flush=True)
            records[row['dataset_id']] = record
            del active[host]
            if record['status'] == 'TIMED_OUT' or record.get('returncode') == 127:
                quarantined.add(host)  # Do not put another experiment on an unresponsive node.
            if record['status'] in ('FAILED', 'TIMED_OUT') and number < retries:
                if row['method']=='RAMBO':owners.pop(row['config'].get('generation_id'),None)
                pending.append((row, number+1, host))
        if not active and (not pending or all(h in quarantined for h in hosts)
                           or time.time() >= deadline-minimum or stop.is_set()):
            break
        await asyncio.sleep(min(poll_seconds, max(.05, next_launch-time.time())) if pending else poll_seconds)
    summary = dict(job=os.environ.get('PBS_JOBID'),
        counts=dict(Counter(r['status'] for r in records.values())),
        pending=[row['dataset_id'] for row, _, _ in pending], quarantined=sorted(quarantined),
        records=list(records.values()), finished=time.time())
    write_json(directory/'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allocation-seconds', type=float, default=3300)
    args = parser.parse_args()
    manifest = read_json(args.manifest)
    hosts = list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    directory = Path(manifest['root'])/'jobs'/f"launch_{os.environ.get('PBS_JOBID', time.time_ns())}"
    deadline = float(os.environ.get('STUDY_STARTED', time.time())) + args.allocation_seconds

    async def run():
        stop = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_running_loop().add_signal_handler(sig, stop.set)
        return await supervise(manifest, hosts, directory, deadline, stop=stop)
    summary = asyncio.run(run())
    print(json.dumps(dict(summary=str(directory/'summary.json'), counts=summary['counts'])), flush=True)
    # A resumable allocation is normal; terminal failures remain visible to PBS.
    return int(any(summary['counts'].get(k, 0) for k in ('FAILED', 'TIMED_OUT')))


if __name__ == '__main__':
    sys.exit(main())
