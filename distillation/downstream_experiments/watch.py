"""Monitor an existing PBS job, resume remaining work within a bound, then collect."""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

from .runtime import pending_runs
from .scripts.campaign import collect
from .scripts.train import source_identity
from distillation.training.T1_single_topo.common import ROOT, claim, file_hash, read_json, write_json


def wait_job(job, report, interval):
    errors = 0
    while True:
        try:
            result = subprocess.run(['qstat', '-xf', job], capture_output=True, text=True, timeout=60)
            match = re.search(r'job_state\s*=\s*(\w)', result.stdout)
            if result.returncode or not match:
                raise RuntimeError(result.stderr.strip() or 'Missing PBS job state')
            errors = 0
            report(status='waiting', job=job, pbs_state=match[1])
            if match[1] == 'F':
                return
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors += 1
            report(status='waiting', job=job, scheduler_error=str(error))
            if errors >= 10:
                raise
        time.sleep(interval)


def submit(manifest_path, nodes, previous):
    root = Path(read_json(manifest_path)['paths']['models'])
    stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    command = ['qsub', '-q', 'debug-scaling', '-l', f'select={nodes}', '-l', 'walltime=01:00:00',
        '-o', str(root/f'pbs-{stamp}.log'), '-v',
        f'DOWNSTREAM_MANIFEST={manifest_path},DOWNSTREAM_SECONDS=3300',
        str(ROOT/'distillation/downstream_experiments/run_downstream.pbs')]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    write_json(root/f'submission-{stamp}.json', dict(job_id=job, submitted=stamp, nodes=nodes,
        walltime='01:00:00', manifest=str(manifest_path), continues=previous,
        command=command, stderr=result.stderr, submitted_by='watch.py'))
    return job


def watch(manifest_path, job, *, max_followups=2, max_nodes=96, interval=60):
    if max_followups < 0 or not 1 <= max_nodes <= 256 or interval <= 0:
        raise ValueError('Require nonnegative follow-ups, 1..256 nodes, and a positive poll interval')
    manifest_path = Path(manifest_path).resolve()
    manifest = read_json(manifest_path)
    original_hash = file_hash(manifest_path)
    root = Path(manifest['paths']['models'])
    jobs = [job]
    with claim(root/'continuation.lock') as acquired:
        if not acquired:
            raise RuntimeError('A continuation monitor is already running')
        def report(**state):
            state.update(jobs=list(jobs), updated=time.time(), max_followups=max_followups,
                         max_nodes=max_nodes, manifest=str(manifest_path))
            write_json(root/'continuation.json', state)
            print(json.dumps(state), flush=True)
        try:
            while True:
                wait_job(job, report, interval)
                report(status='auditing', job=job)
                if file_hash(manifest_path) != original_hash or source_identity() != manifest['sources']:
                    raise ValueError('Frozen manifest/scientific sources changed; refusing continuation')
                pending = pending_runs(manifest)
                counts = dict(completed=len(manifest['runs'])-len(pending), remaining=len(pending))
                if pending and len(jobs)-1 < max_followups:
                    nodes = min(max_nodes, math.ceil(len(pending)/len(manifest['config']['execution']['devices'])))
                    job = submit(manifest_path, nodes, job)
                    jobs.append(job)
                    report(status='submitted', job=job, nodes=nodes, **counts)
                    continue
                report(status='collecting', job=job, **counts)
                collect(manifest_path)
                report(status='complete' if not pending else 'incomplete', job=job, **counts)
                return len(pending)
        except Exception as error:
            report(status='error', job=job, error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--max-followups', type=int, default=2)
    parser.add_argument('--max-nodes', type=int, default=96)
    args = parser.parse_args()
    return int(bool(watch(args.manifest, args.job, max_followups=args.max_followups, max_nodes=args.max_nodes)))


if __name__ == '__main__':
    raise SystemExit(main())
