"""Two-node fault injection: fail one checkpointed student, continue siblings, retry elsewhere.

Use the controller action inside a two-node PBS allocation. It intentionally exits
one training subprocess with code 17 after a cooperative checkpoint; no GPU fault
or physical node shutdown is induced. The ordinary trainer is unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

from distillation.downstream_experiments import runtime
from distillation.downstream_experiments.scripts.campaign import load_config, prepare
from distillation.downstream_experiments.scripts.train import completed, folder_for
from distillation.training.T1_single_topo.common import file_hash, read_json, write_json

MODULE = 'distillation.downstream_experiments.tests.isolation_xpu'


def controller(name):
    hosts = list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    assert len(hosts) == 2, hosts
    config = load_config()
    config.update(name=name, cases=[14], students=['GCN', 'HGT'], seeds=[42,43], methods=['RND777'],
                  creation_initializations=['random'], transfer_pairs=[])
    config['families'] = dict(individual=True, fixed=False, mixed_sizes=[], adaptation_sizes=[], joint_size=0)
    config['training'].update(max_updates=4000, validation_updates=[100,4000], checkpoint_seconds=5, log_updates=10)
    config['execution'].update(devices=[0,2], launch_interval=.2)
    path = prepare(config); manifest = read_json(path)
    root = Path(manifest['paths']['models'])
    fault = next(r for r in manifest['runs'] if r['specification']['model'] == 'GCN' and r['specification']['seed'] == 42)
    write_json(root/'fault-plan.json', dict(run_id=fault['run_id']))
    def command(*args):
        argv = runtime.mpi_command(*args)
        argv[argv.index('-m')+1] = MODULE
        return argv
    assert runtime.launch(path, 2400, command=command) == 0
    injection = read_json(root/'injected.json'); resumed = read_json(root/'resumed.json')
    assert 0 < injection['steps'] < 4000
    assert resumed['checkpoint_sha256'] == injection['checkpoint_sha256']
    assert resumed['host'] != injection['host']
    for run in manifest['runs']:
        folder = folder_for(manifest, run)
        assert completed(folder, run['specification']), folder
        assert read_json(folder/'status.json')['steps'] == 4000
        for split in ('val','test'):
            assert read_json(folder/f'{split}-case14.json')['metrics']['n_samples'] == 30000
    job = root/'jobs'/os.environ['PBS_JOBID']
    summary = read_json(job/'summary.json')
    assert summary['counts'] == {'COMPLETE':2}, summary['counts']
    first = read_json(job/'node-0/attempt1/status.json')
    healthy = read_json(job/'node-1/attempt1/status.json')
    retried = read_json(job/'node-0/attempt2/status.json')
    assert first['status'] == 'FAILED' and first['returncode'] != 0
    assert healthy['status'] == 'COMPLETE' and healthy['finished'] > injection['time']
    assert retried['status'] == 'COMPLETE' and retried['host'] != first['host']
    sibling = next(r for r in manifest['runs'] if r['specification']['model'] == 'GCN' and r['specification']['seed'] == 43)
    assert read_json(folder_for(manifest,sibling)/'status.json')['updated'] > injection['time']
    report = dict(status='PASS', manifest=str(path), students=4, nodes=2,
                  failed_student=injection, resumed_student=resumed, summary=str(job/'summary.json'))
    write_json(root/'qualification.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['controller','node','worker','student'])
    parser.add_argument('--name', default='qualification_isolation_20261016')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--deadline', type=float)
    parser.add_argument('--result', type=Path)
    parser.add_argument('--slot', type=int, default=0)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.action == 'controller':
        controller(args.name); return 0
    # Only this test redirects children through the injection wrapper.
    runtime.MODULE = MODULE
    if args.action == 'node':
        return runtime.node(args.manifest, read_json(args.bundle), args.deadline, args.result)
    if args.action == 'worker':
        results = runtime.worker(args.manifest, read_json(args.bundle), args.slot, args.device, args.deadline)
        return int(any(v == 'failed' for v in results.values()))
    manifest = read_json(args.manifest); root = Path(manifest['paths']['models'])
    fault = read_json(root/'fault-plan.json')['run_id']
    if args.run_id != fault:
        return runtime.student(args.manifest, args.run_id, args.device, args.deadline)
    run = next(r for r in manifest['runs'] if r['run_id'] == fault)
    folder = folder_for(manifest,run)
    if not (root/'injected.json').exists():
        assert runtime.student(args.manifest, fault, args.device, min(args.deadline,time.time()+20)) == 0
        progress = read_json(folder/'status.json')
        assert progress['status'] == 'paused' and progress['steps'] > 0, progress
        write_json(root/'injected.json', dict(run_id=fault, steps=progress['steps'], time=time.time(),
            host=socket.gethostname(), checkpoint_sha256=file_hash(folder/'last.pt')))
        print('EXPECTED TEST FAILURE: exit 17 after saving checkpoint', flush=True)
        return 17
    write_json(root/'resumed.json', dict(host=socket.gethostname(), time=time.time(),
        checkpoint_sha256=file_hash(folder/'last.pt')))
    return runtime.student(args.manifest, fault, args.device, args.deadline)


if __name__ == '__main__':
    sys.exit(main())
