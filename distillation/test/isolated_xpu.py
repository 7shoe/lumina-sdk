"""Two-node fault injection: kill/hang a rank, preserve siblings, resume its checkpoint.

Run inside a two-node allocation from the frozen runtime directory. These are
disposable diagnostic tensors, never scientific campaign datasets.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path.cwd()))
from distillation.experiments.launch import mpi_command, supervise, write_json


def worker(manifest_path, result):
    import torch
    import torch_geometric
    from mpi4py import MPI
    comm=MPI.COMM_WORLD
    assert comm.size==12
    torch.set_num_threads(1);torch.xpu.set_device(comm.rank)
    uuids=comm.allgather(str(torch.xpu.get_device_properties(comm.rank).uuid))
    assert len(set(uuids))==12
    manifest=json.loads(Path(manifest_path).read_text());row=manifest['experiments'][0]
    folder=Path(row['output']);folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'latest.pt'
    step=torch.load(checkpoint,weights_only=False)['step'] if checkpoint.exists() else 0
    before=step;attempt=int(os.environ['STUDY_ATTEMPT'])
    value=torch.ones(32,device=f'xpu:{comm.rank}',dtype=torch.float64,requires_grad=True)
    gradient=torch.autograd.grad(value.square().sum(),value,create_graph=True)[0]
    torch.autograd.grad(gradient.sum(),value)
    torch.xpu.synchronize()
    if comm.rank==0:
        torch.save({'step':step+1},folder/'latest.tmp');(folder/'latest.tmp').replace(checkpoint)
        write_json(folder/f'worker_attempt{attempt}.json',dict(before=before,rank_count=comm.size,uuids=uuids,
            python=sys.executable,prefix=sys.prefix,torch_geometric=torch_geometric.__file__))
    comm.Barrier()
    if row['dataset_id']=='crash' and attempt==1:
        if comm.rank==0:write_json(Path(manifest['root'])/'crash_armed.json',dict(time=time.time()))
        comm.Barrier()
        if comm.rank==3:os.kill(os.getpid(),signal.SIGKILL)
        comm.Barrier()
    elif row['dataset_id']=='hang' and attempt==1:
        if comm.rank==3:
            while True:time.sleep(1)
        comm.Barrier()
    if row['dataset_id']=='healthy':
        limit=time.time()+120
        while not (Path(manifest['root'])/'crash_armed.json').exists():
            if time.time()>limit:raise RuntimeError('Crash worker did not start')
            if comm.rank==0:(folder/'history.jsonl').write_text(json.dumps({'utc':time.time()}))
            time.sleep(1)
        for _ in range(10):
            comm.allreduce(1)
            if comm.rank==0:(folder/'history.jsonl').write_text(json.dumps({'utc':time.time()}))
            time.sleep(1)
    if row['dataset_id'] in ('crash','hang'):assert before>=1
    if comm.rank==0:
        write_json(folder/'artifact.json',{'complete':True,'purpose':'fault-injection diagnostic'})
        write_json(folder/'smoke.json',{'status':'PASS','resumed_step':before})
        write_json(result,{'status':'COMPLETE'})


def suite(root, seconds):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    hosts=list(dict.fromkeys(Path(os.environ['PBS_NODEFILE']).read_text().split()))
    assert len(hosts)==2
    manifest=dict(root=str(root),specification={'execution':dict(max_retries=1,stall_seconds=120,
        minimum_attempt_seconds=30,launch_interval=1)},experiments=[
        dict(dataset_id=name,output=str(root/name),method='DD',config={'case':30},verify_run=True)
        for name in ['healthy','crash','hang']])
    write_json(root/'manifest.json',manifest)
    def command(hostfile,manifest,seconds,result):
        mpi=mpi_command(hostfile,manifest,seconds,result)
        return mpi[:mpi.index(sys.executable)]+[sys.executable,'-u',str(Path(__file__).resolve()),
                                                '--worker',str(manifest),str(result)]
    report=asyncio.run(supervise(manifest,hosts,root/'attempts',time.time()+seconds,
                                command=command,poll_seconds=1))
    assert report['counts']=={'COMPLETE':3},report['counts']
    crash=json.loads((root/'attempts/crash/attempt1/status.json').read_text())
    hang=json.loads((root/'attempts/hang/attempt1/status.json').read_text())
    healthy=json.loads((root/'healthy/launch_status.json').read_text())
    assert crash['returncode']!=0 and hang['status']=='TIMED_OUT'
    assert healthy['attempt']==1 and healthy['finished']>json.loads((root/'crash_armed.json').read_text())['time']
    for name in ('crash','hang'):
        assert json.loads((root/name/'worker_attempt2.json').read_text())['before']==1
    write_json(root/'PASS.json',dict(status='PASS',nodes=hosts,checks=[
        '12 distinct XPUs per MPI world','hard rank death isolated','hung rank timed out',
        'healthy sibling continued','both failed experiments resumed checkpoints']))
    print('FAULT ISOLATION PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--seconds',type=int,default=480)
    args=parser.parse_args()
    if args.worker:worker(*args.worker)
    else:suite(args.root,args.seconds)
