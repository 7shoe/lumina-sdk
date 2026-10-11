"""Independent BAS experiments across XPUs; no gradient or acquisition averaging."""
import argparse
import json
from pathlib import Path
import socket

import torch

from distillation.training.T1_single_topo.common import write_json
from distillation.training.T1_single_topo.state import Stop
from .run import run


def placement(comm):
    """
    Assign each node-local MPI worker a distinct visible cpu
    """
    from mpi4py import MPI
    
    # shared-memory communicator gives each worker its rank witin node
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    if not torch.xpu.is_available() or torch.xpu.device_count() <= local.rank:
        raise RuntimeError(f'Cannot assign local worker {local.rank} a distinct XPU')
    
    # assuming that multiple xpus are visible
    torch.xpu.set_device(local.rank)
    device = torch.device(f'xpu:{local.rank}')
    properties = torch.xpu.get_device_properties(device)
    
    # record device ident
    info = dict(rank=comm.rank,
                local_rank=local.rank,
                host=socket.gethostname(),
                device=str(device),
                uuid=str(properties.uuid),
                name=properties.name)
    local.Free()
    
    return device, info


def main():
    """
    Distribute independent experiments and collect per-worker session results.
    """
    
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--seconds', type=float, default=3300)
    args = parser.parse_args()
    
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    torch.set_num_threads(1)
    device, assignment = placement(comm)
    assignments = comm.allgather(assignment)
    if len({(a['host'], a['uuid']) for a in assignments}) != comm.size:
        raise RuntimeError('Duplicate physical XPU assignment')
    if comm.rank == 0:
        write_json(args.manifest.with_name('assignments.json'), assignments)
    experiments = json.loads(args.manifest.read_text())['experiments']
    outcomes = []
    
    # 1 deadline covers all experiments
    with Stop(args.seconds) as stop:
        for index in range(comm.rank, len(experiments), comm.size):
            if stop():
                break
            experiment = experiments[index]
            try:
                result = run(experiment['config'], experiment['output'], device=device, stop=stop)
            except Exception as error:
                result = dict(status='failed', error=repr(error))
            outcomes.append(dict(index=index, output=experiment['output'], result=result))
            print(json.dumps(outcomes[-1]), flush=True)
    gathered = comm.gather(outcomes, root=0)
    
    # 
    if comm.rank == 0:
        write_json(args.manifest.with_name('session.json'), dict(outcomes=sum(gathered, []), workers=comm.size))
    if comm.allreduce(any(o['result']['status'] == 'failed' for o in outcomes), op=MPI.LOR):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
