"""Independent sequential libraries across CPU workers; optional XPU qualification."""
import argparse
import json
from pathlib import Path
import socket
import traceback

import torch

from distillation.training.T1_single_topo.common import write_json
from distillation.training.T1_single_topo.state import Stop

from .data import coverage
from .run import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--seconds', type=float, default=3300)
    args = parser.parse_args()
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    torch.set_num_threads(1)
    
    manifest = json.loads(args.manifest.read_text())
    
    # only explicit qualification runs need optioal student/checking code
    if manifest['qualify']:
        # Only explicit qualification runs need optional student/checking code.
        from distillation.BAS.campaign import placement
        from .aux.qualify import qualification
        device, assignment = placement(comm)
    else:
        device, assignment = 'cpu', dict(rank=comm.rank, host=socket.gethostname())
    write_json(args.manifest.parent / f'worker{comm.rank}_assignment.json', assignment)
    outcomes = []
    
    # distribute experiments round-robin
    with Stop(args.seconds) as stop:
        for index in range(comm.rank, len(manifest['experiments']), comm.size):
            if stop():
                break
            experiment = manifest['experiments'][index]
            output = Path(experiment['output'])
            try:
                if manifest['qualify']:
                    result = qualification(experiment['config'], output, stop, device, ('MLP', 'GIN', 'HGT')[(index // 2) % 3])
                else:
                    result = run(experiment['config'], output, stop=stop)
                    if result['status'] == 'complete':
                        write_json(output / 'coverage.json', coverage(output / 'dataset.json'))
            except Exception as error:
                # record the failure and allow this rank to continue with later experiments
                result = dict(status='failed', error=repr(error), traceback=traceback.format_exc())
            outcomes.append(dict(index=index, output=str(output), result=result))
            write_json(args.manifest.parent / f'worker{comm.rank}.json', outcomes)
            print(json.dumps(dict(rank=comm.rank, index=index, status=result['status'], error=result.get('error'))), flush=True)
    # rank 0 combines worker records and checks that every experiment completed
    gathered = comm.gather(outcomes, root=0)
    if comm.rank == 0:
        results = sum(gathered, [])
        write_json(args.manifest.with_name('session.json'), dict(workers=comm.size, outcomes=results,
                   complete=len(results) == len(manifest['experiments']) and all(o['result']['status'] == 'complete' for o in results)))
    
    # propagate any recorded failure to a nonzero exit on every rank
    if comm.allreduce(any(o['result']['status'] == 'failed' for o in outcomes), op=MPI.LOR):
        raise SystemExit(1)

if __name__ == '__main__':
    main()
