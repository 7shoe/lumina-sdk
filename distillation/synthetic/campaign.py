"""Run independent protocol experiments on at most one 12-XPU team per node."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import time
import traceback

import torch
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allocation-seconds', type=float, default=3300)
    args = parser.parse_args()
    
    # MPI import
    from mpi4py import MPI
    world = MPI.COMM_WORLD
    
    # world setup
    hosts = world.allgather(socket.gethostname())
    unique = list(dict.fromkeys(hosts))
    node = unique.index(socket.gethostname())
    comm = world.Split(node, world.rank)
    if comm.size != 12:
        raise ValueError('Use exactly 12 ranks per node')
    
    # set torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    torch.xpu.set_device(comm.rank)
    device = torch.device(f'xpu:{comm.rank}')
    assignments = comm.allgather(dict(rank=comm.rank, host=socket.gethostname(), device=str(device),
        uuid=str(torch.xpu.get_device_properties(device).uuid)))
    if len({r['uuid'] for r in assignments}) != 12:
        raise RuntimeError('Expected 12 distinct XPUs')
    manifest = json.loads(args.manifest.read_text())
    
    from distillation.training.T1_single_topo.common import claim, write_json
    
    started = time.monotonic()
    for experiment in manifest['experiments'][node::len(unique)]:
        method = experiment.get('method', manifest['method'])
        if method not in ('DD', 'DCGM', 'HMDC'):
            raise ValueError(f'Unknown method: {method}')
        runner = importlib.import_module({'DD': 'distillation.DD.campaign',
                                         'DCGM': 'distillation.DC.campaign',
                                         'HMDC': 'distillation.HMDC.campaign'}[method])
        folder, config = Path(experiment['output']), experiment['config']
        remaining = args.allocation_seconds - (time.monotonic() - started)
        if remaining < config['reserve_seconds'] + 180:
            break
        folder.mkdir(parents=True, exist_ok=True)
        lock = claim(folder / 'run.lock') if comm.rank == 0 else None
        acquired = lock.__enter__() if lock else None
        if not comm.bcast(acquired, root=0):
            raise RuntimeError(f'Experiment already running: {folder}')
        try:
            if comm.rank == 0:
                write_json(folder / 'assignments.json', assignments)
                from importlib.metadata import version
                source = Path.cwd()
                hashes = source.parent / f'{source.name}_hashes.json'
                write_json(folder / f"provenance_{os.environ.get('PBS_JOBID', 'local')}.json", dict(
                    job=os.environ.get('PBS_JOBID'), method=method, source=str(source),
                    source_hashes=str(hashes) if hashes.exists() else None,
                    python=platform.python_version(), torch=str(torch.__version__),
                    packages={name: version(name) for name in ('numpy', 'scipy', 'torch-geometric', 'pypower', 'mpi4py')},
                    mpi=MPI.Get_library_version(), workers=comm.size))
            state = runner.run(config, folder, comm, device, allocation_seconds=remaining,
                               qualify=config.get('qualify', False),
                               stop_after_updates=experiment.get('stop_after_updates'))
            if comm.rank == 0:
                print(json.dumps(dict(output=str(folder), status=state['status'])), flush=True)
        except Exception:
            (folder / f'failure_rank{comm.rank:02d}.txt').write_text(traceback.format_exc())
            if comm.rank == 0:
                (folder / 'failure.txt').write_text(traceback.format_exc())
                print(traceback.format_exc(), flush=True)
            # Collective failures (including initialization) leave other nodes running.
        finally:
            if lock:
                lock.__exit__(None, None, None)


if __name__ == '__main__':
    main()
