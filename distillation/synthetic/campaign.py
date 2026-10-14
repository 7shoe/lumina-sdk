"""Run independent protocol experiments on at most one 12-XPU team per node."""
import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import socket
import shutil
import time
import traceback
import tempfile
import sys

STARTED = float(os.environ.get('STUDY_STARTED', time.time()))

import torch
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--allocation-seconds', type=float, default=3300)
    parser.add_argument('--result', type=Path, help='Per-attempt result for the independent launcher')
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
    if args.result and (world.size != 12 or len(unique) != 1):
        raise ValueError('Isolated attempts must use exactly one node and 12 ranks')
    
    # set torch
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    torch.xpu.set_device(comm.rank)
    device = torch.device(f'xpu:{comm.rank}')
    torch.xpu.reset_peak_memory_stats(device)
    assignments = comm.allgather(dict(rank=comm.rank, host=socket.gethostname(), device=str(device),
        uuid=str(torch.xpu.get_device_properties(device).uuid)))
    if len({r['uuid'] for r in assignments}) != 12:
        raise RuntimeError('Expected 12 distinct XPUs')
    manifest = json.loads(args.manifest.read_text())
    results = []
    deadline = STARTED + args.allocation_seconds
    
    from distillation.training.T1_single_topo.common import claim, write_json
    from distillation.synthetic.runtime import collectively
    if manifest.get('source_hashes'):
        from distillation.training.T1_single_topo.common import file_hash
        def verify():
            if comm.rank: return
            for relative, expected in manifest['source_hashes'].items():
                if file_hash(Path.cwd()/relative) != expected:
                    raise ValueError(f'Source snapshot changed: {relative}')
            for info in manifest['references'].values():
                if file_hash(info['data_manifest']) != info['data_sha256']:
                    raise ValueError(f'Real-data manifest changed: {info["data_manifest"]}')
            for config in {r['config']['case']:r['config'] for r in manifest['experiments']}.values():
                if file_hash(config['reference_file']) != config['artifact_identity']['reference_sha256']:
                    raise ValueError('Training references changed')
        collectively(comm,verify)
    
    from distillation.experiments.staging import assigned, stage, _dataset
    experiments = assigned(manifest['experiments'], node, len(unique))
    execution = manifest.get('specification', {}).get('execution', {})
    startup = dict(host=socket.gethostname(), job=os.environ.get('PBS_JOBID'),
                   before_staging_seconds=time.time()-STARTED, copper=execution.get('copper', False))
    if execution.get('staging', False) and experiments:
        def prepare_inputs():
            if comm.rank: return None
            cache = Path(tempfile.mkdtemp(prefix='lumina-study-', dir='/tmp'))
            return str(cache/'inputs.json'), stage(experiments, cache)
        prepared = collectively(comm, prepare_inputs)
        mapping, report = comm.bcast(prepared, root=0)
        os.environ['STUDY_INPUT_MAP'] = mapping
        startup['staging'] = report
    startup['ready_seconds'] = time.time()-STARTED
    startup_path = Path(manifest['root'])/'jobs'/f"startup_{os.environ.get('PBS_JOBID', 'local')}_{socket.gethostname()}.json"
    if comm.rank == 0:
        write_json(startup_path, startup)
        print(json.dumps(dict(stage='startup', seconds=startup['ready_seconds'],
                              staged_files=startup.get('staging', {}).get('files', 0))), flush=True)
    for experiment in experiments:
        method = experiment.get('method', manifest['method'])
        if method not in ('DD', 'DCGM', 'HMDC', 'RAMBO'):
            raise ValueError(f'Unknown method: {method}')
        runner = importlib.import_module({'DD': 'distillation.DD.campaign',
                                         'DCGM': 'distillation.DC.campaign',
                                         'HMDC': 'distillation.HMDC.campaign',
                                         'RAMBO': 'distillation.experiments.rambo'}[method])
        folder, config = Path(experiment['output']), experiment['config']
        remaining = deadline - time.time()
        if remaining < config['reserve_seconds'] + 180:
            break
        folder.mkdir(parents=True, exist_ok=True)
        lock = claim(folder / 'run.lock') if comm.rank == 0 else None
        acquired = lock.__enter__() if lock else None
        if not comm.bcast(acquired, root=0):
            raise RuntimeError(f'Experiment already running: {folder}')
        try:
            if comm.rank == 0:
                if 'first_runner_seconds' not in startup:
                    startup['first_runner_seconds'] = time.time()-STARTED
                    write_json(startup_path, startup)
                write_json(folder / 'assignments.json', assignments)
                from importlib.metadata import version
                source = Path.cwd()
                hashes = source.parent / f'{source.name}_hashes.json'
                write_json(folder / f"provenance_{os.environ.get('PBS_JOBID', 'local')}_attempt{os.environ.get('STUDY_ATTEMPT', '1')}.json", dict(
                    job=os.environ.get('PBS_JOBID'), method=method, source=str(source),
                    source_hashes=str(hashes) if hashes.exists() else None,
                    python=platform.python_version(), torch=str(torch.__version__),
                    packages={name: version(name) for name in ('numpy', 'scipy', 'torch-geometric', 'pypower', 'mpi4py')},
                    mpi=MPI.Get_library_version(), workers=comm.size,
                    runtime_revision=manifest.get('runtime_revision'),
                    hmdc_microbatch=int(os.environ.get('STUDY_HMDC_MICROBATCH','0'))))
            if experiment.get('smoke') or experiment.get('verify_run'):
                from distillation.experiments.smoke import run, run_rambo
                if method == 'RAMBO': run = run_rambo
                state = run(runner, experiment, comm, device, remaining)
            else:
                state = runner.run(config, folder, comm, device, allocation_seconds=remaining,
                                   qualify=config.get('qualify', False),
                                   stop_after_updates=experiment.get('stop_after_updates'))
            if manifest.get('schema_version') == 1:
                from distillation.synthetic.artifacts import publish, catalog
                collectively(comm,lambda: publish(experiment) if comm.rank==0 else None)
                if comm.rank==0:
                    catalog(json.loads(Path(manifest['catalog_manifest']).read_text())
                            if manifest.get('catalog_manifest') else manifest)
            results.append(dict(dataset_id=experiment['dataset_id'], status=state['status'],
                iteration=state.get('iteration'), resume_count=state.get('resume_count'),
                accepted_updates=state.get('accepted_updates'),
                peak_xpu_bytes=comm.allgather(torch.xpu.max_memory_allocated(device))))
            if comm.rank == 0:
                print(json.dumps(dict(output=str(folder), status=state['status'])), flush=True)
        except Exception:
            results.append(dict(dataset_id=experiment['dataset_id'], status='FAILED'))
            (folder / f'failure_rank{comm.rank:02d}.txt').write_text(traceback.format_exc())
            if comm.rank == 0:
                (folder / 'failure.txt').write_text(traceback.format_exc())
                print(traceback.format_exc(), flush=True)
                if manifest.get('schema_version') == 1:
                    from distillation.synthetic.artifacts import publish, catalog
                    try:
                        publish(experiment,failure=(folder/'failure.txt').read_text())
                        catalog(json.loads(Path(manifest['catalog_manifest']).read_text())
                                if manifest.get('catalog_manifest') else manifest)
                    except Exception:
                        (folder/'artifact_failure.txt').write_text(traceback.format_exc())
            # Collective failures (including initialization) leave other nodes running.
        finally:
            if lock:
                lock.__exit__(None, None, None)
    startup['dataset_caches'] = comm.allgather(dict(rank=comm.rank, **_dataset.cache_info()._asdict()))
    if comm.rank == 0:
        startup['finished_seconds'] = time.time()-STARTED
        write_json(startup_path, startup)
        if 'staging' in startup:
            shutil.rmtree(Path(mapping).parent)
        if args.result:
            write_json(args.result, dict(status=results[0]['status'] if len(results)==1 else 'INCOMPLETE',
                                         experiments=results))
    return int(any(row['status']=='FAILED' for row in results))


if __name__ == '__main__':
    sys.exit(main())
