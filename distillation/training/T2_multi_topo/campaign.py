"""Schedule unfinished model/collection/seed runs on independent XPU tiles."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import (DEFAULT_CONFIG, claim, completed, load_config, read_json,
                     run_paths, specification)
from distillation.training.T1_single_topo.state import Stop


def task_list(config):
    return [(model, collection, seed) for collection in config["collections"]
            for model in config["models"] for seed in config["seeds"]]


def is_complete(config, task):
    model, collection, seed = task
    manifests = {}
    for case in config["collections"][collection]:
        path = Path(config["paths"]["data"]) / case / "manifest.json"
        if not path.exists():
            return False
        manifests[case] = read_json(path)
    manifest = {"format": "lumina-t2-data-v1", "collection": collection, "cases": manifests}
    spec = specification(config, model, collection, seed, manifest)
    return completed(*run_paths(config, model, collection, seed), spec)


def run_child(command, log, stop):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PMI", "PMIX", "OMPI_", "PALS_"))
           and k not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MPI_LOCALRANKID"}}
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          env=env, start_new_session=True) as child:
        while child.poll() is None:
            if stop():
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    return child.wait()
                try:
                    return child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    return child.wait()
            time.sleep(1)
        return child.returncode


def run_task(config, config_path, task, rank, device, stop):
    model, collection, seed = task
    checkpoints, _ = run_paths(config, model, collection, seed)
    with claim(checkpoints / "worker.lock") as acquired:
        if not acquired or is_complete(config, task):
            return 0
        command = [sys.executable, "-u", "-m", "distillation.training.T2_multi_topo.train",
                   "--config", str(config_path.resolve()), "--model", model,
                   "--collection", collection, "--seed", str(seed), "--device", device,
                   "--seconds", str(max(1, stop.deadline - time.monotonic() - 10))]
        with (checkpoints / "run.log").open("a", buffering=1) as log:
            log.write(f"\nJob {os.environ.get('PBS_JOBID', 'local')}, rank {rank}, {device}\n")
            code = run_child(command, log, stop)
        print(f"rank={rank} {model}/{collection}/seed{seed}: exit={code}", flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", help="Single-worker override, e.g. cpu")
    parser.add_argument("--seconds", type=float, default=3300)
    args = parser.parse_args()
    rank, local_rank, comm = 0, 0, None
    if args.mpi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        local = comm.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = local.Get_rank()
        local.Free()
    if local_rank >= 6:
        raise ValueError("Use at most six workers per node with the FLAT tile mapping")
    device = args.device or f"xpu:{2 * local_rank}"
    config = load_config(args.config)
    tasks = task_list(config)
    report = None
    if rank == 0:
        try:
            missing = [case for case in config["cases"]
                       if not (Path(config["paths"]["data"]) / case / "manifest.json").exists()]
            if missing and not args.dry_run:
                raise ValueError(f"Prepare these cases first: {', '.join(missing)}")
            report = {"pending": [task for task in tasks if not is_complete(config, task)]}
        except Exception as exc:
            report = {"error": str(exc)}
    if comm is not None:
        report = comm.bcast(report, root=0)
    if "error" in report:
        raise ValueError(report["error"])
    pending = report["pending"]
    if rank == 0:
        print(f"{len(tasks) - len(pending)} complete; {len(pending)} unfinished", flush=True)
        if args.dry_run:
            for task in pending:
                print(" / ".join(map(str, task)))
    if args.dry_run or not pending:
        return
    offset = rank % len(pending)
    pending = pending[offset:] + pending[:offset]
    failed = 0
    with Stop(args.seconds) as stop:
        for task in pending:
            if stop():
                break
            try:
                if is_complete(config, task):
                    continue
                code = run_task(config, args.config, task, rank, device, stop)
                failed += int(code != 0)
            except Exception as exc:
                failed += 1
                print(f"rank={rank} FAILED {task}: {exc}", flush=True)
    if comm is not None:
        failed = comm.allreduce(failed)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
