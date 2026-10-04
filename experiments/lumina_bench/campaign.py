"""MPI schedules independent experiments; training itself is single-device.

Each group is prepared in an isolated extraction root, then atomically hardlinked
into a shared cache. No SDK source or shared extraction directory is modified.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from mpi4py import MPI
import yaml

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.trainer.opf.utils import parse_case_name
from .data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split-file", type=Path,
                        help="Reuse a validated split manifest from an earlier campaign")
    parser.add_argument("--models", nargs="+", default=["HGT", "Transformer", "RGAT"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    tasks = [(model, seed) for model in args.models for seed in args.seeds]
    if size != len(tasks):
        raise ValueError(f"Launch exactly {len(tasks)} ranks, one per model/seed")
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    tile = 2 * local.Get_rank()  # use separate physical GPUs before sharing tiles
    config = yaml.safe_load(args.config.read_text())
    case = parse_case_name(config["case"])
    setup_error = None
    if rank == 0:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / "campaign-config.json", config)
        except Exception as exc:
            setup_error = repr(exc)
    setup_error = comm.bcast(setup_error, root=0)
    if setup_error:
        raise RuntimeError(setup_error)
    # Isolate extraction, including the upstream cleanup, per data group.
    error = None
    try:
        destination = args.data_root / "OPFData/processed/dataset_release_1" / case
        destination.mkdir(parents=True, exist_ok=True)
        for group in config["groups"][rank::size]:
            archive = f"{case}_{group}.tar.gz"
            source = args.raw_root / archive
            if not source.is_file():
                raise FileNotFoundError(source)
            canonical_raw = args.data_root / "OPFData/raw/dataset_release_1" / archive
            canonical_raw.parent.mkdir(parents=True, exist_ok=True)
            if not canonical_raw.exists():
                canonical_raw.symlink_to(source.resolve())
            target = destination / f"group_{group}.pt"
            if target.is_file():
                continue
            stage = args.data_root / ".group-staging" / f"{case}-{group}"
            raw = stage / "OPFData/raw/dataset_release_1"
            raw.mkdir(parents=True, exist_ok=True)
            link = raw / archive
            if not link.exists():
                link.symlink_to(source.resolve())
            dataset = OPFDataset(root=str(stage), case_name=case, group_id=group, n_jobs=4)
            os.link(dataset.processed_paths[0], target)
            print(f"rank={rank} prepared group={group} samples={len(dataset)}", flush=True)
            del dataset
            gc.collect()
    except Exception as exc:
        error = repr(exc)
    failures = comm.allgather(error)
    if any(failures):
        if rank == 0:
            write_json(args.output / "preparation-failures.json", failures)
        raise RuntimeError(f"Data preparation failed: {failures}")
    # Parent ranks coordinate; children are independent, not members of MPI/DDP.
    child_env = {key: value for key, value in os.environ.items()
                 if not key.startswith(("PMI", "PMIX", "OMPI_", "PALS_"))
                 and key not in {"RANK", "WORLD_SIZE", "LOCAL_RANK", "MPI_LOCALRANKID"}}
    model, seed = tasks[rank]
    run = args.output / f"{model}-seed{seed}"
    command = [sys.executable, "-u", "-m", "experiments.lumina_bench", "train",
               "--config", str(args.config.resolve()), "--data-root", str(args.data_root),
               "--output", str(run), "--model-type", model, "--seed", str(seed),
               "--device", f"xpu:{tile}"]
    if args.split_file is not None:
        command.extend(["--split-file", str(args.split_file.resolve())])
    assignments = comm.gather({"rank": rank, "host": socket.gethostname(),
                               "device": f"xpu:{tile}", "model": model, "seed": seed,
                               "command": command}, root=0)
    if rank == 0:
        write_json(args.output / "assignments.json", assignments)
    with run.with_suffix(".log").open("w") as log:
        code = subprocess.run(command, env=child_env, stdout=log, stderr=subprocess.STDOUT).returncode
    status = {"model": model, "seed": seed, "exit_code": code, "output": str(run)}
    write_json(args.output / f"{model}-seed{seed}-status.json", status)
    statuses = comm.gather(status, root=0)
    failed = comm.allreduce(int(code != 0), op=MPI.SUM)
    if rank == 0:
        write_json(args.output / "campaign-results.json", statuses)
        print(json.dumps(statuses), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
