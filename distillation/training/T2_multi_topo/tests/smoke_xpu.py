"""Qualify mixed-topology batches, checkpoint recovery, and holdouts on XPUs."""
import argparse
import copy
import json
from pathlib import Path
import socket
import time

import torch

from distillation.data.split import N_SAMPLES
from distillation.training.T1_single_topo.tests.smoke_xpu import SmokeDataset
from distillation.training.T1_single_topo.state import Stop
from distillation.training.T2_multi_topo import data, train
from distillation.training.T2_multi_topo.campaign import task_list
from distillation.training.T2_multi_topo.common import (
    DEFAULT_CONFIG, load_config, read_json, run_paths, write_json)
from experiments.lumina_bench.runner import select_device


class SparseCase(SmokeDataset):
    def __len__(self):
        return N_SAMPLES


def main():
    from mpi4py import MPI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=3300)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    local = comm.Split_type(MPI.COMM_TYPE_SHARED)
    device = select_device(f"xpu:{2 * local.Get_rank()}")
    local.Free()
    assignment = {"rank": rank, "host": socket.gethostname(), "device": str(device),
                  "uuid": str(torch.xpu.get_device_properties(device).uuid)}
    assignments = comm.allgather(assignment)
    assert len({(row["host"], row["uuid"]) for row in assignments}) == size
    config = copy.deepcopy(load_config(args.config))
    config["paths"]["output"] = str(args.output)
    config["training"]["max_samples"] = 512
    datasets = {}
    def sparse_case(root, case):
        if case not in datasets:
            datasets[case] = SparseCase(config, case)
        return datasets[case]
    data.CaseDataset = sparse_case
    tasks = task_list(config)
    report = []
    with Stop(args.seconds) as stop:
        for model, collection, seed in tasks[rank::size]:
            if stop():
                break
            started = time.monotonic()
            row = {**assignment, "model": model, "collection": collection, "seed": seed}
            try:
                torch.xpu.reset_peak_memory_stats(device)
                assert train.train_run(config, model, collection, seed, device=str(device),
                                       stop=stop, session_steps=3) == "paused"
                checkpoints, results = run_paths(config, model, collection, seed)
                partial = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=False)
                assert partial["state"]["samples"] == 192
                assert partial["optimizer"]["state"] and "accelerator" in partial["rng"]
                assert set(partial["samples_per_case"].values()) == {64}
                assert train.train_run(config, model, collection, seed, device=str(device), stop=stop) == "complete"
                final = torch.load(checkpoints / "final.pt", map_location="cpu", weights_only=False)
                assert final["state"]["samples"] == 512 and final["state"]["steps"] == 8
                assert sorted(final["samples_per_case"].values()) == [170, 171, 171]
                assert any(not torch.equal(value, final["model"][key])
                           for key, value in partial["model"].items())
                assert train.train_run(config, model, collection, seed, device="cpu") == "complete"
                for split in ("val", "test"):
                    assert read_json(results / f"{split}.json")["metrics"]["n_samples"] == 384
                    for case in config["collections"][collection]:
                        assert read_json(results / split / f"{case}.json")["metrics"]["n_samples"] == 128
                row.update(status="PASS", peak_gib=torch.xpu.max_memory_allocated(device) / 2**30)
            except Exception as exc:
                row.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
            row["seconds"] = time.monotonic() - started
            report.append(row)
            write_json(args.output / f"rank-{rank}.json", report)
            print(json.dumps(row), flush=True)
            torch.xpu.empty_cache()
    reports = comm.gather(report, root=0)
    success = None
    if rank == 0:
        rows = [row for report in reports for row in report]
        success = len(rows) == len(tasks) and all(row["status"] == "PASS" for row in rows)
        write_json(args.output / "smoke-results.json", {"success": success, "runs": rows})
        print(f"Smoke: {sum(row['status'] == 'PASS' for row in rows)}/{len(tasks)} passed", flush=True)
    if not comm.bcast(success, root=0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
