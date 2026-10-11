"""Exercise real batches, resume, and evaluation on each model/case XPU pair."""
import argparse
import copy
import json
from pathlib import Path
import socket
import tarfile
import time

import numpy as np
import torch

from distillation.data.split import N_SAMPLES, split_indices
from distillation.training.T1_single_topo import train
from distillation.training.T1_single_topo.common import (
    DEFAULT_CONFIG, file_hash, load_config, read_json, run_paths, write_json)
from distillation.training.T1_single_topo.state import Stop
from experiments.lumina_bench.runner import select_device
from lumina.dataset.opf.opf_dataset import build_heterodata_from_grid


class SmokeDataset:
    """Small subsets of the original seed-42 split; original IDs are retained."""
    def __init__(self, config, case):
        original = split_indices(case, N_SAMPLES, seed=42)
        limits = {"train": 256, "val": 128, "test": 128}
        membership = np.empty(N_SAMPLES, dtype=np.int8)
        for label, indices in enumerate(original.values()):
            membership[indices] = label
        names = list(original)
        counts = dict.fromkeys(names, 0)
        self.graphs = {}
        archive = Path(config["paths"]["raw"]) / f"{case}_0.tar.gz"
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if not member.isfile():
                    continue
                index = int(Path(member.name).stem.removeprefix("example_"))
                name = names[membership[index]]
                if counts[name] == limits[name]:
                    continue
                obj = json.load(stream.extractfile(member))
                graph = build_heterodata_from_grid(obj["grid"], obj["metadata"], obj["solution"])
                graph.sample_id = torch.tensor([index])
                self.graphs[index] = graph
                counts[name] += 1
                if counts == limits:
                    break
        if counts != limits:
            raise ValueError(f"Insufficient smoke samples: {case}: {counts}")
        ids = list(self.graphs)
        self.indices = {name: indices[np.isin(indices, ids)] for name, indices in original.items()}
        self.manifest = {"format": "t1-smoke-subset-v1", "case": case, "split_seed": 42,
                         "indices": {key: value.tolist() for key, value in self.indices.items()},
                         "smoke_code_sha256": file_hash(__file__)}

    def __getitem__(self, index):
        return self.graphs[int(index)].clone()

    def splits(self, seed):
        assert seed == 42
        return self.indices


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
    # Only the smoke process substitutes the sparse dataset; production uses full caches.
    def dataset(root, case):
        if case not in datasets:
            datasets[case] = SmokeDataset(config, case)
        return datasets[case]
    train.CaseDataset = dataset
    tasks = [(model, case, seed) for case in config["cases"]
             for model in config["models"] for seed in config["seeds"]]
    report = []
    with Stop(args.seconds) as stop:
        for model, case, seed in tasks[rank::size]:
            if stop():
                break
            started = time.monotonic()
            row = {**assignment, "model": model, "case": case, "seed": seed}
            try:
                torch.xpu.reset_peak_memory_stats(device)
                assert train.train_run(config, model, case, seed, device=str(device),
                                       stop=stop, session_steps=3) == "paused"
                checkpoints, results = run_paths(config, model, case, seed)
                partial = torch.load(checkpoints / "last.pt", map_location="cpu", weights_only=False)
                assert partial["state"]["samples"] == 192
                assert partial["optimizer"]["state"] and "accelerator" in partial["rng"]
                assert train.train_run(config, model, case, seed, device=str(device), stop=stop) == "complete"
                final = torch.load(checkpoints / "final.pt", map_location="cpu", weights_only=False)
                assert final["state"]["samples"] == 512 and final["state"]["steps"] == 8
                assert any(not torch.equal(value, final["model"][key])
                           for key, value in partial["model"].items())
                assert train.train_run(config, model, case, seed, device="cpu") == "complete"
                for split in ("val", "test"):
                    assert read_json(results / f"{split}.json")["metrics"]["n_samples"] == 128
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
