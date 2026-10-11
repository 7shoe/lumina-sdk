"""Train or resume one model on a balanced three-topology collection."""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch

from distillation.training.T1_single_topo.train import evaluate
from experiments.lumina_bench.metrics import MetricAccumulator
from experiments.lumina_bench.runner import build_model, seed_all, select_device
from .common import (DEFAULT_CONFIG, claim, completed, digest, file_hash, load_config,
                     read_json, result_files, run_paths, specification, write_json)
from .data import CollectionDataset, split_record
from distillation.training.T1_single_topo.state import Stop, capture_rng, restore_rng, save_torch


def graph_mse(predictions, batch):
    loss = predictions["bus"].new_zeros(batch.num_graphs)
    for name in ("bus", "generator"):
        graph_ids = batch[name].batch
        errors = (predictions[name] - batch[name].y.float()).square().mean(dim=1)
        totals = errors.new_zeros(batch.num_graphs).index_add(0, graph_ids, errors)
        counts = torch.bincount(graph_ids, minlength=batch.num_graphs)
        loss = loss + totals / counts
    return loss.mean()


def combined_metrics(reports):
    accumulator = MetricAccumulator()
    for report in reports:
        saved = report["accumulator"]
        accumulator.n_samples += saved["n_samples"]
        for key, value in saved["sums"].items():
            accumulator.sums[key] = accumulator.sums.get(key, 0.) + value
            accumulator.counts[key] = accumulator.counts.get(key, 0) + saved["counts"][key]
    metrics = accumulator.result()
    metrics["opf_solution_error"] = metrics["sdk_mse"]
    metrics["constraint_violation"] = metrics["violation"]
    return metrics


def train_run(config, model_name, collection, seed, *, device="xpu:0", stop=None, session_steps=0):
    checkpoints, results = run_paths(config, model_name, collection, seed)
    with claim(checkpoints / "run.lock") as acquired:
        if not acquired:
            return "busy"
        dataset = CollectionDataset(config, collection)
        spec = specification(config, model_name, collection, seed, dataset.manifest)
        if completed(checkpoints, results, spec):
            return "complete"
        (checkpoints / "complete.json").unlink(missing_ok=True)
        if (checkpoints / "config.json").exists():
            if read_json(checkpoints / "config.json") != spec:
                raise ValueError("Configuration/data/code changed; use a new output root")
        else:
            write_json(checkpoints / "config.json", spec)
        indices = dataset.splits(spec["split_seed"])
        split = split_record(dataset)
        split_path = checkpoints / "split.json"
        if split_path.exists() and read_json(split_path) != split:
            raise ValueError("Split membership changed")
        if not split_path.exists():
            write_json(split_path, split)
        stop = stop if stop is not None else Stop()
        device = select_device(device)
        seed_all(seed)
        sample = dataset[int(indices["train"][0])].clone()
        model_config = {"model_type": model_name, "models": {model_name: spec["model_options"]}}
        model = build_model(model_config, sample, device)
        optimizer = torch.optim.AdamW(model.parameters(), **spec["optimizer"])
        training = spec["training"]
        budget = training["max_samples"]
        interval = max(1, math.ceil(len(indices["train"]) * training["checkpoint_epochs"]))
        versions = {"torch": torch.__version__, "pyg": torch_geometric.__version__,
                    "numpy": np.__version__}
        state = {"samples": 0, "steps": 0, "epoch": 0, "offset": 0, "phase": "train",
                 "next_validation": interval, "best_score": float("inf"), "best_samples": 0}
        last = checkpoints / "last.pt"
        if last.exists():
            payload = torch.load(last, map_location="cpu", weights_only=False)
            if payload["specification"] != spec or payload["versions"] != versions:
                raise ValueError("Checkpoint configuration or software versions changed")
            model.load_state_dict(payload["model"], strict=True)
            optimizer.load_state_dict(payload["optimizer"])
            state = payload["state"]
            restore_rng(payload["rng"], device)
            print(f"Resume {model_name}/{collection}: {state['samples']:,}/{budget:,} ({state['phase']})", flush=True)

        def save(path=last):
            save_torch(path, {"specification": spec, "versions": versions, "state": state,
                             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                             "rng": capture_rng(device), "example_graph": sample,
                             "samples_per_case": dataset.sample_counts(state["samples"])})

        def status(name):
            write_json(checkpoints / "status.json", {"status": name, "phase": state["phase"],
                       "samples_seen": state["samples"], "max_samples": budget,
                       "steps": state["steps"], "device": str(device),
                       "samples_per_case": dataset.sample_counts(state["samples"])})

        if not last.exists():
            save()
        initial_steps = state["steps"]
        saved_at = time.monotonic()
        order_epoch, order = None, None
        status("running")
        while state["phase"] != "evaluate":
            if stop() or (session_steps and state["steps"] - initial_steps >= session_steps):
                save()
                status("paused")
                return "paused"
            if state["phase"] == "validate":
                identity = {"specification": digest(spec), "samples": state["samples"], "split": "val"}
                metrics = evaluate(model, dataset, indices["val"], device, training["eval_batch_size"],
                                   checkpoints / "validation-progress.json", identity, stop)
                if metrics is None:
                    save()
                    status("paused")
                    return "paused"
                if metrics["score"] < state["best_score"]:
                    state["best_score"] = metrics["score"]
                    state["best_samples"] = state["samples"]
                    save_torch(checkpoints / "best.pt", {"specification": spec,
                               "model": model.state_dict(), "samples": state["samples"],
                               "validation": metrics, "example_graph": sample})
                write_json(checkpoints / "history" / f"{state['samples']:09d}.json",
                           {"samples": state["samples"], "steps": state["steps"], "validation": metrics})
                print(json.dumps({"samples": state["samples"], "val_mse": metrics["sdk_mse"],
                                  "violation": metrics["violation"]}), flush=True)
                while state["next_validation"] <= state["samples"]:
                    state["next_validation"] += interval
                state["phase"] = "evaluate" if state["samples"] == budget else "train"
                save()
                saved_at = time.monotonic()
                continue
            if order_epoch != state["epoch"]:
                order = dataset.train_order(spec["shuffle_seed"], state["epoch"])
                order_epoch = state["epoch"]
            count = min(training["batch_size"], len(order) - state["offset"], budget - state["samples"])
            selected = indices["train"][order[state["offset"]:state["offset"] + count]]
            batch = Batch.from_data_list([dataset[int(i)] for i in selected]).to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch)
            loss = graph_mse(predictions, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MSE")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            state["samples"] += count
            state["steps"] += 1
            state["offset"] += count
            if state["offset"] == len(order):
                state["epoch"] += 1
                state["offset"] = 0
            if state["samples"] >= state["next_validation"] or state["samples"] == budget:
                state["phase"] = "validate"
            if state["phase"] == "validate" or time.monotonic() - saved_at >= training["checkpoint_seconds"]:
                save()
                status("running")
                saved_at = time.monotonic()

        if not (checkpoints / "final.pt").exists():
            save(checkpoints / "final.pt")
        best = torch.load(checkpoints / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(best["model"], strict=True)
        best_hash = file_hash(checkpoints / "best.pt")
        for split_name in ("val", "test"):
            reports = []
            for heldout_case, selected in dataset.split_groups(split_name).items():
                identity = {"specification": digest(spec), "checkpoint_sha256": best_hash,
                            "split": split_name, "case": heldout_case,
                            "indices_sha256": digest(selected.tolist())}
                path = results / split_name / f"{heldout_case}.json"
                if path.exists() and read_json(path).get("identity") == identity:
                    reports.append(read_json(path))
                    continue
                progress = checkpoints / f"{split_name}-{heldout_case}-progress.json"
                metrics = evaluate(model, dataset, selected, device, training["eval_batch_size"],
                                   progress, identity, stop)
                if metrics is None:
                    status("paused")
                    return "paused"
                report = {"identity": identity, "model": model_name, "collection": collection,
                          "case": heldout_case, "seed": seed, "trained_samples": budget,
                          "samples_per_case": dataset.sample_counts(budget),
                          "checkpoint_samples": best["samples"],
                          "checkpoint": str(checkpoints / "best.pt"), "metrics": metrics,
                          "accumulator": read_json(progress)["accumulator"]}
                write_json(path, report)
                reports.append(report)
            write_json(results / f"{split_name}.json", {
                "identity": {"specification": digest(spec), "checkpoint_sha256": best_hash,
                             "split": split_name, "cases": dataset.cases},
                "model": model_name, "collection": collection, "seed": seed,
                "trained_samples": budget, "samples_per_case": dataset.sample_counts(budget),
                "checkpoint_samples": best["samples"], "checkpoint": str(checkpoints / "best.pt"),
                "metrics": combined_metrics(reports)})
        files = result_files(checkpoints, results, dataset.cases)
        write_json(checkpoints / "complete.json", {"specification": digest(spec),
                   "samples_seen": state["samples"], "files": {k: file_hash(p) for k, p in files.items()}})
        status("complete")
        return "complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--model", required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="xpu:0")
    parser.add_argument("--seconds", type=float, default=3300)
    parser.add_argument("--session-steps", type=int, default=0)
    args = parser.parse_args()
    with Stop(args.seconds) as stop:
        result = train_run(load_config(args.config), args.model, args.collection, args.seed,
                           device=args.device, stop=stop, session_steps=args.session_steps)
    print(result, flush=True)


if __name__ == "__main__":
    main()
