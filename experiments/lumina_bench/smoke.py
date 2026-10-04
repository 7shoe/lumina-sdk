"""Real-data XPU integration check, including CPU checkpoint portability.

python -m experiments.lumina_bench.smoke --data-root CACHE --output FRESH_DIR
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Batch

from .data import write_json
from .runner import build_model, data_splits, evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs/smoke.yaml")
    parser.add_argument("--models", nargs="+", default=["HGT", "Transformer", "RGAT"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    report = {}
    for kind in args.models:
        config["model_type"] = kind
        folder = args.output / kind
        result = train(config, data_root=args.data_root, output=folder, device="xpu:0",
                       split_file=args.output / "shared-split.json")
        initial = torch.load(folder / "initial.pt", map_location="cpu", weights_only=False)
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=False)
        assert any(not torch.equal(value, last["model_state_dict"][name])
                   for name, value in initial["model_state_dict"].items()), "No parameter changed"
        history = [json.loads(line) for line in (folder / "history.jsonl").read_text().splitlines()]
        assert result["validation"]["score"] == min(row["validation"]["score"]
                                                   for row in history if "validation" in row)
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=False)
        splits, _ = data_splits(config, args.data_root, manifest_path=folder / "split.json")
        batch = Batch.from_data_list([splits["test"][i] for i in range(min(8, len(splits["test"])))])
        outputs = {}
        for device in ("cpu", "xpu:0"):
            model = build_model(config, best["example_graph"], torch.device(device))
            model.load_state_dict(best["model_state_dict"], strict=True)
            model.eval()
            with torch.no_grad():
                outputs[device] = {k: v.cpu() for k, v in model(batch.clone().to(device)).items()}
        max_difference = 0.
        for name in outputs["cpu"]:
            cpu, xpu = outputs["cpu"][name], outputs["xpu:0"][name]
            torch.testing.assert_close(cpu, xpu, atol=2e-5, rtol=2e-4)
            max_difference = max(max_difference, (cpu - xpu).abs().max().item())
        # Different, non-dividing eval batch sizes also exercise checkpoint CLI semantics.
        cpu_eval = evaluate_checkpoint(folder / "best.pt", data_root=args.data_root,
                                       device="cpu", batch_size=3)
        xpu_eval = evaluate_checkpoint(folder / "best.pt", data_root=args.data_root,
                                       device="xpu:0", batch_size=5)
        assert cpu_eval["metrics"]["n_samples"] == result["test"]["n_samples"]
        for name in ("mse", "sdk_mse", "violation", "score"):
            torch.testing.assert_close(torch.tensor(cpu_eval["metrics"][name]),
                                       torch.tensor(xpu_eval["metrics"][name]), rtol=2e-4, atol=2e-5)
        write_json(folder / "cpu-evaluation.json", cpu_eval)
        write_json(folder / "xpu-evaluation.json", xpu_eval)
        report[kind] = {"status": "PASS", "steps": result["steps"],
                        "samples_seen": result["samples_seen"],
                        "cpu_xpu_prediction_max_abs": max_difference,
                        "ground_truth_violation": result["ground_truth_test"]["violation"],
                        "elapsed_seconds": result["elapsed_seconds"]}
        write_json(args.output / "smoke-results.json", report)
        print(json.dumps(report[kind]), flush=True)


if __name__ == "__main__":
    main()
