"""python -m experiments.lumina_bench {prepare,train,evaluate}."""
import argparse
import json
from pathlib import Path

import yaml

from .data import write_json
from .models import SUPPORTED_MODELS
from .runner import data_splits, evaluate_checkpoint, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True, type=Path)
        p.add_argument("--data-root", required=True, type=Path)
        p.add_argument("--raw-root", type=Path)
        p.add_argument("--split-file", type=Path)
        p.add_argument("--model-type", choices=SUPPORTED_MODELS)
        p.add_argument("--seed", type=int)
        if command == "train":
            p.add_argument("--output", required=True, type=Path)
            p.add_argument("--device", default="xpu:0")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--case")
    p.add_argument("--groups", nargs="+", type=int)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--split-file", type=Path)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command in {"prepare", "train"}:
        config = yaml.safe_load(args.config.read_text())
        if args.model_type:
            config["model_type"] = args.model_type
        if args.seed is not None:
            config["seed"] = args.seed
        if args.command == "prepare":
            splits, manifest = data_splits(config, args.data_root, args.raw_root, args.split_file)
            print(json.dumps({"identity": manifest["identity"],
                              "effective_split_sizes": {k: len(v) for k, v in splits.items()}}))
        else:
            train(config, data_root=args.data_root, output=args.output, device=args.device,
                  raw_root=args.raw_root, split_file=args.split_file)
    else:
        result = evaluate_checkpoint(args.checkpoint, data_root=args.data_root,
            device=args.device, case=args.case, groups=args.groups, split=args.split,
            split_file=args.split_file, batch_size=args.batch_size)
        if args.output.exists():
            raise FileExistsError(args.output)
        write_json(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
