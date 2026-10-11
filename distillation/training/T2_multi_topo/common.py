"""T2 configuration, run identity, and completion checks."""
from pathlib import Path

from distillation.training.T1_single_topo import common as t1
from distillation.training.T1_single_topo.common import (
    ROOT, atomic_file, canonical_case, claim, digest, file_hash, fixed_split_seed,
    read_json, write_json)

DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def load_config(path=DEFAULT_CONFIG):
    config = t1.load_config(path)
    for name, cases in config["collections"].items():
        if not name or Path(name).name != name:
            raise ValueError("Collection names must be plain directory names")
        cases = [canonical_case(case) for case in cases]
        if len(cases) != 3 or len(set(cases)) != 3 or not set(cases) <= set(config["cases"]):
            raise ValueError(f"Each collection needs three distinct configured cases: {name}")
        config["collections"][name] = cases
    if not config["collections"]:
        raise ValueError("No collections configured")
    return config


def specification(config, model, collection, seed, manifest):
    sources = t1.source_identity()
    sources.update({str(path.relative_to(ROOT)): file_hash(path)
                    for path in sorted(Path(__file__).parent.glob("*.py"))})
    return {"format": "lumina-t2-v1", "model": model, "collection": collection,
            "cases": config["collections"][collection], "seed": seed,
            "split_seed": fixed_split_seed(config), "shuffle_seed": config["shuffle_seed"],
            "model_options": config["models"][model], "training": config["training"],
            "optimizer": config["optimizer"], "data_identity": digest(manifest),
            "sampling": "balanced interleaved shuffled case streams",
            "loss": "mean per-graph bus MSE plus generator MSE", "source_identity": sources}


def run_paths(config, model, collection, seed):
    suffix = Path(model) / collection / f"seed{seed}"
    root = Path(config["paths"]["output"])
    return root / "checkpoints" / suffix, root / "results" / suffix


def result_files(checkpoints, results, cases):
    paths = {"best.pt": checkpoints / "best.pt", "final.pt": checkpoints / "final.pt",
             "val.json": results / "val.json", "test.json": results / "test.json"}
    paths.update({f"{split}/{case}.json": results / split / f"{case}.json"
                  for split in ("val", "test") for case in cases})
    return paths


def completed(checkpoints, results, spec):
    try:
        record = read_json(checkpoints / "complete.json")
    except FileNotFoundError:
        return False
    if record["specification"] != digest(spec):
        raise ValueError(f"Configuration/data/code changed: {checkpoints}; use a new output root")
    missing = False
    for name, path in result_files(checkpoints, results, spec["cases"]).items():
        if not path.exists():
            missing = True
        elif file_hash(path) != record["files"].get(name):
            raise ValueError(f"Completed artifact changed: {path}")
    return not missing
