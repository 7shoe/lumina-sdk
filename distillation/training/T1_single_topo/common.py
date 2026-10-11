"""Configuration, atomic files, and run identity."""
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")
CASE_NAMES = {f"case{n}": f"pglib_opf_case{n}_{suffix}" for n, suffix in
              [(14, "ieee"), (30, "ieee"), (57, "ieee"), (118, "ieee"), (500, "goc")]}


def canonical_case(case):
    if case in CASE_NAMES.values():
        return case
    return CASE_NAMES[case]


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


@contextmanager
def atomic_file(path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open(mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_file(path, "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextmanager
def claim(path):
    """Locks are released by the OS when a worker dies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def fixed_split_seed(config):
    if type(config.get("split_seed")) is not int or config["split_seed"] != 42:
        raise ValueError("T1 split_seed must be 42 for every run")
    return 42


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text())
    fixed_split_seed(config)
    config["cases"] = [canonical_case(case) for case in config["cases"]]
    if not config["cases"] or len(set(config["cases"])) != len(config["cases"]):
        raise ValueError("Cases must be nonempty and unique")
    if not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config["seeds"]):
        raise ValueError("Seeds must be integers in [0, 2**32)")
    if type(config["shuffle_seed"]) is not int or not 0 <= config["shuffle_seed"] < 2**32:
        raise ValueError("shuffle_seed must be an integer in [0, 2**32)")
    supported = {"GCN", "GAT", "GIN", "Transformer", "HGT", "RGAT", "HEAT", "HeteroGNN"}
    if not config["models"] or set(config["models"]) - supported:
        raise ValueError("Choose supported model names in config.models")
    for key, value in config["training"].items():
        if not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"training.{key} must be positive")
    for key in ("max_samples", "batch_size", "eval_batch_size"):
        if type(config["training"][key]) is not int:
            raise ValueError(f"training.{key} must be an integer")
    return config


def source_identity():
    files = list(Path(__file__).parent.glob("*.py"))
    files += [ROOT / name for name in (
        "distillation/data/split.py", "experiments/lumina_bench/models.py",
        "experiments/lumina_bench/metrics.py", "experiments/lumina_bench/runner.py",
        "lumina/model/opf/hetero_model.py", "lumina/model/opf/homo_model.py",
        "lumina/model/opf/losses.py", "lumina/utils/graph_utils.py")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(files)}


def specification(config, model, case, seed, manifest):
    return {"format": "lumina-t1-v1", "model": model, "case": canonical_case(case),
            "seed": seed, "split_seed": fixed_split_seed(config),
            "shuffle_seed": config["shuffle_seed"],
            "model_options": config["models"][model], "training": config["training"],
            "optimizer": config["optimizer"], "data_identity": digest(manifest),
            "source_identity": source_identity()}


def run_paths(config, model, case, seed):
    root = Path(config["paths"]["output"])
    suffix = Path(model) / canonical_case(case) / f"seed{seed}"
    return root / "checkpoints" / suffix, root / "results" / suffix


def completed(checkpoints, results, spec):
    marker = checkpoints / "complete.json"
    try:
        record = read_json(marker)
    except FileNotFoundError:
        return False
    if record["specification"] != digest(spec):
        raise ValueError(f"Configuration/data/code changed: {checkpoints}; use a new output root")
    paths = {"best.pt": checkpoints / "best.pt", "final.pt": checkpoints / "final.pt",
             "val.json": results / "val.json", "test.json": results / "test.json"}
    missing = False
    for name, path in paths.items():
        if not path.is_file():
            missing = True
        elif file_hash(path) != record["files"].get(name):
            raise ValueError(f"Completed artifact changed: {path}")
    return not missing
