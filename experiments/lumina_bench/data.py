"""SDK datasets and persisted, topology-specific train/validation/test indices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Subset

from lumina.dataset.opf.opf_dataset import OPFDataset
from lumina.trainer.opf.utils import parse_case_name


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_data(config, root, raw_root=None):
    """Load every requested group. Optional raw_root contains cached tar.gz files.

    Processing must be performed once before concurrent jobs share this cache;
    the upstream processor uses a shared temporary extraction directory.
    """
    case = parse_case_name(config["case"])
    groups = config["groups"]
    if not groups or len(set(groups)) != len(groups):
        raise ValueError("groups must be nonempty and contain no duplicates")
    if any(not isinstance(g, int) or not 0 <= g < 20 for g in groups):
        raise ValueError("group IDs must be integers in [0, 19]")
    datasets, sources = [], []
    for group in groups:
        if raw_root:
            name = f"{case}_{group}.tar.gz"
            source = Path(raw_root).expanduser().resolve() / name
            if not source.is_file():
                raise FileNotFoundError(source)
            target = Path(root) / "OPFData/raw/dataset_release_1" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(source)
        dataset = OPFDataset(root=str(root), case_name=case, group_id=group,
                             n_jobs=int(config.get("preprocess_workers", 4)))
        processed = Path(dataset.processed_paths[0])
        sources.append({"group": group, "count": len(dataset),
                        "processed_sha256": file_sha256(processed)})
        datasets.append(dataset)
    return ConcatDataset(datasets), {"case": case, "sources": sources}


def split_data(dataset, identity, config, manifest_path=None):
    """The split seed is independent of model seed and case-list position."""
    seed = int(config.get("split_seed", 42))
    ratios = config.get("split", [0.8, 0.1, 0.1])
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("split must contain three positive fractions summing to one")
    requested = {**identity, "seed": seed, "fractions": ratios,
                 "dataset_length": len(dataset), "version": 1}
    if manifest_path and Path(manifest_path).exists():
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest["identity"] != requested:
            raise ValueError("Split manifest does not match dataset/configuration")
    else:
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
        n_train, n_val = int(len(order) * ratios[0]), int(len(order) * ratios[1])
        manifest = {"identity": requested, "indices": {
            "train": order[:n_train], "val": order[n_train:n_train + n_val],
            "test": order[n_train + n_val:]}}
    indices = manifest["indices"]
    combined = sum((indices[name] for name in ("train", "val", "test")), [])
    if len(combined) != len(dataset) or set(combined) != set(range(len(dataset))):
        raise ValueError("Split indices must partition the dataset exactly once")
    if any(not indices[name] for name in ("train", "val", "test")):
        raise ValueError("Every split must be nonempty")
    if manifest_path and not Path(manifest_path).exists():
        write_json(manifest_path, manifest)
    limits = config.get("limits", {})
    subsets = {}
    for name in ("train", "val", "test"):
        limit = int(limits.get(name, 0))
        if limit < 0:
            raise ValueError("split limits must be nonnegative")
        selected = indices[name][:limit] if limit else indices[name]
        subsets[name] = Subset(dataset, selected)
    return subsets, manifest
