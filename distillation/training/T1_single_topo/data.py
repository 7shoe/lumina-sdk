"""Canonical sample IDs and memory-mapped graph groups."""
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.separate import separate

from distillation.data.split import split_indices
from lumina.dataset.opf.opf_dataset import build_heterodata_from_grid
from .common import ROOT, atomic_file, canonical_case, claim, file_hash, read_json, write_json

GROUP_SIZE = 15_000
GROUPS = 20

def processor_identity():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in
            [Path(__file__), ROOT / "lumina/dataset/opf/opf_dataset.py",
             ROOT / "lumina/dataset/opf/utils.py", ROOT / "lumina/dataset/opf/schema.py"]}


def prepare_group(raw_root, data_root, case, group, group_size=GROUP_SIZE):
    case = canonical_case(case)
    folder = Path(data_root) / case
    path = folder / f"group_{group}.pt"
    archive = Path(raw_root) / f"{case}_{group}.tar.gz"
    metadata = path.with_suffix(".json")
    with claim(folder / f"group_{group}.lock") as acquired:
        if not acquired:
            return False
        raw_stat = archive.stat()
        source = {"name": archive.name, "bytes": raw_stat.st_size,
                  "mtime_ns": raw_stat.st_mtime_ns}
        processor = processor_identity()
        if path.exists() and metadata.exists():
            record = read_json(metadata)
            if (record["source"] != source or record["processor"] != processor
                    or record["count"] != group_size):
                raise ValueError(f"Changed source/processor for {path}; use a new data root")
            if file_hash(path) != record["sha256"]:
                raise ValueError(f"Damaged cache: {path}")
            return True
        pattern = re.compile(rf"(?:^|/){case}/group_{group}/example_(\d+)\.json$")
        graphs = {}
        # Stream JSON directly; no shared extraction directory.
        with tarfile.open(archive, "r|gz") as stream:
            for member in stream:
                if not member.isfile():
                    continue
                match = pattern.search(member.name)
                if not match:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                index = int(match[1])
                if index in graphs or not group * group_size <= index < (group + 1) * group_size:
                    raise ValueError(f"Duplicate/out-of-range sample ID: {index}")
                obj = json.load(stream.extractfile(member))
                graph = build_heterodata_from_grid(obj["grid"], obj["metadata"], obj["solution"])
                if any(not torch.isfinite(graph[k].y).all() for k in ("bus", "generator")):
                    raise ValueError(f"Non-finite targets: {member.name}")
                graph.sample_id = torch.tensor([index])
                graphs[index] = graph
        if len(graphs) != group_size:
            raise ValueError(f"Expected {group_size} samples in {archive}; got {len(graphs)}")
        data, slices = InMemoryDataset.collate([graphs[i] for i in sorted(graphs)])
        with atomic_file(path) as stream:
            torch.save((data, slices), stream)
        stat = path.stat()
        write_json(metadata, {"group": group, "count": group_size, "source": source,
                              "raw_sha256": file_hash(archive), "processor": processor,
                              "sha256": file_hash(path), "bytes": stat.st_size,
                              "mtime_ns": stat.st_mtime_ns})
    return True


def finish_case(data_root, case, groups=GROUPS, group_size=GROUP_SIZE):
    folder = Path(data_root) / canonical_case(case)
    records = []
    for group in range(groups):
        path = folder / f"group_{group}.pt"
        metadata = path.with_suffix(".json")
        if not metadata.exists():
            return False
        record = read_json(metadata)
        if record["group"] != group or record["count"] != group_size:
            raise ValueError(f"Invalid group record: {metadata}")
        if record["processor"] != processor_identity():
            raise ValueError(f"Cache was built by a different processor: {path}")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Incomplete group: {path}")
        records.append(record)
    write_json(folder / "manifest.json", {"format": "lumina-t1-data-v1",
               "case": canonical_case(case), "count": groups * group_size,
               "group_size": group_size, "order": "numeric example_ID", "groups": records})
    return True


class CaseDataset:
    def __init__(self, root, case):
        self.folder = Path(root) / canonical_case(case)
        self.manifest = read_json(self.folder / "manifest.json")
        if self.manifest["format"] != "lumina-t1-data-v1":
            raise ValueError("Unsupported data manifest")
        records = self.manifest["groups"]
        if (self.manifest["case"] != canonical_case(case)
                or self.manifest["count"] != len(records) * self.manifest["group_size"]
                or [record["group"] for record in records] != list(range(len(records)))):
            raise ValueError("Invalid case/group manifest")
        processor = processor_identity()
        self.groups = []
        for record in records:
            if record["processor"] != processor:
                raise ValueError("Cache processor changed; prepare a new data root")
            path = self.folder / f"group_{record['group']}.pt"
            stat = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (record["bytes"], record["mtime_ns"]):
                raise ValueError(f"Cache changed after preparation: {path}")
            self.groups.append(torch.load(path, mmap=True, map_location="cpu", weights_only=False))

    def __len__(self):
        return self.manifest["count"]

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        group, offset = divmod(index, self.manifest["group_size"])
        data, slices = self.groups[group]
        graph = separate(data.__class__, data, offset, slices, decrement=False)
        if graph.sample_id.item() != index:
            raise ValueError(f"Sample order mismatch at {index}")
        return graph

    def splits(self, seed=42):
        return split_indices(self.manifest["case"], len(self), seed)


def split_record(case, indices, seed):
    return {"case": canonical_case(case), "seed": seed, "order": "numeric example_ID",
            "indices": {key: np.asarray(value).tolist() for key, value in indices.items()},
            "implementation_sha256": file_hash(ROOT / "distillation/data/split.py")}
