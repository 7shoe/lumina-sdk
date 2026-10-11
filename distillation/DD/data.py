"""Native OPFData loading for DD benchmarks and optional demos."""
from __future__ import annotations

import json
from pathlib import Path
import tarfile

import numpy as np
import torch
from torch_geometric.data import HeteroData

from distillation.data.split import split_indices
from distillation.training.T1_single_topo.common import canonical_case


def native_graph(record):
    """Read the Appendix A schema, retaining JSON precision and empty-store widths."""
    grid, solution = record["grid"], record["solution"]
    data = HeteroData()
    data.x = torch.tensor(grid["context"], dtype=torch.float64).reshape(-1)
    data.objective = torch.tensor(record["metadata"]["objective"], dtype=torch.float64)
    for kind, width in (("bus", 4), ("generator", 11), ("load", 2), ("shunt", 2)):
        data[kind].x = torch.tensor(grid["nodes"][kind], dtype=torch.float64).reshape(-1, width)
    for kind in ("bus", "generator"):
        data[kind].y = torch.tensor(solution["nodes"][kind], dtype=torch.float64).reshape(-1, 2)
    for kind in ("ac_line", "transformer", "generator_link", "load_link", "shunt_link"):
        edge = grid["edges"][kind]
        source = "bus" if kind in ("ac_line", "transformer") else kind.removesuffix("_link")
        store = data[source, kind, "bus"]
        store.edge_index = torch.tensor([edge["senders"], edge["receivers"]], dtype=torch.long).reshape(2, -1)
        if source == "bus":
            width = 9 if kind == "ac_line" else 11
            store.edge_attr = torch.tensor(edge["features"], dtype=torch.float64).reshape(-1, width)
            store.edge_label = torch.tensor(solution["edges"][kind]["features"], dtype=torch.float64).reshape(-1, 4)
    return data


def load_batch(raw_root, case=14, *, count=10, seed=42, group=0, split="test"):
    """Sample the existing campaign split; stream only ten records into memory.

    Archives are read without extraction or changes to the shared data cache.
    The sampling population is the requested split within one 15,000-record group.
    """
    case = canonical_case(f"case{case}")
    if group not in range(20) or not 1 <= count <= 15_000:
        raise ValueError("Expected group in [0, 19] and count in [1, 15000]")
    
    # setup opoluation
    population = split_indices(case, seed=42)[split]
    population = population[(population >= group * 15_000) & (population < (group + 1) * 15_000)]
    
    indices = np.random.default_rng(seed).choice(population, count, replace=False)
    root = Path(raw_root).expanduser()
    paths = [root / case / f"group_{group}" / f"example_{i}.json" for i in indices]
    
    # setup records
    if all(path.is_file() for path in paths):
        records = [json.loads(path.read_text()) for path in paths]
    else:
        path = root / f"{case}_{group}.tar.gz"
        if not path.is_file():
            raise FileNotFoundError(f"OPFData archive missing: {path}. Set RAW_ROOT to dataset_release_1.")
        wanted = {f"{case}/group_{group}/example_{i}.json": int(i) for i in indices}
        found = {}
        with tarfile.open(path, "r|gz") as archive:
            for member in archive:
                name = "/".join(Path(member.name).parts[-3:])
                if member.isfile() and name in wanted:
                    found[wanted[name]] = json.load(archive.extractfile(member))
                    if len(found) == count:
                        break
        if len(found) != count:
            raise ValueError(f"Missing requested sample IDs in {path}: {sorted(set(indices) - found.keys())}")
        records = [found[int(i)] for i in indices]
    
    return [native_graph(record) for record in records], indices
