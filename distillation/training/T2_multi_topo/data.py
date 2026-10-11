"""Reuse T1 caches and balance the three training streams."""
import numpy as np
import torch

from distillation.training.T1_single_topo.data import CaseDataset
from distillation.training.T1_single_topo.common import ROOT, digest, file_hash


class CollectionDataset:
    def __init__(self, config, collection):
        self.cases = config["collections"][collection]
        self.datasets = [CaseDataset(config["paths"]["data"], case) for case in self.cases]
        self.offsets = np.cumsum([0] + [len(dataset) for dataset in self.datasets])
        self.members = [dataset.splits(42) for dataset in self.datasets]
        self.train_sizes = [len(member["train"]) for member in self.members]
        if len(set(self.train_sizes)) != 1:
            raise ValueError("Balanced epochs require equally sized case training splits")
        self.manifest = {"format": "lumina-t2-data-v1", "collection": collection,
                         "cases": {case: dataset.manifest for case, dataset in zip(self.cases, self.datasets)}}

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        case = int(np.searchsorted(self.offsets, index, side="right") - 1)
        graph = self.datasets[case][int(index - self.offsets[case])]
        graph.case_index = torch.tensor([case])
        return graph

    def split_groups(self, name):
        return {case: member[name] + offset for case, member, offset in
                zip(self.cases, self.members, self.offsets)}

    def splits(self, seed=42):
        if seed != 42:
            raise ValueError("T2 split seed must be 42")
        return {name: np.concatenate(list(self.split_groups(name).values()))
                for name in ("train", "val", "test")}

    def train_order(self, shuffle_seed, epoch):
        generator = torch.Generator().manual_seed(shuffle_seed + epoch)
        size = self.train_sizes[0]
        streams = [torch.randperm(size, generator=generator).numpy() + case * size
                   for case in range(len(self.cases))]
        return np.stack(streams, axis=1).reshape(-1)

    def sample_counts(self, samples):
        each, extra = divmod(samples, len(self.cases))
        return {case: each + int(index < extra) for index, case in enumerate(self.cases)}


def split_record(dataset):
    return {"seed": 42, "global_offsets": dataset.offsets.tolist(),
            "indices_by_case": {case: {name: ids.tolist() for name, ids in member.items()}
                                for case, member in zip(dataset.cases, dataset.members)},
            "implementation_sha256": file_hash(ROOT / "distillation/data/split.py"),
            "data_identity": digest(dataset.manifest)}
