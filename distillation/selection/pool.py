"""Train-only candidate access and auditable, ordered subset acquisition."""
import hashlib
import json
from pathlib import Path

import numpy as np
from torch.utils.data import Subset


def ids_hash(ids):
    """hash a consistent integer representation so that ID order affects the hash"""
    return hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()


def input_view(graph):
    """Drop all OPF supervision, including branch targets and objective metadata."""
    graph = graph.clone()
    for store in graph.stores:
        for key in ('y', 'edge_label', 'objective'):
            if key in store:
                del store[key]
    return graph


class TrainingPool:
    """Selection methods can inspect candidate inputs, and reveal acquired labels.

    The underlying processed files already contain labels. This is a logical
    access boundary, not a security boundary or a claim of saved solver work.
    A fixed reference set is labeled but never eligible for model training.
    """
    def __init__(self, dataset, splits, reference_ids=()):
        self.dataset = dataset
        self.splits = {k: np.asarray(splits[k], dtype=np.int64).copy()
                       for k in ('train', 'val', 'test')}
        combined = np.concatenate(list(self.splits.values()))
        if (len(combined) != len(dataset) or
                not np.array_equal(np.sort(combined), np.arange(len(dataset)))):
            raise ValueError('Splits must partition the dataset exactly once')
        self.train_ids = self.splits['train']
        self._train = set(map(int, self.train_ids))
        self.reference_ids = self._check(reference_ids)
        self.selected_ids = []
        self.history = []

    def _check(self, ids):
        ids = list(map(int, ids))
        if len(set(ids)) != len(ids) or not set(ids) <= self._train:
            raise ValueError('IDs must be unique members of the existing training split')
        return ids

    @property
    def remaining_ids(self):
        used = set(self.reference_ids) | set(self.selected_ids)
        return np.asarray([i for i in self.train_ids if i not in used], dtype=np.int64)

    def inputs(self, ids):
        return [input_view(self.dataset[i]) for i in self._check(ids)]

    def labeled(self, ids):
        ids = self._check(ids)
        if not set(ids) <= set(self.reference_ids) | set(self.selected_ids):
            raise ValueError('Candidate labels cannot be read before acquisition')
        return [self.dataset[i].clone() for i in ids]

    def acquire(self, ids, *, reason, details=None):
        ids = self._check(ids)
        if set(ids) & (set(self.reference_ids) | set(self.selected_ids)):
            raise ValueError('Acquisition contains a reserved or already selected ID')
        if ids:
            self.selected_ids.extend(ids)
            self.history.append(dict(ids=ids, reason=reason, details=details or {}))
        return ids

    def state_dict(self):
        return dict(split_hashes={k: ids_hash(v) for k, v in self.splits.items()},
                    reference_ids=self.reference_ids.copy(), selected_ids=self.selected_ids.copy(),
                    history=self.history.copy())

    def load_state_dict(self, state):
        # resume only with the same ordered splits and reference panel
        if state['split_hashes'] != self.state_dict()['split_hashes']:
            raise ValueError('Exact split IDs changed')
        if state['reference_ids'] != self.reference_ids:
            raise ValueError('Reference panel changed')
        selected = self._check(state['selected_ids'])
        if set(selected) & set(self.reference_ids):
            raise ValueError('Reference panel leaked into training')
        
        # the recorded acquisition events must reproduce
        if [i for event in state['history'] for i in event['ids']] != selected:
            raise ValueError('Acquisition history does not match the selected IDs')
        self.selected_ids, self.history = selected, state['history'].copy()


def selected_dataset(dataset, manifest_path):
    """Reload an exported subset without copying or modifying physical graphs."""
    from distillation.training.T1_single_topo.common import digest
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest['data_identity'] != digest(dataset.manifest):
        raise ValueError('Selected subset belongs to a different dataset')
    
    # rebuild and validate the saved pool before exposing its selected samples
    pool = TrainingPool(dataset, dataset.splits(42), manifest['pool']['reference_ids'])
    pool.load_state_dict(manifest['pool'])
    
    return Subset(dataset, pool.selected_ids)
