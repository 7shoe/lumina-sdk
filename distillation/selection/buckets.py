"""Fixed input partitions and capacity-aware proportional integer quotas."""
from dataclasses import dataclass

import numpy as np


def allocate(scores, count, capacities):
    """Capped proportional allocation, with stable largest-remainder rounding.

    Exhausted buckets get zero; all-zero scores fall back to equal bucket
    weights. No allocation exceeds capacity, and the feasible budget is filled.
    """
    scores = np.asarray(scores, dtype=np.float64)
    capacity = np.asarray(capacities, dtype=np.int64)
    if (scores.ndim != 1 or scores.shape != capacity.shape or count < 0
            or not np.isfinite(scores).all() or (scores < 0).any() or (capacity < 0).any()):
        raise ValueError('Finite nonnegative scores, capacities and budget required')
    
    # limit the requested budget to the total number of available examples
    count = min(int(count), int(capacity.sum()))
    result = np.zeros_like(capacity)
    while count:
        available = capacity - result
        weights = np.where(available > 0, scores, 0.)
        weights /= max(float(weights.max(initial=0.)), 1.)
        if weights.sum() == 0:
            weights = (available > 0).astype(float)
        quotas = count * weights / weights.sum()
        
        # fill buckets whose quotas reach capacity
        full = (available > 0) & (quotas >= available)
        if full.any():
            result[full] += available[full]
            count -= int(available[full].sum())
            continue
        
        # round down & award remaining slots
        whole = np.floor(quotas).astype(np.int64)
        result += whole
        count -= int(whole.sum())
        if count:
            eligible = np.flatnonzero(capacity > result)
            
            # stable sorting breaks
            order = eligible[np.argsort(-(quotas - whole)[eligible], kind='stable')]
            result[order[:count]] += 1
            count = 0
    
    return result


@dataclass
class Buckets:
    """One-dimensional partition fitted from training input descriptors only."""
    boundaries: np.ndarray
    strategy: str

    @classmethod
    def fit(cls, values, count=30, strategy='quantile'):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or count < 1:
            raise ValueError('Nonempty finite descriptor vector and positive bucket count required')
        
        # quantiles target equal population
        if strategy == 'quantile':
            inner = np.quantile(values, np.linspace(0, 1, count + 1)[1:-1])
        elif strategy == 'width':
            inner = np.linspace(values.min(), values.max(), count + 1)[1:-1]
        else:
            raise ValueError('Bucket strategy must be quantile or width')
        
        # repeated values can colapse boundaries -> fewer buckets
        inner = np.unique(inner[(inner > values.min()) & (inner < values.max())])
        # Merge empty intervals; every retained bucket needs a scoring example.
        partition = cls(inner, strategy)
        while True:
            empty = np.flatnonzero(np.bincount(partition.assign(values), minlength=partition.count) == 0)
            if not len(empty):
                return partition
            partition.boundaries = np.delete(partition.boundaries, min(int(empty[0]), partition.count - 2))

    @property
    def count(self):
        """number of buckets"""
        return len(self.boundaries) + 1

    def assign(self, values):
        """Maps descriptors to bucket indices"""
        values = np.asarray(values)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite bucket descriptor')
        
        # values exactly on a boundary belong
        return np.searchsorted(self.boundaries, values, side='right')

    def sample(self, ids, membership, scores, count, rng):
        """Sample (bucket) indices w/o replacement given bucket quota"""
        ids, membership = np.asarray(ids), np.asarray(membership)
        if ids.shape != membership.shape or not ((membership >= 0) & (membership < self.count)).all():
            raise ValueError('Invalid bucket membership')
        
        # allocate using current pool sizes
        capacities = np.bincount(membership, minlength=self.count)
        quotas = allocate(scores, count, capacities)
        chosen = [rng.choice(ids[membership == i], int(n), replace=False)
                  for i, n in enumerate(quotas) if n]
        return (rng.permutation(np.concatenate(chosen)).astype(np.int64) if chosen else
                np.empty(0, dtype=np.int64)), quotas

    def state_dict(self):
        """Return bucket boundaries"""
        return dict(boundaries=self.boundaries.tolist(), strategy=self.strategy)
