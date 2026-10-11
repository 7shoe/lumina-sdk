"""Cached input-only descriptors for the canonical memory-mapped OPFData pool."""
import numpy as np
import torch

from .pool import ids_hash
from distillation.training.T1_single_topo.common import atomic_file, digest

def load_descriptors(dataset, train_ids, cache):
    """Total active demand (p.u.); no solution, objective or holdout statistics.

    Contiguous processed-group tensors avoid constructing 240k PyG objects.
    Only training rows are reduced; the cache contains exactly those IDs.
    """
    identity = dict(version=1, data=digest(dataset.manifest), train=ids_hash(train_ids),
                    descriptor='sum(load.pd), per-unit')
    if cache.exists():
        # reuse cache descriptors
        saved = torch.load(cache, map_location='cpu', weights_only=False)
        if saved['identity'] != identity or not np.array_equal(saved['ids'], train_ids):
            raise ValueError('Descriptor cache identity changed')
        values = saved['values']
    else:
        # compute each training sample's total active demand
        values = np.empty(len(train_ids), dtype=np.float64)
        group_size = dataset.manifest['group_size']
        for group, (data, slices) in enumerate(dataset.groups):
            # locate training IDs in this group
            positions = np.flatnonzero(np.asarray(train_ids) // group_size == group)
            local = np.asarray(train_ids)[positions] % group_size
            offsets = slices['load']['x'].numpy()
            pd = data['load'].x[:, 0].numpy()
            
            # one-time CPU scan-> do not materialize graph dictionaries.
            values[positions] = [pd[offsets[i]:offsets[i + 1]].sum(dtype=np.float64) for i in local]
            
        # write completed cache
        with atomic_file(cache) as stream:
            torch.save(dict(identity=identity, ids=np.asarray(train_ids), values=values), stream)
    
    # validate both loaded and newly computed descriptors
    if values.shape != (len(train_ids),) or not np.isfinite(values).all():
        raise ValueError('Invalid demand descriptors')
    return values, identity
