"""DREAM-style nearest-cluster representatives in current model feature space.

Unlike image class partitions, each AC-OPF case is one regression population.
The candidate pool is a seeded, rotating training-only subset for tractability.
"""

import numpy as np
import torch

from distillation.synthetic.runtime import graph_batch
from .features import (forward_features,
                       node_graph_ids,
                       pooled)


def representatives(features, count, seed=42, iterations=30):
    """Deterministic k-means++ and nearest actual points; returns unique positions."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or not 0 < count <= len(x) or not np.isfinite(x).all():
        raise ValueError('Finite feature matrix with 0 < clusters <= samples required')
    x = x / max(float(np.abs(x).max()), 1.)  # Common rescaling preserves Euclidean assignments.
    rng = np.random.default_rng(seed)
    centers, chosen = [], []
    distances = np.full(len(x), np.inf)
    
    # seed centers with probability proportional to squared distance from existing centers
    for _ in range(count):
        available = np.ones(len(x), dtype=bool); available[chosen] = False
        weights = np.where(available, distances, 0.)
        
        # coose uniformly for the first center when possible
        if not np.isfinite(weights.sum()) or weights.sum() == 0:
            index = int(rng.choice(np.flatnonzero(available)))
        else:
            index = int(rng.choice(len(x), p=weights / weights.sum()))
        centers.append(x[index].copy()); chosen.append(index)
        distances = np.minimum(distances, ((x - x[index]) ** 2).sum(1))
    centers = np.stack(centers)
    
    # alternate nearest-center assignments and means; retain centers of empty clusters
    for _ in range(iterations):
        distances = ((x[:, None] - centers[None]) ** 2).sum(2)
        assignments = distances.argmin(1)
        updated = np.stack([x[assignments == i].mean(0) if np.any(assignments == i) else centers[i]
                            for i in range(count)])
        if np.array_equal(updated, centers):
            break
        centers = updated
    
    # replace each center with its nearest unused actual candidate
    distances = ((x[:, None] - centers[None]) ** 2).sum(2)
    selected = []
    for i in range(count):
        distances[selected, i] = np.inf
        selected.append(int(distances[:, i].argmin()))
    return selected


@torch.no_grad()
def select_real(model, dataset, candidate_ids, count, *, batch_size=64, seed=42):
    device = next(model.parameters()).device
    embeddings = []
    
    # pool final-layer node features into one embedding per candidate graph
    for start in range(0, len(candidate_ids), batch_size):
        batch = graph_batch(dataset, candidate_ids[start:start + batch_size]).to(device)
        hidden = forward_features(model.eval(), batch)[1][-1]
        embeddings.append(pooled(hidden, node_graph_ids(batch), batch.num_graphs).cpu())
        
    # map selected embedding positions back to dataset indices
    chosen = representatives(torch.cat(embeddings).numpy(), count, seed)
    
    return [int(candidate_ids[i]) for i in chosen]
