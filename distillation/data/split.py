"""Deterministic 80/10/10 split specific to dataset_id.

This is *the* authoritative dset split for all cases.
While it deviates from other papers it maintains integrity of held-out metrics.

Reference: 
It is inspired by lumina-bench although this work does not share exact indices.
Source: LUMINA (GridFM Benchmark): https://arxiv.org/abs/2605.02133

Reproducibility: 80/10/10% split; N=300_000 per case
random indices specific to case{14,...,2000}

# Usage
    splits = split_indices("pglib_opf_case118_ieee", len(ds))
    train_ds = ds[splits["train"].tolist()]
    val_ds   = ds[splits["val"].tolist()]
    test_ds  = ds[splits["test"].tolist()]
    
# Meta info
    seed: 42
    python: 3.12.12
    numpy: 2.3.5
    rng: PCG64
    dataset_seed_hash: SHA-256
    split_ratios: [0.8, 0.1, 0.1]
"""

import hashlib
import numpy as np

# dataset-specific to pglib_opf_case{XXX}_ieee
N_SAMPLES=300_000
SEED_CONSTANT=42

def split_indices(dataset_id,
                  n:int=N_SAMPLES,
                  seed:int=SEED_CONSTANT):
    """
    One 80%/10%/10% split across case{14,30,57,118,500,2000}
    """
    key = f"{seed}:{dataset_id}".encode()
    dataset_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(dataset_seed))
    indices = rng.permutation(n)
    
    # hard-coded
    a, b = int(0.8 * n), int(0.9 * n)
    
    return {
        "train": indices[:a],
        "val": indices[a:b],
        "test": indices[b:],
    }