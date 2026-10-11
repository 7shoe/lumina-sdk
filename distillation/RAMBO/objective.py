"""The paper's sum of log-L1 block distances, not nearest-neighbor max-min."""
import numpy as np


def sample_subsets(library, network, rng, history_size=30, variable_size=30):
    """
    Sample history entries and coordinates without replacement per pg/pd/vm block.
    Return 1-based sample and variable indices for the Julia backend.
    """
    sets = {}
    for key, width in (('pg', network['g']), ('pd', network['l']), ('vm', network['n'])):
        chosen = rng.choice(len(library), min(history_size, len(library)), replace=False)
        
        # sample coordinates per entry
        sets[key] = [dict(sample=int(j) + 1,
                         variables=(rng.choice(width, min(width, variable_size), replace=False) + 1).tolist())
                     for j in chosen]
    return sets


def distance(point, library, subsets, delta=1e-8):
    """
    Sum log(delta + L1 distance) over the selected history entries and coordinates.
    Subset indices are 1-based; positive delta prevents log(0).
    """
    value = 0.
    for key, terms in subsets.items():
        for term in terms:
            # convert stored indices back to NumPy's 0-based convention
            idx = np.asarray(term['variables']) - 1
            diff = np.asarray(point[key])[idx] - np.asarray(library[term['sample'] - 1][key])[idx]
            
            # sum log-L1 distances across sampled blocks
            value += np.log(delta + np.abs(diff).sum())
    return float(value)
