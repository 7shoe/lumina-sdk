"""Shared initialization and launch plumbing; no DD/DCGM optimization objective."""
from pathlib import Path

import numpy as np
import torch

from . import generate, bus_limits, diversity, describe


def case_bus_range(case):
    """New campaign default: +/-25%, rounding half buses upward, as requested."""
    bus_limits(case)  # Validate the positive integer anchor.
    return [(3 * int(case) + 2) // 4, (5 * int(case) + 2) // 4]


def balanced_sizes(case, batches, batch_size, seed=42, bus_range=None):
    """Cover both size limits while keeping EXACTLY batch_size * case buses per batch."""
    low, high = case_bus_range(case) if bus_range is None else bus_range
    if not low <= case <= high:
        raise ValueError('Bus range must contain the base case')
    if batch_size < 4:
        raise ValueError("At least four graphs are needed to span limits at a matched bus budget")
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(batches):
        sizes = np.rint(np.geomspace(low, high, batch_size)).astype(int)
        while sizes.sum() != case * batch_size:
            direction = 1 if sizes.sum() < case * batch_size else -1
            eligible = [i for i in range(1, batch_size - 1) if low <= sizes[i] + direction <= high]
            sizes[rng.choice(eligible)] += direction
        result.extend(rng.permutation(sizes).tolist())
    return result


def initialize(config, folder, comm):
    """Distribute expensive solver calls; checkpoint each accepted initial sample."""
    from distillation.synthetic.runtime import save
    
    source = torch.load(config['reference_file'], weights_only=False, map_location='cpu') if comm.rank == 0 else None
    source = comm.bcast(source, root=0)
    count = config['synthetic_count']
    
    if len(source['graphs']) < count or len(source['ids']) < count:
        raise ValueError('The reference file needs at least synthetic_count graphs and IDs for a matched real-data control')
    original, ids = source['graphs'][:count], [int(i) for i in source['ids'][:count]]
    protocol = config['generation_protocol']
    limits = config.get('bus_range', bus_limits(config['case']))  # Preserve existing runs on resume.
    sizes = (balanced_sizes(config['case'], config['batches'], config['batch_size'], config['data_seed'], limits)
             if protocol == 'random_size' else [config['case']] * count)
    
    folder = Path(folder)
    cache = folder / 'initial_samples'
    cache.mkdir(exist_ok=True)
    local, errors = [], []
    for index in range(comm.rank, count, comm.size):
        path = cache / f'{index:04d}.pt'
        try:
            if path.exists():
                item = torch.load(path, weights_only=False, map_location='cpu')
                if item['protocol'] != protocol or item['case'] != config['case']:
                    raise ValueError('Cached initialization does not match this protocol')
                if item['metadata']['bus_limits'] != list(limits):
                    raise ValueError('Cached initialization uses different bus limits')
            else:
                mode = 'conservative' if protocol == 'conservative' else 'random'
                result = generate(original, mode=mode, original_buses=config['case'], count=1,
                    sizes=None if mode == 'conservative' else [sizes[index]],
                    min_topologies=1, seed=config['data_seed'] + index * 104729,
                    attempts=config.get('generation_attempts', 64), max_edits=2,
                    strength=0.05 if mode == 'conservative' else 0.5, reference_ids=ids,
                    bus_range=limits)
                item = dict(graph=result.physical_graphs[0], row=result.report.iloc[0].to_dict(),
                            metadata=result.metadata, protocol=protocol, case=config['case'])
                save(path, item)
            if not limits[0] <= item['graph']['bus'].num_nodes <= limits[1]:
                raise ValueError('Generated graph outside this experiment bus range')
            local.append((index, item['graph'], item['row']))
        except Exception as exc:
            errors.append(dict(index=index, error=f'{type(exc).__name__}: {exc}'))
            if hasattr(exc, 'failures'):
                save(cache / f'{index:04d}_failure.pt', exc.failures)
    failures = [r for group in comm.allgather(errors) for r in group]
    if failures:
        raise RuntimeError(f'Initialization failed (accepted samples are cached): {failures}')
    ordered = sorted(r for group in comm.allgather(local) for r in group)
    graphs, changes = [g for _, g, _ in ordered], [r for _, _, r in ordered]
    
    if min(diversity(graphs, config['batches'])) < min(3, config['batch_size']):
        raise RuntimeError('Initial per-batch diversity requirement failed')
    if comm.rank == 0:
        save(folder / 'original.pt', dict(graphs=original, ids=ids))
        save(folder / 'initial.pt', dict(graphs=graphs, changes=changes,
            statistics=[describe(g) for g in graphs], protocol=protocol, requested_sizes=sizes))
    return original, ids, graphs, changes
