"""Shared initialization and launch plumbing; no DD/DCGM optimization objective."""
from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
from distillation.experiments.staging import local_path

from . import generate, bus_limits, diversity, describe


def common_config(case):
    """New experiments only; explicit saved configurations are never upgraded on resume."""
    root = Path('/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets')
    return dict(case=case, batches=10, batch_size=10, synthetic_count=100,
        strength=.10, target_mode='solver', target_rate=.003, model_precision='float64',
        initialization='random', model_seed=1_000_000, data_seed=42, split_seed=42,
        sampler_seed=20261010, generation_protocol='conservative', bus_range=case_bus_range(case),
        generation_attempts=128, real_batch=64, eval_batch=128,
        evaluation_epochs=10, validation_samples=256, final_samples=1024, retention_samples=0,
        student_rates=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
        calibration_samples=128, calibration_seed=314159, calibration_model_seed=30_000_000,
        calibration_panels=2, student_rate_factor=1., trace_every=10, trace_samples=32,
        training_seconds=2400, max_updates=1000, checkpoint_seconds=120, eval_seconds=300,
        reserve_seconds=300, evaluation_reserve_seconds=90, qualify=True,
        raw_root=str(root/'raw/OPFData/dataset_release_1'), data_root=str(root/'processed/T1'))


def validate_condensation(config):
    if config.get('split_seed', 42) != 42:
        raise ValueError('The benchmark split remains fixed at seed 42')
    if config.get('target_mode', 'solver') not in ('solver', 'frozen', 'learned'):
        raise ValueError('target_mode must be solver, frozen or learned')
    if config.get('topology_every', 0):
        raise ValueError('DCGM/HMDC keep topology fixed after initialization')
    for key in ('batches', 'batch_size', 'synthetic_count', 'evaluation_epochs', 'restart_rounds',
                'real_batch', 'eval_batch', 'validation_samples', 'final_samples', 'max_updates'):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config['synthetic_count'] != config['batches'] * config['batch_size']:
        raise ValueError('Synthetic count and batch schedule differ')
    if not 0 < config['strength'] < 1:
        raise ValueError('strength must be in (0, 1)')
    for key in ('data_rate', 'target_rate'):
        value = config.get(key, config['data_rate'])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not config['student_rates'] or any(not math.isfinite(v) or v <= 0 for v in config['student_rates']):
        raise ValueError('student_rates must contain finite positive rates')


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


def generation_settings(config, ids):
    """Content identity of an initialization, independent of method and teaching targets."""
    protocol, count = config['generation_protocol'], config['synthetic_count']
    limits = config.get('bus_range', bus_limits(config['case']))
    sizes = (balanced_sizes(config['case'], config['batches'], config['batch_size'], config['data_seed'], limits)
             if protocol == 'random_size' else [config['case']] * count)
    return dict(protocol=protocol, case=config['case'], count=count, sizes=sizes,
        seed=config['data_seed'], limits=list(limits), reference_ids=[int(i) for i in ids],
        reference_sha256=hashlib.sha256(local_path(config['reference_file']).read_bytes()).hexdigest(),
        strength=config.get('generation_strength', 0.05 if protocol == 'conservative' else 0.5),
        max_edits=config.get('generation_max_edits', 2), attempts=config.get('generation_attempts', 64))


def initialize(config, folder, comm):
    """Serialize shared-cache generation across concurrently launched methods."""
    import fcntl
    from .runtime import collectively
    cache = Path(config.get('initial_cache', Path(folder)/'initial_samples'))
    def acquire():
        if comm.rank: return None
        cache.mkdir(parents=True, exist_ok=True)
        handle = (cache/'initialization.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX)
        return handle
    handle = collectively(comm, acquire)
    try:
        return _initialize(config, folder, comm)
    finally:
        if handle: handle.close()


def _initialize(config, folder, comm):
    """Distribute expensive solver calls; checkpoint each accepted initial sample."""
    from distillation.synthetic.runtime import save
    
    source = torch.load(local_path(config['reference_file']), weights_only=False, map_location='cpu') if comm.rank == 0 else None
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
    cache = Path(config.get('initial_cache', folder / 'initial_samples'))
    cache.mkdir(parents=True, exist_ok=True)
    generation = generation_settings(config, ids)
    fingerprint = hashlib.sha256(json.dumps(generation, sort_keys=True).encode()).hexdigest()
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
                if ('target_mode' in config and 'generation_fingerprint' not in item or
                        item.get('generation_fingerprint', fingerprint) != fingerprint):
                    raise ValueError('Cached initialization uses different generation settings or references')
            else:
                mode = 'conservative' if protocol == 'conservative' else 'random'
                result = generate(original, mode=mode, original_buses=config['case'], count=1,
                    sizes=None if mode == 'conservative' else [sizes[index]],
                    min_topologies=1, seed=config['data_seed'] + index * 104729,
                    attempts=generation['attempts'], max_edits=generation['max_edits'],
                    strength=generation['strength'], reference_ids=ids,
                    bus_range=limits)
                item = dict(graph=result.physical_graphs[0], row=result.report.iloc[0].to_dict(),
                            metadata=result.metadata, protocol=protocol, case=config['case'],
                            generation_fingerprint=fingerprint)
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
    
    if min(diversity(graphs, config['batches'])) < min(config.get('min_topologies', 3), config['batch_size']):
        raise RuntimeError('Initial per-batch diversity requirement failed')
    if comm.rank == 0:
        save(folder / 'original.pt', dict(graphs=original, ids=ids))
        save(folder / 'initial.pt', dict(graphs=graphs, changes=changes,
            statistics=[describe(g) for g in graphs], protocol=protocol, requested_sizes=sizes))
    return original, ids, graphs, changes
