"""Shared sampling, MPI reductions and atomic I/O; no learning objective."""
import json
import time
import math
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from distillation.solver import ACOPF
from distillation.synthetic import audit_networks
from distillation.training.T1_single_topo.common import atomic_file


def save(path, payload):
    with atomic_file(path) as stream:
        torch.save(payload, stream)



def log(path, event):
    event = {'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **event}
    with Path(path).open('a') as stream:
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()
    print(json.dumps(event, allow_nan=False), flush=True)



def graph_batch(dataset, ids):
    return Batch.from_data_list([dataset[int(i)] for i in ids])



def sampler_ids(population, iteration, rank, world, count, seed=20261010):
    """Stateless cyclic shuffled epochs; all workers sample disjoint positions.

    The iteration checkpoint is the sampler cursor. A new deterministic shuffle
    is drawn for each epoch, so the full population is accessible on resume.
    """
    start = (iteration * world + rank) * count
    result = []
    while len(result) < count:
        epoch, offset = divmod(start, len(population))
        order = np.random.default_rng(seed + epoch).permutation(len(population))
        take = min(count - len(result), len(population) - offset)
        result.extend(population[order[offset:offset + take]].tolist())
        start += take
    return result



def averaged_gradients(synthetic, comm):
    """CPU float64 reduction; only identically shaped shared synthetic data enter."""
    from mpi4py import MPI
    parameters = [p for p in synthetic.parameters() if p.requires_grad]
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        valid = False
    else:
        valid = True
    if not comm.allreduce(valid, op=MPI.LAND):
        raise FloatingPointError('Missing/nonfinite meta-gradient on at least one worker')
    vector = np.concatenate([p.grad.detach().double().numpy().ravel() for p in parameters])
    comm.Allreduce(MPI.IN_PLACE, vector, op=MPI.SUM)
    vector /= comm.size
    offset = 0
    for p in parameters:
        p.grad.copy_(torch.from_numpy(vector[offset:offset + p.numel()].copy()).reshape(p.shape))
        offset += p.numel()
    return float(np.linalg.norm(vector))



def collective_audit(synthetic, comm):
    local = []
    with torch.no_grad():
        for i in range(comm.rank, len(synthetic.anchors), comm.size):
            matrix = synthetic.edits[i](synthetic.anchors[i], synthetic.strength)
            checks, solutions = audit_networks([matrix])
            row = checks.iloc[0].to_dict()
            row['sample'] = i
            graph = solutions[0].to_hetero(ACOPF(matrix)) if row['success'] else None
            local.append((i, row, graph))
    records = sorted(item for group in comm.allgather(local) for item in group)
    return all(row['success'] for _, row, _ in records), [g for _, _, g in records], [r for _, r, _ in records]



def score_summary(scores):
    finite = all(math.isfinite(x) for x in scores)
    # Calibration can produce enormous but finite losses in FP64. Scaling
    # avoids overflow in the sum and squared deviations used by numpy.std.
    scale = max((abs(x) for x in scores), default=0.) or 1.
    normalized = np.asarray(scores) / scale if finite else None
    return {'mean': float(np.mean(normalized) * scale) if finite else None,
            'std': float(np.std(normalized) * scale) if finite else None,
            'per_initialization': [x if math.isfinite(x) else None for x in scores],
            'nonfinite_initializations': sum(not math.isfinite(x) for x in scores)}

