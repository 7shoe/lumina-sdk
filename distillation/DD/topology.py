"""DD replay scoring; physical topology utilities live in distillation.synthetic.

Re-export the original API so existing notebooks and campaign artifacts keep working.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from distillation.DD.mvp import SyntheticOPF, unroll
from distillation.synthetic import topology as physical_topology
from distillation.synthetic.physical import audit_networks
from distillation.synthetic.topology import (
    OPERATIONS, batch_report, bus_limits, diversity, propose, refresh_seeds, topology_signature,
)


@dataclass
class Candidate:
    row: dict
    graphs: object = None


def solve_prefixes(matrices, original, *, solver="pypower"):
    return physical_topology.solve_prefixes(matrices, original, solver=solver, audit=audit_networks)


def generate_dataset(candidate, verified_graphs, **options):
    return physical_topology.generate_dataset(candidate, verified_graphs, audit=audit_networks, **options)


def evaluate_candidate(candidate, verified_graphs, model, real_batch, real_targets, *,
                       min_buses, max_buses, seed=42, synthetic_options=None, solver="pypower",
                       max_edits=2, attempts_per_graph=6, min_topologies_per_batch=3):
    """Score a whole diverse synthetic dataset from the same model initialization."""
    started = perf_counter()
    options = synthetic_options or {}
    steps = options.get("steps", 3)
    diversity(verified_graphs, steps)  # Validate the partition before proposing expensive solves.
    required = min(len(verified_graphs) // steps, min_topologies_per_batch)
    if required < 1:
        raise ValueError("At least one topology per nonempty batch is required")
    graphs, changes = generate_dataset(candidate, verified_graphs, min_buses=min_buses, max_buses=max_buses,
                                      seed=seed, max_edits=max_edits, attempts_per_graph=attempts_per_graph, solver=solver)
    counts = diversity(graphs, steps)
    row = dict(candidate=candidate, operation="baseline" if candidate == 0 else "mixed",
               status="rejected", eligible=min(counts) >= required, required_topologies=required,
               bus_min=min(g["bus"].num_nodes for g in graphs), bus_max=max(g["bus"].num_nodes for g in graphs),
               topologies_per_batch=counts, changed_graphs=sum(item["changed"] for item in changes),
               rejected_attempts=sum(item["rejected_attempts"] for item in changes),
               backtracked_graphs=sum(item["backtracked_edits"] > 0 for item in changes),
               backtracked_edits=sum(item["backtracked_edits"] for item in changes),
               real_mse=None, solver_success=True, changes=changes, error="")
    if row["eligible"] or candidate == 0:
        synthetic = SyntheticOPF(graphs, **options)
        _, loss, _ = unroll(model, synthetic, real_batch, real_targets, create_graph=False)
        if torch.isfinite(loss):
            row.update(status="scored", real_mse=float(loss.detach()))
        else:
            row["error"] = "Nonfinite replay loss"
    if not row["eligible"]:
        row["error"] = f"Requires {required} distinct topologies in each inner batch; observed {counts}"
    row["seconds"] = perf_counter() - started
    return Candidate(row, graphs)


def choose_candidate(candidates):
    """Deterministic objective selection; a failed topology can never win."""
    scored = [item for item in candidates if item.row["status"] == "scored" and item.row["eligible"]
              and item.row["solver_success"] and item.row["real_mse"] is not None
              and np.isfinite(item.row["real_mse"])]
    if not scored:
        raise RuntimeError("No topology candidate passed the solver and replay checks")
    return min(scored, key=lambda item: (item.row["real_mse"], item.row["candidate"]))


def search_topology(seed_graphs, model, real_batch, real_targets, *, trials=4, **options):
    """Refresh feasible seeds, then select a diverse proposal for the complete dataset."""
    if trials < 1:
        raise ValueError("At least one trial (the baseline) is required")
    seeds = refresh_seeds(seed_graphs, solver=options.get("solver", "pypower"))
    if trials == 1:
        options = options | {"min_topologies_per_batch": 1}
    candidates = [evaluate_candidate(i, seeds, model, real_batch, real_targets, **options)
                  for i in range(trials)]
    winner = choose_candidate(candidates)
    table = pd.DataFrame([item.row for item in candidates])
    table["selected"] = table.candidate == winner.row["candidate"]
    return winner.graphs, table
