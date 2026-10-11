"""Discrete proposals, feasibility backtracking and batch diversity; no training objective."""

import numpy as np
import pandas as pd
import torch

from distillation.conversion.matrix_form import (NetworkEditor,
                                                 mat2opf,
                                                 opf2mat,
                                                 validate_network)
from distillation.solver import ACOPF
from .physical import audit_networks


OPERATIONS = ("drop_line",
              "add_line",
              "rewire_line",
              "split_line",
              "merge_buses")


def bus_limits(original_buses):
    """Hard limits anchored to the original case, including after repeated edits."""
    if original_buses < 1 or int(original_buses) != original_buses:
        raise ValueError("The original bus count must be a positive integer")
    return (int(original_buses) + 2) // 3, 3 * int(original_buses)


def _line_values(params, index):
    return {name: float(getattr(params, name)[index]) for name in
            ("br_r", "br_x", "b_fr", "b_to", "rate_a", "rate_b", "rate_c", "angmin", "angmax")}


def propose(graph, candidate, *, min_buses, max_buses, seed=42):
    """One reproducible discrete edit; connectedness and bus-budget violations reject it."""
    if candidate < 0 or min_buses < 1 or max_buses < min_buses:
        raise ValueError("Invalid candidate ID or bus-count budget")
    original = opf2mat(graph.clone().detach().cpu())
    p = original.params
    if candidate == 0:
        edited, operation = original, "baseline"
    else:
        if p.n_line == 0:
            raise ValueError("Topology proposals need an AC line")
        operation = OPERATIONS[(candidate - 1) % len(OPERATIONS)]
        rng = np.random.default_rng(seed + 1009 * candidate)
        index = int(rng.integers(p.n_line))
        fr, to = int(p.fr[index]), int(p.to[index])
        values = _line_values(p, index)
        editor = NetworkEditor(original)
        if operation == "drop_line":
            editor.remove_branch(index)
        elif operation == "add_line":
            existing = {tuple(sorted(pair)) for pair in zip(p.fr.tolist(), p.to.tolist())}
            kv = p.base_kv.numpy()
            pairs = np.transpose(np.nonzero(np.triu(np.isclose(kv[:, None], kv[None, :]), 1)))
            pairs = [tuple(map(int, pair)) for pair in pairs if tuple(pair) not in existing]
            if not pairs:
                raise ValueError("No unused pair of equal-voltage buses for a new AC line")
            fr, to = pairs[int(rng.integers(len(pairs)))]
            editor.add_branch(fr, to, **values)
        elif operation == "rewire_line":
            choices = [bus for bus in range(p.N) if bus not in (fr, to)
                       and bool(torch.isclose(p.base_kv[bus], p.base_kv[to]))]
            if not choices:
                raise ValueError("No compatible endpoint for rewiring")
            editor.rewire_branch(index, fr, choices[int(rng.integers(len(choices)))])
        elif operation == "split_line":
            if not torch.isclose(p.base_kv[fr], p.base_kv[to]):
                raise ValueError("Cannot insert a bus into a cross-voltage AC line")
            bus = editor.add_bus(base_kv=p.base_kv[fr], vmin=torch.maximum(p.vmin[fr], p.vmin[to]),
                                 vmax=torch.minimum(p.vmax[fr], p.vmax[to]))
            editor.remove_branch(index)
            # Each half-length pi section gets half the impedance and charging.
            # Equal endpoint charging stays representable by the PYPOWER backend.
            half = values | {name: values[name] / 2 for name in
                             ("br_r", "br_x", "b_fr", "b_to", "angmin", "angmax")}
            editor.add_branch(fr, bus, **half)
            editor.add_branch(bus, to, **half)
        else:
            editor.merge_buses(fr, to)
        edited = editor.build()
    q = edited.params
    validate_network(q)
    if not min_buses <= q.N <= max_buses:
        raise ValueError(f"Proposed {q.N} buses outside [{min_buses}, {max_buses}]")
    for name in ("pd", "qd", "pmin", "pmax", "qmin", "qmax", "c2", "c1", "c0"):
        if not torch.equal(getattr(p, name), getattr(q, name)):
            raise ValueError(f"A topology proposal must preserve {name} records")
    return edited, operation


def refresh_seeds(graphs, *, solver="pypower"):
    """Verify the unchanged fallback graphs too; every returned label is freshly solved."""
    if not graphs:
        return []
    matrices = [opf2mat(graph) for graph in graphs]
    checks, solutions = audit_networks(matrices, backend=solver)
    if not bool(checks.success.all()):
        failures = checks.loc[~checks.success, ["sample", "converged", "feasible", "error"]]
        raise RuntimeError(f"Seed graphs failed AC-OPF checks: {failures.to_dict(orient='records')}")
    return [solution.to_hetero(ACOPF(matrix)) for matrix, solution in zip(matrices, solutions)]


def topology_signature(graph):
    """Connection/equipment signature with fixed bus identities; ignore electrical values."""
    p = opf2mat(graph).params
    lines = tuple(sorted(tuple(sorted(pair)) for pair in zip(p.fr[:p.n_line].tolist(), p.to[:p.n_line].tolist())))
    transformers = tuple(sorted(zip(p.fr[p.n_line:].tolist(), p.to[p.n_line:].tolist())))
    return (p.N, lines, transformers, tuple(p.gen_bus.tolist()), tuple(p.load_bus.tolist()), tuple(p.shunt_bus.tolist()))


def diversity(graphs, steps):
    if steps < 1 or not graphs or len(graphs) % steps:
        raise ValueError("Graphs must form K equal-size batches")
    batch_size = len(graphs) // steps
    return [len({topology_signature(graph) for graph in graphs[k * batch_size:(k + 1) * batch_size]})
            for k in range(steps)]


def batch_report(graphs, steps):
    counts = diversity(graphs, steps)
    batch_size = len(graphs) // steps
    rows = []
    for step, count in enumerate(counts):
        sizes = [graph["bus"].num_nodes for graph in graphs[step * batch_size:(step + 1) * batch_size]]
        rows.append(dict(step=step + 1, graphs=batch_size, topologies=count,
                         bus_min=min(sizes), bus_max=max(sizes), bus_counts=sorted(set(sizes))))
    return pd.DataFrame(rows)


def solve_prefixes(matrices, original, *, solver="pypower", audit=None):
    """Undo the last discrete edits until a changed, solver-feasible prefix remains."""
    audit = audit_networks if audit is None else audit
    original_signature = topology_signature(original)
    failures = []
    for length in range(len(matrices), 0, -1):
        matrix = matrices[length - 1]
        if topology_signature(mat2opf(matrix, include_solution=False)) == original_signature:
            failures.append(dict(edits=length, error="Edits returned to the original topology"))
            continue
        checks, solutions = audit([matrix], backend=solver)
        if bool(checks.success.all()):
            return solutions[0].to_hetero(ACOPF(matrix)), length, failures
        failures.append(dict(edits=length, error=checks.iloc[0].error or "No converged, feasible AC-OPF solution",
                             solver_warnings=checks.iloc[0].get("solver_warnings", "")))
    return None, 0, failures


def generate_dataset(candidate, verified_graphs, *, min_buses, max_buses, seed=42,
                     max_edits=2, attempts_per_graph=6, solver="pypower", audit=None):
    """Independently propose and solve edits for EVERY synthetic graph.

    Failed edit sequences backtrack to shorter prefixes before drawing a new
    sequence. If none is feasible, retain the verified seed and record the reason.
    The caller then enforces diversity for each complete inner batch.
    Budgets refer to the original case, not to each graph's current size.
    """
    if candidate < 0 or max_edits < 1 or attempts_per_graph < 1:
        raise ValueError("Edit and attempt budgets must be positive")
    if any(not min_buses <= graph["bus"].num_nodes <= max_buses for graph in verified_graphs):
        raise ValueError("Seed graph outside the original-case bus-count limits")
    rng = np.random.default_rng(seed + 104729 * candidate)
    graphs, changes = [], []
    for index, original in enumerate(verified_graphs):
        chosen = original
        row = dict(sample=index, buses_before=original["bus"].num_nodes, buses=original["bus"].num_nodes,
                   operations=[], changed=False, rejected_attempts=0, last_rejection="",
                   proposed_edits=0, backtracked_edits=0, prefix_rejections=[])
        if candidate:
            for attempt in range(attempts_per_graph):
                operations, matrices = [], []
                graph = original
                planned = int(rng.integers(1, max_edits + 1))
                structural_error = ""
                try:
                    for _ in range(planned):
                        matrix, operation = propose(graph, int(rng.integers(1, 2**30)),
                                                    min_buses=min_buses, max_buses=max_buses,
                                                    seed=seed + index)
                        operations.append(operation)
                        matrices.append(matrix)
                        graph = mat2opf(matrix, include_solution=False)
                except ValueError as exc:
                    structural_error = str(exc)
                solved, accepted_edits, failures = solve_prefixes(matrices, original, solver=solver, audit=audit)
                if structural_error:
                    failures.insert(0, dict(edits=len(matrices) + 1, error=structural_error))
                if failures:
                    row["rejected_attempts"] += 1
                    row["last_rejection"] = failures[-1]["error"]
                    row["prefix_rejections"].extend(dict(attempt=attempt + 1, **failure) for failure in failures)
                if solved is not None:
                    chosen = solved
                    row.update(buses=chosen["bus"].num_nodes, operations=operations[:accepted_edits], changed=True,
                               proposed_edits=planned, backtracked_edits=planned - accepted_edits)
                    break
        graphs.append(chosen)
        changes.append(row)
    return graphs, changes

