"""Method-independent synthetic grids: physical inputs, certificates and labels."""
from dataclasses import dataclass, replace
import hashlib

import numpy as np
import pandas as pd
import torch

from distillation.conversion.matrix_form import (mat2opf,
                                                 opf2mat,
                                                 validate_network)
from distillation.solver import ACOPF
from .physical import Perturbation, audit_networks
from .random import random_network, sample_sizes
from .topology import (bus_limits,
                       diversity,
                       generate_dataset,
                       refresh_seeds)


@dataclass
class SyntheticDataset:
    inputs: list
    physical_graphs: list
    teaching_targets: list
    report: pd.DataFrame
    metadata: dict


class GenerationError(RuntimeError):
    """Finite proposal budget exhausted; partial contains only accepted grids."""
    def __init__(self, message, partial, failures):
        super().__init__(message)
        self.partial, self.failures = partial, failures


def teaching_targets(graphs, *, source="solver"):
    """Copy detached supervision. Alternate labels are deliberately unimplemented."""
    if source in ("teacher", "learned"):
        raise NotImplementedError("Independent/teacher labels are a placeholder; use source='solver'")
    if source != "solver":
        raise ValueError(f"Unknown label source: {source}")
    return [{kind: graph[kind].y.detach().clone() for kind in ("bus", "generator")} for graph in graphs]


def describe(graph):
    """Structural and operational statistics, supplementing fixed-ID signatures."""
    p = opf2mat(graph).params
    degrees = torch.bincount(torch.cat((p.fr, p.to)), minlength=p.N)
    return dict(buses=p.N, branches=p.E, generators=p.G, loads=p.L, shunts=p.S,
                transformers=p.E - p.n_line, cycle_rank=p.E - p.N + 1,
                mean_degree=float(degrees.double().mean()), max_degree=int(degrees.max()),
                degree_histogram=torch.bincount(degrees).tolist(),
                demand_pu=float(p.pd.sum()), capacity_pu=float(p.pmax.sum()))


def _fingerprint(graph):
    digest = hashlib.sha256()
    for name, value in vars(opf2mat(graph).params).items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes()
                      if isinstance(value, torch.Tensor) else str(value).encode())
    return digest.hexdigest()


def _dataset(graphs, rows, metadata):
    return SyntheticDataset(
        [mat2opf(opf2mat(g), include_solution=False).detach() for g in graphs],
        [g.clone().detach() for g in graphs], teaching_targets(graphs),
        pd.DataFrame(rows), metadata.copy(),
    )


def perturb_operating(graph, *, strength=0.1, seed=42, solver="pypower",
                      fractions=(1., 0.5, 0.25, 0.)):
    """Backtrack a bounded operational edit; even the unchanged endpoint is solved."""
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("Use 0 <= strength < 1")
    if not fractions or any(not 0 <= x <= 1 for x in fractions):
        raise ValueError("Operational backtracking fractions must be in [0, 1]")
    matrix = opf2mat(graph.clone().detach().cpu())
    edit = Perturbation(matrix, seed=seed)
    failures = []
    for fraction in fractions:
        changed = edit(matrix, strength * fraction)
        checks, solutions = audit_networks([changed], backend=solver)
        failures.append(dict(fraction=fraction, **checks.iloc[0].to_dict()))
        if bool(checks.success.all()):
            return solutions[0].to_hetero(ACOPF(changed)), dict(
                operational_fraction=fraction, operating_attempts=failures)
    raise RuntimeError(f"No feasible operational step, including requested fallback: {failures}")


def generate(references, *, mode="conservative", count=None, sizes=None, original_buses=None,
             seed=42, steps=1, min_topologies=2, attempts=12, max_edits=2,
             strength=None, density=(1.0, 1.6), demand_fractions=(1., 0.8, 0.6),
             placement="electrical", solver="pypower", label_source="solver", reference_ids=None,
             bus_range=None):
    """Generate fully solver-verified native graphs for DD, DC or HMDC.

    Supply training-only native OPFData references, ideally from the same case.
    Random mode honors EVERY requested size or raises with feasible partial work;
    absent sizes, each batch spans the original hard limits. Conservative mode
    preserves the existing prefix-backtracking algorithm and verified fallback.
    It also supports bounded operational changes (strength=0 disables those).
    Neither mode evaluates a learning objective or updates a student model.
    """
    teaching_targets([], source=label_source)  # Reject placeholder modes before solving.
    if mode not in ("conservative", "random") or not references:
        raise ValueError("Choose conservative/random and provide training references")
    if solver not in ("pypower", "sparse"):
        raise ValueError("solver must be pypower or sparse")
    if attempts < 1 or int(attempts) != attempts or max_edits < 1 or int(max_edits) != max_edits:
        raise ValueError("Positive integer attempt/edit budgets required")
    attempts, max_edits = int(attempts), int(max_edits)
    references = [g.clone().detach().cpu() for g in references]
    if original_buses is None and len({g["bus"].num_nodes for g in references}) != 1:
        raise ValueError("Mixed-size references require the original_buses anchor explicitly")
    original = references[0]["bus"].num_nodes if original_buses is None else original_buses
    low, high = bus_limits(original)
    if bus_range is not None:
        if (len(bus_range) != 2 or any(int(n) != n for n in bus_range)
                or not low <= bus_range[0] <= original <= bus_range[1] <= high):
            raise ValueError("bus_range must contain the original case within the legacy hard limits")
        low, high = map(int, bus_range)
    base_mva = opf2mat(references[0]).params.base_mva
    for graph in references:
        p = opf2mat(graph).params
        validate_network(p)
        if not low <= p.N <= high:
            raise ValueError("Reference outside original-case bus limits")
        if not torch.equal(p.base_mva, base_mva):
            raise ValueError("References must share a per-unit base")
    if sizes is not None:
        if mode != "random":
            raise ValueError("Exact requested sizes apply to random initialization only")
        sizes = list(sizes)
        if count is not None and count != len(sizes):
            raise ValueError("count and sizes disagree")
        count = len(sizes)
        if any(int(n) != n or not max(2, low) <= n <= high for n in sizes):
            raise ValueError("Requested bus counts outside original-case hard limits")
    count = len(references) if count is None else count
    if count < 1 or int(count) != count or steps < 1 or int(steps) != steps or count % steps:
        raise ValueError("Need a positive integer count forming equal-size batches")
    count, steps = int(count), int(steps)
    if min_topologies < 1 or int(min_topologies) != min_topologies:
        raise ValueError("min_topologies must be a positive integer")
    strength = (0.5 if mode == "random" else 0.1) if strength is None else strength
    if not np.isfinite(strength) or not 0 <= strength < 1:
        raise ValueError("Use 0 <= strength < 1")
    if placement not in ("electrical", "uniform"):
        raise ValueError("placement must be electrical or uniform")
    if len(density) != 2 or not np.isfinite(density).all() or not 0 < density[0] <= density[1]:
        raise ValueError("density must be a positive ordered pair")
    if (not demand_fractions or demand_fractions[0] != 1 or
            any(not 0 < f <= 1 for f in demand_fractions) or
            any(a <= b for a, b in zip(demand_fractions, demand_fractions[1:]))):
        raise ValueError("Demand fractions must start at 1 and strictly decrease within (0,1]")
    if reference_ids is not None and len(reference_ids) != len(references):
        raise ValueError("One reference ID is required per graph")
    rng = np.random.default_rng(seed)
    settings = dict(mode=mode, original_buses=int(original), bus_limits=[low, high], seed=seed,
                    count=count, steps=steps, min_topologies=min_topologies, attempts=attempts,
                    max_edits=max_edits, strength=strength, density=list(density),
                    demand_fractions=list(demand_fractions), placement=placement,
                    solver=solver, label_source=label_source,
                    format="synthetic-opf-v1", reference_ids=reference_ids,
                    reference_sha256=[_fingerprint(g) for g in references])
    graphs, rows = [], []
    if mode == "conservative":
        indices = rng.integers(len(references), size=count)
        seeds = refresh_seeds([references[i] for i in indices], solver=solver)
        graphs, rows = generate_dataset(1, seeds, min_buses=low, max_buses=high, seed=seed,
                                       max_edits=max_edits, attempts_per_graph=attempts, solver=solver)
        for i, graph in enumerate(graphs):
            if strength:
                graphs[i], operating = perturb_operating(graph, strength=strength, seed=seed + i, solver=solver)
                rows[i].update(operating)
            rows[i].update(reference_index=int(indices[i]), success=True, **describe(graphs[i]))
    else:
        sizes = sample_sizes(count, original, steps=steps, seed=seed, bus_range=(low, high)) if sizes is None else sizes
        settings["sizes"] = list(sizes)
        for sample, n in enumerate(sizes):
            rng = np.random.default_rng(np.random.SeedSequence([seed, sample]))
            failures, accepted = [], False
            for attempt in range(attempts):
                index, draw_seed = int(rng.integers(len(references))), int(rng.integers(2**31))
                try:
                    matrix, proposal = random_network(references[index], n, original_buses=original,
                                                      seed=draw_seed, spread=strength, density=density,
                                                      placement=placement)
                except ValueError as exc:
                    failures.append(dict(attempt=attempt + 1, seed=draw_seed,
                                         reference_index=index, error=str(exc)))
                    continue
                for fraction in demand_fractions:
                    # Keep sampled topology, bounds and ratings; reduce P and Q together.
                    p = matrix.params
                    trial = replace(matrix, params=replace(p, pd=p.pd * fraction, qd=p.qd * fraction))
                    checks, solutions = audit_networks([trial], backend=solver)
                    entry = checks.iloc[0].to_dict() | dict(sample=sample, attempt=attempt + 1,
                                 seed=draw_seed, reference_index=index, demand_fraction=fraction)
                    failures.append(entry)
                    if bool(checks.success.all()):
                        graph = solutions[0].to_hetero(ACOPF(trial))
                        rows.append(proposal | describe(graph) | dict(sample=sample, reference_index=index,
                                    attempts=attempt + 1, demand_fraction=fraction,
                                    solve_attempts=failures, success=True))
                        graphs.append(graph)
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                raise GenerationError(f"No feasible random grid for sample {sample}, requested buses={n}; "
                                      f"exhausted {attempts} attempts without changing requested size",
                                      _dataset(graphs, rows, settings), failures)
    counts = diversity(graphs, steps)
    settings["topologies_per_batch"] = counts
    result = _dataset(graphs, rows, settings)
    if min(counts) < min(min_topologies, count // steps):
        raise GenerationError(f"Insufficient per-batch topology diversity: {counts}", result, [])
    return result
