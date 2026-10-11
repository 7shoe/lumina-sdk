"""Shared bounded physical parameters and detached AC-OPF feasibility checks."""
from dataclasses import replace
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import torch

from distillation.conversion.matrix_form import mat2opf, validate_network
from distillation.solver import ACOPF, PyPowerSolver


class Perturbation(torch.nn.Module):
    """Bounded, differentiable relative edits of physically representable parameters.

    A positive per-branch admittance scale induces Y_tilde = Y_new - Y. Unlike an
    arbitrary dense additive matrix, this retains taps, sparsity and an exact
    mapping back to branch parameters. Fixed intervals and zero ratings stay fixed.
    """
    def __init__(self, matrices, *, bounds=True, admittance=True, loads=True, seed=42):
        super().__init__()
        p = matrices.params
        shapes = {}
        if bounds:
            shapes.update(voltage=p.N, active=p.G, reactive=p.G, thermal=p.E)
        if admittance:
            shapes["admittance"] = p.E
        if loads:
            shapes["load"] = p.L
        rng = torch.Generator(device="cpu").manual_seed(seed)
        self.delta = torch.nn.ParameterDict({
            key: torch.nn.Parameter(torch.randn(size, generator=rng, dtype=p.vmin.dtype))
            for key, size in shapes.items()
        })

    def forward(self, matrices, strength=0.02):
        if not np.isfinite(strength) or not 0 <= strength < 1:
            raise ValueError("Use 0 <= STRENGTH < 1; tanh limits each relative change.")
        p = matrices.params
        scale = {key: 1 + strength * value.tanh() for key, value in self.delta.items()}
        if "admittance" in scale:
            # Scaling admittance by s divides R and X by s. This equivalent
            # form also makes the zero edit exact, avoiding reciprocal roundoff
            # that can change convergence of a nearly degenerate OPF solve.
            p = replace(p, br_r=p.br_r / scale["admittance"], br_x=p.br_x / scale["admittance"])
        values = {}
        for key, lo, hi in (("voltage", "vmin", "vmax"), ("active", "pmin", "pmax"),
                            ("reactive", "qmin", "qmax")):
            if key in scale:
                a, b = getattr(p, lo), getattr(p, hi)
                expansion = (b - a) / 2 * (scale[key] - 1)
                values[lo], values[hi] = a - expansion, b + expansion
        if "thermal" in scale:
            values["rate_a"] = p.rate_a * scale["thermal"]
        if "load" in scale:
            values.update(pd=p.pd * scale["load"], qd=p.qd * scale["load"])
        return replace(matrices, params=replace(p, **values), sol=None)


@torch.no_grad()
def audit_networks(matrices, *, backend="pypower", sparse_seconds=10):
    """Detached feasibility gate on every candidate, with fresh solutions kept separate."""
    if backend == "pypower":
        solver = PyPowerSolver()
    elif backend == "sparse":
        from distillation.solver.sparse_solver import SparseSolver
        solver = SparseSolver(time_limit=sparse_seconds)
    else:
        raise ValueError("SOLVER must be 'pypower' or 'sparse'")
    rows, solutions = [], []
    for i, matrix in enumerate(matrices):
        started = perf_counter()
        row = dict(sample=i, success=False, converged=False, feasible=False,
                   balance=float("nan"), thermal=float("nan"), bounds=float("nan"),
                   angle=float("nan"), reference=float("nan"), cost_relative=float("nan"), error="")
        solution = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                validate_network(matrix.params)
                graph = mat2opf(matrix, include_solution=False).detach()
                problem = ACOPF(graph)
                solution = solver.solve(problem if backend == "pypower" else graph)
                row.update(solution.checks)
                row.update(success=bool(solution.success), converged=bool(solution.converged))
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
        row["solver_warnings"] = "; ".join(f"{item.category.__name__}: {item.message}" for item in caught)
        rows.append(row | {"seconds": perf_counter() - started})
        solutions.append(solution)
    return pd.DataFrame(rows), solutions

