"""CPU AC-OPF solvers, adapted from acopf/core (commit 71fd17e63b30).

    result = solve(data)                 # native OPFData HeteroData
    result = solve(edited_matrices)      # OPFMatrices from conversion
    if result.success:
        solved_data = result.to_hetero(ACOPF(data))

PYPOWER is the default. SciPy and experimental torch routines live in backends/;
new routines can implement solve(model, x0=None) -> Solution. Model functions and
NewtonPF support autograd; solve() returns detached results. No global dtype changes.
Inputs, solutions and generator rows retain their original order. Check converged,
feasible and checks separately; success requires both convergence and feasibility.
"""
from .problem import ACOPF
from .solution import Solution
from .backends.reference import PyPowerSolver, ScipySolver
from .backends.torch_solvers import NewtonPF, ReducedGradient
from . import validate

__all__ = ["ACOPF", "Solution", "PyPowerSolver", "ScipySolver", "NewtonPF",
           "ReducedGradient", "solve", "validate"]


def solve(network, *, method="pypower", x0=None, **options):
    """Solve one HeteroData, OPFMatrices, PYPOWER case, or existing ACOPF model."""
    if method not in ("pypower", "scipy", "reduced_gradient"):
        raise ValueError(f"Unknown solver: {method}")
    model = network if isinstance(network, ACOPF) else ACOPF(network)
    if method == "reduced_gradient":
        return ReducedGradient(model, **options).solve(x0=x0)
    backend = PyPowerSolver if method == "pypower" else ScipySolver
    return backend(**options).solve(model, x0=x0)
