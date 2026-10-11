"""CPU reference optimizers adapted from acopf/core/solvers/reference.py."""
from copy import deepcopy

import numpy as np
import torch
from torch.func import grad, hessian, jacrev
from scipy.optimize import Bounds, NonlinearConstraint, minimize
from pypower.api import ppoption, runopf

from ..solution import Solution
from ..validate import feasibility


class PyPowerSolver:
    """CPU AC-OPF reference solver using PYPOWER's default PIPS backend.

    PIPS (Python Interior Point Solver) uses a primal-dual interior-point
    method to optimize generator dispatch and bus voltages subject to
    nonlinear power balance, equipment bounds, and branch limits.
    AC-OPF is nonconvex; convergence does not certify a global optimum.

    PYPOWER's PIPS adapter constructs its own initial point, so this wrapper
    rejects x0. Returned solutions include independent feasibility checks;
    tol controls those checks, while options control solver tolerances.
    
    FYI: Scipy also offers a variant with 
        ```
        scipy.optimize.linprog(method='interior-point')
        ```
    """
    name = "pypower"

    def __init__(self, *, tol=1e-6, **options):
        self.tol = tol
        
        # keep PYPOWER's constraint conventions consistent with the model
        if options.get("OPF_FLOW_LIM", 0) != 0 or options.get("OPF_IGNORE_ANG_LIM", False):
            raise ValueError("Use apparent-power limits and enforce angle bounds")
        self.options = dict(VERBOSE=0, OUT_ALL=0, PDIPM_MAX_IT=150, OPF_VIOLATION=1e-8,
                            PDIPM_FEASTOL=1e-8, PDIPM_GRADTOL=1e-8,
                            PDIPM_COMPTOL=1e-8, PDIPM_COSTTOL=1e-9) | options

    def solve(self, model, x0=None):
        if x0 is not None:
            raise ValueError("PYPOWER/PIPS does not honor a supplied x0; use ScipySolver for warm starts")
        if model.lo.device.type != "cpu":
            raise ValueError("PYPOWER requires a CPU model")
        
        # isolate solver-side modifications from the original case
        ppc = deepcopy(model.ppc)
        redundant_limit = None
        if not model.monitored.any():
            if not model.nl:
                raise ValueError("PYPOWER needs a branch; use ScipySolver for a single-bus problem")
            
            # avoid empty thermal constraints by adding a nonbinding limit
            # PIPS mishandles empty thermal constraints. This Vmax bound is redundant.
            vmax = model.hi[model.nb:2*model.nb]
            from_bound = vmax[model.f] * (model.Yf.abs() @ vmax)
            to_bound = vmax[model.t] * (model.Yt.abs() @ vmax)
            ceilings = torch.maximum(from_bound, to_bound)
            index = int(ceilings.argmin())
            
            # add a margin and convert the selected bound from p.u. to MVA
            limit = 1.01 * max(1., float(ceilings[index])) * model.base
            ppc["branch"][index, 5] = limit
            redundant_limit = dict(branch=index, mva=limit)
        
        result = runopf(ppc, ppoption(**self.options))
        
        # check bus IDs and generator bus assignments
        if not np.array_equal(result["bus"][:, 0], model.ppc["bus"][:, 0]):
            raise RuntimeError("PYPOWER changed the external bus order")
        if not np.array_equal(result["gen"][:, 0], model.ppc["gen"][:, 0]):
            raise RuntimeError("PYPOWER changed the external generator order")
        
        # convert PYPOWER's degrees and MW/MVAr back 
        solution = Solution(
            np.radians(result["bus"][:, 8]), result["bus"][:, 7].copy(),
            result["gen"][:, 1].copy() / model.base, result["gen"][:, 2].copy() / model.base,
            float(result["f"]), bool(result["success"]), self.name,
            dict(elapsed=result.get("et"),
                 output=result.get("raw", {}).get("output", {}),
                 redundant_thermal_limit=redundant_limit),
        )
        solution.checks = feasibility(model, solution, self.tol)
        return solution


class ScipySolver:
    """Dense trust-constr with torch derivatives; intended for small networks."""
    name = "scipy"

    def __init__(self, *, tol=1e-6, **options):
        self.tol = tol
        self.options = dict(xtol=1e-10, gtol=1e-9, barrier_tol=1e-10, maxiter=500, verbose=0) | options

    def solve(self, model, x0=None):
        m = model
        if m.lo.device.type != "cpu":
            raise ValueError("SciPy requires a CPU model")
        
        # use a flat start unless a caller-supplied solution provides one
        start = m.flat_start() if x0 is None else x0.x(m)
        if start.shape != m.lo.shape or not torch.isfinite(start).all():
            raise ValueError("Invalid initial solution")
        
        # optimize only free variables
        free = m.lo < m.hi
        fixed = torch.where(free, torch.zeros_like(m.lo), m.lo)
        indices = free.nonzero().flatten()

        def expand(z):
            """reconstruct the full state for objective and constraint evaluation"""
            return fixed.index_copy(0, indices, z)

        def numpy_fn(fn):
            """adapt torch calculations to SciPy's NumPy callback interface"""
            return lambda z: fn(torch.as_tensor(z, dtype=torch.float64)).detach().numpy()

        def constraint(fn, low, high):
            fun = lambda z: fn(expand(z))
            curvature = lambda z, v: hessian(lambda t: torch.as_tensor(v) @ fun(t))(torch.as_tensor(z)).numpy()
            return NonlinearConstraint(numpy_fn(fun), low, high, jac=numpy_fn(jacrev(fun)), hess=curvature)

        # scale cost for numerical conditioning
        objective = lambda z: m.cost(expand(z)) / m.cost_scale
        constraints = [constraint(m.balance, 0, 0)]
        
        # follow inequal. convention of g(x) <= 0
        if m.inequalities(start).numel():
            constraints.append(constraint(m.inequalities, -np.inf, 0))
        
        # supply torch-derived gradients, Jacobians, and dense Hessians
        result = minimize(numpy_fn(objective), start[free].numpy(), jac=numpy_fn(grad(objective)),
                          hess=numpy_fn(hessian(objective)), method="trust-constr",
                          bounds=Bounds(m.lo[free].numpy(), m.hi[free].numpy()),
                          constraints=constraints, options=self.options)
        
        # restore fixed variables and retain solver diagnostics
        return Solution.from_x(expand(torch.as_tensor(result.x)), m, result.success, self.name,
                               tol=self.tol, iterations=result.nit, status=result.status,
                               message=str(result.message), optimality=float(result.optimality))
