"""Independent post-checks shared by every backend; no global optimality claim."""
import math

import numpy as np
import torch


def maximum(value):
    return float(value.max()) if value.numel() else 0.


def feasibility(model, solution, tol=1e-6):
    """Max residuals: balance/thermal/bounds in p.u.; angle/reference in radians."""
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("Tolerance must be positive and finite")
    x = solution.x(model)
    if x.shape != model.lo.shape:
        raise ValueError("Solution size does not match the model")
    keys = ("balance", "thermal", "bounds", "angle", "reference", "cost_relative")
    if not torch.isfinite(x).all() or not math.isfinite(solution.objective):
        return dict.fromkeys(keys, math.inf) | {"feasible": False, "tolerance": tol}
    with torch.no_grad():
        sf, st = model.branch_flows(x)
        flows = torch.cat((sf[model.monitored].abs(), st[model.monitored].abs()))
        rates = model.rate[model.monitored].repeat(2)
        angles = model.angles(x)
        checks = dict(
            balance=maximum(model.balance(x).abs()),
            thermal=maximum((flows - rates).clamp_min(0)),
            bounds=maximum(torch.maximum(model.lo - x, x - model.hi).clamp_min(0)),
            angle=maximum(torch.maximum(model.angle_lo - angles, angles - model.angle_hi).clamp_min(0)),
            reference=abs(float(x[model.ref])),
            cost_relative=abs(float(model.cost(x)) - solution.objective) / max(1., abs(solution.objective)),
        )
    checks["feasible"] = all(math.isfinite(v) and v <= tol for v in checks.values())
    checks["tolerance"] = tol
    return checks


def compare(solution, reference):
    """Absolute relative objective gap and maximum state/dispatch deviations."""
    result = {f"d_{key}": float(np.max(np.abs(getattr(solution, key) - getattr(reference, key))))
              for key in ("theta", "vm", "pg", "qg")}
    result["rel_obj"] = abs(solution.objective - reference.objective) / max(1., abs(reference.objective))
    return result
