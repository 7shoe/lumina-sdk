"""Shared detached results; arrays retain input row order, powers p.u., angles rad."""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Solution:
    theta: np.ndarray
    vm: np.ndarray
    pg: np.ndarray
    qg: np.ndarray
    objective: float
    converged: bool
    source: str
    info: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @property
    def feasible(self):
        return self.checks.get("feasible", False)

    @property
    def success(self):
        return self.converged and self.feasible

    def x(self, model=None):
        device = model.lo.device if model is not None else "cpu"
        return torch.as_tensor(np.r_[self.theta, self.vm, self.pg, self.qg], dtype=torch.float64, device=device)

    @classmethod
    def from_label(cls, data):
        bus = data["bus"].y.detach().cpu().double().numpy()
        gen = data["generator"].y.detach().cpu().double().numpy()
        return cls(bus[:, 0].copy(), bus[:, 1].copy(), gen[:, 0].copy(), gen[:, 1].copy(),
                   float(data.objective), True, "opfdata")

    @classmethod
    def from_x(cls, x, model, converged, source, *, tol=1e-6, **info):
        from .validate import feasibility
        x = torch.as_tensor(x, dtype=model.lo.dtype, device=model.lo.device).detach()
        arrays = [value.cpu().numpy().copy() for value in model.unpack(x)]
        solution = cls(*arrays, float(model.cost(x)), bool(converged), source, info)
        solution.checks = feasibility(model, solution, tol)
        return solution

    def to_hetero(self, model):
        """Attach fresh float64 labels to a copy of the original problem inputs."""
        if not self.success:
            raise ValueError("Only a converged, feasible solution can be exported as labels")
        if model.template is None:
            raise ValueError("HeteroData export needs a model constructed from HeteroData or OPFMatrices")
        from .validate import feasibility
        if not feasibility(model, self, self.checks["tolerance"])["feasible"]:
            raise ValueError("Solution is not feasible for the supplied model")
        data = model.template.clone()
        theta, vm, pg, qg = model.unpack(self.x(model))
        data["bus"].y = torch.stack((theta, vm), dim=-1).detach().cpu()
        data["generator"].y = torch.stack((pg, qg), dim=-1).detach().cpu()
        sf, st = model.branch_flows(self.x(model))
        values = dict(pf=sf.real, qf=sf.imag, pt=st.real, qt=st.imag)
        labels = torch.stack([values[key] for key in model.label_columns], dim=-1).detach().cpu()
        nline = data["bus", "ac_line", "bus"].num_edges
        data["bus", "ac_line", "bus"].edge_label = labels[:nline]
        data["bus", "transformer", "bus"].edge_label = labels[nline:]
        data.objective = torch.tensor(self.objective, dtype=torch.float64)
        return data
