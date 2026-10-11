"""Differentiable AC-OPF in original row order; x=[theta, Vm, Pg, Qg], rad/p.u.

Adapted from acopf/core/problem.py. One connected, active network, one reference
bus, apparent-power limits and polynomial active-generation costs of degree <=2.
Model buffers use float64/complex128 without changing torch's default dtype.
"""
from copy import deepcopy

import numpy as np
import torch
from pypower.makeYbus import makeYbus
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from distillation.conversion import hetero_to_ppc
from distillation.conversion.matrix_form import (
    OPFMatrices, LINK_TYPES, mat2opf, opf2mat, validate_network,
)


def as_ppc(network):
    """Reuse conversion, ignoring labels and refusing unsupported lossy encodings."""
    if isinstance(network, dict):
        return deepcopy(network), None, ("pt", "qt", "pf", "qf")
    columns = (network.meta["schema"]["branch_y"] if isinstance(network, OPFMatrices)
               else ("pt", "qt", "pf", "qf"))
    data = mat2opf(network, include_solution=False) if isinstance(network, OPFMatrices) else network.clone()
    data = data.detach().cpu()
    for store in data.stores:
        for key in ("y", "edge_label", "objective"):
            if key in store:
                del store[key]
    mats = opf2mat(data)
    p = mats.params
    validate_network(p)
    if not torch.equal(p.b_fr, p.b_to):
        raise ValueError("PYPOWER's branch format requires equal charging at both ends")
    if ((p.angmin == 0) | (p.angmax == 0)).any():
        raise ValueError("A strict zero angle bound cannot be represented by PYPOWER's zero sentinel")
    for edge, name in LINK_TYPES.items():
        buses = getattr(p, name)
        data[edge].edge_index = torch.stack((torch.arange(len(buses)), buses))
    return hetero_to_ppc(data), data, columns


class ACOPF(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.ppc, self.template, self.label_columns = as_ppc(network)
        self.base = float(self.ppc["baseMVA"])
        bus, gen, br, gc = (np.array(self.ppc[k], dtype=float, copy=True)
                            for k in ("bus", "gen", "branch", "gencost"))
        self.nb, self.ng, self.nl = len(bus), len(gen), len(br)
        if not np.isfinite(self.base) or self.base <= 0 or not self.nb or not self.ng:
            raise ValueError("Need positive baseMVA, buses and generators")
        if any(not np.isfinite(a).all() for a in (bus, gen, br, gc)):
            raise ValueError("PYPOWER case arrays must be finite")
        if set(self.ppc) & {"A", "N", "H", "Cw", "userfcn"}:
            raise ValueError("Extended PYPOWER objectives/constraints are not supported")
        if np.any(~np.isin(bus[:, 1], [1, 2, 3])) or np.any(gen[:, 7] <= 0) or np.any(br[:, 10] <= 0):
            raise ValueError("Remove offline equipment/buses before constructing ACOPF")
        refs = np.flatnonzero(bus[:, 1] == 3)
        if len(refs) != 1:
            raise ValueError("Expected exactly one reference bus")
        self.ref = int(refs[0])
        ids = bus[:, 0].copy()
        if len(set(ids)) != self.nb or np.any(ids != ids.astype(int)) or np.any(ids < 0):
            raise ValueError("Bus IDs must be unique nonnegative integers")
        lookup = {value: index for index, value in enumerate(ids)}
        try:
            gen[:, 0] = [lookup[value] for value in gen[:, 0]]
            br[:, 0] = [lookup[value] for value in br[:, 0]]
            br[:, 1] = [lookup[value] for value in br[:, 1]]
        except KeyError as exc:
            raise ValueError(f"Equipment refers to unknown bus {exc.args[0]}") from exc
        bus[:, 0] = np.arange(self.nb)
        if np.any((br[:, 2] == 0) & (br[:, 3] == 0)) or np.any(br[:, 8] < 0) or np.any(br[:, 5] < 0):
            raise ValueError("Invalid impedance, tap or thermal rating")
        graph = coo_matrix((np.ones(self.nl), (br[:, 0].astype(int), br[:, 1].astype(int))),
                           shape=(self.nb, self.nb))
        if connected_components(graph, directed=False, return_labels=False) != 1:
            raise ValueError("Solve connected networks separately")
        if len(gc) != self.ng or np.any(gc[:, 0] != 2) or np.any(~np.isin(gc[:, 3], [1, 2, 3])):
            raise ValueError("Only polynomial active-generation costs of degree <=2 are supported")
        if np.any((gen[:, 8] == 0) & (gen[:, 9] < 0)):
            raise ValueError("PYPOWER dispatchable loads are not supported")
        coefficients = np.zeros((self.ng, 3))
        for index, row in enumerate(gc):
            count = int(row[3])
            coefficients[index, -count:] = row[4:4 + count]
        self.ppc["bus"] = np.array(self.ppc["bus"], dtype=float, copy=True)
        self.ppc["bus"][self.ref, 8] = 0

        def buffer(name, array, dtype=torch.float64):
            self.register_buffer(name, torch.as_tensor(np.array(array, copy=True), dtype=dtype))

        ybus, yf, yt = makeYbus(self.base, bus, br)
        for name, matrix in (("Ybus", ybus), ("Yf", yf), ("Yt", yt)):
            buffer(name, matrix.toarray(), torch.complex128)
        buffer("gen_bus", gen[:, 0], torch.long)
        cg = np.zeros((self.nb, self.ng), dtype=complex)
        cg[gen[:, 0].astype(int), np.arange(self.ng)] = 1
        buffer("Cg", cg, torch.complex128)
        buffer("Sd", (bus[:, 2] + 1j * bus[:, 3]) / self.base, torch.complex128)
        buffer("f", br[:, 0], torch.long)
        buffer("t", br[:, 1], torch.long)
        buffer("rate", br[:, 5] / self.base)
        buffer("monitored", br[:, 5] > 0, torch.bool)
        buffer("c", coefficients)
        amin = np.where((br[:, 11] != 0) & (br[:, 11] > -360), np.radians(br[:, 11]), -np.inf)
        amax = np.where((br[:, 12] != 0) & (br[:, 12] < 360), np.radians(br[:, 12]), np.inf)
        buffer("angle_lo", amin)
        buffer("angle_hi", amax)
        lo = np.r_[np.full(self.nb, -np.inf), bus[:, 12], gen[:, 9] / self.base, gen[:, 4] / self.base]
        hi = np.r_[np.full(self.nb, np.inf), bus[:, 11], gen[:, 8] / self.base, gen[:, 3] / self.base]
        lo[self.ref] = hi[self.ref] = 0
        if np.any(lo > hi) or np.any(amin > amax) or np.any(bus[:, 12] <= 0):
            raise ValueError("Invalid voltage, dispatch or angle bounds")
        buffer("lo", lo)
        buffer("hi", hi)
        initial = np.r_[np.zeros(self.nb), np.ones(self.nb), gen[:, 1] / self.base, gen[:, 2] / self.base]
        buffer("x_setpoint", np.clip(initial, lo, hi))
        self.cost_scale = max(1., abs(float(self.cost(self.x_setpoint))))

    def unpack(self, x):
        n, g = self.nb, self.ng
        return x[..., :n], x[..., n:2*n], x[..., 2*n:2*n+g], x[..., 2*n+g:]

    def pack(self, theta, vm, pg, qg):
        return torch.cat((theta, vm, pg, qg), dim=-1)

    def flat_start(self):
        return self.x_setpoint.clone()

    def cost(self, x):
        power = self.unpack(x)[2] * self.base
        return (self.c[:, 0] * power.square() + self.c[:, 1] * power + self.c[:, 2]).sum(-1)

    def balance(self, x):
        theta, vm, pg, qg = self.unpack(x)
        voltage = torch.polar(vm, theta)
        residual = torch.complex(pg, qg) @ self.Cg.T - self.Sd - voltage * (voltage @ self.Ybus.T).conj()
        return torch.cat((residual.real, residual.imag), dim=-1)

    def branch_flows(self, x):
        theta, vm, _, _ = self.unpack(x)
        voltage = torch.polar(vm, theta)
        return (voltage[..., self.f] * (voltage @ self.Yf.T).conj(),
                voltage[..., self.t] * (voltage @ self.Yt.T).conj())

    def flows(self, x):
        """Squared apparent-power residuals at monitored ends, p.u.²; <=0 feasible."""
        sf, st = self.branch_flows(x)
        rate = self.rate[self.monitored].square()
        return torch.cat((sf[..., self.monitored].abs().square() - rate,
                          st[..., self.monitored].abs().square() - rate), dim=-1)

    def angles(self, x):
        theta = self.unpack(x)[0]
        return theta[..., self.f] - theta[..., self.t]

    def inequalities(self, x):
        angles = self.angles(x)
        low, high = self.angle_lo.isfinite(), self.angle_hi.isfinite()
        return torch.cat((self.flows(x), self.angle_lo[low] - angles[..., low],
                          angles[..., high] - self.angle_hi[high]), dim=-1)
