"""Sample-weighted OPF metrics, computed on CPU in float64.

OPFData release-1 processed features encode powers, ratings and shunts in p.u.,
angles in radians, and polynomial costs on p.u. generation. Do not apply baseMVA
a second time. The schema's MVA/cost descriptions disagree with these archived
values; ground-truth and PYPOWER checks in tests establish the convention.

Paper §2.5: mean per-sample family L2 norms, then divide their sum by sqrt(Nbus).
Thermal residuals use relu(|S|^2 - rate_a^2) at both branch ends. Zero ratings
denote unconstrained branches. Box violations are diagnostics, excluded from Viol.
"""
from __future__ import annotations

import torch


def _sum(value, graph, count):
    result = value.new_zeros(count)
    return result.index_add_(0, graph, value)


def physical_residuals(predictions, batch):
    """Return bus complex balance residuals and both-end thermal excesses."""
    bus, gen = predictions["bus"].double(), predictions["generator"].double()
    voltage = torch.polar(bus[:, 1], bus[:, 0])
    residual = torch.zeros_like(voltage)
    edge = batch["generator", "generator_link", "bus"].edge_index
    residual.index_add_(0, edge[1], torch.complex(gen[edge[0], 0], gen[edge[0], 1]))
    for kind in ("load", "shunt"):
        if kind not in batch.node_types or batch[kind].num_nodes == 0:
            continue
        edge = batch[kind, f"{kind}_link", "bus"].edge_index
        features = batch[kind].x.double()[edge[0]]
        demand = (torch.complex(features[:, 0], features[:, 1]) if kind == "load"
                  else torch.complex(features[:, 1], -features[:, 0]) * bus[edge[1], 1].square())
        residual.index_add_(0, edge[1], -demand)
    thermal, thermal_graph = [], []
    for kind in ("ac_line", "transformer"):
        key = ("bus", kind, "bus")
        if key not in batch.edge_types:
            continue
        edge, features = batch[key].edge_index, batch[key].edge_attr.double()
        if features.shape[0] == 0:
            continue
        if kind == "ac_line":
            r, x, bfr, bto, rate = (features[:, i] for i in (4, 5, 2, 3, 6))
            tap = torch.ones_like(r, dtype=torch.complex128)
        else:
            r, x, bfr, bto, rate = (features[:, i] for i in (2, 3, 9, 10, 4))
            ratio = features[:, 7]
            ratio = torch.where(ratio == 0, torch.ones_like(ratio), ratio)
            tap = torch.polar(ratio, features[:, 8])
        impedance = torch.complex(r, x)
        if torch.any(impedance.abs() == 0):
            raise ValueError("Zero-impedance branch cannot be evaluated")
        admittance = impedance.reciprocal()
        yff = (admittance + 1j * bfr) / tap.abs().square()
        ytt = admittance + 1j * bto
        yft, ytf = -admittance / tap.conj(), -admittance / tap
        vf, vt = voltage[edge[0]], voltage[edge[1]]
        sf = vf * (yff * vf + yft * vt).conj()
        st = vt * (ytf * vf + ytt * vt).conj()
        residual.index_add_(0, edge[0], -sf)
        residual.index_add_(0, edge[1], -st)
        for flow in (sf, st):
            excess = torch.relu(flow.abs().square() - rate.square())
            thermal.append(torch.where(rate > 0, excess, torch.zeros_like(excess)))
            thermal_graph.append(batch["bus"].batch[edge[0]])
    return residual, (torch.cat(thermal) if thermal else bus.new_empty(0)), (
        torch.cat(thermal_graph) if thermal_graph else torch.empty(0, dtype=torch.long))


def sample_metrics(predictions, batch):
    """Return a vector of length num_graphs for each metric; never average batches."""
    batch = batch.cpu()
    predictions = {k: v.detach().cpu().double() for k, v in predictions.items()}
    count = batch.num_graphs
    bgraph, ggraph = batch["bus"].batch, batch["generator"].batch
    nb = torch.bincount(bgraph, minlength=count).double()
    ng = torch.bincount(ggraph, minlength=count).double()
    by, gy = batch["bus"].y.double(), batch["generator"].y.double()
    bp, gp = predictions["bus"], predictions["generator"]
    for value in (by, gy, bp, gp):
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite label/prediction; refusing to omit samples")
    be, ge = (bp - by).square(), (gp - gy).square()
    bus_mse = _sum(be.sum(1), bgraph, count) / (2 * nb)
    gen_mse = _sum(ge.sum(1), ggraph, count) / (2 * ng)
    balance, thermal, thermal_graph = physical_residuals(predictions, batch)
    balance_l2 = _sum(balance.abs().square(), bgraph, count).sqrt()
    thermal_l2 = _sum(thermal.square(), thermal_graph, count).sqrt()
    gx, bx = batch["generator"].x.double(), batch["bus"].x.double()
    gen_bounds = (torch.relu(gx[:, [2, 5]] - gp) + torch.relu(gp - gx[:, [3, 6]]))
    bus_bounds = torch.relu(bx[:, 1] - bp[:, 1]) + torch.relu(bp[:, 1] - bx[:, 2])
    bounds_l2 = (_sum(gen_bounds.square().sum(1), ggraph, count)
                 + _sum(bus_bounds.square(), bgraph, count)).sqrt()
    def cost(pg):
        return _sum(gx[:, 8] * pg.square() + gx[:, 9] * pg + gx[:, 10], ggraph, count)
    pred_cost, target_cost = cost(gp[:, 0]), cost(gy[:, 0])
    result = {
        "mse": (2 * nb * bus_mse + 2 * ng * gen_mse) / (2 * (nb + ng)),
        "sdk_mse": bus_mse + gen_mse,
        "va_mse": _sum(be[:, 0], bgraph, count) / nb,
        "vm_mse": _sum(be[:, 1], bgraph, count) / nb,
        "pg_mse": _sum(ge[:, 0], ggraph, count) / ng,
        "qg_mse": _sum(ge[:, 1], ggraph, count) / ng,
        "balance_l2": balance_l2, "thermal_l2": thermal_l2,
        "violation": (balance_l2 + thermal_l2) / nb.sqrt(),
        "bounds_l2": bounds_l2,
        "cost_difference": pred_cost - target_cost,
        "reference_cost": target_cost,
    }
    if torch.all(target_cost.abs() > 1e-12):
        result["cost_difference_pct"] = 100 * (pred_cost - target_cost) / target_cost
    objective = getattr(batch, "objective", None)
    if objective is not None:
        result["label_cost_objective_abs_error"] = (target_cost - objective.reshape(-1).double()).abs()
    for name, value in result.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite physical metric: {name}")
    return result


class MetricAccumulator:
    def __init__(self):
        self.sums, self.counts = {}, {}
        self.n_samples = 0

    def update(self, values):
        self.n_samples += len(values["mse"])
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + value.sum().item()
            self.counts[key] = self.counts.get(key, 0) + value.numel()

    def result(self, selection_mse="sdk_mse"):
        if not self.n_samples:
            raise ValueError("Cannot evaluate an empty split")
        result = {key: value / self.counts[key] for key, value in self.sums.items()}
        result["n_samples"] = self.n_samples
        result["score"] = result[selection_mse] + result["violation"]
        return result
