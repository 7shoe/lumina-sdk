"""Upstream Frontier AL objective/policy with checked fixed-topology physics.

Source: https://github.com/argonne-gridfm/lumina-sdk/tree/ed/frontier
lumina/model/opf/augmented_lagrangian.py. Residual transformations, normalization,
EMA signal, signs, improvement gate, clipping and epoch penalty policy follow
that module. Integration differences: explicit [B,N] graph layout, persistent
physical-constraint duals, read-only forward, successful-step state commits.

SOURCE DETAILS:
- REPO: https://github.com/argonne-gridfm/lumina-sdk
- BRANCH: ed/frontier
- FILE: ~/lumina/model/opf/augmented_lagrangian.py
- Modified by Carlo Siebenschuh

IMPL DETAILS: 
- earlier variants deviated more severaly (no EWM, uncombined residuals, etc.)
- reverted to be closer to paper: https://arxiv.org/pdf/2605.02133
- Table 8: "... hree different loss functions"

Copyright (c) 2025, Argonne National Laboratory. All rights reserved.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn

# upstream frontier this implementation is based
SOURCE_COMMIT = "f53a7a35ec362121ae39b1c9781bd66b5b9e1b5d"
# AL formulation identifier
FORMULA = "upstream_frontier_al_physical_constraints_v1"

@dataclass(frozen=True)
class ALConfig:
    formula: str = FORMULA
    mu_0: float = 1.0
    mu_increase_factor: float = 1.5
    max_mu: float = 1000.0
    normalize_by_rms: bool = False
    normalize_by_size: bool = True
    warmup_epochs: int = 0
    ema_beta: float = 0.9
    penalty_check_interval: int = 5
    penalty_drop_ratio: float = 0.5
    penalty_increase_factor: float | None = None
    min_penalty: float | None = None
    max_penalty: float | None = None
    multiplier_clip: float | None = 1000.0
    multiplier_improve_ratio: float = 0.0
    multiplier_check_interval: int = 1
    multiplier_min_improve: float = 0.0
    verbose: bool = False
    units: str = "opfdata_release1_pu_radians"
    zero_rating: str = "unmonitored"

    def __post_init__(self):
        if self.formula != FORMULA:
            raise ValueError("AL requires the upstream Frontier formula; literal-loss states/configs are incompatible")
        for name in ('mu_0', 'mu_increase_factor', 'max_mu', 'penalty_increase_factor',
                     'min_penalty', 'max_penalty', 'multiplier_clip'):
            value = getattr(self, name)
            if value is None and name in ('penalty_increase_factor', 'min_penalty', 'max_penalty', 'multiplier_clip'):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ('ema_beta', 'penalty_drop_ratio', 'multiplier_improve_ratio', 'multiplier_min_improve'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if self.ema_beta > 1 or self.penalty_drop_ratio > 1 or self.multiplier_improve_ratio > 1:
            raise ValueError("EMA and improvement/drop ratios must lie in [0,1]")
        for name in ('warmup_epochs', 'penalty_check_interval', 'multiplier_check_interval'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'warmup_epochs' else 1):
                raise ValueError(f"Invalid epoch/interval setting: {name}")
        if (self.min_penalty or self.mu_0) > (self.max_penalty or self.max_mu):
            raise ValueError("min_penalty exceeds max_penalty")
        for name in ('normalize_by_rms', 'normalize_by_size', 'verbose'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be boolean")
        if self.units != "opfdata_release1_pu_radians" or self.zero_rating != "unmonitored":
            raise ValueError("Only explicit release-1 p.u./radians and zero-rating=unmonitored are supported")

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("lagrangian_config must be an explicit mapping")
        unknown = set(values) - cls.__dataclass_fields__.keys()
        if unknown:
            raise ValueError(f"Unknown AL configuration keys: {sorted(unknown)}")
        return cls(**values)


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite {name}")


def al_terms(mse, signal, multipliers, penalty, n_eq, active=True):
    """Upstream MSE - lambda.dot(signal) + mu/2 * signal.square().sum()."""
    if mse.ndim != 0 or signal.ndim != 1 or signal.shape != multipliers.shape:
        raise ValueError("Require scalar MSE and matching 1D constraints/multipliers")
    for value in (mse, signal, multipliers):
        if not value.is_floating_point() or value.dtype != signal.dtype or value.device != signal.device:
            raise ValueError("Loss tensors must share a floating dtype/device")
        _finite(value, "AL term input")
    eq, ineq = signal[:n_eq], signal[n_eq:]
    lam = multipliers.detach()
    zero = mse.new_zeros(())
    terms = {'objective': mse,
             'linear_eq': -torch.dot(lam[:n_eq], eq) if active else zero,
             'linear_ineq': -torch.dot(lam[n_eq:], ineq) if active else zero,
             'quadratic_eq': penalty / 2 * eq.square().sum(),
             'quadratic_ineq': penalty / 2 * ineq.square().sum()}
    total = mse + terms['quadratic_eq'] + terms['quadratic_ineq'] + terms['linear_eq'] + terms['linear_ineq']
    _finite(total, "AL total")
    return total, terms


class FixedTopologyResiduals:
    """Checked base Ybus with [B,N] voltages and canonical component indices.

    Immutable CPU constants are kep at archive precision for identity checks.
    Sparse power-balance multiplication uses the source's cpu fallback, including
    on XPU. Transfers of preds retain autograd. No predicted tensor is
    hashed, converted to numpy, or detached here.
    """
    BRANCHES = {
        "ac_line": (4, 5, 2, 3, 6, None, None),
        "transformer": (2, 3, 9, 10, 4, 7, 8),
    }

    def __init__(self, graph):
        if getattr(graph, "num_graphs", 1) != 1 or not hasattr(graph, "node_types"):
            raise ValueError("Initialize physics with one original heterogeneous graph")
        graph = graph.clone().cpu()
        self.counts = {}
        self.links = {}
        self.branches = {}
        for name in ("bus", "generator", "load", "shunt"):
            if name not in graph.node_types:
                raise ValueError(f"Missing physical node type {name}; use an explicit empty store if absent")
            self.counts[name] = int(graph[name].num_nodes)
        if self.counts["bus"] == 0:
            raise ValueError("Require at least one bus")
        for name in ("generator", "load", "shunt"):
            key = (name, f"{name}_link", "bus")
            if key not in graph.edge_types:
                raise ValueError(f"Missing physical relation {key}")
            edge = graph[key].edge_index.detach().clone()
            self._check_edge(edge, self.counts[name], self.counts["bus"], key)
            if edge.shape[1] != self.counts[name] or not torch.equal(edge[0].sort().values, torch.arange(self.counts[name])):
                raise ValueError(f"Every {name} must attach to exactly one bus")
            self.links[name] = edge
        self.shunts = self._features(graph, "shunt", 2).clone()
        _finite(self.shunts, "shunts")
        self.base_mva = torch.as_tensor(getattr(graph, "baseMVA", float("nan"))).double().reshape(-1)
        if self.base_mva.numel() != 1 or not bool(torch.isfinite(self.base_mva).all()) or bool((self.base_mva <= 0).any()):
            raise ValueError("Require finite positive baseMVA metadata (no implicit conversion)")
        self._features(graph, "load", 2)
        found = False
        for kind, columns in self.BRANCHES.items():
            key = ("bus", kind, "bus")
            if key not in graph.edge_types:
                continue
            found = True
            edge = graph[key].edge_index.detach().clone()
            self._check_edge(edge, self.counts["bus"], self.counts["bus"], key)
            attr = graph[key].edge_attr.detach()
            width = 9 if kind == "ac_line" else 11
            if edge.shape[1] == 0 and attr.numel() == 0:
                attr = attr.reshape(0, width)
            if attr.ndim != 2 or attr.shape != (edge.shape[1], width):
                raise ValueError(f"Invalid {kind} feature shape")
            # All archived fields participate in identity, even unused ratings.
            self.branches[kind] = (edge, attr.clone())
        if not found:
            raise ValueError("Require explicit branch stores, possibly empty")
        self.identity = self._identity()
        self._build_electrical()
        self.validate_batch(graph)

    @staticmethod
    def _check_edge(edge, n_src, n_dst, name):
        if edge.dtype != torch.long or edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError(f"Invalid edge index for {name}")
        if bool((edge < 0).any()) or bool((edge[0] >= n_src).any()) or bool((edge[1] >= n_dst).any()):
            raise ValueError(f"Out-of-range physical link for {name}")

    @staticmethod
    def _features(graph, name, width):
        x = graph[name].x
        if x.numel() == 0 and graph[name].num_nodes == 0:
            return x.reshape(0, width)
        if x.ndim != 2 or x.shape != (graph[name].num_nodes, width):
            raise ValueError(f"Invalid {name} physical feature shape")
        return x

    def _identity(self):
        digest = hashlib.sha256(json.dumps(self.counts, sort_keys=True).encode())
        tensors = {"baseMVA": self.base_mva, "shunts": self.shunts, **self.links}
        for name, (edge, attr) in self.branches.items():
            tensors[name + ".edge"] = edge
            tensors[name + ".attr"] = attr
        for name, value in sorted(tensors.items()):
            digest.update(name.encode())
            digest.update(str(tuple(value.shape)).encode())
            # Constants only: normalize archive floating representation, no prediction path.
            digest.update(value.double().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def _build_electrical(self):
        indices, ff, ft, tf, tt, limits = [], [], [], [], [], []
        self.branch_ids = []
        for kind, (edge, raw) in self.branches.items():
            cols = self.BRANCHES[kind]
            r, x, bfr, bto, rate = (raw[:, i].double() for i in cols[:5])
            for name, value in (("r", r), ("x", x), ("b_fr", bfr), ("b_to", bto)):
                _finite(value, name)
            if bool((torch.isnan(rate) | (rate < 0)).any()):
                raise ValueError("Ratings must be nonnegative; positive infinity is unmonitored")
            denom = r.square() + x.square()
            if bool((denom == 0).any()):
                raise ValueError("Zero-impedance branch is unsupported")
            # Source _build_branch_admittance formulas, without silent sanitization/clamping.
            y = torch.complex(r / denom, -x / denom)
            tap = raw[:, cols[5]].double() if cols[5] is not None else torch.ones_like(r)
            shift = raw[:, cols[6]].double() if cols[6] is not None else torch.zeros_like(r)
            _finite(tap, "tap"); _finite(shift, "shift")
            tap = torch.where(tap == 0, torch.ones_like(tap), tap)
            if bool((tap < 0).any()):
                raise ValueError("Tap ratios must be nonnegative (zero denotes unity)")
            tap_complex = torch.complex(tap * torch.cos(shift), tap * torch.sin(shift))
            indices.append(edge)
            ff.append((y + torch.complex(torch.zeros_like(bfr), bfr)) / tap.square())
            tt.append(y + torch.complex(torch.zeros_like(bto), bto))
            ft.append(-y / tap_complex.conj()); tf.append(-y / tap_complex)
            limits.append(rate)
            self.branch_ids.extend(f"{kind}:{i}" for i in range(edge.shape[1]))
        self.edge = torch.cat(indices, dim=1)
        self.yff, self.yft, self.ytf, self.ytt = (torch.cat(parts) for parts in (ff, ft, tf, tt))
        rate = torch.cat(limits)
        self.monitored = torch.isfinite(rate) & (rate > 0)
        self.ratings = rate[self.monitored]  # mask BEFORE subtraction/squaring
        i, j = self.edge
        row = torch.cat((i, i, j, j, self.links["shunt"][1]))
        col = torch.cat((i, j, i, j, self.links["shunt"][1]))
        shunt = self.shunts[self.links["shunt"][0]].double()
        values = torch.cat((self.yff, self.yft, self.ytf, self.ytt,
                            torch.complex(shunt[:, 1], shunt[:, 0])))
        n = self.counts["bus"]
        idx = torch.stack((row, col))
        self.y_real = torch.sparse_coo_tensor(idx, values.real, (n, n), check_invariants=True).coalesce()
        self.y_imag = torch.sparse_coo_tensor(idx, values.imag, (n, n), check_invariants=True).coalesce()
        _finite(self.y_real.values(), "Ybus real"); _finite(self.y_imag.values(), "Ybus imag")
        monitored_ids = [name for name, keep in zip(self.branch_ids, self.monitored.tolist()) if keep]
        self.constraint_order = {
            "eq": [f"{part}:bus:{i}" for part in ("P", "Q") for i in range(n)],
            "ineq": [f"{end}:{name}" for end in ("from", "to") for name in monitored_ids],
        }

    def validate_batch(self, batch):
        if not hasattr(batch, "node_types"):
            raise ValueError("AL requires the original heterogeneous physical graph")
        b = int(getattr(batch, "num_graphs", 1))
        if b < 1:
            raise ValueError("Empty batch")
        for name, n in self.counts.items():
            if name not in batch.node_types or batch[name].num_nodes != b * n:
                raise ValueError(f"Changed topology/node count: {name}")
            ptr = getattr(batch[name], "ptr", None)
            if ptr is not None and not torch.equal(ptr.cpu(), torch.arange(b + 1) * n):
                raise ValueError(f"Changed graph-local ordering/count: {name}")
        base = torch.as_tensor(getattr(batch, "baseMVA", float("nan"))).detach().cpu().double().reshape(-1)
        if base.shape != (b,) or not torch.equal(base, self.base_mva.expand(b)):
            raise ValueError("Changed/missing baseMVA metadata")
        for name, expected in self.links.items():
            key = (name, f"{name}_link", "bus")
            self._validate_edges(batch, key, expected, b, self.counts[name], self.counts["bus"])
        shunt = self._features(batch, "shunt", 2).detach().cpu()
        if not torch.equal(shunt.reshape(b, self.counts["shunt"], 2), self.shunts.expand(b, -1, -1)):
            raise ValueError("Changed cached shunt parameters")
        load = self._features(batch, "load", 2)
        _finite(load, "demand")
        for kind in self.BRANCHES:
            key = ("bus", kind, "bus")
            if (key in batch.edge_types) != (kind in self.branches):
                raise ValueError("Changed branch types")
            if kind not in self.branches:
                continue
            edge, attr = self.branches[kind]
            self._validate_edges(batch, key, edge, b, self.counts["bus"], self.counts["bus"])
            actual = batch[key].edge_attr.detach().cpu()
            if actual.numel() != b * attr.numel() or (actual.numel() and actual.shape != (b * attr.shape[0], attr.shape[1])):
                raise ValueError(f"Changed {kind} feature dimensions")
            if not torch.equal(actual.reshape(b, *attr.shape), attr.expand(b, -1, -1)):
                raise ValueError(f"Changed cached {kind} electrical parameters/order")
        return b

    @staticmethod
    def _validate_edges(batch, key, expected, b, n_src, n_dst):
        if key not in batch.edge_types:
            raise ValueError(f"Missing relation {key}")
        actual = batch[key].edge_index.detach().cpu()
        offsets = torch.arange(b).reshape(1, b, 1) * torch.tensor([n_src, n_dst]).reshape(2, 1, 1)
        want = (expected[:, None, :] + offsets).reshape(2, -1)
        if actual.dtype != torch.long or not torch.equal(actual, want):
            raise ValueError(f"Changed physical connectivity/order: {key}")

    def residuals(self, predictions, batch):
        b = self.validate_batch(batch)
        bus, gen = predictions["bus"], predictions["generator"]
        for name, value in (("bus", bus), ("generator", gen)):
            if value.shape != (b * self.counts[name], 2) or value.dtype not in (torch.float32, torch.float64):
                raise ValueError(f"AL requires full FP32/FP64 [va,vm]/[pg,qg] predictions: {name}")
            if value.device != bus.device or value.dtype != bus.dtype:
                raise ValueError("Prediction dtype/device mismatch")
            _finite(value, name + " predictions")
        n = self.counts["bus"]
        bus = bus.reshape(b, n, 2)
        gen = gen.reshape(b, self.counts["generator"], 2)
        va, vm = bus[..., 0], bus[..., 1]
        vr, vi = vm * va.cos(), vm * va.sin()
        p, q = torch.zeros_like(vr), torch.zeros_like(vi)
        src, dst = self.links["generator"].to(bus.device)
        p = p.index_add(1, dst, gen[:, src, 0]); q = q.index_add(1, dst, gen[:, src, 1])
        demand = self._features(batch, "load", 2).to(device=bus.device, dtype=bus.dtype).reshape(b, self.counts["load"], 2)
        src, dst = self.links["load"].to(bus.device)
        p = p.index_add(1, dst, -demand[:, src, 0]); q = q.index_add(1, dst, -demand[:, src, 1])
        # Same four sparse products as source; CPU fallback remains differentiable.
        yr, yi = self.y_real.to(dtype=bus.dtype), self.y_imag.to(dtype=bus.dtype)
        vrc, vic = vr.to("cpu").T, vi.to("cpu").T
        ir = (torch.sparse.mm(yr, vrc) - torch.sparse.mm(yi, vic)).T.to(bus.device)
        ii = (torch.sparse.mm(yi, vrc) + torch.sparse.mm(yr, vic)).T.to(bus.device)
        r = torch.cat((p - (vr * ir + vi * ii), q - (vi * ir - vr * ii)), dim=1)
        sf, st = self.terminal_powers(vr, vi)
        mask = self.monitored.to(bus.device)
        limit2 = self.ratings.to(device=bus.device, dtype=bus.dtype).square()
        sf, st = sf[:, mask], st[:, mask]
        h = torch.cat((sf.real.square() + sf.imag.square() - limit2,
                       st.real.square() + st.imag.square() - limit2), dim=1)
        _finite(r, "balance residuals"); _finite(h, "thermal residuals")
        return r, h

    def terminal_powers(self, vr, vi):
        """Both physical ends, before max/ReLU/sqrt (source line-flow formulas)."""
        i, j = self.edge.to(vr.device)
        voltage = torch.complex(vr, vi)
        ctype = voltage.dtype
        ff, ft, tf, tt = (v.to(device=vr.device, dtype=ctype) for v in (self.yff, self.yft, self.ytf, self.ytt))
        vf, vt = voltage[:, i], voltage[:, j]
        return vf * (ff * vf + ft * vt).conj(), vt * (tf * vf + tt * vt).conj()


class _LagrangianSchedule:
    """Epoch-based update schedule used by Frontier training."""

    def __init__(self, warmup_epochs, penalty_check_interval, multiplier_check_interval):
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.penalty_check_interval = max(1, int(penalty_check_interval))
        self.multiplier_check_interval = max(1, int(multiplier_check_interval))
        self.current_epoch = 0
        self.epochs_since_penalty_check = 0
        self.multiplier_steps_since_update = 0

    def warmup_complete(self):
        return self.current_epoch >= self.warmup_epochs

    def advance_epoch(self):
        self.current_epoch += 1

    def penalty_check_due(self):
        self.epochs_since_penalty_check += 1
        if self.epochs_since_penalty_check < self.penalty_check_interval:
            return False
        self.epochs_since_penalty_check = 0
        return True

    def force_penalty_check(self):
        self.epochs_since_penalty_check = 0

    def multiplier_interval_ready(self):
        if self.multiplier_steps_since_update < self.multiplier_check_interval - 1:
            self.multiplier_steps_since_update += 1
            return False
        return True

    def reset_multiplier_interval(self):
        self.multiplier_steps_since_update = 0

    def reset_penalty_interval(self):
        self.epochs_since_penalty_check = 0

    def reset(self):
        self.current_epoch = 0
        self.epochs_since_penalty_check = 0
        self.multiplier_steps_since_update = 0

    def training_state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "epochs_since_penalty_check": self.epochs_since_penalty_check,
            "multiplier_steps_since_update": self.multiplier_steps_since_update,
        }

    def load_training_state_dict(self, state):
        if not isinstance(state, dict):
            return
        for name in self.training_state_dict():
            if name in state:
                setattr(self, name, int(state[name]))



class AugmentedLagrangianLoss(nn.Module):
    """Source numerical policy, with state committed only after successful steps."""
    def __init__(self, config):
        super().__init__()
        self.config = ALConfig.from_dict(config)
        for name, value in asdict(self.config).items():
            setattr(self, name, value)
        self.penalty_increase_factor = self.penalty_increase_factor or self.mu_increase_factor
        self.min_penalty = self.min_penalty if self.min_penalty is not None else self.mu_0
        self.max_penalty = self.max_penalty if self.max_penalty is not None else self.max_mu
        self.max_mu = self.max_penalty
        self.mu_k = float(np.clip(self.mu_0, self.min_penalty, self.max_penalty))
        self._schedule = _LagrangianSchedule(self.warmup_epochs, self.penalty_check_interval,
                                             self.multiplier_check_interval)
        self.physics = None
        self.register_buffer('lambda_k', torch.empty(0))
        self.register_buffer('constraint_ema', torch.empty(0))
        self.register_buffer('successful_steps', torch.zeros((), dtype=torch.long))
        self.register_buffer('dual_updates', torch.zeros((), dtype=torch.long))
        self._raw_violation = self._ema_violation = None
        self._last_penalty_check_violation = self._last_multiplier_violation = None
        self._last_multiplier_norm = None
        self._last_multiplier_updated = False

    @property
    def current_epoch(self):
        return self._schedule.current_epoch

    def initialize_constraints(self, graph, device=None, dtype=torch.float32):
        if self.physics is not None:
            raise RuntimeError("AL constraints already initialized; refuse to reset duals")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("AL requires FP32 or FP64")
        self.physics = FixedTopologyResiduals(graph)
        self.n_eq = len(self.physics.constraint_order['eq'])
        self.n_line = len(self.physics.constraint_order['ineq']) // 2
        self.lambda_k = torch.zeros(self.n_eq + self.n_line, device=device, dtype=dtype)
        self.constraint_ema = torch.zeros_like(self.lambda_k)
        self.successful_steps = self.successful_steps.to(device=device)
        self.dual_updates = self.dual_updates.to(device=device)

    def _require_initialized(self):
        if self.physics is None:
            raise RuntimeError("Initialize AL constraints explicitly before forward/load")

    def constraint_vector(self, r, h):
        # Source compute_power_flow_constraints: RMS across graphs, epsilon 1e-8.
        eq = torch.sqrt(r.square().mean(dim=0) + 1e-8)
        # Source compute_line_flow_constraints: worst end, ReLU, sqrt, mean graphs.
        worst = torch.maximum(h[:, :self.n_line], h[:, self.n_line:])
        ineq = torch.sqrt(torch.relu(worst)).mean(dim=0)
        return torch.cat((self._normalize_constraint_vector(eq), self._normalize_constraint_vector(ineq)))

    def forward(self, mse, predictions, batch):
        self._require_initialized()
        r, h = self.physics.residuals(predictions, batch)
        constraints = self.constraint_vector(r, h)
        signal = self.ema_beta * self.constraint_ema + (1 - self.ema_beta) * constraints
        total, info = al_terms(mse, signal, self.lambda_k, self.mu_k, self.n_eq, self._should_use_multipliers())
        info.update(penalty_parameter=self.mu_k, multipliers_active=self._should_use_multipliers(),
                    constraint_violation=signal.detach().norm(),
                    al_observation={'c': constraints.detach(), 'r': r.detach(), 'h': h.detach(),
                                    'step': int(self.successful_steps.item()) + 1,
                                    'epoch': self.current_epoch, 'identity': self.physics.identity})
        return total, info

    @torch.no_grad()
    def on_successful_step(self, observation, successful_steps):
        self._require_initialized()
        if (successful_steps != int(self.successful_steps.item()) + 1 or
                observation['step'] != successful_steps or observation['epoch'] != self.current_epoch or
                observation['identity'] != self.physics.identity):
            raise ValueError("Stale/incompatible AL observation or successful-step counter")
        constraints = observation['c']
        if (constraints.shape != self.lambda_k.shape or constraints.dtype != self.lambda_k.dtype or
                constraints.device != self.lambda_k.device or bool((constraints < 0).any())):
            raise ValueError("Invalid AL constraint observation")
        _finite(constraints, 'AL constraint observation')
        self.constraint_ema.copy_(self.ema_beta * self.constraint_ema + (1 - self.ema_beta) * constraints)
        self._raw_violation = self._compute_violation_norm(constraints)
        self._ema_violation = self._compute_violation_norm(self.constraint_ema)
        self._last_multiplier_updated = False
        if self._should_use_multipliers() and self._should_update_multipliers_now(self._ema_violation):
            self._schedule.reset_multiplier_interval()
            self._apply_multiplier_update(self.constraint_ema)
            self._last_multiplier_norm = self._compute_violation_norm(self.lambda_k)
            self._last_multiplier_violation = self._ema_violation
            self._last_multiplier_updated = True
            self.dual_updates.add_(1)
        self.successful_steps.fill_(successful_steps)
        return self._last_multiplier_updated

    def _normalize_constraint_vector(self, constraints: torch.Tensor) -> torch.Tensor:
        """Apply configured RMS/size normalization to a 1D constraint vector."""
        if constraints.numel() == 0:
            return constraints

        normalized = constraints
        if self.normalize_by_rms:
            rms_value = torch.sqrt(torch.mean(normalized**2)) + 1e-8
            normalized = normalized / rms_value
        if self.normalize_by_size:
            scale_factor = torch.sqrt(normalized.new_tensor(float(normalized.numel())))
            normalized = normalized / scale_factor
        return normalized


    def _warmup_complete(self) -> bool:
        return self._schedule.warmup_complete()


    def _should_use_multipliers(self) -> bool:
        """Return True when multiplier updates are allowed (post-warmup)."""
        return self._warmup_complete()


    def _compute_violation_norm(self, constraints: torch.Tensor | None) -> float:
        """Compute L2 norm of a constraint vector (safe for None or empty)."""
        if constraints is None or constraints.numel() == 0:
            return 0.0
        return float(torch.norm(constraints, p=2).detach().item())


    def _maybe_update_penalty(self, current_violation: float | None, force: bool = False):
        """Increase μ if violations are not shrinking enough based on EMA trend."""
        if current_violation is None:
            return
        if torch.is_tensor(current_violation):
            current_violation = float(current_violation.detach().item())

        # Always ensure μ stays within bounds
        self.mu_k = float(np.clip(self.mu_k, self.min_penalty, self.max_penalty))

        # During warmup, just seed the baseline and skip updates
        if not self._warmup_complete() and not force:
            self._last_penalty_check_violation = current_violation
            return

        if not force:
            if not self._schedule.penalty_check_due():
                return
        else:
            self._schedule.force_penalty_check()

        if self._last_penalty_check_violation is None:
            self._last_penalty_check_violation = current_violation
            return

        # If violation did not drop enough, bump μ
        if current_violation > self.penalty_drop_ratio * self._last_penalty_check_violation:
            old_mu = self.mu_k
            self.mu_k = min(max(self.mu_k * self.penalty_increase_factor, self.min_penalty), self.max_penalty)
            if self.mu_k > old_mu and self.verbose:
                print(f"Increased penalty parameter from {old_mu:.3e} to μ = {self.mu_k:.3e}")

        self._last_penalty_check_violation = current_violation


    def _should_update_multipliers_now(self, violation: float | None) -> bool:
        """Decide whether to update λ based on violation improvement and interval."""
        if violation is None:
            return False

        # Default: always update when gating disabled
        gating_disabled = self.multiplier_improve_ratio <= 0.0 and self.multiplier_min_improve <= 0.0
        if gating_disabled:
            return True

        # First observation, allow an update to establish baseline
        if self._last_multiplier_violation is None:
            return True

        # Respect minimum interval between updates
        if not self._schedule.multiplier_interval_ready():
            return False

        ratio_ok = (
            self.multiplier_improve_ratio <= 0.0
            or violation <= self.multiplier_improve_ratio * self._last_multiplier_violation
        )
        abs_ok = (
            self.multiplier_min_improve <= 0.0
            or violation <= self._last_multiplier_violation - self.multiplier_min_improve
        )

        return ratio_ok and abs_ok


    def _apply_multiplier_update(self, constraint_signal: torch.Tensor):
        update = self.mu_k * constraint_signal.detach()
        self.lambda_k = self.lambda_k - update

        if self.multiplier_clip is not None:
            self.lambda_k = torch.clamp(self.lambda_k, -self.multiplier_clip, self.multiplier_clip)


    def step_epoch(self):
        """Advance epoch counter and run scheduled penalty updates."""
        current_violation = self._ema_violation if self._ema_violation is not None else self._raw_violation
        self._maybe_update_penalty(current_violation)
        self._schedule.advance_epoch()


    def get_extra_state(self):
        return {'version': 2, 'config': asdict(self.config), 'source_commit': SOURCE_COMMIT,
                'identity': self.physics.identity if self.physics else None,
                'constraint_order': ({'eq': self.physics.constraint_order['eq'],
                    'ineq': ['worst:' + key.removeprefix('from:')
                             for key in self.physics.constraint_order['ineq'][:self.n_line]]}
                    if self.physics else None),
                'algorithm': {'mu_k': self.mu_k, 'schedule': self._schedule.training_state_dict(),
                    **{name: getattr(self, name) for name in self._state_fields()}}}

    @staticmethod
    def _state_fields():
        return ('_raw_violation', '_ema_violation', '_last_penalty_check_violation',
                '_last_multiplier_violation', '_last_multiplier_norm', '_last_multiplier_updated')

    def _validate_extra_state(self, state):
        expected = self.get_extra_state()
        if not isinstance(state, dict) or state.keys() != expected.keys() or any(
                state[k] != expected[k] for k in expected if k != 'algorithm'):
            raise ValueError('Incompatible AL formula/config/topology/order state')
        alg = state['algorithm']
        if not isinstance(alg, dict) or alg.keys() != expected['algorithm'].keys():
            raise ValueError('Invalid AL algorithm state')
        if not isinstance(alg['mu_k'], (float, int)) or not math.isfinite(alg['mu_k']) or not self.min_penalty <= alg['mu_k'] <= self.max_penalty:
            raise ValueError('Invalid AL penalty state')
        schedule = alg['schedule']
        if not isinstance(schedule, dict) or schedule.keys() != self._schedule.training_state_dict().keys():
            raise ValueError('Invalid AL schedule state')
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in schedule.values()):
            raise ValueError('Invalid AL schedule counters')
        if schedule['epochs_since_penalty_check'] >= self.penalty_check_interval or schedule['multiplier_steps_since_update'] >= self.multiplier_check_interval:
            raise ValueError('Invalid AL schedule counters')
        for name in self._state_fields():
            value = alg[name]
            if name == '_last_multiplier_updated':
                if not isinstance(value, bool): raise ValueError('Invalid AL update flag')
            elif value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
                raise ValueError('Invalid AL violation/norm state')

    def set_extra_state(self, state):
        self._validate_extra_state(state)
        alg = state['algorithm']
        self.mu_k = float(alg['mu_k'])
        self._schedule.load_training_state_dict(alg['schedule'])
        for name in self._state_fields():
            setattr(self, name, alg[name])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._require_initialized()
        if state_dict.keys() != self.state_dict().keys():
            raise ValueError('Incompatible AL state keys')
        self._validate_extra_state(state_dict.get('_extra_state'))
        for key in ('lambda_k', 'constraint_ema'):
            value = state_dict[key]
            if value.shape != self.lambda_k.shape or value.dtype != self.lambda_k.dtype:
                raise ValueError(f'Incompatible AL {key} shape/dtype')
            _finite(value, key)
        if bool((state_dict['lambda_k'] > 0).any()) or bool((state_dict['constraint_ema'] < 0).any()):
            raise ValueError('Invalid multiplier/EMA sign')
        if self.multiplier_clip is not None and bool((state_dict['lambda_k'].abs() > self.multiplier_clip).any()):
            raise ValueError('Multipliers exceed configured clip')
        for key in ('successful_steps', 'dual_updates'):
            value = state_dict[key]
            if value.shape != torch.Size([]) or value.dtype != torch.long or value.item() < 0:
                raise ValueError(f'Invalid AL {key}')
        if state_dict['dual_updates'].item() > state_dict['successful_steps'].item():
            raise ValueError('Inconsistent AL schedule counters')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def training_state_dict(self):
        self._require_initialized()
        return {k: v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.state_dict().items()}
