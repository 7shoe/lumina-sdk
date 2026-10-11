"""Feasible OPF data on CPU, with a differentiable inner loop on CPU or XPU."""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.func import functional_call

from distillation.conversion.matrix_form import mat2opf, opf2mat
from distillation.synthetic.physical import Perturbation, audit_networks
from distillation.training.T1_single_topo.common import canonical_case
from experiments.lumina_bench.runner import build_model


from distillation.synthetic.learning import benchmark_batch, benchmark_graph, targets, mse


class SyntheticOPF(nn.Module):
    """K ordered batches of independent, bounded OPF parameters and teaching targets.

    Each graph's incidence/equipment matrices stay fixed within an unroll;
    graph sizes may differ after an outer topology-search step. Derived Y is a
    differentiable function of physical branch parameters, not a free dense matrix.
    Zero offsets reproduce the seed inputs/targets up to reciprocal roundoff.
    Teaching targets are synthetic regression supervision, never exported as an
    AC-OPF solution. A separate, detached solver audit checks the edited inputs.
    """

    def __init__(self, graphs, *, steps=3, strength=0.02, target_strength=0.02,
                 learn_targets=True, initial_rate=1e-6, rate_min=1e-8, rate_max=1e-4):
        super().__init__()
        if not graphs or steps < 1 or len(graphs) % steps:
            raise ValueError("Need a nonempty dataset evenly divided into K inner batches")
        if not (0 < strength < 1 and 0 <= target_strength < 1):
            raise ValueError("Expected 0 < strength < 1 and 0 <= target_strength < 1")
        if not 0 < rate_min < initial_rate < rate_max or not math.isfinite(rate_max):
            raise ValueError("Expected finite 0 < rate_min < initial_rate < rate_max")
        graphs = [graph.clone().detach().cpu() for graph in graphs]
        self.anchors = [opf2mat(graph) for graph in graphs]
        if any(matrix.params.N < 1 or matrix.params.G < 1 for matrix in self.anchors):
            raise ValueError("Every graph needs buses and generators")
        self.steps, self.batch_size = steps, len(graphs) // steps
        self.strength, self.target_strength = strength, target_strength
        self.rate_min, self.rate_max = rate_min, rate_max
        self.edits = nn.ModuleList([Perturbation(matrix) for matrix in self.anchors])
        self.target_offsets = nn.ModuleList([
            nn.ParameterDict({kind: nn.Parameter(torch.zeros_like(graph[kind].y),
                                                requires_grad=learn_targets)
                              for kind in ("bus", "generator")})
            for graph in graphs
        ])
        self.seed_targets = [{kind: graph[kind].y.clone() for kind in ("bus", "generator")}
                             for graph in graphs]
        probability = (initial_rate - rate_min) / (rate_max - rate_min)
        self.rate_logits = nn.Parameter(torch.full(
            (steps,), math.log(probability / (1 - probability)), dtype=torch.float64))
        with torch.no_grad():
            for edit in self.edits:
                for parameter in edit.parameters():
                    parameter.zero_()

    @property
    def rates(self):
        return self.rate_min + (self.rate_max - self.rate_min) * self.rate_logits.sigmoid()

    def matrices(self):
        return [edit(matrix, self.strength) for edit, matrix in zip(self.edits, self.anchors)]

    def batch(self, step):
        if step not in range(self.steps):
            raise IndexError(step)
        indices = range(step * self.batch_size, (step + 1) * self.batch_size)
        graphs, teaching = [], {kind: [] for kind in ("bus", "generator")}
        for i in indices:
            matrix = self.edits[i](self.anchors[i], self.strength)
            graphs.append(mat2opf(matrix, include_solution=False))
            for kind in teaching:
                base = self.seed_targets[i][kind]
                delta = self.target_strength * base.abs().clamp_min(0.1) * self.target_offsets[i][kind].tanh()
                if kind == "bus":
                    reference = self.anchors[i].params.bus_type == 3
                    delta = torch.stack((delta[:, 0] * (~reference), delta[:, 1]), dim=1)
                teaching[kind].append(base + delta)
        return benchmark_batch(graphs), {kind: torch.cat(value) for kind, value in teaching.items()}


def make_model(sample, *, initialization="scratch", checkpoint_root=None, case=30,
               seed=42, hidden_channels=16):
    """Use the actual benchmark HeteroGNN/GAT; load only its requested checkpoint."""
    kind = "HeteroGNN"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if initialization == "scratch":
            options = dict(hidden_channels=hidden_channels, num_layers=2, backend="gat")
            model = build_model({"model_type": kind, "models": {kind: options}},
                                benchmark_graph(sample), torch.device("cpu"))
        elif initialization == "checkpoint":
            case_name = canonical_case(f"case{case}")
            path = Path(checkpoint_root).expanduser() / kind / case_name / f"seed{seed}" / "best.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Requested checkpoint missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            spec = payload["specification"]
            if spec["model"] != kind or spec["case"] != case_name:
                raise ValueError("Checkpoint model/case does not match the request")
            model = build_model({"model_type": kind, "models": {kind: spec["model_options"]}},
                                payload["example_graph"].cpu(), torch.device("cpu"))
            model.load_state_dict(payload["model"], strict=True)
        else:
            raise ValueError("INITIALIZATION must be 'scratch' or 'checkpoint'")
    # Disable stochastic dropout for this reproducible trajectory; gradients stay enabled.
    return model.cpu().eval()


def unroll(model, synthetic, real_batch, real_targets, *, create_graph=True):
    """Reset once to theta_0, then differentiate through K ordered SGD steps.

    Physical parameters remain on CPU. Tensor copies to the model's device keep
    their autograd links, including second derivatives back to the CPU leaves.
    No discrete topology operation or external OPF solve enters this graph.
    """
    device = next(model.parameters()).device
    theta = {name: p.detach().clone().requires_grad_(True) for name, p in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rates = synthetic.rates.to(device)
    trace = []
    for step in range(synthetic.steps):
        batch, teaching_targets = synthetic.batch(step)
        batch = batch.to(device)
        prediction = functional_call(model, (theta, buffers), (batch,), strict=True)
        inner_loss = mse(prediction, teaching_targets, batch)
        gradients = torch.autograd.grad(inner_loss, tuple(theta.values()), create_graph=create_graph, allow_unused=True)
        theta = {name: p if g is None else p - rates[step] * g for (name, p), g in zip(theta.items(), gradients)}
        if not create_graph:
            theta = {name: p.detach().requires_grad_(True) for name, p in theta.items()}
        trace.append(dict(step=step + 1, inner_mse=float(inner_loss.detach()), rate=float(rates[step].detach())))
    real_batch = real_batch.clone().to(device)
    prediction = functional_call(model, (theta, buffers), (real_batch,), strict=True)
    return theta, mse(prediction, real_targets, real_batch), pd.DataFrame(trace)


def gradient_summary(synthetic):
    rows = {}
    for name, parameter in synthetic.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "rates" if name == "rate_logits" else f"{name.split('.')[0]}/{name.split('.')[-1]}"
        row = rows.setdefault(family, dict(parameters=0, gradient_squared=0., max_abs_gradient=0.))
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Missing or nonfinite outer gradient: {name}")
        row["parameters"] += parameter.numel()
        row["gradient_squared"] += float(parameter.grad.square().sum())
        row["max_abs_gradient"] = max(row["max_abs_gradient"],
                                      float(parameter.grad.abs().max()) if parameter.numel() else 0.)
    return pd.DataFrame([dict(family=family, parameters=row["parameters"],
                              gradient_norm=math.sqrt(row["gradient_squared"]),
                              max_abs_gradient=row["max_abs_gradient"])
                         for family, row in rows.items()]).set_index("family")



def backtrack(synthetic, before, proposal, score, reference_loss, *, audit=audit_networks,
              fractions=(1., 0.5, 0.25, 0.125, 0.0625)):
    """Accept one scaled Adam proposal, or restore ALL synthetic parameters and rates.

    Replays start from the same model initialization. This feasibility/descent gate
    is an OPF-specific addition to Wang's update, not a differentiable OPF solver.
    Optimizer moments are not rewound: the notebook makes exactly one outer update.
    """
    attempts = []
    accepted = False
    try:
        for fraction in fractions:
            if not 0 < fraction <= 1:
                raise ValueError("Backtracking fractions must be in (0, 1]")
            synthetic.load_state_dict({key: value + fraction * (proposal[key] - value)
                                       for key, value in before.items()})
            loss = float(score())
            row = dict(fraction=fraction, real_mse=loss, audited=0, solved=0, accepted=False)
            if math.isfinite(loss) and loss <= reference_loss:
                checks, solutions = audit(synthetic.matrices())
                row.update(audited=len(checks), solved=int(checks.success.sum()))
                if len(checks) == len(synthetic.anchors) and bool(checks.success.all()):
                    accepted = row["accepted"] = True
                    attempts.append(row)
                    return pd.DataFrame(attempts), checks, solutions
            attempts.append(row)
        raise RuntimeError(f"No feasible descent step accepted; initial data/rates restored. Trials: {attempts}")
    finally:
        if not accepted:
            synthetic.load_state_dict(before)


def _gradient_vectors(synthetic):
    groups = {}
    gradient_summary(synthetic)  # Missing/nonfinite gradients are errors, including on CPU.
    for name, p in synthetic.named_parameters():
        if p.requires_grad:
            family = "rates" if name == "rate_logits" else f"{name.split('.')[0]}/{name.split('.')[-1]}"
            groups.setdefault(family, []).append(p.grad.detach().cpu().flatten())
    return {name: torch.cat(values) for name, values in groups.items()}


def check_derivatives(cpu_model, graphs, real_batch, real_targets, options, device):
    """Compare loss and every family of meta-gradients, not just the first backward."""
    results = []
    for target_device in (torch.device("cpu"), device):
        model = deepcopy(cpu_model).to(target_device)
        synthetic = SyntheticOPF(graphs, **options)
        _, loss, _ = unroll(model, synthetic, real_batch, real_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite unroll on {target_device}")
        loss.backward()
        vectors = _gradient_vectors(synthetic)
        if vectors["rates"].norm() == 0 or not any(v.norm() > 0 for k, v in vectors.items() if k.startswith("edits/")):
            raise RuntimeError(f"Missing input/rate meta-gradient on {target_device}")
        results.append((float(loss.detach()), vectors))
        del model, synthetic, loss
    cpu_loss, reference = results[0]
    device_loss, actual = results[1]
    rows = []
    for family in reference:
        a, b = reference[family], actual[family]
        error = float((a - b).abs().max()) if a.numel() else 0.
        scale = float(a.abs().max()) if a.numel() else 0.
        rows.append(dict(family=family, cpu_gradient_norm=float(a.norm()), device_gradient_norm=float(b.norm()),
                         max_abs_error=error, tolerance=1e-10 + 0.02 * scale,
                         error_l2=float((a - b).norm()),
                         cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-30))
                         if a.numel() and a.norm() > 0 and b.norm() > 0 else None,
                         passed=error <= 1e-10 + 0.02 * scale))
    loss_ok = abs(cpu_loss - device_loss) <= 1e-6 + 2e-4 * abs(cpu_loss)
    return dict(device=str(device), cpu_loss=cpu_loss, device_loss=device_loss,
                loss_passed=loss_ok, gradients=rows, passed=loss_ok and all(row["passed"] for row in rows))
