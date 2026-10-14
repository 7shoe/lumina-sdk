"""Test the notebook's shared unroll, bounded OPF inputs, and transactional gate."""
from copy import deepcopy

import pandas as pd
import pytest
import torch

from distillation.synthetic.learning import benchmark_batch
from distillation.DD.mvp import (
    SyntheticOPF, audit_networks, backtrack, gradient_summary, make_model, targets,
    unroll as unroll_core,
)
from distillation.conversion.matrix_form import equal_records, mat2opf, validate_network
from distillation.test import test_dd_demo as reference

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


@pytest.fixture(scope="module")
def unroll():
    return unroll_core


class SmallRegressor(torch.nn.Module):
    """Smooth double-precision model for a second-order finite-difference check."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

    def forward(self, batch):
        bus, gen = batch["bus"].x, batch["generator"].x
        return dict(bus=torch.stack((bus[:, 1] * self.weight[0], bus[:, 2] * self.weight[1]), dim=1),
                    generator=torch.stack((gen[:, 3] * self.weight[2], gen[:, 6] * self.weight[3]), dim=1))


def test_bounded_parameters_keep_physical_structure_and_drop_stale_labels(graph):
    original = graph.clone()
    synthetic = SyntheticOPF([graph], steps=1)
    batch, teaching = synthetic.batch(0)
    for kind in teaching:
        torch.testing.assert_close(teaching[kind], graph[kind].y)
        assert "y" not in batch[kind]
    with torch.no_grad():
        for p in synthetic.edits.parameters():
            p.fill_(2.)
    matrix = synthetic.matrices()[0]
    validate_network(matrix.params)
    torch.testing.assert_close(matrix.params.A, synthetic.anchors[0].params.A, rtol=0, atol=0)
    ratio = (matrix.params.y / synthetic.anchors[0].params.y).real
    assert bool(((ratio >= 0.98) & (ratio <= 1.02)).all())
    assert matrix.params.Y.requires_grad
    exported = mat2opf(matrix, include_solution=False)
    assert "objective" not in exported
    assert all("edge_label" not in exported[edge] for edge in exported.edge_types)
    assert equal_records(graph, original)


@pytest.mark.parametrize("parameter,index", [("input", 0), ("target", (0, 1)), ("rate", 1)])
def test_notebook_meta_gradient_matches_finite_difference(graph, unroll, parameter, index):
    synthetic = SyntheticOPF([graph] * 3, steps=3, initial_rate=0.03, rate_min=1e-6, rate_max=0.2)
    model = SmallRegressor().eval()
    before = deepcopy(model.state_dict())
    batch, real_targets = benchmark_batch([graph]), targets([graph])
    variable = {"input": synthetic.edits[0].delta["voltage"],
                "target": synthetic.target_offsets[0]["bus"], "rate": synthetic.rate_logits}[parameter]
    theta, loss, trace = unroll(model, synthetic, batch, real_targets)
    exact = torch.autograd.grad(loss, variable)[0][index].item()
    epsilon = 1e-4
    values = []
    for sign in (1, -1):
        with torch.no_grad():
            variable[index] += sign * epsilon
        values.append(float(unroll(model, synthetic, batch, real_targets, create_graph=False)[1].detach()))
        with torch.no_grad():
            variable[index] -= sign * epsilon
    estimate = (values[0] - values[1]) / (2 * epsilon)
    assert abs(exact) > 1e-10
    assert exact == pytest.approx(estimate, rel=2e-5, abs=1e-10)
    assert len(trace) == 3
    assert torch.equal(theta["unused"], model.unused)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_real_benchmark_model_has_second_order_input_and_rate_gradients(graph, unroll):
    synthetic = SyntheticOPF([graph] * 3, steps=3)
    model = make_model(graph, hidden_channels=8)
    _, loss, _ = unroll(model, synthetic, benchmark_batch([graph]), targets([graph]))
    assert torch.isfinite(loss)
    loss.backward()
    table = gradient_summary(synthetic)
    assert table.loc["rates", "gradient_norm"] > 0
    assert table.loc["edits/admittance", "gradient_norm"] > 0
    assert table.loc["target_offsets/bus", "gradient_norm"] > 0
    assert all(p.grad is None for p in model.parameters())


def test_positive_per_step_rates_and_fixed_teaching_targets(graph):
    synthetic = SyntheticOPF([graph] * 3, steps=3, learn_targets=False)
    with torch.no_grad():
        synthetic.rate_logits.copy_(torch.tensor([-1000., 0., 1000.]))
    assert bool((synthetic.rates >= synthetic.rate_min).all())
    assert bool((synthetic.rates <= synthetic.rate_max).all())
    assert synthetic.rates.unique().numel() == 3
    assert all(not p.requires_grad for p in synthetic.target_offsets.parameters())
    _, teaching = synthetic.batch(0)
    assert all(not value.requires_grad for value in teaching.values())


def test_gate_backtracks_data_targets_and_rates_together(graph):
    synthetic = SyntheticOPF([graph], steps=1)
    before = deepcopy(synthetic.state_dict())
    proposal = {key: value + 0.1 for key, value in before.items()}
    calls = []

    def audit(matrices):
        calls.append(len(matrices))
        return pd.DataFrame([dict(success=len(calls) == 2)]), [None]

    attempts, checks, _ = backtrack(synthetic, before, proposal, lambda: 0., 1., audit=audit)
    assert calls == [1, 1]
    assert attempts.fraction.tolist() == [1., 0.5] and checks.success.all()
    for name, value in synthetic.state_dict().items():
        torch.testing.assert_close(value, before[name] + 0.5 * (proposal[name] - before[name]))


@pytest.mark.parametrize("reason", ["worse_loss", "failed_solver", "exception"])
def test_rejected_or_interrupted_update_restores_entire_state(graph, reason):
    synthetic = SyntheticOPF([graph], steps=1)
    before = deepcopy(synthetic.state_dict())
    proposal = {key: value + 0.1 for key, value in before.items()}

    def audit(matrices):
        if reason == "worse_loss":
            pytest.fail("An increasing objective must be rejected before the solver audit")
        if reason == "exception":
            raise RuntimeError("audit interrupted")
        return pd.DataFrame([dict(success=False)]), [None]

    with pytest.raises(RuntimeError):
        backtrack(synthetic, before, proposal, lambda: 2. if reason == "worse_loss" else 0.,
                  1., audit=audit, fractions=(1., 0.5))
    assert all(torch.equal(value, synthetic.state_dict()[name]) for name, value in before.items())


def test_solver_audit_checks_all_edited_inputs(graph):
    synthetic = SyntheticOPF([graph, graph], steps=1)
    with torch.no_grad():
        for p in synthetic.edits.parameters():
            p.fill_(0.1)
    checks, solutions = audit_networks(synthetic.matrices())
    assert len(checks) == 2 and checks.success.all()
    assert all(solution.success for solution in solutions)
    assert (checks.balance < 1e-6).all()
