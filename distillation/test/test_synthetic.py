"""Shared generation contracts, real solver checks, failure paths and DD reuse."""
import numpy as np
import pandas as pd
import pytest
import torch

from distillation.conversion.matrix_form import equal_records, mat2opf, opf2mat, validate_network
from distillation.solver import ACOPF, Solution, validate
from distillation.synthetic import (
    GenerationError, Perturbation, generate, perturb_operating, random_network,
    sample_sizes, teaching_targets, topology_signature,
)
from distillation.synthetic import pipeline
from distillation.test import test_dd_demo as reference

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def assert_feasible(graph):
    validate_network(opf2mat(graph).params)
    assert validate.feasibility(ACOPF(graph), Solution.from_label(graph))["feasible"]


@pytest.mark.parametrize("case,limits", [(30, (10, 90)), (57, (19, 171)), (118, (40, 354))])
def test_every_batch_covers_original_bus_limits(case, limits):
    sizes = sample_sizes(15, case, steps=3)
    for batch in np.array(sizes).reshape(3, 5):
        assert (min(batch), max(batch)) == limits
        assert len(set(batch)) == 5
    assert sizes == sample_sizes(15, case, steps=3)
    assert sample_sizes(1, case) == [case]


@pytest.mark.parametrize("size", [3, 9, 27])
@pytest.mark.parametrize("placement", ["uniform", "electrical"])
def test_random_inputs_are_fresh_reproducible_and_unlabeled(graph, size, placement):
    original = graph.clone()
    a, report = random_network(graph, size, seed=6, placement=placement)
    b, _ = random_network(graph, size, seed=6, placement=placement)
    out = mat2opf(a, include_solution=False)
    assert a.params.N == size and report["buses"] == size
    assert equal_records(out, mat2opf(b, include_solution=False))
    assert equal_records(graph, original)
    assert "objective" not in out and all("y" not in out[kind] for kind in out.node_types)
    assert all("edge_label" not in out[edge] for edge in out.edge_types)
    assert topology_signature(out) != topology_signature(graph)
    validate_network(a.params)
    assert a.params.pmin.min() >= 0 and (a.params.qmin <= a.params.qmax).all()
    pairs = {tuple(sorted(pair)) for pair in zip(a.params.fr.tolist(), a.params.to.tolist())}
    assert len(pairs) == a.params.E
    if size == 9:
        other, _ = random_network(graph, size, seed=7, placement=placement)
        assert topology_signature(out) != topology_signature(mat2opf(other, include_solution=False))


@pytest.mark.parametrize("size", [2, 28, 9.5])
def test_random_size_bounds_cannot_drift(graph, size):
    with pytest.raises(ValueError, match="integer"):
        random_network(graph, size)


@pytest.fixture(scope="module")
def random_dataset(graph):
    return generate([graph], mode="random", sizes=[6, 9, 12], seed=16, attempts=12,
                    reference_ids=["train:case9:0"])


def test_random_generation_returns_exact_sizes_and_independent_labels(random_dataset):
    result = random_dataset
    assert [g["bus"].num_nodes for g in result.inputs] == [6, 9, 12]
    assert result.report.success.all() and result.metadata["topologies_per_batch"] == [3]
    assert result.metadata["reference_ids"] == ["train:case9:0"]
    for inputs, solved, target in zip(result.inputs, result.physical_graphs, result.teaching_targets):
        assert_feasible(solved)
        assert "y" not in inputs["bus"] and "objective" not in inputs
        torch.testing.assert_close(target["bus"], solved["bus"].y)
        assert not target["bus"].requires_grad
        before = target["bus"].clone()
        target["bus"].fill_(123)
        assert not (solved["bus"].y == 123).all()
        target["bus"].copy_(before)


@pytest.mark.parametrize("kind", ["GAT", "GIN", "HeteroGNN"])
def test_random_mixed_size_grids_work_with_dd_and_all_three_models(graph, random_dataset, kind):
    from distillation.DD.campaign import model_for, model_options
    from distillation.DD.demo import benchmark_batch
    from distillation.DD.mvp import SyntheticOPF, gradient_summary, targets, unroll

    model = model_for(graph, dict(model=kind, model_options=model_options(kind, width=8)), 31, "cpu")
    synthetic = SyntheticOPF(random_dataset.physical_graphs, steps=1, target_strength=0.)
    _, loss, _ = unroll(model, synthetic, benchmark_batch([graph]), targets([graph]))
    loss.backward()
    summary = gradient_summary(synthetic)
    assert torch.isfinite(loss) and summary.loc["edits/admittance", "gradient_norm"] > 0


def test_conservative_generation_reuses_verified_edits_and_operational_backtracking(graph):
    result = generate([graph], count=3, seed=11, strength=0.1, max_edits=3)
    assert len(result.inputs) == 3 and result.report.changed.any()
    for solved in result.physical_graphs:
        assert_feasible(solved)
        assert 3 <= solved["bus"].num_nodes <= 27
    assert result.report.operational_fraction.between(0, 1).all()


def test_operational_rejection_backtracks_to_verified_unchanged_grid(graph, monkeypatch):
    audit = pipeline.audit_networks
    calls = []

    def reject_changed(matrices, **kwargs):
        calls.append(matrices)
        if len(calls) == 1:
            return pd.DataFrame([dict(success=False, error="injected failure")]), [None]
        return audit(matrices, **kwargs)

    monkeypatch.setattr(pipeline, "audit_networks", reject_changed)
    solved, row = perturb_operating(graph, fractions=(1., 0.))
    assert row["operational_fraction"] == 0 and len(row["operating_attempts"]) == 2
    torch.testing.assert_close(opf2mat(solved).params.pd, opf2mat(graph).params.pd)
    assert_feasible(solved)


def test_failed_random_solves_never_fall_back_to_reference_or_shrink_requested_size(graph, monkeypatch):
    calls = []

    def reject(matrices, **kwargs):
        calls.extend(matrices)
        return pd.DataFrame([dict(success=False, error="injected infeasibility")]), [None]

    monkeypatch.setattr(pipeline, "audit_networks", reject)
    with pytest.raises(GenerationError, match="requested buses=12") as captured:
        generate([graph], mode="random", sizes=[12], attempts=2, demand_fractions=(1., 0.5))
    assert len(calls) == 4 and all(matrix.params.N == 12 for matrix in calls)
    assert not captured.value.partial.inputs and len(captured.value.failures) == 4
    torch.testing.assert_close(calls[1].params.pd, calls[0].params.pd * 0.5)
    torch.testing.assert_close(calls[1].params.qd, calls[0].params.qd * 0.5)
    torch.testing.assert_close(calls[1].params.rate_a, calls[0].params.rate_a)


@pytest.mark.parametrize("source", ["teacher", "learned"])
def test_future_label_sources_fail_explicitly_before_solving(graph, source):
    with pytest.raises(NotImplementedError, match="placeholder"):
        generate([graph], label_source=source)
    with pytest.raises(NotImplementedError):
        teaching_targets([graph], source=source)


def test_shared_continuous_parameters_retain_second_derivatives(graph):
    anchor = opf2mat(graph)
    edit = Perturbation(anchor)
    matrix = edit(anchor, 0.2)
    loss = matrix.params.y.abs().square().sum() + matrix.params.pd.square().sum()
    gradients = torch.autograd.grad(loss, (edit.delta["admittance"], edit.delta["load"]), create_graph=True)
    second = torch.autograd.grad(sum(g.square().sum() for g in gradients),
                                 (edit.delta["admittance"], edit.delta["load"]))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in second)


def test_invalid_inputs_fail_before_generation(graph):
    for options in (dict(sizes=[2], mode="random"), dict(count=3, steps=2),
                    dict(sizes=[9]), dict(strength=1.), dict(demand_fractions=(0.8, 1.))):
        with pytest.raises(ValueError):
            generate([graph], **options)


@pytest.mark.parametrize('case,limits', [(30, [23, 38]), (57, [43, 71]), (118, [89, 148])])
def test_campaign_size_range_and_matched_bus_budget(case, limits):
    from distillation.synthetic.experiments import balanced_sizes, case_bus_range
    assert case_bus_range(case) == limits
    for batch in np.array(balanced_sizes(case, 3, 16)).reshape(3, 16):
        assert [min(batch), max(batch)] == limits
        assert sum(batch) == 16 * case


def test_explicit_bus_range_is_enforced_and_recorded(graph):
    with pytest.raises(ValueError, match='bus counts'):
        generate([graph], mode='random', sizes=[12], bus_range=(7, 11))
    result = generate([graph], count=2, min_topologies=1, bus_range=(9, 9))
    assert result.metadata['bus_limits'] == [9, 9]
    assert all(g['bus'].num_nodes == 9 for g in result.inputs)
