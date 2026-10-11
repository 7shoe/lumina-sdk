"""Physical topology constraints, diverse batches, and mixed-size meta-gradients."""
from copy import deepcopy

import pandas as pd
import pytest
import torch

from distillation.DD.demo import benchmark_batch
from distillation.DD.mvp import SyntheticOPF, audit_networks, gradient_summary, make_model, mse, targets, unroll
from distillation.DD.run_topology import outer_update
from distillation.DD import topology
from distillation.conversion.matrix_form import equal_records, mat2opf, opf2mat, validate_network
from distillation.solver import ACOPF, Solution, validate
from distillation.test import test_dd_demo as reference

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


@pytest.mark.parametrize("original,expected", [(9, (3, 27)), (14, (5, 42)), (30, (10, 90)),
                                               (57, (19, 171)), (118, (40, 354)), (500, (167, 1500))])
def test_hard_limits_round_inward(original, expected):
    assert topology.bus_limits(original) == expected


@pytest.mark.parametrize("candidate,bound", [(4, 27), (5, 3)])
def test_repeated_bus_edits_cannot_move_original_case_limits(graph, candidate, bound):
    current = graph
    for _ in range(abs(bound - graph["bus"].num_nodes)):
        matrix, _ = topology.propose(current, candidate, min_buses=3, max_buses=27)
        current = mat2opf(matrix, include_solution=False)
    assert current["bus"].num_nodes == bound
    with pytest.raises(ValueError, match="outside"):
        topology.propose(current, candidate, min_buses=3, max_buses=27)


@pytest.fixture(scope="module")
def mixed_graphs(graph):
    matrices = [topology.propose(graph, candidate, min_buses=3, max_buses=27)[0] for candidate in (4, 5)]
    checks, solutions = audit_networks(matrices)
    assert checks.success.all()
    return [solution.to_hetero(ACOPF(matrix)) for matrix, solution in zip(matrices, solutions)]


def test_split_and_merge_preserve_equipment_and_have_fresh_feasible_labels(graph, mixed_graphs):
    original = graph.clone()
    p = opf2mat(graph).params
    assert [g["bus"].num_nodes for g in mixed_graphs] == [10, 8]
    for changed in mixed_graphs:
        q = opf2mat(changed).params
        validate_network(q)
        for field in ("pd", "qd", "pmin", "pmax", "qmin", "qmax", "c2", "c1", "c0"):
            torch.testing.assert_close(getattr(p, field), getattr(q, field), rtol=0, atol=0)
        assert validate.feasibility(ACOPF(changed), Solution.from_label(changed))["feasible"]
    assert equal_records(graph, original)


def test_splitting_charged_lines_preserves_total_charging_and_pypower_support(graph):
    charged = opf2mat(graph)
    charged.params.b_fr.fill_(0.02)
    charged.params.b_to.fill_(0.02)
    source = mat2opf(charged, include_solution=False)
    split, _ = topology.propose(source, 4, min_buses=3, max_buses=27)
    p, q = charged.params, split.params
    torch.testing.assert_close((q.b_fr + q.b_to).sum(), (p.b_fr + p.b_to).sum())
    assert torch.equal(q.b_fr, q.b_to)
    checks, _ = audit_networks([split])
    assert checks.success.all(), checks.error.tolist()


def test_topology_backtracking_recovers_last_feasible_prefix(graph):
    first, _ = topology.propose(graph, 4, min_buses=3, max_buses=27)
    second, _ = topology.propose(mat2opf(first, include_solution=False), 4, min_buses=3, max_buses=27)
    second.params.pd *= 1000.
    solved, edits, failures = topology.solve_prefixes([first, second], graph)
    assert edits == 1 and solved["bus"].num_nodes == 10
    assert len(failures) == 1 and failures[0]["edits"] == 2
    assert validate.feasibility(ACOPF(solved), Solution.from_label(solved))["feasible"]
    torch.testing.assert_close(opf2mat(solved).params.pd, first.params.pd)


def test_topology_backtracking_can_recover_after_edits_cancel(graph):
    first, _ = topology.propose(graph, 4, min_buses=3, max_buses=27)
    solved, edits, failures = topology.solve_prefixes([first, opf2mat(graph)], graph)
    assert edits == 1 and solved["bus"].num_nodes == 10
    assert "original topology" in failures[0]["error"]


def test_graph_weight_is_independent_of_its_bus_count(mixed_graphs):
    batch = benchmark_batch(mixed_graphs)
    expected = targets(mixed_graphs)
    prediction = deepcopy(expected)
    prediction["bus"][:10] += 1.
    prediction["bus"][10:] += 3.
    assert float(mse(prediction, expected, batch)) == pytest.approx(2.5)
    assert float(mse(prediction, expected)) != pytest.approx(2.5)


def test_second_order_unroll_accepts_mixed_bus_counts(graph, mixed_graphs):
    graphs = [mixed_graphs[0], graph, mixed_graphs[1], mixed_graphs[0]]
    synthetic = SyntheticOPF(graphs, steps=2)
    model = make_model(graph, hidden_channels=8)
    original_weights = deepcopy(model.state_dict())
    _, loss, trace = unroll(model, synthetic, benchmark_batch([graph]), targets([graph]))
    loss.backward()
    gradients = gradient_summary(synthetic)
    assert torch.isfinite(loss) and len(trace) == 2
    assert gradients.loc["rates", "gradient_norm"] > 0
    assert gradients.loc["edits/admittance", "gradient_norm"] > 0
    assert gradients.loc["target_offsets/bus", "gradient_norm"] > 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in original_weights.items())
    assert [g["bus"].num_nodes for g in graphs] == [10, 9, 8, 10]


def test_whole_candidate_is_diverse_in_every_batch_and_solver_feasible(graph):
    model = make_model(graph, hidden_channels=8)
    result = topology.evaluate_candidate(1, [graph] * 6, model, benchmark_batch([graph]), targets([graph]),
                                         min_buses=3, max_buses=27, synthetic_options=dict(steps=2),
                                         min_topologies_per_batch=3)
    assert result.row["eligible"] and result.row["solver_success"] and result.row["status"] == "scored"
    assert result.row["topologies_per_batch"] == [3, 3]
    assert result.row["changed_graphs"] == 6
    assert topology.batch_report(result.graphs, 2).topologies.tolist() == [3, 3]
    for changed in result.graphs:
        assert validate.feasibility(ACOPF(changed), Solution.from_label(changed))["feasible"]


def test_infeasible_proposals_retain_verified_seeds_and_cannot_bypass_diversity(graph, monkeypatch):
    calls = []

    def reject(matrices, **kwargs):
        calls.extend(matrices)
        return pd.DataFrame([dict(success=False, error="injected infeasibility")]), [None]

    monkeypatch.setattr(topology, "audit_networks", reject)
    result = topology.evaluate_candidate(1, [graph] * 4, None, None, None, min_buses=3, max_buses=27,
                                         synthetic_options=dict(steps=2), attempts_per_graph=3, max_edits=1,
                                         min_topologies_per_batch=2)
    assert calls
    assert all(changed is graph for changed in result.graphs)
    assert not result.row["eligible"] and result.row["status"] == "rejected"
    assert result.row["rejected_attempts"] == 12 and result.row["changed_graphs"] == 0
    with pytest.raises(RuntimeError, match="No topology candidate"):
        topology.choose_candidate([result])


def test_objective_cannot_select_failed_solver_or_collapsed_diversity():
    good = dict(candidate=2, status="scored", eligible=True, solver_success=True, real_mse=4.)
    winner = topology.Candidate(good)
    candidates = [topology.Candidate(good | dict(candidate=0, eligible=False, real_mse=1.)),
                  topology.Candidate(good | dict(candidate=1, solver_success=False, real_mse=2.)), winner]
    assert topology.choose_candidate(candidates) is winner


def test_topology_only_update_preserves_offsets_rates_and_exports_solved_inputs(graph, mixed_graphs, tmp_path):
    model = make_model(graph, hidden_channels=8)
    options = dict(steps=1)
    initial = SyntheticOPF(mixed_graphs, **options).state_dict()
    result = outer_update(model, mixed_graphs, benchmark_batch([graph]), targets([graph]), options,
                          solver="pypower", output=tmp_path, metadata={}, update_kind="topology")
    artifact = torch.load(tmp_path / "distilled.pt", weights_only=False, map_location="cpu")
    assert result["continuous_updates"] == 0 and result["solved_graphs"] == 2
    assert result["real_mse_before"] == result["real_mse_after"]
    assert all(torch.equal(value, artifact["synthetic_state"][name]) for name, value in initial.items())
    assert [g["bus"].num_nodes for g in artifact["physical_solved_graphs"]] == [10, 8]
    assert artifact["schedule"][0]["inputs"].num_graphs == 2


def test_seed_feasibility_is_required_even_when_topology_is_unchanged(graph):
    impossible = graph.clone()
    impossible["load"].x *= 1000.
    with pytest.raises(RuntimeError, match="Seed graphs failed"):
        topology.refresh_seeds([impossible])
    assert topology.refresh_seeds([]) == []
