"""DD integration checks on a small built-in case, with no downloads/checkpoints."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
from pypower.api import case9
import pytest
import torch
from threadpoolctl import threadpool_limits

from distillation.DD import demo
from distillation.conversion.matrix_form import equal_records, mat2opf, opf2mat
from distillation.solver import ACOPF, PyPowerSolver, Solution, validate
from lumina.dataset.opf.opf_dataset import build_heterodata_from_grid

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def record():
    ppc = case9()
    base = ppc["baseMVA"]
    bus, gen, branch, cost = (ppc[key] for key in ("bus", "gen", "branch", "gencost"))
    bx = bus[:, [9, 1, 12, 11]]
    gx = gen[:, [6, 1, 9, 8, 2, 4, 3, 5]].astype(float)
    gx[:, [1, 2, 3, 4, 5, 6]] /= base
    gx = np.c_[gx, cost[:, 4] * base**2, cost[:, 5] * base, cost[:, 6]]
    loads = np.flatnonzero(np.any(bus[:, 2:4] != 0, axis=1))
    attrs = np.c_[np.radians(branch[:, 11:13]), branch[:, 4] / 2, branch[:, 4] / 2,
                  branch[:, 2:4], branch[:, 5:8] / base]
    def link(indices):
        return {"senders": list(range(len(indices))), "receivers": indices.tolist()}
    grid = dict(context=[base], nodes=dict(bus=bx.tolist(), generator=gx.tolist(),
                                         load=(bus[loads, 2:4] / base).tolist(), shunt=[]),
                edges={"ac_line": dict(senders=(branch[:, 0].astype(int)-1).tolist(),
                                       receivers=(branch[:, 1].astype(int)-1).tolist(), features=attrs.tolist()),
                       "transformer": dict(senders=[], receivers=[], features=[]),
                       "generator_link": link(gen[:, 0].astype(int)-1),
                       "load_link": link(loads), "shunt_link": link(np.array([], dtype=int))})
    solution = dict(nodes=dict(bus=np.c_[np.zeros(len(bus)), np.ones(len(bus))].tolist(),
                               generator=(gen[:, 1:3] / base).tolist()),
                    edges={"ac_line": {"features": np.zeros((len(branch), 4)).tolist()},
                           "transformer": {"features": []}})
    return dict(grid=grid, solution=solution, metadata={"objective": 0.})


@pytest.fixture(scope="module")
def graph(record):
    raw = demo.native_graph(record)
    model = ACOPF(raw)
    result = PyPowerSolver().solve(model)
    assert result.success
    return result.to_hetero(model)


def test_archive_sampling_matches_requested_split(tmp_path, monkeypatch, record):
    monkeypatch.setattr(demo, "split_indices", lambda *args, **kwargs: {"test": np.array([1, 3, 8])})
    case = "pglib_opf_case14_ieee"
    with tarfile.open(tmp_path / f"{case}_0.tar.gz", "w:gz") as archive:
        for index in range(9):
            obj = deepcopy(record)
            obj["metadata"]["objective"] = float(index)
            payload = json.dumps(obj).encode()
            info = tarfile.TarInfo(f"gridopt-dataset-tmp/dataset_release_1/{case}/group_0/example_{index}.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    graphs, indices = demo.load_batch(tmp_path, count=3, seed=7)
    assert set(indices) == {1, 3, 8}
    assert [int(g.objective) for g in graphs] == indices.tolist()
    again, repeated = demo.load_batch(tmp_path, count=3, seed=7)
    np.testing.assert_array_equal(indices, repeated)
    assert all(equal_records(a, b) for a, b in zip(graphs, again))


def test_adapter_matches_training_processor(record):
    native = demo.native_graph(record)
    actual = demo.benchmark_graph(native)
    expected = build_heterodata_from_grid(record["grid"], record["metadata"], record["solution"])
    assert actual.metadata() == expected.metadata()
    assert actual.baseMVA == expected.baseMVA
    for kind in actual.node_types:
        target = expected[kind].x.reshape(actual[kind].x.shape)
        torch.testing.assert_close(actual[kind].x.to(target.dtype), target, rtol=0, atol=0)
        assert "y" not in actual[kind]
    for edge in actual.edge_types:
        torch.testing.assert_close(actual[edge].edge_index, expected[edge].edge_index.long(), rtol=0, atol=0)
        if "edge_attr" in actual[edge]:
            target = expected[edge].edge_attr.reshape(actual[edge].edge_attr.shape)
            torch.testing.assert_close(actual[edge].edge_attr.to(target.dtype), target, rtol=0, atol=0)
        assert "edge_label" not in actual[edge]
    assert "objective" not in actual


@pytest.mark.parametrize("bounds,admittance,loads", [(False, False, False), (True, False, False),
                                                    (False, True, False), (False, False, True)])
def test_edit_modes_do_not_cross_parameter_families(graph, bounds, admittance, loads):
    before = graph.clone()
    original = opf2mat(graph)
    delta = demo.Perturbation(original, bounds=bounds, admittance=admittance, loads=loads)
    edited = delta(original, 0.1)
    exported = mat2opf(edited, include_solution=False)
    assert equal_records(graph, before)
    assert equal_records(exported, mat2opf(opf2mat(exported)))
    assert "objective" not in exported and "y" not in exported["bus"]
    assert all("edge_label" not in exported[edge] for edge in exported.edge_types)
    p, q = original.params, edited.params
    torch.testing.assert_close(p.fr, q.fr)
    torch.testing.assert_close(p.to, q.to)
    for enabled, field in ((bounds, "pmax"), (admittance, "Y"), (loads, "pd")):
        assert torch.equal(getattr(p, field), getattr(q, field)) is not enabled
    assert bool((q.pmin <= q.pmax).all() and (q.vmin > 0).all())
    assert torch.equal(q.c2, p.c2)


def test_gradients_through_roundtrip_batch_and_neural_model(graph):
    original = [opf2mat(graph), opf2mat(graph)]
    deltas = torch.nn.ModuleList([demo.Perturbation(m, seed=i) for i, m in enumerate(original)])
    changed = [delta(m) for delta, m in zip(deltas, original)]
    exported = [mat2opf(m, include_solution=False) for m in changed]
    batch = demo.benchmark_batch(exported)
    assert batch.num_graphs == 2
    config = {"model_type": "GCN", "models": {"GCN": dict(hidden_dim=8, num_layers=2, dropout=0.)}}
    model = demo.build_model(config, demo.benchmark_graph(graph), torch.device("cpu"))
    metrics, _, errors = demo.infer({"GCN": model}, exported, variant="edited")
    assert errors.empty and len(metrics) == 2
    assert all(item["load"].x.requires_grad for item in exported)
    prediction = model(batch)
    loss = prediction["generator"].square().mean() + prediction["bus"].square().mean()
    gradients = torch.autograd.grad(loss, tuple(deltas.parameters()), allow_unused=True, retain_graph=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    physics = sum(m.params.Y.abs().square().mean() for m in changed)
    admittance_grads = torch.autograd.grad(physics, [d.delta["admittance"] for d in deltas])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in admittance_grads)


@pytest.mark.parametrize("options", [dict(add_bus=True), dict(add_line=True), dict(remove_lines=[1]),
                                     dict(remove_buses=[1])])
def test_topology_edits_and_warm_start_identity(graph, options):
    model = ACOPF(graph)
    previous = Solution.from_label(graph)
    previous.checks = validate.feasibility(model, previous)
    changed, origins = demo.edit_topology(opf2mat(graph), **options)
    exported = mat2opf(changed, include_solution=False)
    start = demo.warm_start(previous, exported, origins)
    target = ACOPF(exported)
    assert torch.isfinite(start.x(target)).all()
    assert (start.x(target) >= target.lo).all() and (start.x(target) <= target.hi).all()
    assert len(start.pg) == len(origins[1])
    np.testing.assert_allclose(start.pg, previous.pg[origins[1]])
    if options.get("add_bus"):
        assert exported["bus"].num_nodes == graph["bus"].num_nodes + 1
        assert origins[0][-1] == -1
    assert "y" not in exported["bus"]


def test_disconnected_proposal_and_capacity_shortage(graph):
    with pytest.raises(ValueError, match="islands"):
        demo.edit_topology(opf2mat(graph), remove_lines=[0])
    matrix = opf2mat(graph)
    matrix.params.pd *= 1000
    screened = demo.screen_batch([matrix])
    assert not bool(screened.iloc[0].necessary_checks)
    assert "Active capacity" in screened.iloc[0].assessment


def test_solver_statuses_and_fresh_labels(graph):
    label = Solution.from_label(graph)
    table, solutions, best = demo.run_solvers(graph, reference=label, warm=True, scipy_max_buses=0)
    assert best is not None and best.success
    assert table.loc[table.solver == "SciPy", "status"].eq("skipped").all()
    pypower_warm = table[(table.solver == "PYPOWER") & (table.start == "warm")].iloc[0]
    assert pypower_warm.status == "skipped" and "unsupported" in pypower_warm.note
    assert abs(table[(table.solver == "PYPOWER") & (table.start == "cold")].iloc[0].cost_gap_pct) < 1e-6
    fresh = best.to_hetero(ACOPF(graph))
    assert validate.feasibility(ACOPF(fresh), Solution.from_label(fresh))["feasible"]


def test_inference_reference_coverage_and_training_copy(graph):
    config = {"model_type": "GCN", "models": {"GCN": dict(hidden_dim=8, num_layers=2, dropout=0.)}}
    model = demo.build_model(config, demo.benchmark_graph(graph), torch.device("cpu"))
    models = {"GCN": model}
    before = deepcopy(model.state_dict())
    records, predictions, errors = demo.infer(models, [graph, graph], variant="edited",
                                             references={0: Solution.from_label(graph)})
    assert errors.empty and len(records) == 2 and "GCN" in predictions
    assert records.has_reference.tolist() == [True, False]
    assert np.isfinite(records.iloc[0].dispatch_l2) and np.isnan(records.iloc[1].dispatch_l2)
    assert np.isfinite(records.cost).all() and np.isfinite(records.balance_pu).all()
    result = demo.training_step(models, graph)
    assert result["gradient_norm"] > 0 and result["graphs"] == 1
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not model.training


def test_missing_checkpoints_are_not_random_models(tmp_path):
    models, status = demo.load_models(tmp_path)
    assert not models and len(status) == 8 and status.status.eq("missing").all()
    assert "skipped" in demo.training_step(models, None)["status"]


def test_unavailable_sparse_solver_keeps_reference_working(graph, monkeypatch):
    monkeypatch.setitem(sys.modules, "distillation.solver.sparse_solver", None)
    table, _, best = demo.run_solvers(graph, scipy_max_buses=0)
    assert best is not None and best.success
    sparse = table[table.solver == "Sparse CPU"].iloc[0]
    assert sparse.status == "skipped" and "unavailable" in sparse.note


def test_notebook_structure():
    notebook = json.loads((ROOT / "distillation/DD/demo_dd.ipynb").read_text())
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) <= 12
    for index, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] == "code"
        source = "".join(cell["source"])
        compile(source, f"cell-{index}", "exec")
