"""CPU solver checks: built-in networks and small, read-only OPFData samples.

Run: python -m pytest distillation/test/test_acopf_solver.py -q
OPF_RAW_ROOT overrides the archive directory; missing archives skip only data tests.
"""
from copy import deepcopy
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest
import torch
from pypower.api import case9, case14, case30
from torch_geometric.data import HeteroData

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "distillation/solver/problem.py").is_file())
sys.path.insert(0, str(ROOT))

from distillation.conversion.matrix_form import (
    E_LINE, E_XFMR, NetworkEditor, equal_records, mat2opf, opf2mat, residuals,
)
from distillation.solver import (
    ACOPF, NewtonPF, PyPowerSolver, ReducedGradient, ScipySolver, Solution, solve, validate,
)

RAW_ROOT = Path(os.environ.get("OPF_RAW_ROOT", "/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/raw/OPFData/dataset_release_1"))
CASES = ("pglib_opf_case14_ieee", "pglib_opf_case30_ieee", "pglib_opf_case57_ieee",
         "pglib_opf_case118_ieee", "pglib_opf_case500_goc")


@lru_cache(maxsize=None)
def raw_samples(case):
    path = RAW_ROOT / f"{case}_0.tar.gz"
    if not path.exists():
        pytest.skip(f"Missing local archive: {path}")
    samples = []
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            obj = json.load(archive.extractfile(member))
            grid, solution = obj["grid"], obj["solution"]
            data = HeteroData()
            data.x = torch.tensor(grid["context"], dtype=torch.float64).reshape(-1)
            data.objective = torch.tensor(obj["metadata"]["objective"], dtype=torch.float64)
            for node, width in (("bus", 4), ("generator", 11), ("load", 2), ("shunt", 2)):
                data[node].x = torch.tensor(grid["nodes"][node], dtype=torch.float64).reshape(-1, width)
            for node in ("bus", "generator"):
                data[node].y = torch.tensor(solution["nodes"][node], dtype=torch.float64)
            for kind, edge in grid["edges"].items():
                source = "bus" if kind in ("ac_line", "transformer") else kind.removesuffix("_link")
                store = data[source, kind, "bus"]
                store.edge_index = torch.tensor([edge["senders"], edge["receivers"]], dtype=torch.long).reshape(2, -1)
                if kind in ("ac_line", "transformer"):
                    width = 9 if kind == "ac_line" else 11
                    store.edge_attr = torch.tensor(edge["features"], dtype=torch.float64).reshape(-1, width)
                    store.edge_label = torch.tensor(solution["edges"][kind]["features"], dtype=torch.float64).reshape(-1, 4)
            samples.append(data)
            if len(samples) == 2:
                break
    assert len(samples) == 2
    return samples


@pytest.mark.parametrize("factory", (case9, case14, case30))
def test_pypower_builtin_networks(factory):
    ppc = factory()
    original = deepcopy(ppc)
    dtype = torch.get_default_dtype()
    model = ACOPF(ppc)
    result = PyPowerSolver().solve(model)
    assert result.converged and result.feasible and result.success, result.checks
    assert torch.get_default_dtype() == dtype
    for key in ("bus", "gen", "branch", "gencost"):
        np.testing.assert_array_equal(ppc[key], original[key])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("index", (0, 1))
def test_pypower_opfdata(case, index):
    data = raw_samples(case)[index].clone()
    original = data.clone()
    model = ACOPF(data)
    result = PyPowerSolver().solve(model)
    assert result.success, result.checks
    assert validate.compare(result, Solution.from_label(data))["rel_obj"] < 1e-5
    solved = result.to_hetero(model)
    checks = residuals(opf2mat(solved), zero_rating="unmonitored")
    for key in ("balance", "flow_f", "flow_t", "thermal", "angle", "ref_angle",
                "v_bounds", "pg_bounds", "qg_bounds"):
        assert checks[key] < 1e-6, (case, key, checks[key])
    assert checks["cost_rel"] < 1e-10
    assert equal_records(data, original)


def test_unlabeled_edited_network():
    data = raw_samples(CASES[0])[0].clone()
    editor = NetworkEditor(opf2mat(data))
    new_bus = editor.add_bus(base_kv=1, vmin=0.95, vmax=1.05, load={"pd": 0.01, "qd": 0.002})
    editor.add_branch(0, new_bus, br_r=0.01, br_x=0.1, rate_a=1, angmin=-0.5, angmax=0.5)
    edited = editor.build()
    assert edited.sol is None
    result = solve(edited)
    assert result.success, result.checks
    model = ACOPF(edited)
    solved = result.to_hetero(model)
    assert solved["bus"].num_nodes == data["bus"].num_nodes + 1
    unlabeled = mat2opf(opf2mat(data), include_solution=False)
    for edge in ("generator", "load", "shunt"):
        store = unlabeled[edge, f"{edge}_link", "bus"]
        store.edge_index = store.edge_index.flip(1)
    assert solve(unlabeled).success


def test_external_generator_order():
    ppc = case14()
    reference = solve(ppc)
    ppc["gen"], ppc["gencost"] = ppc["gen"][::-1].copy(), ppc["gencost"][::-1].copy()
    mapping = {bus: 100 + 10 * i for i, bus in enumerate(ppc["bus"][:, 0])}
    for array, columns in ((ppc["bus"], (0,)), (ppc["gen"], (0,)), (ppc["branch"], (0, 1))):
        for col in columns:
            array[:, col] = [mapping[value] for value in array[:, col]]
    result = solve(ppc)
    assert result.success, result.checks
    np.testing.assert_allclose(result.pg[::-1], reference.pg, atol=1e-5)
    assert abs(result.objective / reference.objective - 1) < 1e-7


@pytest.mark.parametrize("factory", (case9, case14))
def test_scipy_from_flat_start(factory):
    model = ACOPF(factory())
    reference = PyPowerSolver().solve(model)
    result = ScipySolver(maxiter=300).solve(model)
    assert result.success, (result.info, result.checks)
    assert validate.compare(result, reference)["rel_obj"] < 1e-5


def test_scipy_warm_start_and_iteration_limit():
    model = ACOPF(case9())
    reference = PyPowerSolver().solve(model)
    result = ScipySolver(maxiter=150).solve(model, x0=reference)
    assert result.success, (result.info, result.checks)
    failed = ScipySolver(maxiter=1).solve(model)
    assert not failed.converged and not failed.success
    with pytest.raises(ValueError, match="x0"):
        PyPowerSolver().solve(model, x0=reference)


@pytest.mark.parametrize("factory", (case9, case14))
def test_newton_pf_and_gradients(factory):
    model = ACOPF(factory())
    reference = PyPowerSolver().solve(model)
    controls = [torch.tensor(value, dtype=torch.float64, requires_grad=True)
                for value in (reference.pg, reference.qg)]
    pf = NewtonPF(model)
    state = pf(controls)
    x = pf.assemble(state, controls)
    assert model.balance(x).abs().max() < 1e-9
    torch.testing.assert_close(x, reference.x(model), rtol=1e-5, atol=1e-5)
    model.unpack(x)[1].sum().backward()
    assert all(u.grad is not None and torch.isfinite(u.grad).all() for u in controls)
    assert controls[1].grad.abs().max() > 0


def test_reduced_gradient_reports_limits_and_slack_bounds():
    model = ACOPF(case9())
    reference = PyPowerSolver().solve(model)
    optimizer = ReducedGradient(model, iters=5, lr=1e-4)
    result = optimizer.solve(x0=reference)
    assert np.isfinite(result.x().numpy()).all()
    assert not result.converged and not result.success
    assert result.info["iterations"] == 5
    assert not np.allclose(result.x().numpy(), reference.x().numpy(), rtol=0, atol=1e-8)
    x = reference.x(model)
    x[2*model.nb + optimizer.pf.slack] = model.hi[2*model.nb + optimizer.pf.slack] + 1
    assert optimizer.penalty(x) >= 1


def test_angle_bounds_and_unmonitored_lines():
    ppc = case9()
    ppc["branch"][:, 5] = 0
    model = ACOPF(ppc)
    assert model.flows(model.flat_start()).numel() == 0
    result = PyPowerSolver().solve(model)
    assert result.success
    assert result.info["redundant_thermal_limit"] is not None
    assert np.all(model.ppc["branch"][:, 5] == 0)
    model.angle_lo[:] = -0.001
    model.angle_hi[:] = 0.001
    assert model.inequalities(result.x(model)).max() > 0
    assert not validate.feasibility(model, result)["feasible"]
    result = Solution.from_x(result.x(model), model, True, "candidate")
    assert result.converged and not result.success


def test_failures_are_not_usable_labels():
    data = raw_samples(CASES[0])[0].clone()
    model = ACOPF(data)
    candidate = Solution.from_x(model.flat_start(), model, False, "test")
    assert not candidate.success
    with pytest.raises(ValueError, match="converged, feasible"):
        candidate.to_hetero(model)
    ppc = case9()
    ppc["gen"][:, 9], ppc["gen"][:, 8] = 0, 0.1
    assert not solve(ppc, PDIPM_MAX_IT=30).success
    with pytest.raises(ValueError, match="equal charging"):
        data[E_LINE].edge_attr[0, 2] += 0.01
        ACOPF(data)
