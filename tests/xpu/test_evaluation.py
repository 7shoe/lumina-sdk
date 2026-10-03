import math

import pytest
import torch
from torch_geometric.data import Batch

from lumina.evaluator.opf import evaluator as evaluator_module
from lumina.evaluator.opf import utils as evaluation_utils
from lumina.evaluator.opf.utils import Modeler
from lumina.trainer.opf.utils import select_device

from test_device_selection import load_example
from test_numerics import TARGETS, assert_close, make_graph


@pytest.mark.parametrize("target", TARGETS)
def test_evaluator_defaults_and_real_bound_metrics(monkeypatch, target):
    device = select_device(device=target)
    monkeypatch.setattr(evaluator_module, "select_device", lambda: device)
    monkeypatch.setattr(evaluation_utils, "select_device", lambda: device)
    batch = Batch.from_data_list([make_graph(1), make_graph(2)])
    default = evaluator_module.ACOPFConstraintEvaluator()
    assert default.device == device
    assert evaluator_module.ACOPFConstraintEvaluator(device=torch.device("cpu")).device.type == "cpu"
    # Default helper paths must produce nonempty results on the selected device.
    network = evaluation_utils.extract_network_parameters_from_batch(batch)
    assert {"pd", "qd", "gen_bus_indices", "line_edge_index"} <= network.keys()
    voltage, generation = evaluation_utils.extract_voltage_and_generation_limits_from_batch(batch)
    costs = evaluation_utils.extract_generation_costs_from_batch(batch)
    assert voltage and generation and costs is not None
    for tensor in [*network.values(), *voltage.values(), *generation.values(), costs]:
        assert tensor.device == device
        assert torch.isfinite(tensor).all()
    # Actual bound calculations, including nonzero violations, on CPU and target.
    results = []
    for destination in [torch.device("cpu"), device]:
        current = batch.clone().to(destination)
        limits_v = Modeler.derive_voltage_limits(current["bus"].x, destination)
        limits_g = Modeler.derive_generation_limits(current["generator"].x, destination)
        evaluator = evaluator_module.ACOPFConstraintEvaluator(
            voltage_limits=limits_v, generation_limits=limits_g, device=destination
        )
        predictions = {name: current[name].y + 2 for name in ["bus", "generator"]}
        violations = evaluator.evaluate_all_constraints(
            predictions, current, return_individual=False
        )
        assert violations
        assert any(torch.count_nonzero(value).item() > 0 for value in violations.values())
        summary = evaluator.get_violation_summary(violations)
        assert all(math.isfinite(value) for value in summary.values())
        results.append(violations)
    assert results[0].keys() == results[1].keys()
    for key in results[0]:
        assert_close(results[1][key], results[0][key], atol=2e-5, rtol=2e-4)


def test_evaluation_device_cli_choices(monkeypatch):
    module = load_example(monkeypatch, "evaluate_out_of_sample")
    for device in ["auto", "cpu", "cuda", "xpu"]:
        monkeypatch.setattr("sys.argv", ["evaluate_out_of_sample.py", "--checkpoint-file",
                                        "unused.pt", "--test-cases", "case14", "--device", device])
        assert module.parse_args().device == device


def test_standalone_evaluator_does_not_import_trainers():
    import subprocess
    import sys
    code = (
        "import sys; sys.modules['lumina.trainer'] = None; "
        "from lumina.evaluator.opf.evaluator import ACOPFConstraintEvaluator; "
        "import torch; ACOPFConstraintEvaluator(device=torch.device('cpu'))"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True)
