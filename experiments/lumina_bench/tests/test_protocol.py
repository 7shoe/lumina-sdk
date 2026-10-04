"""Independent numerical and protocol checks, local to this experiment."""
import json
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, HeteroData

from experiments.lumina_bench.data import split_data
from experiments.lumina_bench.metrics import MetricAccumulator, physical_residuals, sample_metrics
from experiments.lumina_bench.models import apply_bounds
from experiments.lumina_bench.runner import build_model


def graph():
    data = HeteroData()
    data["bus"].x = torch.tensor([[230, .9, 1.1, 0, 0, 1, 0],
                                  [230, .9, 1.1, 1, 0, 0, 0]], dtype=torch.float64)
    data["bus"].y = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64)
    data["generator"].x = torch.tensor([[100, .5, 0, 2, .1, -1, 1, 1, 2, 3, 4]], dtype=torch.float64)
    data["generator"].y = torch.tensor([[.5, .1]], dtype=torch.float64)
    data["load"].x = torch.tensor([[.5, .1]], dtype=torch.float64)
    data["shunt"].x = torch.tensor([[0., 0.]], dtype=torch.float64)
    for name in ("generator", "load", "shunt"):
        data[name, f"{name}_link", "bus"].edge_index = torch.tensor([[0], [0]])
        data["bus", f"{name}_link", name].edge_index = torch.tensor([[0], [0]])
    data["bus", "ac_line", "bus"].edge_index = torch.tensor([[0], [1]])
    data["bus", "ac_line", "bus"].edge_attr = torch.tensor([
        [-.5, .5, 0, 0, .02, .1, 1, 1, 1]], dtype=torch.float64)
    data.baseMVA = torch.tensor([100.])
    data.objective = torch.tensor([6.])
    return data


def predictions(batch):
    return {name: batch[name].y.clone() for name in ("bus", "generator")}


def test_ground_truth_and_controlled_power_violation():
    batch = Batch.from_data_list([graph()])
    pred = predictions(batch)
    metrics = sample_metrics(pred, batch)
    assert metrics["mse"].item() == 0
    assert metrics["violation"].item() == 0
    assert metrics["label_cost_objective_abs_error"].item() == 0
    pred["generator"][0, 0] += 1
    metrics = sample_metrics(pred, batch)
    assert metrics["balance_l2"].item() == pytest.approx(1.)
    assert metrics["violation"].item() == pytest.approx(1 / np.sqrt(2))
    assert metrics["cost_difference"].item() == pytest.approx(7.)


def test_metrics_are_sample_weighted_and_batch_invariant():
    graphs = [graph() for _ in range(5)]
    # Unequal sample errors expose mean-of-batch-means mistakes.
    for i, item in enumerate(graphs):
        item["load"].x[0, 0] += i / 10
    def collect(batch_size):
        acc = MetricAccumulator()
        for start in range(0, len(graphs), batch_size):
            batch = Batch.from_data_list(graphs[start:start + batch_size])
            acc.update(sample_metrics(predictions(batch), batch))
        return acc.result()
    assert collect(1) == pytest.approx(collect(3), abs=1e-12)
    assert collect(3)["balance_l2"] == pytest.approx(.2)


@pytest.mark.parametrize("transformer", [False, True])
def test_branch_physics_against_pypower(transformer):
    from pypower.makeYbus import makeYbus
    data = graph()
    data["shunt"].x[:] = torch.tensor([[.03, .02]])
    features = data["bus", "ac_line", "bus"].edge_attr
    features[0, 2:4] = .02
    features[0, 6] = .3
    ratio, shift = (1.03, .07) if transformer else (1., 0.)
    if transformer:
        del data["bus", "ac_line", "bus"]
        data["bus", "transformer", "bus"].edge_index = torch.tensor([[0], [1]])
        data["bus", "transformer", "bus"].edge_attr = torch.tensor([
            [-.5, .5, .02, .1, .3, .3, .3, ratio, shift, .02, .02]], dtype=torch.float64)
    batch = Batch.from_data_list([data])
    pred = predictions(batch)
    pred["bus"] = torch.tensor([[.04, 1.02], [-.03, .96]], dtype=torch.float64)
    actual_balance, actual_thermal, _ = physical_residuals(pred, batch)
    bus = np.zeros((2, 13)); bus[:, 0] = [0, 1]
    bus[0, 4:6] = [2., 3.]  # MW/MVAr shunts at baseMVA=100
    branch = np.array([[0, 1, .02, .1, .04, 30, 30, 30, ratio, np.degrees(shift), 1, -30, 30]])
    ybus, yf, yt = makeYbus(100., bus, branch)
    voltage = np.array([1.02 * np.exp(.04j), .96 * np.exp(-.03j)])
    expected_balance = -voltage * (ybus @ voltage).conj()  # generation == load
    sf, st = voltage[0] * (yf @ voltage).conj(), voltage[1] * (yt @ voltage).conj()
    expected_thermal = np.maximum(np.abs(np.r_[sf, st]) ** 2 - .3 ** 2, 0)
    np.testing.assert_allclose(actual_balance.numpy(), expected_balance, atol=2e-9)
    np.testing.assert_allclose(actual_thermal.numpy(), expected_thermal, atol=1e-12)


def test_zero_rating_is_unconstrained():
    data = graph()
    data["bus", "ac_line", "bus"].edge_attr[:, 6] = 0
    batch = Batch.from_data_list([data])
    pred = predictions(batch)
    pred["bus"][1, 0] = 1.
    _, thermal, _ = physical_residuals(pred, batch)
    assert torch.count_nonzero(thermal).item() == 0


def test_shared_bounds_and_zero_width_generator_interval():
    data = graph()
    data["generator"].x[0, 2:4] = 0
    batch = Batch.from_data_list([data])
    pred = {"bus": torch.tensor([[2., -100.], [-1., 100.]]),
            "generator": torch.tensor([[100., -100.]])}
    bounded = apply_bounds(pred, batch)
    torch.testing.assert_close(bounded["bus"][:, 0], pred["bus"][:, 0])
    torch.testing.assert_close(bounded["bus"][:, 1], torch.tensor([.9, 1.1]))
    assert bounded["generator"][0, 0].item() == 0


def test_splits_do_not_depend_on_training_seed_and_reject_changed_data(tmp_path):
    config = {"split_seed": 42, "seed": 1}
    dataset = list(range(100))
    path = tmp_path / "split.json"
    first, _ = split_data(dataset, {"case": "case30"}, config, path)
    config["seed"] = 999
    second, _ = split_data(dataset, {"case": "case30"}, config, path)
    assert first["test"].indices == second["test"].indices
    assert not set(first["train"].indices) & set(first["test"].indices)
    with pytest.raises(ValueError, match="does not match"):
        split_data(dataset, {"case": "case57"}, config, path)
    manifest = json.loads(path.read_text())
    manifest["indices"]["test"][0] = manifest["indices"]["train"][0]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly once"):
        split_data(dataset, {"case": "case30"}, config, path)


def test_real_case30_ground_truth():
    root = os.environ.get("LUMINA_BENCH_TEST_DATA")
    if not root:
        pytest.skip("set LUMINA_BENCH_TEST_DATA to the prepared case30 cache")
    from lumina.dataset.opf.opf_dataset import OPFDataset
    dataset = OPFDataset(root=root, case_name="pglib_opf_case30_ieee", group_id=0)
    batch = Batch.from_data_list([dataset[i] for i in (0, 19, 137, 2000)])
    values = sample_metrics(predictions(batch), batch)
    assert values["balance_l2"].max() < 1e-3
    assert values["thermal_l2"].max() < 1e-3
    assert values["label_cost_objective_abs_error"].max() < .05


def test_nonfinite_predictions_are_not_silently_dropped():
    batch = Batch.from_data_list([graph()])
    pred = predictions(batch)
    pred["bus"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="Non-finite"):
        sample_metrics(pred, batch)


def test_model_is_fp32_without_changing_callers_default_dtype():
    before = torch.get_default_dtype()
    try:
        config = {"model_type": "HGT", "models": {"HGT": {
            "hidden_channels": 8, "num_layers": 1, "num_heads": 2}}}
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(9)
        reference = build_model(config, graph(), torch.device("cpu"))
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(9)
        model = build_model(config, graph(), torch.device("cpu"))
        assert all(p.dtype == torch.float32 for p in model.parameters())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[key], rtol=0, atol=0)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(before)


def test_training_honors_partial_batch_budget_and_selects_validation_score(tmp_path, monkeypatch):
    from experiments.lumina_bench import runner
    dataset = [graph() for _ in range(20)]
    config = {"case": "case30", "groups": [0], "seed": 3,
              "model_type": "HGT", "models": {"HGT": {
                  "hidden_channels": 8, "num_layers": 1, "num_heads": 2}},
              "optimizer": {"AdamW": {"lr": .001}},
              "training": {"max_samples": 11, "batch_size": 8,
                           "validate_every_samples": 8, "eval_batch_size": 3}}
    subsets, manifest = split_data(dataset, {"case": "case30"}, config)
    monkeypatch.setattr(runner, "data_splits", lambda *a, **kw: (subsets, manifest))
    output = tmp_path / "run"
    result = runner.train(config, data_root=tmp_path, output=output, device="cpu")
    assert result["samples_seen"] == 11
    assert result["steps"] == 2
    assert result["test"]["n_samples"] == len(subsets["test"])
    history = [json.loads(line) for line in (output / "history.jsonl").read_text().splitlines()]
    assert result["validation"]["score"] == min(row["validation"]["score"] for row in history)
    with pytest.raises(FileExistsError, match="Refusing"):
        runner.train(config, data_root=tmp_path, output=output, device="cpu")


def test_explicit_evaluation_manifest_cannot_change_training_split(tmp_path):
    from experiments.lumina_bench.runner import evaluate_checkpoint
    checkpoint = tmp_path / "best.pt"
    torch.save({"format": "lumina-paper-experiment-v1", "config": {"case": "case30"},
                "split_sha256": "original-checksum"}, checkpoint)
    other = tmp_path / "different-split.json"
    other.write_text("{}")
    with pytest.raises(ValueError, match="checksum changed"):
        evaluate_checkpoint(checkpoint, data_root=tmp_path, split_file=other,
                            case="pglib_opf_case30_ieee")


@pytest.mark.parametrize('mse_epochs,warmup,k', [(0, 0, 3), (0, 1, 2), (1, 0, 2), (1, 1, 2)])
def test_al_runner_short_batch_epoch_schedule_state_and_readonly_validation(tmp_path, monkeypatch, mse_epochs, warmup, k):
    import copy
    from experiments.lumina_bench import runner
    from lumina.model.opf.losses import OPFLossManager
    dataset = [graph() for _ in range(20)]
    config = {'case': 'case30', 'groups': [0], 'seed': 3, 'model_type': 'HGT',
              'loss_type': 'augmented_lagrangian', 'evaluate_test': False,
              'lagrangian': {'mu_0': .001, 'multiplier_check_interval': k, 'ema_beta': .4,
                             'multiplier_improve_ratio': .9, 'warmup_epochs': warmup},
              'models': {'HGT': {'hidden_channels': 8, 'num_layers': 1, 'num_heads': 2}},
              'optimizer': {'AdamW': {'lr': .001}},
              'training': {'loss_schedule': {'enabled': True, 'initial_loss_type': 'mse',
                           'switch_loss_type': 'augmented_lagrangian', 'mse_epochs': mse_epochs},
                           'max_samples': 55, 'batch_size': 8, 'validate_every_samples': 8, 'eval_batch_size': 3,
                           'gradient_diagnostics_every_steps': 1, 'checkpoint_samples': [16]}}
    subsets, manifest = split_data(dataset, {'case': 'case30'}, config)
    monkeypatch.setattr(runner, 'data_splits', lambda *a, **kw: (subsets, manifest))
    managers = []
    def factory(*args, **kwargs):
        mgr = OPFLossManager(*args, **kwargs); managers.append(mgr); return mgr
    monkeypatch.setattr(runner, 'OPFLossManager', factory)
    evaluator = runner.evaluate
    def checked_evaluate(*args, **kwargs):
        assert args[1] is subsets['val']  # debug never evaluates held-out test labels
        before = managers[0].loss_state_dict()
        value = evaluator(*args, **kwargs)
        after = managers[0].loss_state_dict()
        for key in before:
            if torch.is_tensor(before[key]): torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
            else: assert before[key] == after[key]
        return value
    monkeypatch.setattr(runner, 'evaluate', checked_evaluate)
    output = tmp_path / 'al'
    result = runner.train(config, data_root=tmp_path, output=output, device='cpu')
    assert result['steps'] == 7 and result['samples_seen'] == 55
    assert result['test'] is None and result['ground_truth_test'] is None
    initial = torch.load(output / 'initial.pt', weights_only=False)
    last = torch.load(output / 'last.pt', weights_only=False)
    assert initial['loss_state_dict']['lambda_k'].count_nonzero() == 0
    assert last['loss_state_dict']['successful_steps'].item() == 7 - 2*mse_epochs
    assert last['loss_state_dict']['_extra_state']['algorithm']['schedule']['current_epoch'] == 3 - mse_epochs
    assert 0 < last['loss_state_dict']['dual_updates'].item() <= 7 - 2*mse_epochs
    assert last['loss_state_dict']['lambda_k'].shape == (5,)
    review = torch.load(output / 'sample_16.pt', weights_only=False)
    assert review['samples_seen'] == 16
    if mse_epochs or warmup: assert review['loss_state_dict']['dual_updates'].item() == 0
    assert torch.load(output / 'best_through_16.pt', weights_only=False)['samples_seen'] <= 16
    history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
    assert history[-1]['epoch'] == 3
    assert history[-1]['loss_phase'] == 'dual'
    assert all(row['loss_phase'] == 'mse' for row in history[:2*mse_epochs])
    if warmup: assert history[2*mse_epochs]['loss_phase'] == 'penalty_only'
    for row in history:
        assert 'gradient_components' in row and 'gradient_clipped' in row
        assert row['train_loss'] == pytest.approx(sum(row[k] for k in (
            'train_sdk_mse', 'linear_eq', 'linear_ineq', 'quadratic_eq', 'quadratic_ineq')), rel=1e-6)
    restored = OPFLossManager('augmented_lagrangian', lagrangian_config=config['lagrangian'])
    restored.initialize_constraints(graph())
    restored.load_loss_state_dict(last['loss_state_dict'])
    for key, value in managers[0].loss_state_dict().items():
        if torch.is_tensor(value): torch.testing.assert_close(restored.loss_state_dict()[key], value, atol=0, rtol=0)
        else: assert restored.loss_state_dict()[key] == value


def test_gradient_diagnostics_preserve_backward_and_rng():
    from experiments.lumina_bench.runner import component_gradients
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad(): model.weight.copy_(torch.tensor([[1., 2.]]))
    p = model.weight
    info = {'objective': .5 * p.square().sum(), 'linear_eq': -3 * p.sum()}
    rng = torch.get_rng_state().clone()
    result = component_gradients(info, model)
    assert model.weight.grad is None
    assert torch.equal(rng, torch.get_rng_state())
    assert result['norms']['objective'] == pytest.approx(5 ** .5)
    assert result['norms']['constraints'] == pytest.approx(18 ** .5)
    assert result['constraint_mse_cosine'] == pytest.approx(-9 / (90 ** .5))
    sum(info.values()).backward()
    torch.testing.assert_close(model.weight.grad, torch.tensor([[-2., -1.]], dtype=torch.float64))


def test_al_runner_rejects_unsupported_configuration():
    from experiments.lumina_bench.runner import validate_config
    for training in ({'accumulate_grad_batches': 2}, {'amp': True}, {'precision': 'bf16'}):
        with pytest.raises(ValueError):
            validate_config({'loss_type': 'augmented_lagrangian', 'lagrangian': {}, 'training': training})
