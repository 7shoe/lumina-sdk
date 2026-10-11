"""Exercise data ordering, interruption, recovery, and completion on CPU."""
import copy
import io
import json
import tarfile

import numpy as np
import pytest
import torch
import yaml

from distillation.data.split import split_indices
from distillation.training.T1_single_topo import campaign, train
from distillation.training.T1_single_topo.common import (
    canonical_case, claim, load_config, read_json, run_paths, write_json)
from distillation.training.T1_single_topo.data import CaseDataset, finish_case, prepare_group
from experiments.lumina_bench.tests.test_protocol import graph

CASE = canonical_case("case14")


def raw_sample(index):
    sample = graph()
    pg = .5 + index / 1000
    sample["generator"].x[0, 1] = pg
    sample["generator"].y[0, 0] = pg
    sample["load"].x[0, 0] = pg
    nodes = {name: sample[name].x.tolist() for name in sample.node_types}
    nodes["bus"] = [[230, 3, .9, 1.1], [230, 1, .9, 1.1]]
    edges = {f"{name}_link": {"senders": [0], "receivers": [0]}
             for name in ("generator", "load", "shunt")}
    edges["ac_line"] = {"senders": [0], "receivers": [1],
                        "features": sample["bus", "ac_line", "bus"].edge_attr.tolist()}
    edges["transformer"] = {"senders": [1], "receivers": [0],
                            "features": [[-.5, .5, .02, .1, 1, 1, 1, 1, 0, 0, 0]]}
    return {"grid": {"context": [[100.]], "nodes": nodes, "edges": edges},
            "solution": {"nodes": {name: sample[name].y.tolist() for name in ("bus", "generator")},
                         "edges": {name: {"features": [[0., 0., 0., 0.]]}
                                   for name in ("ac_line", "transformer")}},
            "metadata": {"objective": 2 * pg**2 + 3 * pg + 4}}


def archive(root, group, indices):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{CASE}_{group}.tar.gz"
    with tarfile.open(path, "w:gz") as stream:
        for index in indices:
            payload = json.dumps(raw_sample(index)).encode()
            member = tarfile.TarInfo(f"release/{CASE}/group_{group}/example_{index}.json")
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    return path


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config["cases"] = [CASE]
    config["seeds"] = [42]
    config["paths"] = {name: str(tmp_path / name) for name in ("raw", "data", "output")}
    for group in range(2):
        archive(tmp_path / "raw", group, reversed(range(group * 20, (group + 1) * 20)))
        assert prepare_group(config["paths"]["raw"], config["paths"]["data"], CASE, group, 20)
    assert finish_case(config["paths"]["data"], CASE, groups=2, group_size=20)
    config["training"].update(max_samples=75, batch_size=7, eval_batch_size=1)
    for options in config["models"].values():
        options["hidden_dim" if "hidden_dim" in options else "hidden_channels"] = 8
        options["num_layers"] = 2
    return config


def run(config, model="HGT", **kwargs):
    return train.train_run(config, model, CASE, 42, device="cpu", **kwargs)


def checkpoint(config, model="HGT", name="final.pt"):
    folder, _ = run_paths(config, model, CASE, 42)
    return torch.load(folder / name, map_location="cpu", weights_only=False)


def assert_same(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


def test_numeric_sample_order_and_authoritative_split(config):
    dataset = CaseDataset(config["paths"]["data"], CASE)
    assert [dataset[i].sample_id.item() for i in range(40)] == list(range(40))
    assert dataset[37]["generator"].y[0, 0].item() == pytest.approx(.537)
    expected = split_indices(CASE, 40, 42)
    assert_same(dataset.splits(), expected)
    assert not set(expected["train"]) & set(expected["test"])
    before = dataset.folder.joinpath("group_0.pt").stat().st_mtime_ns
    assert prepare_group(config["paths"]["raw"], config["paths"]["data"], CASE, 0, 20)
    assert dataset.folder.joinpath("group_0.pt").stat().st_mtime_ns == before


@pytest.mark.parametrize("indices, message", [([0, 0], "Duplicate"), ([0], "Expected"), ([0, 2], "out-of-range")])
def test_preparation_rejects_invalid_sample_ids(tmp_path, indices, message):
    archive(tmp_path / "raw", 0, indices)
    with pytest.raises(ValueError, match=message):
        prepare_group(tmp_path / "raw", tmp_path / "data", CASE, 0, 2)
    assert not (tmp_path / "data" / CASE / "group_0.json").exists()


@pytest.mark.parametrize("model", load_config()["models"])
def test_resume_matches_uninterrupted_training_for_each_model(config, model):
    assert run(config, model) == "complete"
    expected = checkpoint(config, model)
    restarted = copy.deepcopy(config)
    restarted["paths"]["output"] += "-resumed"
    assert run(restarted, model, session_steps=7) == "paused"
    partial = checkpoint(restarted, model, "last.pt")
    assert partial["state"]["samples"] == 46
    assert partial["state"]["epoch"] == 1 and partial["state"]["offset"] == 14
    assert run(restarted, model) == "complete"
    actual = checkpoint(restarted, model)
    assert actual["state"]["samples"] == 75
    for key in ("model", "optimizer", "state", "rng"):
        assert_same(expected[key], actual[key])
    folder, results = run_paths(restarted, model, CASE, 42)
    for split in ("val", "test"):
        metrics = read_json(results / f"{split}.json")["metrics"]
        assert metrics["n_samples"] == 4
        assert {"opf_solution_error", "constraint_violation", "cost_difference"} <= metrics.keys()
    assert checkpoint(restarted, model, "best.pt")["validation"]["score"] == min(
        read_json(path)["validation"]["score"] for path in (folder / "history").glob("*.json"))


def test_crash_rolls_back_to_last_committed_checkpoint(config, monkeypatch):
    assert run(config) == "complete"
    expected = checkpoint(config)
    config["paths"]["output"] += "-crashed"
    save = train.save_torch
    def crash(path, value):
        if path.name == "last.pt" and value["state"]["samples"] >= 32:
            raise RuntimeError("simulated termination before checkpoint write")
        return save(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(train, "save_torch", crash)
        with pytest.raises(RuntimeError, match="simulated termination"):
            run(config)
    assert checkpoint(config, name="last.pt")["state"]["samples"] == 21
    assert run(config) == "complete"
    actual = checkpoint(config)
    for key in ("model", "optimizer", "state", "rng"):
        assert_same(expected[key], actual[key])


def test_final_evaluation_resumes_without_training_again(config, monkeypatch):
    evaluate = train.evaluate
    def interrupted(*args):
        if args[-2].get("split") != "test":
            return evaluate(*args)
        calls = iter([False, True])
        return evaluate(*args[:-1], lambda: next(calls))
    with monkeypatch.context() as patch:
        patch.setattr(train, "evaluate", interrupted)
        assert run(config) == "paused"
    folder, results = run_paths(config, "HGT", CASE, 42)
    assert read_json(folder / "test-progress.json")["cursor"] == 1
    final = checkpoint(config)
    assert not (folder / "complete.json").exists()
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("Unexpected optimizer step"))
        assert run(config) == "complete"
    assert read_json(results / "test.json")["metrics"]["n_samples"] == 4
    assert_same(final["model"], checkpoint(config)["model"])
    dataset = CaseDataset(config["paths"]["data"], CASE)
    model = train.build_model({"model_type": "HGT", "models": config["models"]}, dataset[0], torch.device("cpu"))
    model.load_state_dict(checkpoint(config, name="best.pt")["model"])
    expected = evaluate(model, dataset, dataset.splits()["test"], torch.device("cpu"), 1,
                        folder / "reference-eval.json", {}, lambda: False)
    assert read_json(results / "test.json")["metrics"] == expected


def test_interrupted_validation_preserves_training_trajectory(config, monkeypatch):
    assert run(config) == "complete"
    expected = checkpoint(config)
    config["paths"]["output"] += "-validation"
    evaluate = train.evaluate
    def interrupted(*args):
        calls = iter([False, True])
        return evaluate(*args[:-1], lambda: next(calls))
    with monkeypatch.context() as patch:
        patch.setattr(train, "evaluate", interrupted)
        assert run(config) == "paused"
    folder, _ = run_paths(config, "HGT", CASE, 42)
    assert checkpoint(config, name="last.pt")["state"]["phase"] == "validate"
    assert read_json(folder / "validation-progress.json")["cursor"] == 1
    assert run(config) == "complete"
    actual = checkpoint(config)
    for key in ("model", "optimizer", "state", "rng"):
        assert_same(expected[key], actual[key])


def test_completed_runs_are_skipped_and_missing_results_are_repaired(config, monkeypatch):
    assert run(config) == "complete"
    assert campaign.is_complete(config, ("HGT", CASE, 42))
    with monkeypatch.context() as patch:
        patch.setattr(train, "select_device", lambda *a: pytest.fail("Completed run used a device"))
        assert run(config) == "complete"
    _, results = run_paths(config, "HGT", CASE, 42)
    (results / "test.json").unlink()
    assert not campaign.is_complete(config, ("HGT", CASE, 42))
    with monkeypatch.context() as patch:
        patch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("Unexpected optimizer step"))
        assert run(config) == "complete"
    assert campaign.is_complete(config, ("HGT", CASE, 42))


def test_changed_configuration_and_concurrent_run_are_rejected(config):
    folder, _ = run_paths(config, "HGT", CASE, 42)
    with claim(folder / "run.lock") as acquired:
        assert acquired
        assert run(config) == "busy"
    assert run(config, session_steps=1) == "paused"
    config["optimizer"]["lr"] *= 2
    with pytest.raises(ValueError, match="changed"):
        run(config)


def test_changed_completed_result_is_not_accepted(config):
    assert run(config) == "complete"
    _, results = run_paths(config, "HGT", CASE, 42)
    path = results / "test.json"
    report = read_json(path)
    report["metrics"]["sdk_mse"] = -1
    write_json(path, report)
    with pytest.raises(ValueError, match="Completed artifact changed"):
        campaign.is_complete(config, ("HGT", CASE, 42))


def test_default_campaign_has_120_runs_and_one_million_samples():
    config = load_config()
    assert len(set(campaign.task_list(config))) == 120
    assert config["training"]["max_samples"] == 1_000_000
    assert config["seeds"] == [42, 43, 44]
    assert config["split_seed"] == config["shuffle_seed"] == 42


def test_model_seeds_change_weights_but_not_splits_or_training_sequence(config, monkeypatch):
    weights, seen, splits = {}, {}, {}
    build_model = train.build_model
    def observe_model(*args):
        model = build_model(*args)
        weights[seed] = {key: value.clone() for key, value in model.state_dict().items()}
        def observe_batch(model, inputs):
            if model.training:
                seen[seed].extend(inputs[0].sample_id.tolist())
        model.register_forward_pre_hook(observe_batch)
        return model
    monkeypatch.setattr(train, "build_model", observe_model)
    for seed in (42, 43, 44):
        seen[seed] = []
        assert train.train_run(config, "HGT", CASE, seed, device="cpu", session_steps=6) == "paused"
        folder, _ = run_paths(config, "HGT", CASE, seed)
        splits[seed] = read_json(folder / "split.json")
    assert splits[42] == splits[43] == splits[44]
    assert seen[42] == seen[43] == seen[44]
    assert len(seen[42]) == 39  # One full epoch and the next batch.
    assert set(seen[42][:32]) == set(splits[42]["indices"]["train"])
    for seed in (43, 44):
        assert any(not torch.equal(weights[42][key], value) for key, value in weights[seed].items())


@pytest.mark.parametrize("seed", [43, 42.0])
def test_alternate_split_seed_is_rejected(config, tmp_path, seed):
    config["split_seed"] = seed
    path = tmp_path / "invalid-split.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="split_seed must be 42"):
        load_config(path)
    with pytest.raises(ValueError, match="split_seed must be 42"):
        run(config)


def test_campaign_launches_a_child_and_skips_it_on_resubmission(config, tmp_path, monkeypatch, capsys):
    config["models"] = {"HGT": config["models"]["HGT"]}
    config["training"]["max_samples"] = 7
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("sys.argv", ["campaign", "--config", str(path), "--device", "cpu", "--seconds", "120"])
    campaign.main()
    assert campaign.is_complete(config, ("HGT", CASE, 42))
    monkeypatch.setattr(campaign, "run_child", lambda *a: pytest.fail("Completed run was launched"))
    campaign.main()
    assert "1 complete; 0 unfinished" in capsys.readouterr().out


def test_campaign_continues_after_a_worker_error(config, tmp_path, monkeypatch):
    config["models"] = {name: config["models"][name] for name in ("HGT", "GCN")}
    config["training"]["max_samples"] = 7
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setattr("sys.argv", ["campaign", "--config", str(path), "--device", "cpu"])
    launch = campaign.run_child
    def fail_first(command, log, stop):
        if command[command.index("--model") + 1] == "HGT":
            raise OSError("simulated launch failure")
        return launch(command, log, stop)
    monkeypatch.setattr(campaign, "run_child", fail_first)
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 1
    assert not campaign.is_complete(config, ("HGT", CASE, 42))
    assert campaign.is_complete(config, ("GCN", CASE, 42))
