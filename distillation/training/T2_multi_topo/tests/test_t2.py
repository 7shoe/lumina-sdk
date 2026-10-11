"""Check balanced data, mixed graph sizes, and exact checkpoint recovery."""
import copy
import io
import json
import tarfile

import numpy as np
import pytest
import torch
import yaml
from torch_geometric.data import Batch

from distillation.data.split import split_indices
from distillation.training.T1_single_topo.data import finish_case, prepare_group
from distillation.training.T1_single_topo.tests.test_t1 import assert_same, raw_sample
from distillation.training.T2_multi_topo import campaign, train
from distillation.training.T2_multi_topo.common import load_config, read_json, run_paths
from distillation.training.T2_multi_topo.data import CollectionDataset, split_record

COLLECTION = "case14_case30_case57"


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config["collections"] = {COLLECTION: config["collections"][COLLECTION]}
    config["cases"] = config["collections"][COLLECTION]
    config["seeds"] = [43]
    config["paths"] = {name: str(tmp_path / name) for name in ("raw", "data", "output")}
    (tmp_path / "raw").mkdir()
    for extra, case in enumerate(config["cases"]):
        for group in range(2):
            path = tmp_path / "raw" / f"{case}_{group}.tar.gz"
            with tarfile.open(path, "w:gz") as stream:
                for index in reversed(range(group * 20, (group + 1) * 20)):
                    sample = raw_sample(index)
                    sample["grid"]["nodes"]["bus"].extend([[230, 1, .9, 1.1]] * extra)
                    sample["solution"]["nodes"]["bus"].extend([[0., 1.]] * extra)
                    payload = json.dumps(sample).encode()
                    member = tarfile.TarInfo(f"release/{case}/group_{group}/example_{index}.json")
                    member.size = len(payload)
                    stream.addfile(member, io.BytesIO(payload))
            assert prepare_group(config["paths"]["raw"], config["paths"]["data"], case, group, 20)
        assert finish_case(config["paths"]["data"], case, groups=2, group_size=20)
    config["training"].update(max_samples=123, batch_size=7, eval_batch_size=1)
    for options in config["models"].values():
        options["hidden_dim" if "hidden_dim" in options else "hidden_channels"] = 8
        options["num_layers"] = 2
    return config


def run(config, model="HGT", **kwargs):
    return train.train_run(config, model, COLLECTION, 43, device="cpu", **kwargs)


def checkpoint(config, model="HGT", name="final.pt"):
    folder, _ = run_paths(config, model, COLLECTION, 43)
    return torch.load(folder / name, map_location="cpu", weights_only=False)


def test_uniform_sampling_and_original_splits(config):
    dataset = CollectionDataset(config, COLLECTION)
    indices = dataset.splits()
    assert {name: len(ids) for name, ids in indices.items()} == {"train": 96, "val": 12, "test": 12}
    record = split_record(dataset)
    for case, member in zip(dataset.cases, dataset.members):
        assert_same(member, split_indices(case, 40, seed=42))
        assert record["indices_by_case"][case] == {k: v.tolist() for k, v in member.items()}
    first = dataset.train_order(42, 0)
    assert sorted(first) == list(range(96))
    assert not np.array_equal(first, dataset.train_order(42, 1))
    graphs = [dataset[int(index)] for index in indices["train"][first]]
    assert [graph.case_index.item() for graph in graphs] == [0, 1, 2] * 32
    assert [graph["bus"].num_nodes for graph in graphs[:3]] == [2, 3, 4]
    assert dataset.sample_counts(1_000_000) == dict(zip(dataset.cases, [333334, 333333, 333333]))


def test_mse_weights_graphs_equally(config):
    dataset = CollectionDataset(config, COLLECTION)
    batch = Batch.from_data_list([dataset[0], dataset[40], dataset[80]])
    predictions = {}
    for name in ("bus", "generator"):
        error = (batch[name].batch + 1).float().unsqueeze(1)
        predictions[name] = (batch[name].y + error).requires_grad_(True)
    loss = train.graph_mse(predictions, batch)
    assert loss.item() == pytest.approx(2 * (1 + 4 + 9) / 3)
    loss.backward()
    assert all(torch.isfinite(value.grad).all() for value in predictions.values())


@pytest.mark.parametrize("model", list(load_config()["models"]))
def test_all_models_resume_exactly_across_mixed_epoch(config, model):
    assert run(config, model) == "complete"
    expected = checkpoint(config, model)
    restarted = copy.deepcopy(config)
    restarted["paths"]["output"] += "-resumed"
    assert run(restarted, model, session_steps=16) == "paused"
    partial = checkpoint(restarted, model, "last.pt")["state"]
    assert (partial["epoch"], partial["offset"], partial["samples"]) == (1, 14, 110)
    assert run(restarted, model) == "complete"
    actual = checkpoint(restarted, model)
    for key in ("model", "optimizer", "state", "rng", "samples_per_case"):
        assert_same(expected[key], actual[key])
    assert actual["state"]["samples"] == 123
    assert set(actual["samples_per_case"].values()) == {41}
    assert campaign.is_complete(restarted, (model, COLLECTION, 43))
    folder, results = run_paths(restarted, model, COLLECTION, 43)
    for split in ("val", "test"):
        assert read_json(results / f"{split}.json")["metrics"]["n_samples"] == 12
        for case in config["cases"]:
            metrics = read_json(results / split / f"{case}.json")["metrics"]
            assert metrics["n_samples"] == 4
            assert {"opf_solution_error", "constraint_violation", "cost_difference"} <= metrics.keys()
    assert len(read_json(folder / "complete.json")["files"]) == 10


def test_evaluation_resumes_and_missing_case_result_is_repaired(config, monkeypatch):
    evaluate = train.evaluate
    def interrupted(*args):
        if args[-2].get("split") != "test":
            return evaluate(*args)
        calls = iter([False, True])
        return evaluate(*args[:-1], lambda: next(calls))
    with monkeypatch.context() as patch:
        patch.setattr(train, "evaluate", interrupted)
        assert run(config) == "paused"
    folder, results = run_paths(config, "HGT", COLLECTION, 43)
    first_case = config["cases"][0]
    assert read_json(folder / f"test-{first_case}-progress.json")["cursor"] == 1
    before = checkpoint(config)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda *a, **kw: pytest.fail("Unexpected optimizer step"))
    assert run(config) == "complete"
    (results / "test" / f"{first_case}.json").unlink()
    assert not campaign.is_complete(config, ("HGT", COLLECTION, 43))
    assert run(config) == "complete"
    assert_same(before["model"], checkpoint(config)["model"])
    dataset = CollectionDataset(config, COLLECTION)
    model = train.build_model({"model_type": "HGT", "models": config["models"]}, dataset[0], torch.device("cpu"))
    model.load_state_dict(checkpoint(config, name="best.pt")["model"])
    expected = evaluate(model, dataset, dataset.splits()["test"], torch.device("cpu"), 1,
                        folder / "reference-eval.json", {}, lambda: False)
    actual = read_json(results / "test.json")["metrics"]
    assert actual == pytest.approx(expected)
    monkeypatch.setattr(train, "select_device", lambda *a: pytest.fail("Completed run used a device"))
    assert run(config) == "complete"


def test_initialization_seeds_preserve_data_and_shuffle(config, monkeypatch):
    weights, seen, splits = {}, {}, {}
    build = train.build_model
    def observe(*args):
        model = build(*args)
        weights[seed] = {key: value.clone() for key, value in model.state_dict().items()}
        def batch_seen(model, inputs):
            if model.training:
                batch = inputs[0]
                seen[seed].extend(zip(batch.case_index.tolist(), batch.sample_id.tolist()))
        model.register_forward_pre_hook(batch_seen)
        return model
    monkeypatch.setattr(train, "build_model", observe)
    for seed in (42, 43, 44):
        seen[seed] = []
        assert train.train_run(config, "HGT", COLLECTION, seed, device="cpu", session_steps=16) == "paused"
        folder, _ = run_paths(config, "HGT", COLLECTION, seed)
        splits[seed] = read_json(folder / "split.json")
    assert splits[42] == splits[43] == splits[44]
    assert seen[42] == seen[43] == seen[44]
    assert len(seen[42]) == 110
    for seed in (43, 44):
        assert any(not torch.equal(weights[42][key], value) for key, value in weights[seed].items())


def test_combined_metrics_use_metric_specific_counts():
    reports = [{"accumulator": {"n_samples": n, "sums": {"sdk_mse": n, "violation": 2 * n,
                "cost_difference_pct": total}, "counts": {"sdk_mse": n, "violation": n,
                "cost_difference_pct": count}}} for n, total, count in [(4, 2., 2), (4, 12., 4)]]
    result = train.combined_metrics(reports)
    assert result["cost_difference_pct"] == pytest.approx(14 / 6)
    assert result["n_samples"] == 8 and result["score"] == 3


def test_default_campaign():
    config = load_config()
    assert len(set(campaign.task_list(config))) == 72
    assert config["training"]["max_samples"] == 1_000_000
    assert config["seeds"] == [42, 43, 44]
    assert config["split_seed"] == config["shuffle_seed"] == 42


def test_campaign_child_launch_and_completed_skip(config, tmp_path, monkeypatch, capsys):
    config["models"] = {"HGT": config["models"]["HGT"]}
    config["training"]["max_samples"] = 7
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(config))
    monkeypatch.setattr("sys.argv", ["campaign", "--config", str(path), "--device", "cpu", "--seconds", "120"])
    campaign.main()
    assert campaign.is_complete(config, ("HGT", COLLECTION, 43))
    monkeypatch.setattr(campaign, "run_child", lambda *a: pytest.fail("Completed run was launched"))
    campaign.main()
    assert "1 complete; 0 unfinished" in capsys.readouterr().out
