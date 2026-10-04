"""Backbone integration: trainability, checkpoint reconstruction and real edge inputs."""
import pytest
import torch
from torch_geometric.data import Batch

from experiments.lumina_bench.models import SUPPORTED_MODELS
from experiments.lumina_bench.runner import build_model
from experiments.lumina_bench.tests.test_protocol import graph


def configuration(kind):
    if kind in {"GCN", "GAT", "GIN", "Transformer"}:
        options = {"hidden_dim": 16, "num_layers": 3, "dropout": 0.1}
    else:
        options = {"hidden_channels": 16, "num_layers": 3}
        if kind in {"HGT", "RGAT"}:
            options["num_heads"] = 2
        elif kind == "HEAT":
            options["attention_heads"] = 2
        else:
            options["backend"] = "gat"
    return {"model_type": kind, "models": {kind: options}}


@pytest.mark.parametrize("kind", SUPPORTED_MODELS)
def test_model_mse_update_and_checkpoint_reconstruction(kind, tmp_path):
    torch.manual_seed(42)
    sample = graph()
    config = configuration(kind)
    model = build_model(config, sample, torch.device("cpu"))
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    batch = Batch.from_data_list([sample, sample.clone()])
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    outputs = model(batch)
    assert set(outputs) == {"bus", "generator"}
    assert all(outputs[k].shape == batch[k].y.shape for k in outputs)
    loss = sum((outputs[k] - batch[k].y.float()).square().mean() for k in outputs)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert any(not torch.equal(v, before[k]) for k, v in model.state_dict().items())
    model.eval()
    expected = model(batch)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": config, "example_graph": sample, "weights": model.state_dict()}, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    restored = build_model(payload["config"], payload["example_graph"], torch.device("cpu"))
    restored.load_state_dict(payload["weights"], strict=True)
    restored.eval()
    actual = restored(batch)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert ((actual["bus"][:, 1] >= .9) & (actual["bus"][:, 1] <= 1.1)).all()
    assert ((actual["generator"][:, 0] >= 0) & (actual["generator"][:, 0] <= 2)).all()


@pytest.mark.parametrize("kind", ["HEAT", "HeteroGNN"])
def test_edge_features_affect_predictions_and_receive_gradients(kind):
    torch.manual_seed(42)
    sample = graph()
    # Two incoming branch edges are needed to expose GAT's attention weights.
    sample["bus", "ac_line", "bus"].edge_index = torch.tensor([[0, 1], [1, 1]])
    sample["bus", "ac_line", "bus"].edge_attr = torch.tensor([
        [-.5, .5, 0, 0, .02, .1, 1, 1, 1],
        [-.4, .4, .02, .02, .04, .2, .5, .5, .5]], dtype=torch.float32)
    model = build_model(configuration(kind), sample, torch.device("cpu"))
    model.eval()
    batch = Batch.from_data_list([sample])
    edge_attr = batch["bus", "ac_line", "bus"].edge_attr.requires_grad_()
    outputs = model(batch)
    sum(v.square().sum() for v in outputs.values()).backward()
    assert edge_attr.grad is not None and torch.isfinite(edge_attr.grad).all()
    assert edge_attr.grad.abs().sum() > 0
    changed = batch.clone()
    changed["bus", "ac_line", "bus"].edge_attr = torch.zeros_like(edge_attr)
    altered = model(changed)
    assert any(not torch.equal(outputs[k], altered[k]) for k in outputs)
