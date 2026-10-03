"""Real trainer integration. Run as a fresh process, optionally under torchrun/MPI."""
import argparse
import copy
import importlib
import io
import math
import os
import socket
from contextlib import redirect_stdout
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset

from lumina.model.opf.losses import OPFLossManager
from lumina.evaluator.opf.utils import Modeler
from lumina.trainer.opf.trainer import MultiCaseOPFTrainer, OPFTrainer
from lumina.trainer.opf.utils import init_distributed_runtime, select_device

from test_numerics import assert_close, make_graph


class SyntheticDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return make_graph(1000 + int(index))

    def metadata(self):
        graph = self[0]
        return {
            "nodes": {name: graph[name].x.size(1) for name in graph.node_types},
            "edges": {
                name: graph[name].edge_attr.size(1)
                if "edge_attr" in graph[name] else 0
                for name in graph.edge_types
            },
        }


class SyntheticOPFTrainer(OPFTrainer):
    def _load_data(self):
        self.dataset = SyntheticDataset(24 * self.world_size)


class SyntheticMultiTrainer(MultiCaseOPFTrainer):
    def _load_dataset(self, case_name):
        return SyntheticDataset(24 * self.world_size)


def assert_rank_equal(tensor):
    assert torch.isfinite(tensor).all()
    if dist.get_world_size() > 1:
        values = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(values, tensor.contiguous())
        for value in values:
            torch.testing.assert_close(value, tensor, atol=1e-7, rtol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-device", choices=["cpu", "cuda", "xpu"], required=True)
    parser.add_argument("--output", required=True, help="Shared path for all ranks.")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--model-type", choices=["HGT", "RGAT"], default="HGT")
    parser.add_argument("--minmax-scaling", action=argparse.BooleanOptionalAction,
                        default=True)
    args = parser.parse_args()
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
    else:
        from mpi4py import MPI
        rank, world = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
        local = next((int(os.environ[name]) for name in
                      ["MPI_LOCALRANKID", "SLURM_LOCALID", "LOCAL_RANK", "PALS_LOCAL_RANKID"]
                      if name in os.environ), 0)
    print(f"rank={rank} local={local}: initializing distributed runtime", flush=True)
    local, rank, world, index = init_distributed_runtime(local, rank, world)
    device = select_device(local)
    assert device.type == args.expect_device, (device, args.expect_device)
    assert dist.get_backend() == {"cpu": "gloo", "cuda": "nccl", "xpu": "xccl"}[device.type]
    if device.type != "cpu":
        assert getattr(torch, device.type).current_device() == index
    print(f"rank={rank} local={local} device={device} backend={dist.get_backend()}", flush=True)
    if device.type == "xpu":
        properties = torch.xpu.get_device_properties(device)
        assignment = dict(host=socket.gethostname(), rank=rank, local_rank=local,
                          index=device.index, uuid=str(properties.uuid),
                          mask=os.environ.get("ZE_AFFINITY_MASK"))
        print(f"XPU assignment: {assignment}; properties={properties}", flush=True)
        assignments = [None] * world
        dist.all_gather_object(assignments, assignment)
        assert len({(item["host"], item["uuid"]) for item in assignments}) == world, assignments
    # Exercise entry-point and throughput object collectives on this group.
    objects = [{"seed": 42, "models": [args.model_type]}] if rank == 0 else [None]
    dist.broadcast_object_list(objects, src=0)
    assert objects == [{"seed": 42, "models": [args.model_type]}]
    gathered = [None] * world
    dist.all_gather_object(gathered, rank)
    assert gathered == list(range(world))
    flag = torch.tensor(rank, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    assert flag.item() == world - 1

    torch.manual_seed(42)
    output = Path(args.output)
    config = {
        "root": str(output), "checkpoint_dir": str(output), "logging_dir": str(output),
        "train_split": 0.5, "val_split": 0.25,
        "loader": {"batch_size": 2, "num_workers": 0, "shuffle": False, "pin_memory": True},
        "models": {args.model_type: {"hidden_channels": 16, "num_layers": 2,
                           "num_heads": 2, "dropout": 0.0}},
        "optimizer": {"AdamW": {"lr": 1e-3, "eps": 1e-8, "weight_decay": 0.01}},
        "scheduler": {"type": "cosine", "t_max": 24, "eta_min": 0.0},
        "training": {
            "max_epochs": 2, "patience": 100, "accumulate_grad_batches": 1,
            "gradient_clip_val": 1.0, "fail_on_nonfinite": True,
            "log_every_n_samples": 0, "val_every_n_samples": 4 * world,
            "violation_eval_p": 1.0, "validation_timing": True,
            "throughput_enabled": True, "throughput_warmup_steps": 0,
            "throughput_measure_steps": 3,
        },
        "checkpointing": {"every_n_samples": 4 * world},
    }
    options = dict(config=config, group_ids=[0], model_type=args.model_type, loss_type="mse",
                   minmax_scaling=args.minmax_scaling, local_rank=local, global_rank=rank,
                   world_size=world, wandb_requested=False)
    initialization = io.StringIO()
    try:
        with redirect_stdout(initialization):
            trainer = (
                SyntheticMultiTrainer(case_names=["synthetic_a", "synthetic_b"], **options)
                if args.multi else SyntheticOPFTrainer(case_name="synthetic", **options)
            )
    finally:
        print(initialization.getvalue(), end="", flush=True)
    assert "Warning: Model initialization failed:" not in initialization.getvalue()
    assert trainer.device == device
    assert all(not isinstance(p, torch.nn.parameter.UninitializedParameter)
               for p in trainer.model.parameters())
    samplers = (list(trainer.train_samplers.values()) if args.multi
                else [trainer.train_sampler])
    for sampler in samplers:
        indices = [None] * world
        dist.all_gather_object(indices, list(sampler))
        assert all(indices)
        for first in range(world):
            for second in range(first):
                assert set(indices[first]).isdisjoint(indices[second])
    start = [p.detach().clone() for p in trainer.model.parameters()]
    step_checks = []

    def check_gradients(optimizer, positional, keywords):
        grads = [p.grad.flatten() for p in trainer.model.parameters() if p.grad is not None]
        assert grads
        assert_rank_equal(torch.cat(grads))
        step_checks.append(1)

    hook = trainer.optimizer.register_step_pre_hook(check_gradients)
    trainer.train()  # Production batching, forward, backward, updates, validation, save.
    hook.remove()
    assert len(step_checks) >= 3
    assert trainer.throughput_tracker.has_run
    assert len(trainer.throughput_tracker.samples) == 3
    assert all(math.isfinite(sample["throughput/samples_per_sec"])
               and sample["throughput/samples_per_sec"] > 0
               for sample in trainer.throughput_tracker.samples)
    assert trainer.nonfinite_loss_skips == trainer.nonfinite_grad_skips == 0
    assert any(not torch.equal(p.detach(), initial)
               for p, initial in zip(trainer.model.parameters(), start))
    assert_rank_equal(torch.cat([p.detach().flatten() for p in trainer.model.parameters()]))
    val_loss, _, metrics = trainer.validate()
    assert val_loss is not None and torch.isfinite(torch.tensor(val_loss))
    assert metrics["val/perf/eval_batches"] > 0
    assert trainer.best_val_loss < float("inf")
    if rank == 0:
        assert list(output.glob("best-*.pt"))
        assert list(output.glob("checkpoint-*.pt"))
    checkpoint_path = output / "roundtrip.pt"
    trainer.save_checkpoint(str(checkpoint_path))
    dist.barrier()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    expected_keys = {
        "epoch", "config", "run_metadata", "model_class", "model_kwargs", "loss_type",
        "model_state_dict", "optimizer_state_dict", "best_val_loss",
    }
    if args.multi:
        expected_keys.add("case_names")
    assert set(checkpoint) == expected_keys
    assert all(value.device.type == "cpu"
               for value in checkpoint["model_state_dict"].values())
    module_name, class_name = checkpoint["model_class"].rsplit(".", 1)
    model_cls = getattr(importlib.import_module(module_name), class_name)
    batch_cpu = make_graph(2026)
    model_cpu = model_cls(**checkpoint["model_kwargs"])
    model_cpu.load_state_dict(checkpoint["model_state_dict"])
    model_cpu.eval()
    model_device = model_cls(**checkpoint["model_kwargs"]).to(device)
    model_device.load_state_dict(checkpoint["model_state_dict"])
    model_device.eval()
    trainer.model.eval()
    batch_device = batch_cpu.clone().to(device)
    with torch.no_grad():
        cpu_pred = model_cpu(batch_cpu.x_dict, batch_cpu.edge_index_dict,
                             minmax_scaling=args.minmax_scaling)
        restored_pred = model_device(batch_device.x_dict, batch_device.edge_index_dict,
                                     minmax_scaling=args.minmax_scaling)
        original_pred = trainer.forward(batch_device)
    for name in cpu_pred:
        assert_close(restored_pred[name], cpu_pred[name], atol=2e-5, rtol=2e-4)
        assert_close(restored_pred[name], original_pred[name], atol=2e-5, rtol=2e-4)
    # Exercise the existing public checkpoint/inference API on both devices.
    loaded_predictions = []
    for destination in [torch.device("cpu"), device]:
        modeler = Modeler(destination, fail_on_missing=True, verbose=False)
        modeler.load_model_from_training_checkpoint(checkpoint_path, strict=True)
        assert next(modeler.model.parameters()).device == destination
        with torch.no_grad():
            pred, returned_batch = modeler.predict_batch(
                batch_cpu.clone(), minmax_scaling=args.minmax_scaling
            )
        assert all(x.device.type == "cpu" for x in returned_batch.x_dict.values())
        assert all(value.device.type == "cpu" for value in pred.values())
        stats = modeler.evaluate_from_predictions([(pred, returned_batch)])
        assert stats
        assert all(math.isfinite(value) for entry in stats.values() for value in entry.values())
        assert all(entry["weight"] > 0 for entry in stats.values())
        loaded_predictions.append(pred)
    for name in loaded_predictions[0]:
        assert_close(loaded_predictions[1][name], loaded_predictions[0][name],
                     atol=2e-5, rtol=2e-4)
    # Restore AdamW moments and compare one continuation update on CPU and device.
    continued = []
    for model, batch in [(model_cpu, batch_cpu), (model_device, batch_device)]:
        optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"]["AdamW"])
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state_dict"]))
        assert optimizer.state
        optimizer.zero_grad(set_to_none=True)
        loss, _ = OPFLossManager().compute_loss(
            model(batch.x_dict, batch.edge_index_dict,
                  minmax_scaling=args.minmax_scaling), batch
        )
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        continued.append(dict(model.named_parameters()))
    for name in continued[0]:
        assert_close(continued[1][name], continued[0][name], atol=5e-6, rtol=2e-4)
    dist.barrier()
    print(f"rank={rank}: trainer, validation, checkpoint and continuation passed", flush=True)
    if not (dist.get_backend() == "xccl"
            and os.environ.get("LUMINA_SKIP_XCCL_DESTROY") == "1"):
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
