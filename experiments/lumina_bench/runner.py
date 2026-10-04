"""Small single-device runner; the SDK owns the datasets, backbones and MSE loss."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import subprocess
import time

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.utils import parse_case_name
from .data import file_sha256, load_data, split_data, write_json
from .metrics import MetricAccumulator, sample_metrics
from .models import BenchmarkModel, SUPPORTED_MODELS


def select_device(requested):
    device = torch.device(requested)
    if device.type == "xpu":
        if not torch.xpu.is_available():
            raise RuntimeError("XPU was explicitly requested but is unavailable")
        torch.xpu.set_device(device.index or 0)
    elif device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        torch.cuda.set_device(device.index or 0)
    elif device.type != "cpu":
        raise ValueError("Supported devices: cpu, cuda[:index], xpu[:index]")
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.xpu.is_available():
        torch.xpu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")


def build_model(config, sample, device):
    # Initialize lazy layers on CPU so CPU and XPU can start with identical weights.
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"):
            model = BenchmarkModel(sample, config["model_type"], config["models"][config["model_type"]])
    finally:
        torch.set_default_dtype(previous_dtype)
    model.eval()
    with torch.no_grad():
        model(Batch.from_data_list([sample.clone()]))
    return model.to(device)


def provenance(device):
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    files = {str(p.relative_to(repo)): file_sha256(p)
             for p in Path(__file__).parent.rglob("*")
             if p.is_file() and p.suffix in {".py", ".yaml", ".sh"}}
    loss_files = {str(p.relative_to(repo)): file_sha256(p)
                  for p in (repo / 'lumina/model/opf').glob('*lagrangian*.py')}
    loss_files['lumina/model/opf/losses.py'] = file_sha256(repo / 'lumina/model/opf/losses.py')
    return {"sdk_commit": commit, "experiment_file_sha256": files, "loss_file_sha256": loss_files,
            "torch": torch.__version__, "torch_geometric": torch_geometric.__version__,
            "python": platform.python_version(), "host": platform.node(),
            "device": str(device), "pbs_jobid": os.environ.get("PBS_JOBID"),
            "device_name": (torch.xpu.get_device_name(device) if device.type == "xpu"
                            else torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")}


def protocol(config):
    result = {"version": 1, "status": "reconstructed protocol; not an exact paper replication",
            "paper": "https://arxiv.org/abs/2605.02133v1",
            "precision": "FP32 model; CPU FP64 physical metrics",
            "loss": "SDK bus-MSE + generator-MSE; no dummy load/shunt targets",
            "bounds": "shared sigmoid Vm/Pg/Qg head; raw voltage angles",
            "homogeneous_conversion": "SDK convert_opf_to_homo(node_dim=64, edge_dim=32); preserves edge attributes",
            "selection": config.get("selection_mse", "sdk_mse") + " + violation",
            "violation": "mean_s [(||P/Q balance||_2 + ||positive squared-flow excess at both ends||_2) / sqrt(Nbus)]",
            "units": "processed OPFData p.u. powers/ratings/cost polynomial; radians",
            "split": "persisted seeded random indices; seed independent of training seed",
            "limitations": ["author split IDs, winning hyperparameters and exact MSE reduction unavailable",
                            "MSE only; single process; no AL, DDP or resume"]}
    if config.get('loss_type') == 'augmented_lagrangian':
        from dataclasses import asdict
        from lumina.model.opf.augmented_lagrangian import ALConfig, SOURCE_COMMIT
        result.update(loss='SDK MSE + upstream Frontier RMS/worst-end AL with physical-constraint duals',
                      lagrangian=asdict(ALConfig.from_dict(config.get('lagrangian'))),
                      loss_schedule=config['training'].get('loss_schedule'),
                      al_source_commit=SOURCE_COMMIT,
                      precision='FP32 model/residuals/duals; CPU sparse balance; CPU FP64 evaluation',
                      limitations=[result['limitations'][0],
                                   'fixed topology/order/electrical parameters; no AMP, accumulation, DDP or resume',
                                   'source objective/policy on [B,N]; not upstream collated-batch slot duals',
                                   'EMA/duals commit after optimizer success; validation is read-only'])
    return result


def validate_config(config):
    if config.get("loss_type", "mse") not in {'mse', 'augmented_lagrangian'}:
        raise ValueError('Supported losses: mse, augmented_lagrangian')
    if config.get('loss_type') == 'augmented_lagrangian':
        from lumina.model.opf.augmented_lagrangian import ALConfig
        ALConfig.from_dict(config.get('lagrangian'))
        training = config['training']
        if training.get('accumulate_grad_batches', 1) != 1:
            raise ValueError('AL requires accumulate_grad_batches=1')
        if training.get('precision', 'fp32') != 'fp32' or training.get('amp', False):
            raise ValueError('AL requires FP32 without AMP')
    elif config.get('lagrangian') is not None:
        raise ValueError('lagrangian configuration requires AL')
    if not isinstance(config.get('evaluate_test', True), bool):
        raise ValueError('evaluate_test must be a boolean')
    if config.get("selection_mse", "sdk_mse") not in {"mse", "sdk_mse"}:
        raise ValueError("selection_mse must be mse or sdk_mse")
    if config["model_type"] not in SUPPORTED_MODELS:
        raise ValueError(f"Supported models: {', '.join(SUPPORTED_MODELS)}")
    training = config["training"]
    schedule = training.get('loss_schedule')
    if schedule is not None:
        if (config.get('loss_type') != 'augmented_lagrangian' or not isinstance(schedule, dict)
                or set(schedule) != {'enabled', 'initial_loss_type', 'switch_loss_type', 'mse_epochs'}
                or schedule['enabled'] is not True or schedule['initial_loss_type'] != 'mse'
                or schedule['switch_loss_type'] != 'augmented_lagrangian'
                or isinstance(schedule['mse_epochs'], bool) or not isinstance(schedule['mse_epochs'], int)
                or schedule['mse_epochs'] < 0):
            raise ValueError('Unsupported MSE-to-AL loss_schedule')
    for key in ("max_samples", "validate_every_samples", "batch_size"):
        if int(training[key]) <= 0:
            raise ValueError(f"training.{key} must be positive")
    if int(training.get("max_steps", 0)) < 0:
        raise ValueError("max_steps must be nonnegative")
    if float(training.get("gradient_clip", 1.0)) <= 0:
        raise ValueError("gradient_clip must be positive")
    interval = training.get('gradient_diagnostics_every_steps', 0)
    if isinstance(interval, bool) or not isinstance(interval, int) or interval < 0:
        raise ValueError('gradient_diagnostics_every_steps must be a nonnegative integer')
    if any(isinstance(s, bool) or not isinstance(s, int) or s <= 0
           for s in training.get('checkpoint_samples', [])):
        raise ValueError('checkpoint_samples must contain positive integers')
    if config.get('scheduler') is not None or config.get('lr_scheduler') is not None:
        raise ValueError('This benchmark runner uses constant learning rate; scheduler configuration is unsupported')


def component_gradients(info, model):
    """Observe the existing forward graph without changing .grad, RNG or duals."""
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = {}
    for name in ('objective', 'linear_eq', 'linear_ineq', 'quadratic_eq', 'quadratic_ineq'):
        if name not in info:
            continue
        grads = (torch.autograd.grad(info[name], params, retain_graph=True, allow_unused=True)
                 if info[name].requires_grad else [None] * len(params))
        vectors[name] = torch.cat([(g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
                                   for p, g in zip(params, grads)])
    mse = vectors['objective']
    constraints = sum((v for k, v in vectors.items() if k != 'objective'), torch.zeros_like(mse))
    vectors['constraints'] = constraints
    norms = {k: v.norm().item() for k, v in vectors.items()}
    denominator = norms['objective'] * norms['constraints']
    return {'norms': norms,
            'constraint_to_mse_ratio': norms['constraints'] / norms['objective'] if norms['objective'] else None,
            'constraint_mse_cosine': (torch.dot(mse, constraints).item() / denominator) if denominator else None}


def data_splits(config, data_root, raw_root=None, manifest_path=None):
    dataset, identity = load_data(config, data_root, raw_root)
    splits, manifest = split_data(dataset, identity, config, manifest_path)
    return splits, manifest


def evaluate(model, dataset, device, batch_size=64, selection_mse="sdk_mse", ground_truth=False):
    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            # Preserve original CPU features for FP64 physics rather than rounding them.
            if ground_truth:
                predictions = {name: batch[name].y for name in ("bus", "generator")}
            else:
                predictions = model(batch.clone().to(device))
            accumulator.update(sample_metrics(predictions, batch))
    return accumulator.result(selection_mse)


def train(config, *, data_root, output, device="xpu:0", raw_root=None, split_file=None):
    """Train one T1 experiment. Paths/device are supplied by the caller, not YAML.

    Returns the JSON-serializable result. Refuses to reuse a nonempty output
    directory. This API is intended for future acopf orchestration.
    """
    config = copy.deepcopy(config)
    validate_config(config)
    if any(int(os.environ.get(name, "1")) > 1 for name in ("WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE")):
        raise ValueError("Use one process per independent experiment; DDP is not implemented here")
    device = select_device(device)
    output = Path(output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(int(config.get("seed", 42)))
    splits, manifest = data_splits(config, data_root, raw_root, split_file)
    write_json(output / "split.json", manifest)
    write_json(output / "config.json", config)
    write_json(output / "protocol.json", protocol(config))
    write_json(output / "environment.json", provenance(device))
    sample = splits["train"][0].clone().cpu()
    model = build_model(config, sample, device)
    loss_manager = OPFLossManager(loss_type=config.get('loss_type', 'mse'),
                                  lagrangian_config=config.get('lagrangian'), device=device)
    if loss_manager.lagrangian is not None:
        loss_manager.initialize_constraints(sample, device=device, dtype=torch.float32)
        # Preflight every selected graph before reusing fixed electrical constants.
        checked = {}
        for split, subset in splits.items():
            count = 0
            for batch in DataLoader(subset, batch_size=1024, shuffle=False,
                                    generator=torch.Generator().manual_seed(0)):
                count += loss_manager.lagrangian.physics.validate_batch(batch)
            checked[split] = count
        write_json(output / 'physics-preflight.json', {
            'checked_graphs': checked, **loss_manager.lagrangian.get_extra_state()})
        write_json(output / 'resolved_protocol.json', {
            **protocol(config), 'config': config, 'loss': loss_manager.lagrangian.get_extra_state(),
            'split_sha256': file_sha256(output / 'split.json'), 'environment': provenance(device)})
    # Persist the initialization to enable matched CPU/XPU trajectory comparisons.
    def checkpoint(path, samples, steps, validation):
        payload = {"format": "lumina-paper-experiment-v1", "config": config,
                   "example_graph": sample,
                   "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                   "samples_seen": samples, "steps": steps, "validation": validation,
                   "split_sha256": file_sha256(output / "split.json"),
                   "protocol": protocol(config)}
        if loss_manager.lagrangian is not None:
            payload['loss_state_dict'] = loss_manager.loss_state_dict()
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(path)
    checkpoint(output / "initial.pt", 0, 0, None)
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"]["AdamW"])
    training = config["training"]
    mse_epochs = training.get('loss_schedule', {}).get('mse_epochs', 0)
    mse_manager = OPFLossManager('mse', device=device) if mse_epochs else None
    batch_size = int(training["batch_size"])
    max_samples = int(training["max_samples"])
    max_steps = int(training.get("max_steps", 0))
    interval = int(training["validate_every_samples"])
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    selection = config.get("selection_mse", "sdk_mse")
    loader = DataLoader(splits["train"], batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(int(config.get("seed", 42))),
                        num_workers=0)
    samples = steps = epoch = 0
    best_score = float("inf")
    next_validation, last_validation = interval, None
    started = time.monotonic()
    with (output / "history.jsonl").open("w", buffering=1) as history:
        while samples < max_samples and (not max_steps or steps < max_steps):
            for batch_index, batch in enumerate(loader):
                # Advance the AL epoch only when the entire training split is consumed.
                epoch_complete = batch_index + 1 == len(loader)
                if batch.num_graphs > max_samples - samples:
                    batch = Batch.from_data_list(batch.to_data_list()[:max_samples - samples])
                    epoch_complete = False
                model.train()
                optimizer.zero_grad(set_to_none=True)
                batch = batch.to(device)
                for name in ("bus", "generator"):
                    batch[name].y = batch[name].y.float()
                predictions = model(batch)
                active_manager = mse_manager if epoch < mse_epochs else loss_manager
                loss, info = active_manager.compute_loss(predictions, batch, return_info=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                diagnostic_interval = training.get('gradient_diagnostics_every_steps', 0)
                diagnostic = (component_gradients(info, model) if diagnostic_interval and
                    (steps == 0 or (steps + 1) % diagnostic_interval == 0
                     or batch_index == 0) else None)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                    float(training.get("gradient_clip", 1.0)), error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite parameter after optimizer step")
                samples += batch.num_graphs
                steps += 1
                if active_manager.lagrangian is not None:
                    active_manager.on_successful_step(info['al_observation'],
                        successful_steps=int(active_manager.lagrangian.successful_steps.item()) + 1)
                    if epoch_complete:
                        active_manager.step_epoch()
                finished = samples >= max_samples or (max_steps and steps >= max_steps)
                row = {"samples": samples, "steps": steps, "epoch": epoch,
                       "train_sdk_mse": info['objective'].item(), "gradient_norm": grad_norm.item(),
                       "gradient_clipped": grad_norm.item() > float(training.get('gradient_clip', 1.0))}
                if diagnostic is not None:
                    row['gradient_components'] = diagnostic
                if loss_manager.lagrangian is not None:
                    al = loss_manager.lagrangian
                    row.update(train_loss=loss.item(), penalty_parameter=info.get('penalty_parameter', 0.),
                               next_penalty_parameter=al.mu_k, al_epoch=al.current_epoch,
                               al_successful_steps=int(al.successful_steps.item()),
                               loss_phase=('mse' if epoch < mse_epochs else
                                           'dual' if info['multipliers_active'] else 'penalty_only'),
                               multiplier_updated=al._last_multiplier_updated if epoch >= mse_epochs else False,
                               dual_updates=int(al.dual_updates.item()),
                               **{k: info[k].item() if k in info else 0. for k in
                                  ('linear_eq', 'linear_ineq', 'quadratic_eq', 'quadratic_ineq')})
                    if 'al_observation' in info:
                        observation = info['al_observation']
                        balance = observation['r'].norm(dim=1).mean().item()
                        thermal = observation['h'].clamp_min(0).norm(dim=1).mean().item()
                        row.update(balance_l2=balance, thermal_l2=thermal,
                                   violation=(balance + thermal) / al.physics.counts['bus'] ** 0.5,
                                   constraint_signal_norm=info['constraint_violation'].item())
                    for name in ('lambda_k', 'constraint_ema'):
                        value = getattr(al, name)
                        row[name] = {'min': value.min().item() if value.numel() else 0.,
                                     'max': value.max().item() if value.numel() else 0.,
                                     'norm': value.norm().item()}
                if samples >= next_validation or finished:
                    last_validation = evaluate(model, splits["val"], device, eval_batch_size, selection)
                    row["validation"] = last_validation
                    if last_validation["score"] < best_score:
                        best_score = last_validation["score"]
                        checkpoint(output / "best.pt", samples, steps, last_validation)
                    checkpoint(output / "last.pt", samples, steps, last_validation)
                    for requested in training.get('checkpoint_samples', []):
                        path = output / f'sample_{requested}.pt'
                        if samples >= requested and not path.exists():
                            checkpoint(path, samples, steps, last_validation)
                            shutil.copyfile(output / 'best.pt', output / f'best_through_{requested}.pt')
                    print(json.dumps(row), flush=True)
                    while next_validation <= samples:
                        next_validation += interval
                row["elapsed_seconds"] = time.monotonic() - started
                if loss_manager.lagrangian is not None:
                    row['samples_per_second'] = samples / row['elapsed_seconds']
                    row['host_peak_memory_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
                    row['device_peak_memory_bytes'] = (torch.xpu.max_memory_allocated(device) if device.type == 'xpu'
                        else torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
                history.write(json.dumps(row, allow_nan=False) + "\n")
                if finished:
                    break
            epoch += 1
    best = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"], strict=True)
    test = truth = None
    if config.get('evaluate_test', True):
        test = evaluate(model, splits["test"], device, eval_batch_size, selection)
        truth = evaluate(model, splits["test"], device, eval_batch_size, selection, ground_truth=True)
    result = {"status": "complete", "model_type": config["model_type"], "case": config["case"],
              "samples_seen": samples, "steps": steps, "best_samples_seen": best["samples_seen"],
              "validation": best["validation"], "test": test, "ground_truth_test": truth,
              "elapsed_seconds": time.monotonic() - started,
              "effective_split_sizes": {k: len(v) for k, v in splits.items()},
              "protocol": protocol(config), "device": str(device)}
    write_json(output / "results.json", result)
    print(json.dumps(result), flush=True)
    return result


def evaluate_checkpoint(checkpoint, *, data_root, device="cpu", case=None, groups=None,
                        split="test", split_file=None, batch_size=None):
    """Evaluate experiment checkpoints only; never silently accept SDK head differences."""
    path = Path(checkpoint).expanduser().resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "lumina-paper-experiment-v1":
        raise ValueError("Expected a lumina-paper-experiment-v1 checkpoint")
    config = copy.deepcopy(payload["config"])
    held_out = case is not None and parse_case_name(case) != parse_case_name(config["case"])
    if case is not None:
        config["case"] = case
    if groups is not None:
        config["groups"] = groups
    if not held_out:
        split_file = Path(split_file) if split_file is not None else path.parent / "split.json"
        if not split_file.is_file() or file_sha256(split_file) != payload["split_sha256"]:
            raise ValueError("Training split manifest is missing or its checksum changed")
    splits, manifest = data_splits(config, data_root, manifest_path=split_file)
    device = select_device(device)
    model = build_model(config, payload["example_graph"], device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    metrics = evaluate(model, splits[split], device,
                       batch_size or config["training"].get("eval_batch_size", 64),
                       config.get("selection_mse", "sdk_mse"))
    return {"checkpoint": str(path), "checkpoint_sha256": file_sha256(path),
            "device": str(device), "case": config["case"], "split": split,
            "metrics": metrics, "split_identity": manifest["identity"],
            "evaluated_indices_sha256": hashlib.sha256(
                json.dumps(splits[split].indices).encode()).hexdigest(),
            "environment": provenance(device), "protocol": protocol(config)}
