"""Allocated-XPU gates: physical residual autograd and actual benchmark models."""
import copy

import pytest
import torch
from torch_geometric.data import Batch

from experiments.lumina_bench.tests.test_protocol import graph, predictions
from experiments.lumina_bench.runner import build_model, component_gradients
from lumina.model.opf.losses import OPFLossManager

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason='Allocated Intel XPU required')


def close(actual, reference, atol=1e-5, rtol=1e-4):
    torch.testing.assert_close(actual.detach().cpu(), reference.detach().cpu(), atol=atol, rtol=rtol)


def test_xpu_residuals_loss_duals_and_prediction_gradients():
    batch = Batch.from_data_list([graph(), graph()])
    pred = {k: v.float() for k, v in predictions(batch).items()}
    pred['bus'][1, 0] = .02
    outputs = []
    for device in ('cpu', 'xpu:0'):
        manager = OPFLossManager('augmented_lagrangian', lagrangian_config={'mu_0': .01, 'ema_beta': 0.})
        manager.initialize_constraints(graph(), device=device)
        b = batch.clone().to(device)
        for name in ('bus', 'generator'): b[name].y = b[name].y.float()
        p = {k: v.clone().to(device).requires_grad_() for k, v in pred.items()}
        value, info = manager.compute_loss(p, b)
        value.backward()
        manager.on_successful_step(info['al_observation'], 1)
        outputs.append((value, info, manager, p))
    ref, actual = outputs
    close(actual[0], ref[0])
    for key in ('r', 'h'): close(actual[1]['al_observation'][key], ref[1]['al_observation'][key])
    for key in ('lambda_k', 'constraint_ema'): close(getattr(actual[2].lagrangian, key), getattr(ref[2].lagrangian, key))
    for key in ('bus', 'generator'): close(actual[3][key].grad, ref[3][key].grad)


@pytest.mark.parametrize('kind', ['HGT', 'Transformer', 'RGAT'])
def test_xpu_benchmark_models_al_multiple_steps(kind):
    torch.manual_seed(32)
    config = {'model_type': kind, 'models': {
        'HGT': {'hidden_channels': 16, 'num_layers': 2, 'num_heads': 2, 'dropout': 0.},
        'RGAT': {'hidden_channels': 16, 'num_layers': 2, 'num_heads': 2},
        'Transformer': {'hidden_dim': 16, 'num_layers': 2, 'dropout': 0.}}}
    cpu = build_model(config, graph(), torch.device('cpu'))
    xpu = copy.deepcopy(cpu).to('xpu:0')
    batch = Batch.from_data_list([graph(), graph()])
    batch['load'].x[1, 0] += .1
    managers = [OPFLossManager('augmented_lagrangian', lagrangian_config={
        'mu_0': .001, 'ema_beta': 0., 'warmup_epochs': 1}) for _ in range(2)]
    devices = [torch.device('cpu'), torch.device('xpu:0')]
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (cpu, xpu)]
    for mgr, device in zip(managers, devices): mgr.initialize_constraints(graph(), device=device)
    for step in range(1, 4):
        outputs = []
        for model, mgr, device, opt in zip((cpu, xpu), managers, devices, optimizers):
            b = batch.clone().to(device)
            for name in ('bus', 'generator'): b[name].y = b[name].y.float()
            opt.zero_grad(); pred = model(b); value, info = mgr.compute_loss(pred, b)
            if step == 3:
                diagnostics = component_gradients(info, model)
                assert all(torch.isfinite(torch.tensor(v)) for v in diagnostics['norms'].values())
                assert all(p.grad is None for p in model.parameters())
            value.backward()
            outputs.append((value, info))
        # Compare gradients at identical initial weights; later AdamW trajectories can
        # diverge at near-zero attention gradients, already documented in XPU tests.
        if step == 1:
            close(outputs[1][0], outputs[0][0], atol=2e-5, rtol=2e-4)
            for (name, ref), (_, actual) in zip(cpu.named_parameters(), xpu.named_parameters()):
                if ref.grad is not None: close(actual.grad, ref.grad, atol=2e-5, rtol=2e-3)
        for model, mgr, opt, (_, info) in zip((cpu, xpu), managers, optimizers, outputs):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); mgr.on_successful_step(info['al_observation'], step)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert torch.isfinite(mgr.lagrangian.lambda_k).all()
            assert torch.isfinite(mgr.lagrangian.constraint_ema).all()
            assert mgr.lagrangian.dual_updates.item() == max(0, step - 1)
            if step == 1: mgr.step_epoch()
    assert managers[1].lagrangian.dual_updates.item() == 2
