"""DD target policies, all benchmark backbones, and the 100-update curriculum."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from distillation.DD.campaign import (Schedule, default_config, inner, model_for, model_options,
    optimizer_for, outer_loss, proposal_step, student_seed, transplant, validate_config,
    export, replay_export, restore, final_evaluation)
from distillation.DD.mvp import SyntheticOPF
from distillation.synthetic.learning import benchmark_graph
from distillation.test.test_dd_mvp import SmallRegressor
from distillation.test import test_dd_demo as reference
from experiments.lumina_bench.models import SUPPORTED_MODELS

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def configuration(mode='solver'):
    return default_config(30, 'HeteroGNN') | dict(batches=1, batch_size=1, passes=2,
        synthetic_count=1, target_mode=mode, inner_rate=.003, rate_max=.2)


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_target_semantics_and_unrestricted_offsets(graph, mode):
    data = SyntheticOPF([graph], steps=1, target_mode=mode)
    with torch.no_grad():
        data.target_offsets[0]['bus'].fill_(5.)
        data.edits[0].delta['load'].fill_(1.)
    batch, target = data.batch(0)
    assert 'y' not in batch['bus']
    expected = graph['bus'].y.clone()
    if mode == 'learned':
        expected[:, 1] += 5.
        expected[:, 0] += 5. * (data.anchors[0].params.bus_type != 3)
    torch.testing.assert_close(target['bus'], expected)
    assert target['bus'].requires_grad == (mode == 'learned')


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_gate_uses_correct_targets_and_audit_order(graph, mode):
    cfg = configuration(mode)
    data = Schedule([graph], cfg)
    opt = optimizer_for(data, cfg)
    for p in data.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    solved = graph.clone()
    solved['bus'].y += .1
    calls = []
    def audit():
        calls.append('audit')
        return True, [solved], [dict(success=True, converged=True)]
    def score():
        calls.append('score')
        if mode == 'solver':
            # A stale-label pre-screen would incorrectly reject this proposal.
            return 0. if torch.equal(data.seed_targets[0]['bus'], solved['bus'].y) else 10.
        return 0.
    accepted, _, _, _ = proposal_step(data, opt, cfg, 1., score, audit)
    assert accepted
    assert calls == (['audit', 'score'] if mode == 'solver' else ['score', 'audit'])
    torch.testing.assert_close(data.seed_targets[0]['bus'], solved['bus'].y if mode == 'solver' else graph['bus'].y)


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_gate_restores_targets_parameters_and_adam_on_rejection(graph, mode):
    cfg = configuration(mode)
    data = Schedule([graph], cfg)
    opt = optimizer_for(data, cfg)
    before = deepcopy(data.state_dict())
    labels = deepcopy(data.seed_targets)
    for p in data.parameters():
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    solved = graph.clone()
    solved['bus'].y += 1.
    accepted, _, _, _ = proposal_step(data, opt, cfg, 0., lambda: 10.,
                                      lambda: (True, [solved], [dict(success=True)]))
    assert not accepted and not opt.state
    for key, value in before.items():
        torch.testing.assert_close(data.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(data.seed_targets[0]['bus'], labels[0]['bus'], rtol=0, atol=0)


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
def test_all_models_have_finite_dd_meta_gradients(graph, kind):
    cfg = configuration('learned') | dict(model=kind, model_options=model_options(kind, width=8),
                                         inner_rate=1e-7)
    data = Schedule([graph], cfg)
    model = model_for(graph, cfg, 123, 'cpu')
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    loss = outer_loss(model, data, real, create_graph=True)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in data.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert data.rate_logits.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in data.target_offsets.parameters()) > 0


def test_learned_target_meta_gradient_matches_finite_difference(graph):
    data = Schedule([graph], configuration('learned'))
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    model = SmallRegressor()
    p = data.target_offsets[0]['bus']
    exact = torch.autograd.grad(outer_loss(model, data, real, create_graph=True), p)[0][0, 1].item()
    values = []
    for value in (1e-4, -1e-4):
        with torch.no_grad():
            p[0, 1] = value
        values.append(float(outer_loss(model, data, real, create_graph=False).detach()))
    assert exact == pytest.approx((values[0] - values[1]) / 2e-4, rel=1e-5, abs=1e-10)


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
def test_checkpointed_inner_preserves_loss_and_all_meta_gradients(graph, kind):
    cfg = configuration('learned') | dict(model=kind, model_options=model_options(kind, width=8),
        inner_rate=1e-7, batches=2, synthetic_count=2, checkpoint_inner=True)
    data = Schedule([graph, graph], cfg)
    model = model_for(graph, cfg, 123, 'cpu')
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    parameters = tuple(p for p in data.parameters() if p.requires_grad)
    records = []
    for enabled in (False, True):
        data.checkpoint_inner = enabled
        loss = outer_loss(model, data, real, create_graph=True)
        gradients = torch.autograd.grad(loss, parameters)
        records.append((loss.detach(), tuple(g.detach() for g in gradients)))
    torch.testing.assert_close(records[0], records[1], rtol=1e-9, atol=1e-12)


def test_topology_replacement_preserves_refreshed_unchanged_targets(graph):
    cfg = configuration() | dict(batch_size=2, synthetic_count=2)
    data = Schedule([graph, graph], cfg)
    data.seed_targets[0]['bus'].add_(.1)
    new, _ = transplant(data, [graph, graph], cfg, {1})
    torch.testing.assert_close(new.seed_targets[0]['bus'], data.seed_targets[0]['bus'])


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_modes_resume_and_export_exact_curriculum(graph, tmp_path, mode):
    cfg = configuration(mode)
    data = Schedule([graph], cfg)
    opt = optimizer_for(data, cfg)
    model = SmallRegressor()
    real = benchmark_graph(graph, labels=True)
    batch = Batch.from_data_list([real])
    outer_loss(model, data, batch, create_graph=True).backward()
    opt.step()
    payload = deepcopy(dict(anchor_graphs=[graph], config=cfg, seed_targets=data.seed_targets,
        synthetic_state=data.state_dict(), optimizer=opt.state_dict(),
        rng_torch=torch.get_rng_state(), rng_numpy=np.random.get_state()))
    resumed, resumed_opt = restore(payload)
    for module, optimizer in ((data, opt), (resumed, resumed_opt)):
        optimizer.zero_grad(set_to_none=True)
        outer_loss(model, module, batch, create_graph=True).backward()
        optimizer.step()
    for name, value in data.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    export(tmp_path, data, cfg, {}, [graph], [], {})
    artifact = torch.load(tmp_path / 'distilled.pt', weights_only=False)
    assert artifact['target_mode'] == mode
    expected = float(outer_loss(model, data, batch, create_graph=False).detach())
    assert replay_export(model, artifact, [real], [0], 1) == pytest.approx(expected, abs=1e-14)


def test_default_curriculum_and_seed_regimes(graph):
    cfg = default_config(30, 'GIN')
    validate_config(cfg)
    data = Schedule([graph] * 100, cfg)
    steps = []
    inner(SmallRegressor(), data, create_graph=False, observe=lambda step, *_: steps.append(step))
    assert steps == list(range(101)) and len(data.rates) == 100
    assert student_seed(cfg, 2, 1, 12) != student_seed(cfg, 1, 1, 12)
    fixed = cfg | {'initialization': 'fixed'}
    assert student_seed(fixed, 2, 1, 12) == student_seed(fixed, 1, 0, 12)
    with pytest.raises(ValueError, match='correspondence'):
        validate_config(cfg | {'target_mode': 'frozen', 'topology_every': 1})


def test_final_evaluation_resumes_fixed_initialization_and_real_control(graph, tmp_path):
    class Solo:
        rank, size = 0, 1
        def __init__(self, stop=False):
            self.stop, self.checks = stop, 0
        def allgather(self, value):
            if isinstance(value, bool):
                self.checks += 1
                return [value and not (self.stop and self.checks > 1)]
            return [value]
        def Barrier(self):
            pass
    cfg = configuration('learned') | dict(initialization='fixed', model_options=model_options('HeteroGNN', width=8),
        inner_rate=1e-7, validation_samples=2, final_samples=2, trace_every=1, trace_samples=1,
        calibration_samples=4, real_control_rates=[1e-7, 1e-6])
    data = Schedule([graph], cfg)
    state = dict(best_iteration=0, iteration=0, seen_ids=set(), seed_ids=[0])
    export(tmp_path, data, cfg, state, [graph], [], {}, name='best.pt')
    export(tmp_path, data, cfg, state, [graph], [], {}, name='distilled.pt')
    torch.save({'graphs': [graph]}, tmp_path / 'initial.pt')
    dataset = [benchmark_graph(graph, labels=True)] * 24
    split = {'train': np.arange(20), 'val': np.arange(20, 24)}
    assert not final_evaluation(tmp_path, data, [graph], cfg, dataset, split, state, Solo(stop=True), 'cpu')
    partial = json.loads((tmp_path / 'evaluation.partial.json').read_text())
    assert 'best_distilled' in partial and 'real_subset_tuned_rates' not in partial
    assert final_evaluation(tmp_path, data, [graph], cfg, dataset, split, state, Solo(), 'cpu')
    report = json.loads((tmp_path / 'evaluation.json').read_text())
    assert report['best_distilled'] == partial['best_distilled']
    assert 'best_distilled_unseen' in report and 'real_subset_tuned_unseen' in report
    assert set(report['real_control_tuning']['training_ids']) <= set(split['train'])
