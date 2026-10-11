"""Meaningful resume, scheduling, and topology-state regression checks."""
from copy import deepcopy

import numpy as np
import pytest
import torch

from distillation.DD.campaign import (
    Schedule, default_config, export, inner, optimizer_for, outer_loss, replay_export, restore,
    sampler_ids, score_summary, transplant,
)
from distillation.DD.demo import benchmark_graph
from distillation.DD.replay_campaign import fit_schedule
from distillation.DD.mvp import mse
from distillation.test.test_dd_mvp import SmallRegressor
from distillation.test import test_dd_demo as reference
from torch_geometric.data import Batch

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def config():
    return default_config(30, 'HeteroGNN') | {'batches': 2, 'batch_size': 1, 'synthetic_count': 2,
        'passes': 2, 'inner_rate': 0.03, 'rate_min': 1e-6, 'rate_max': 0.2}


def test_sampler_covers_population_and_resumes_without_overlap():
    population = np.arange(100, 160)
    ids = [sampler_ids(population, step, rank, 3, 5) for step in range(4) for rank in range(3)]
    assert sorted(x for batch in ids for x in batch) == population.tolist()
    assert sampler_ids(population, 2, 1, 3, 5) == ids[7]
    assert sampler_ids(population, 0, 0, 3, 5) != sampler_ids(population, 4, 0, 3, 5)


def test_nonfinite_initialization_is_reported_not_silently_dropped():
    result = score_summary([1.0, float('inf')])
    assert result['mean'] is None and result['std'] is None
    assert result['nonfinite_initializations'] == 1
    assert result['per_initialization'] == [1.0, None]


def test_multipass_rate_meta_gradient_matches_finite_difference(graph):
    synthetic = Schedule([graph] * 2, config())
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    model = SmallRegressor()
    loss = outer_loss(model, synthetic, real, create_graph=True)
    exact = torch.autograd.grad(loss, synthetic.rate_logits)[0][3].item()
    values = []
    epsilon = 1e-4
    before = synthetic.rate_logits.detach().clone()
    for sign in (1, -1):
        with torch.no_grad():
            synthetic.rate_logits.copy_(before)
            synthetic.rate_logits[3] += sign * epsilon
        values.append(float(outer_loss(model, synthetic, real, create_graph=False).detach()))
    assert abs(exact) > 1e-10
    assert exact == pytest.approx((values[0] - values[1]) / (2 * epsilon), rel=2e-5, abs=1e-10)
    assert len(synthetic.rates) == 4


def test_resume_preserves_next_adam_update_exactly(graph):
    cfg = config()
    synthetic = Schedule([graph] * 2, cfg)
    optimizer = optimizer_for(synthetic, cfg)
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    model = SmallRegressor()
    outer_loss(model, synthetic, real, create_graph=True).backward()
    optimizer.step()
    payload = deepcopy({'anchor_graphs': [graph] * 2, 'config': cfg,
        'seed_targets': synthetic.seed_targets, 'synthetic_state': synthetic.state_dict(),
        'optimizer': optimizer.state_dict(), 'rng_torch': torch.get_rng_state(), 'rng_numpy': np.random.get_state()})
    resumed, resumed_optimizer = restore(payload)
    for module, opt in ((synthetic, optimizer), (resumed, resumed_optimizer)):
        opt.zero_grad(set_to_none=True)
        outer_loss(model, module, real, create_graph=True).backward()
        opt.step()
    for name, value in synthetic.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)


def test_topology_rebase_retains_other_parameters_and_moments(graph):
    cfg = config()
    synthetic = Schedule([graph] * 2, cfg)
    optimizer = optimizer_for(synthetic, cfg)
    for p in synthetic.parameters():
        p.grad = torch.ones_like(p)
    optimizer.step()
    replacement, opt = transplant(synthetic, [graph] * 2, cfg, {1}, optimizer)
    old = dict(synthetic.named_parameters())
    for name, p in replacement.named_parameters():
        if name.startswith(('edits.1.', 'target_offsets.1.')):
            assert p not in opt.state
            assert torch.count_nonzero(p) == 0
        else:
            torch.testing.assert_close(p, old[name], rtol=0, atol=0)
            torch.testing.assert_close(opt.state[p]['exp_avg'], optimizer.state[old[name]]['exp_avg'], rtol=0, atol=0)


def test_inner_resets_model_and_does_not_modify_weights(graph):
    model = SmallRegressor()
    synthetic = Schedule([graph] * 2, config())
    before = deepcopy(model.state_dict())
    a, _ = inner(model, synthetic, create_graph=False)
    b, _ = inner(model, synthetic, create_graph=False)
    for name in before:
        torch.testing.assert_close(before[name], model.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)


def test_export_replays_exact_training_schedule(graph, tmp_path):
    cfg = config()
    synthetic = Schedule([graph] * 2, cfg)
    model = SmallRegressor()
    with torch.no_grad():
        synthetic.rate_logits.add_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        synthetic.target_offsets[0]['generator'].fill_(0.2)
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    expected = float(outer_loss(model, synthetic, real, create_graph=False).detach())
    export(tmp_path, synthetic, cfg, {}, [graph] * 2, [], {})
    artifact = torch.load(tmp_path / 'distilled.pt', weights_only=False)
    value = replay_export(model, artifact, [benchmark_graph(graph, labels=True)], [0], 1)
    assert value == pytest.approx(expected, abs=1e-14)
    assert [x['batch'] for x in artifact['schedule']] == [0, 1, 0, 1]
    trained = fit_schedule(model, artifact)
    value = float(mse(trained(real), {k: real[k].y for k in ('bus', 'generator')}, real).detach())
    assert value == pytest.approx(expected, abs=1e-14)
