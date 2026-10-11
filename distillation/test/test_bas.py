"""Numerical, access-boundary, selection and resume qualification for BAS."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from distillation.selection import Buckets, TrainingPool, allocate, input_view, selected_dataset
from distillation.BAS import run as runner
from distillation.BAS.models import make_model, model_options, output_weights, SUPPORTED_MODELS
from distillation.BAS.schedule import Patience
from distillation.BAS.scores import acquisition_scores, per_graph_loss
from distillation.training.T1_single_topo.data import CaseDataset, prepare_group, finish_case
from distillation.training.T1_single_topo.tests.test_t1 import archive, CASE, assert_same
from experiments.lumina_bench.tests.test_protocol import graph


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def examples():
    result = [graph() for _ in range(3)]
    for i, g in enumerate(result):
        g['load'].x *= 1 + i * .1
    return result


def test_proportional_rounding_and_capacity_conservation():
    np.testing.assert_equal(allocate([1, 2, 3], 11, [1, 10, 20]), [1, 4, 6])
    np.testing.assert_equal(allocate([0, 0, 0], 5, [0, 10, 10]), [0, 3, 2])
    rng = np.random.default_rng(42)
    for _ in range(100):
        capacities = rng.integers(0, 20, 7)
        count = int(rng.integers(0, 150))
        allocation = allocate(rng.random(7), count, capacities)
        assert allocation.sum() == min(count, capacities.sum())
        assert ((allocation >= 0) & (allocation <= capacities)).all()


@pytest.mark.parametrize('strategy', ['quantile', 'width'])
def test_bucket_degeneracy_boundaries_and_no_empty_panel_buckets(strategy):
    values = np.r_[np.zeros(20), 1, 100]
    buckets = Buckets.fit(values, 30, strategy)
    memberships = buckets.assign(values)
    assert np.bincount(memberships, minlength=buckets.count).min() > 0
    assert Buckets.fit(np.ones(100), 30, strategy).count == 1
    selected, quotas = buckets.sample(np.arange(len(values)), memberships, np.ones(buckets.count),
                                      10, np.random.default_rng(42))
    assert len(set(selected)) == 10 and quotas.min() > 0


def test_train_only_pool_and_unrevealed_candidate_labels():
    dataset = [graph() for _ in range(12)]
    splits = dict(train=np.arange(8), val=np.arange(8, 10), test=np.arange(10, 12))
    pool = TrainingPool(dataset, splits, [0, 1])
    for bad in ([8], [10], [2, 2]):
        with pytest.raises(ValueError):
            pool.inputs(bad)
    with pytest.raises(ValueError, match='before acquisition'):
        pool.labeled([2])
    clean = pool.inputs([2])[0]
    assert all('y' not in store and 'objective' not in store for store in clean.stores)
    assert 'y' in dataset[2]['bus']  # views cannot mutate cached source tensors
    pool.acquire([2, 3], reason='test')
    for ids in ([0], [2], [9]):
        with pytest.raises(ValueError):
            pool.acquire(ids, reason='invalid')
    assert len(pool.labeled([2])) == 1
    restored = TrainingPool(dataset, splits, [0, 1])
    restored.load_state_dict(pool.state_dict())
    assert_same(restored.state_dict(), pool.state_dict())


def test_patience_thresholds_and_cumulative_second_counter():
    schedule = Patience(patience_lr=1, patience_acquire=2)
    assert not schedule.step(5.)  # rho2=0 is not a new acquisition event
    assert not schedule.step(5.)
    assert not schedule.step(4.)
    assert schedule.rho2 == 1  # Improvement resets only rho1 in Algorithm 4.
    assert not schedule.step(4.)
    assert schedule.step(4.)
    assert (schedule.rho1, schedule.rho2, schedule.lr) == (0, 0, .001)
    for _ in range(10):
        assert not schedule.step(4., can_acquire=False)
    assert schedule.lr < .001  # No boost when the data budget is exhausted.


@pytest.mark.parametrize('kind', ['MLP', *SUPPORTED_MODELS])
def test_labeled_scores_match_individual_graphs_for_all_models(kind):
    graphs = examples()
    config = runner.configuration(model=kind, model_options=model_options(kind, width=8, layers=2))
    model = make_model(config, graphs[0], torch.device('cpu'))
    batch = Batch.from_data_list(graphs)
    for metric in ('le', 'ig', 'lg'):
        scores = acquisition_scores(model, batch, metric)
        individual = np.concatenate([acquisition_scores(model, Batch.from_data_list([g]), metric) for g in graphs])
        np.testing.assert_allclose(scores, individual, rtol=5e-4, atol=5e-5)
        assert np.isfinite(scores).all() and (scores >= 0).all()
    assert all(p.grad is None for p in model.parameters())


def test_input_gradient_against_finite_difference():
    config = runner.configuration(model='MLP', model_options=dict(layers=1, width=8, dropout=0.))
    model = make_model(config, graph(), torch.device('cpu')).eval()
    batch = Batch.from_data_list([graph()])
    score = acquisition_scores(model, batch, 'ig')[0]
    derivatives = []
    for column in range(2):
        low, high = batch.clone(), batch.clone()
        low['load'].x[0, column] -= .001
        high['load'].x[0, column] += .001
        derivatives.append(float((per_graph_loss(model(input_view(high)), high) -
                                  per_graph_loss(model(input_view(low)), low)).detach()) / .002)
    assert score == pytest.approx(np.linalg.norm(derivatives), rel=.005, abs=.0002)


def test_last_layer_norm_is_per_example_not_cancelling_batch_gradient():
    config = runner.configuration(model='MLP')
    model = make_model(config, graph(), torch.device('cpu')).eval()
    a, b = graph(), graph()
    with torch.no_grad():
        predictions = model(Batch.from_data_list([a]))
        for kind in ('bus', 'generator'):
            a[kind].y, b[kind].y = predictions[kind] + 1, predictions[kind] - 1
    batch = Batch.from_data_list([a, b])
    scores = acquisition_scores(model, batch, 'lg')
    gradients = torch.autograd.grad(per_graph_loss(model(input_view(batch)), batch).mean(), output_weights(model))
    assert sum(g.square().sum() for g in gradients) < 1e-10
    assert (scores > 1).all()


@pytest.mark.parametrize('metric', ['mcv_p', 'mcv_l'])
def test_mc_is_stochastic_but_restores_model_flags_and_training_rng(metric):
    model = make_model(runner.configuration(model='MLP'), graph(), torch.device('cpu'))
    model.train()
    batch = Batch.from_data_list(examples())
    rng = torch.get_rng_state().clone()
    buffers = {k: v.clone() for k, v in model.named_buffers()}
    first = acquisition_scores(model, batch, metric, draws=8, seed=100)
    second = acquisition_scores(model, batch, metric, draws=8, seed=100)
    np.testing.assert_array_equal(first, second)
    assert (first > 0).all() and model.training
    assert torch.equal(rng, torch.get_rng_state())
    assert_same(buffers, dict(model.named_buffers()))


def test_no_dropout_is_not_silently_reported_as_uncertainty():
    config = runner.configuration(model='MLP', model_options=dict(layers=1, dropout=0.))
    model = make_model(config, graph(), torch.device('cpu'))
    with pytest.raises(ValueError, match='dropout'):
        acquisition_scores(model, Batch.from_data_list([graph()]), 'mcv_p')


@pytest.fixture
def tiny_case(tmp_path):
    for group in range(2):
        archive(tmp_path / 'raw', group, reversed(range(group * 20, (group + 1) * 20)))
        prepare_group(tmp_path / 'raw', tmp_path / 'data', CASE, group, 20)
    finish_case(tmp_path / 'data', CASE, groups=2, group_size=20)
    return runner.configuration(data_root=str(tmp_path / 'data'), case=CASE, model='MLP',
         model_options=dict(layers=1, width=8, dropout=.1), buckets=2, panel_size=8,
         initial_size=4, max_selected=12, acquisition_size=4, candidate_size=16, mc_draws=3,
         batch_size=3, score_batch_size=4, max_rounds=5, validation_size=4,
         schedule=dict(minimum=1e-6, maximum=.01, decrease=.5, increase=2., patience_lr=2, patience_acquire=0))


def test_resume_across_epochs_and_acquisitions_preserves_every_training_state(tiny_case, tmp_path, monkeypatch):
    original = runner.score_ids
    def constant_panel(model, pool, ids, metric, config, **kwargs):
        # Deterministic plateau triggers real IG acquisitions, even if training improves.
        return np.ones(len(ids)) if metric == 'le' else original(model, pool, ids, metric, config, **kwargs)
    monkeypatch.setattr(runner, 'score_ids', constant_panel)
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    assert runner.run(tiny_case, direct)['status'] == 'complete'
    assert runner.run(tiny_case, resumed, session_steps=5)['status'] == 'paused'
    assert runner.run(tiny_case, resumed)['status'] == 'complete'
    a, b = [torch.load(p / 'latest.pt', weights_only=False) for p in (direct, resumed)]
    for key in ('model', 'optimizer', 'rng', 'pool', 'schedule'):
        assert_same(a[key], b[key])
    for saved in (a, b):
        saved['state'].pop('seconds')
        saved['state'].pop('wall_seconds')
    assert_same(a['state'], b['state'])
    assert len(a['pool']['selected_ids']) == 12
    assert len(a['pool']['history']) == 3
    dataset = CaseDataset(tiny_case['data_root'], tiny_case['case'])
    selected = selected_dataset(dataset, resumed / 'selection.json')
    assert len(selected) == 12
    assert set(selected.indices) <= set(dataset.splits()['train'])
    assert not set(selected.indices) & set(a['pool']['reference_ids'])
    with pytest.raises(ValueError, match='changed'):
        runner.run(dict(tiny_case, model_seed=100), resumed)


@pytest.mark.parametrize('method', ['bas', 'random', 'stratified', 'mcdue', 'inactive'])
def test_controls_and_input_index_do_not_read_test_records(tiny_case, tmp_path, monkeypatch, method):
    config = dict(tiny_case, method=method, max_rounds=2)
    dataset = CaseDataset(config['data_root'], config['case'])
    test_ids = set(dataset.splits()['test'])
    original = CaseDataset.__getitem__
    def guarded(self, index):
        assert int(index) not in test_ids
        return original(self, index)
    monkeypatch.setattr(CaseDataset, '__getitem__', guarded)
    original_score = runner.score_ids
    def plateau(model, pool, ids, metric, config, **kwargs):
        return np.ones(len(ids)) if metric == 'le' else original_score(model, pool, ids, metric, config, **kwargs)
    monkeypatch.setattr(runner, 'score_ids', plateau)
    # Three rounds provide one acquisition followed by actual training on it.
    config['max_rounds'] = 3
    folder = tmp_path / method
    assert runner.run(config, folder, dataset=dataset)['status'] == 'complete'
    selected = json.loads((folder / 'selection.json').read_text())
    assert selected['test_used'] is False and selected['solver_calls'] == 0
    assert selected['labeled_count'] == selected['selected_count'] + 8
    cache = torch.load(tmp_path / f'{CASE}_train_inputs.pt', weights_only=False)
    assert set(cache['ids']) == set(dataset.splits()['train'])
    expected = [dataset[int(i)]['load'].x[:, 0].sum().item() for i in cache['ids']]
    np.testing.assert_allclose(cache['values'], expected)


def test_fresh_student_and_matched_random_control_resume(tiny_case, tmp_path):
    from distillation.BAS.aux.transfer import train_subset
    selected = tmp_path / 'selected'
    runner.run(dict(tiny_case, max_rounds=1), selected)
    for random_control in (False, True):
        folder = tmp_path / f'transfer_{random_control}'
        options = dict(model='MLP', seed=123, epochs=2, random_control=random_control)
        assert train_subset(selected / 'selection.json', folder, session_steps=1, **options)['status'] == 'paused'
        result = train_subset(selected / 'selection.json', folder, **options)
        assert result['status'] == 'complete' and result['selected'] == 4 and not result['test_used']
        ids = json.loads((folder / 'config.json').read_text())['train_ids']
        manifest = json.loads((selected / 'selection.json').read_text())
        assert not set(ids) & set(manifest['pool']['reference_ids'])
