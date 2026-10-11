"""Pinned-upstream objective/policy parity and integration regression gates."""
import copy
import importlib.util
import subprocess
from dataclasses import asdict

import pytest
import torch
from torch_geometric.data import Batch

from experiments.lumina_bench.tests.test_protocol import graph, predictions
from lumina.model.opf.augmented_lagrangian import ALConfig, AugmentedLagrangianLoss, al_terms, FORMULA
from lumina.model.opf.losses import OPFLossManager


def tensor(value):
    return torch.tensor(value, dtype=torch.float64)


def make_loss(**config):
    loss = AugmentedLagrangianLoss({'mu_0': .2, 'ema_beta': 0., **config})
    loss.initialize_constraints(graph(), dtype=torch.float64)
    return loss


def snapshot_equal(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if torch.is_tensor(a[key]): torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
        else: assert a[key] == b[key]


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / 'upstream_al.py'
    path.write_bytes(subprocess.check_output(['git', 'show',
        'f53a7a35ec362121ae39b1c9781bd66b5b9e1b5d:lumina/model/opf/augmented_lagrangian.py']))
    spec = importlib.util.spec_from_file_location('upstream_reference', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.AugmentedLagrangianACOPF


def test_independent_objective_and_feasible_thermal_gradient():
    total, terms = al_terms(tensor(.3), tensor([1., 2., 3.]), tensor([-.5, -1., -2.]), .2, 2)
    assert total.item() == pytest.approx(10.2, abs=1e-12)
    assert terms['linear_eq'] == 2.5 and terms['linear_ineq'] == 6
    loss = make_loss()
    r = tensor([[3., 4., 0., 0.], [-3., 4., 0., 0.]])
    h = tensor([[-9., 4.], [-1., 9.]]).requires_grad_()
    c = loss.constraint_vector(r, h)
    torch.testing.assert_close(c, torch.cat((torch.sqrt(tensor([9., 16., 0., 0.])+1e-8)/2, tensor([2.5]))))
    for value in (-1., 0.):
        # Exact feasibility and a strictly feasible line produce finite zero gradients.
        h = tensor([[value, value]]).requires_grad_()
        c = loss.constraint_vector(tensor([[0., 0., 0., 0.]]), h)
        total, terms = al_terms(tensor(0.), c, tensor([0., 0., 0., 0., -1.]), .2, 4)
        assert terms['linear_ineq'] == 0 and terms['quadratic_ineq'] == 0
        torch.testing.assert_close(torch.autograd.grad(total, h)[0], torch.zeros_like(h))


@pytest.mark.parametrize('beta,rms,size', [(0., False, True), (.4, False, True), (.6, True, False), (1., False, False)])
def test_complete_objective_gradients_and_policy_match_upstream(reference, beta, rms, size):
    from tests.model.opf.test_lagrangian_physics import fixture
    g = fixture(True)
    cfg = {'mu_0': .2, 'ema_beta': beta, 'normalize_by_rms': rms, 'normalize_by_size': size,
           'warmup_epochs': 1, 'penalty_check_interval': 1, 'penalty_drop_ratio': .9,
           'penalty_increase_factor': 1.5, 'max_penalty': .8, 'multiplier_clip': .3,
           'multiplier_improve_ratio': .9, 'multiplier_check_interval': 3}
    loss = AugmentedLagrangianLoss(cfg); loss.initialize_constraints(g, dtype=torch.float64)
    physics = loss.physics
    src = reference(**cfg)
    kwargs = {'Y_real_sparse': physics.y_real, 'Y_imag_sparse': physics.y_imag,
              'line_edge_index': physics.edge, 'line_limits': physics.ratings}
    for name in ('ff', 'ft', 'tf', 'tt'):
        value = getattr(physics, 'y' + name)
        kwargs['line_y_' + name + '_real'] = value.real
        kwargs['line_y_' + name + '_imag'] = value.imag
    src.set_network_parameters(**kwargs)
    src.lambda_k = torch.zeros_like(loss.lambda_k); src.constraint_ema = torch.zeros_like(loss.lambda_k)
    for step in range(1, 17):
        batch = Batch.from_data_list([g, g])
        pred = predictions(batch); pred['bus'][:, 0] = tensor([.1, -.1, .09, -.12]) / (1 + .08 * step)
        pred['generator'] += .03 * (step % 3)
        pred = {k: v.requires_grad_() for k, v in pred.items()}
        expected_pred = {k: v.detach().clone().reshape(2, -1, 2).requires_grad_() for k, v in pred.items()}
        demand = batch['load'].x.reshape(2, 1, 2)
        data = {'pd': demand[..., 0], 'qd': demand[..., 1], 'gen_bus_indices': physics.links['generator'][1],
                'load_bus_indices': physics.links['load'][1], 'line_edge_index': physics.edge}
        before = loss.training_state_dict()
        actual, info = loss(tensor(.3), pred, batch)
        snapshot_equal(before, loss.training_state_dict())
        expected, source_info = src(tensor(.3), expected_pred, data)
        torch.testing.assert_close(info['al_observation']['c'], source_info['constraints'], atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        ga = torch.autograd.grad(actual, tuple(pred.values()))
        ge = torch.autograd.grad(expected, tuple(expected_pred.values()))
        for a, e in zip(ga, ge): torch.testing.assert_close(a, e.reshape_as(a), atol=1e-11, rtol=1e-9)
        src.update_multipliers(source_info['constraints'])
        changed = loss.on_successful_step(info['al_observation'], step)
        assert changed == src._last_multiplier_updated
        torch.testing.assert_close(loss.lambda_k, src.lambda_k, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(loss.constraint_ema, src.constraint_ema, atol=1e-12, rtol=1e-10)
        if step % 3 == 0:
            loss.step_epoch(); src.step_epoch()
        assert loss.mu_k == src.mu_k
        assert loss._schedule.training_state_dict() == src._schedule.training_state_dict()
        assert loss._last_multiplier_violation == pytest.approx(src._last_multiplier_violation) if src._last_multiplier_violation is not None else loss._last_multiplier_violation is None
        restored = AugmentedLagrangianLoss(cfg); restored.initialize_constraints(g, dtype=torch.float64)
        restored.load_state_dict(loss.training_state_dict())
        snapshot_equal(loss.training_state_dict(), restored.training_state_dict())


@pytest.mark.parametrize('k,ratio', [(1, .9), (3, .9), (3, 0.)])
def test_gate_intervals_clipping_and_stagnation_match_upstream(reference, k, ratio):
    cfg = {'multiplier_improve_ratio': ratio, 'multiplier_check_interval': k, 'multiplier_clip': .25}
    loss = make_loss(**cfg); src = reference(mu_0=.2, ema_beta=0., **cfg)
    src.lambda_k = torch.zeros_like(loss.lambda_k); src.constraint_ema = torch.zeros_like(loss.lambda_k)
    changes = []
    for step, amplitude in enumerate([1., 2., .95, .8, .7, .7, .5, .4], 1):
        c = torch.full_like(loss.lambda_k, amplitude)
        src.compute_augmented_lagrangian(tensor(.3), c); src.update_multipliers(c)
        obs = {'c': c, 'step': step, 'epoch': 0, 'identity': loss.physics.identity}
        changes.append(loss.on_successful_step(obs, step))
        assert changes[-1] == src._last_multiplier_updated
        torch.testing.assert_close(loss.lambda_k, src.lambda_k)
    assert changes[0]
    if ratio:
        assert not changes[1]  # worsening violations do not accumulate multipliers
    else:
        assert all(changes)  # preserve upstream bypass of K when gating is disabled
    assert loss.lambda_k.min() >= -.25


def test_forward_validation_and_rejected_steps_are_readonly():
    loss = make_loss(ema_beta=.4)
    batch = Batch.from_data_list([graph(), graph()])
    pred = predictions(batch); pred['generator'] += .3
    before = loss.training_state_dict()
    total, info = loss(tensor(.3), pred, batch)
    snapshot_equal(before, loss.training_state_dict())
    loss.eval()
    with torch.no_grad(): loss(tensor(.3), pred, batch)
    snapshot_equal(before, loss.training_state_dict())
    bad = copy.deepcopy(info['al_observation']); bad['c'][0] = float('nan')
    with pytest.raises(FloatingPointError): loss.on_successful_step(bad, 1)
    snapshot_equal(before, loss.training_state_dict())
    loss.on_successful_step(info['al_observation'], 1)
    after = loss.training_state_dict()
    with pytest.raises(ValueError, match='Stale'): loss.on_successful_step(info['al_observation'], 1)
    snapshot_equal(after, loss.training_state_dict())
    for key, value in [('formula', 'literal_al_raw_h_v1'), ('ema_beta', .8)]:
        bad = copy.deepcopy(after); bad['_extra_state']['config'][key] = value
        with pytest.raises(ValueError, match='Incompatible'): loss.load_state_dict(bad)
        snapshot_equal(after, loss.training_state_dict())
    bad = copy.deepcopy(after); bad['dual_updates'].fill_(100)
    with pytest.raises(ValueError, match='schedule counters'): loss.load_state_dict(bad)
    snapshot_equal(after, loss.training_state_dict())
    bad = copy.deepcopy(after); bad['lambda_k'][0] = 1
    with pytest.raises(ValueError, match='sign'): loss.load_state_dict(bad)
    snapshot_equal(after, loss.training_state_dict())


@pytest.mark.parametrize('bad', [{'rho': .1}, {'mu_0': 0}, {'mu_0': float('nan')},
                                {'multiplier_check_interval': True}, {'warmup_epochs': -1},
                                {'warmup_epochs': 1.5}, {'ema_beta': 1.1}, {'normalize_by_size': 1},
                                {'min_penalty': 2, 'max_penalty': 1}, {'inequality_quadratic': 'raw_h'}])
def test_strict_config(bad):
    with pytest.raises(ValueError): ALConfig.from_dict(bad)


def test_manager_requires_explicit_initialization_and_rejects_unsupported_paths():
    batch = Batch.from_data_list([graph()])
    mgr = OPFLossManager('augmented_lagrangian', lagrangian_config={})
    assert mgr.get_loss_info()['loss_type'] == 'augmented_lagrangian'
    assert mgr.get_loss_info()['lagrangian']['formula'] == FORMULA
    with pytest.raises(RuntimeError, match='Initialize'): mgr.compute_loss(predictions(batch), batch)
    with pytest.raises(ValueError): OPFLossManager('augmented_lagrangian', lagrangian_config={}, reduction='sum')
    with pytest.raises(ValueError): OPFLossManager('mse', lagrangian_config={})
    mgr.initialize_constraints(graph(), dtype=torch.float64)
    with pytest.raises(RuntimeError, match='already initialized'): mgr.initialize_constraints(graph())
    mgr.to(dtype=torch.float32)
    assert mgr.lagrangian.lambda_k.dtype == torch.float32


def test_frozen_duals_small_gradient_step_descends():
    loss = make_loss()
    batch = Batch.from_data_list([graph()]); pred = predictions(batch)
    pred['bus'][1, 0] = .03
    pred = {k: v.requires_grad_() for k, v in pred.items()}
    value, _ = loss(tensor(0.), pred, batch)
    grads = torch.autograd.grad(value, tuple(pred.values()))
    assert all(torch.isfinite(g).all() for g in grads)
    after, _ = loss(tensor(0.), {k: v - 1e-5*g for (k,v),g in zip(pred.items(),grads)}, batch)
    assert after < value


@pytest.mark.parametrize('homogeneous', [False, True])
def test_mse_manager_matches_pinned_original_values_gradients_and_steps(tmp_path, homogeneous):
    # Preserve exact pre-integration code as the regression oracle, not a reimplementation.
    path = tmp_path / 'original_losses.py'
    path.write_bytes(subprocess.check_output(['git', 'show',
        'b3c53ea8593c364ec5a8e79918542bced9dd5184:lumina/model/opf/losses.py']))
    spec = importlib.util.spec_from_file_location('original_losses', path)
    original = importlib.util.module_from_spec(spec); spec.loader.exec_module(original)
    batch = Batch.from_data_list([graph(), graph()])
    batch['generator'].y[0, 0] = float('nan')  # existing supervised finite-target filtering
    gen_pred = batch['generator'].y.nan_to_num()
    if homogeneous:
        from lumina.utils.graph_utils import convert_opf_to_homo
        batch = convert_opf_to_homo(batch)
        # Conversion does not supply a mask; exercise the manager's optional
        # dataset-provided mask explicitly, including a deliberately excluded bus.
        batch.y_mask = batch.node_type < 2
        batch.y_mask[torch.where(batch.node_type == 0)[0][0]] = False
    managers = [original.OPFLossManager(), OPFLossManager()]
    parameters = [torch.nn.Parameter(tensor([[.12, 1.01]]).expand(4, 2).clone()) for _ in managers]
    optimizers = [torch.optim.AdamW([p], lr=.001) for p in parameters]
    for _ in range(3):
        values = []
        for mgr, p, opt in zip(managers, parameters, optimizers):
            opt.zero_grad()
            value, _ = mgr.compute_loss({'bus': p, 'generator': gen_pred}, batch)
            value.backward(); values.append(value.detach()); opt.step()
        torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
        torch.testing.assert_close(parameters[0].grad, parameters[1].grad, atol=0, rtol=0)
        torch.testing.assert_close(parameters[0], parameters[1], atol=0, rtol=0)
    assert managers[1].lagrangian is None
