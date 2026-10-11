"""Physical oracles, batching and differentiability, independent of AL total."""
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from experiments.lumina_bench.tests.test_protocol import graph, predictions
from experiments.lumina_bench.metrics import physical_residuals
from lumina.model.opf.augmented_lagrangian import FixedTopologyResiduals, AugmentedLagrangianLoss


def fixture(transformer=False):
    g = graph()
    g['shunt'].x[:] = torch.tensor([[.03, .02]], dtype=torch.float64)
    if transformer:
        del g['bus', 'ac_line', 'bus']
        g['bus', 'transformer', 'bus'].edge_index = torch.tensor([[0], [1]])
        g['bus', 'transformer', 'bus'].edge_attr = torch.tensor([
            [-.5, .5, .02, .1, .3, .3, .3, 1.03, .07, .02, .02]], dtype=torch.float64)
    else:
        g['bus', 'ac_line', 'bus'].edge_attr[:, 2:4] = .02
        g['bus', 'ac_line', 'bus'].edge_attr[:, 6] = .3
    return g


@pytest.mark.parametrize('transformer', [False, True])
def test_signed_physics_against_pypower(transformer):
    from pypower.makeYbus import makeYbus
    g = fixture(transformer)
    physics = FixedTopologyResiduals(g)
    batch = Batch.from_data_list([g])
    pred = predictions(batch)
    pred['bus'] = torch.tensor([[.04, 1.02], [-.03, .96]], dtype=torch.float64)
    r, h = physics.residuals(pred, batch)
    bus = np.zeros((2, 13)); bus[:, 0] = [0, 1]; bus[0, 4:6] = [2., 3.]
    ratio, shift = (1.03, .07) if transformer else (1., 0.)
    branch = np.array([[0, 1, .02, .1, .04, 30, 30, 30, ratio, np.degrees(shift), 1, -30, 30]])
    ybus, yf, yt = makeYbus(100., bus, branch)
    v = np.array([1.02 * np.exp(.04j), .96 * np.exp(-.03j)])
    balance = -v * (ybus @ v).conj()
    flow = np.r_[v[0] * (yf @ v).conj(), v[1] * (yt @ v).conj()]
    np.testing.assert_allclose(r[0].numpy(), np.r_[balance.real, balance.imag], atol=1e-12)
    np.testing.assert_allclose(h[0].numpy(), abs(flow)**2 - .3**2, atol=1e-12)
    independent_r, independent_h, _ = physical_residuals(pred, batch)
    torch.testing.assert_close(r[0], torch.cat((independent_r.real, independent_r.imag)))
    torch.testing.assert_close(h.clamp_min(0).reshape(-1), independent_h)


def test_batch_permutation_duplication_and_short_final_batch():
    gs = [fixture(True) for _ in range(3)]
    for i, g in enumerate(gs):
        g['load'].x += .04 * i
        g['bus'].y[1, 0] -= .02 * i
    loss = AugmentedLagrangianLoss({'mu_0': .01, 'ema_beta': 0.})
    loss.initialize_constraints(gs[0], dtype=torch.float64)
    def compute(items):
        batch = Batch.from_data_list(items)
        return loss(torch.tensor(.3, dtype=torch.float64), predictions(batch), batch)
    value, info = compute(gs)
    r, h = info['al_observation']['r'], info['al_observation']['h']
    for i, g in enumerate(gs):
        _, one = compute([g])
        torch.testing.assert_close(one['al_observation']['r'][0], r[i])
        torch.testing.assert_close(one['al_observation']['h'][0], h[i])
    for items in (gs[::-1], gs * 2):
        other, obs = compute(items)
        torch.testing.assert_close(value, other)
        torch.testing.assert_close(r.mean(0), obs['al_observation']['r'].mean(0))
        torch.testing.assert_close(h.clamp_min(0).mean(0), obs['al_observation']['h'].clamp_min(0).mean(0))
    loss.on_successful_step(info['al_observation'], 1)
    state = loss.lambda_k.clone()
    _, short = compute(gs[:1])
    torch.testing.assert_close(loss.lambda_k, state, rtol=0, atol=0)
    loss.on_successful_step(short['al_observation'], 2)
    assert loss.lambda_k.shape == (5,)


def test_gradcheck_and_each_prediction_and_demand_path():
    g = fixture(True)
    batch = Batch.from_data_list([g, g])
    physics = FixedTopologyResiduals(g)
    pred = predictions(batch)
    pred['bus'][:, 0] = torch.tensor([.04, -.03, .02, -.01], dtype=torch.float64)
    bus, gen = (pred[k].requires_grad_() for k in ('bus', 'generator'))
    assert torch.autograd.gradcheck(lambda a, b: physics.residuals({'bus': a, 'generator': b}, batch),
                                    (bus, gen), eps=1e-6, atol=2e-5, rtol=1e-4)
    r, h = physics.residuals({'bus': bus, 'generator': gen}, batch)
    gb, gg = torch.autograd.grad(r.square().sum() + h.square().sum(), (bus, gen))
    assert all(gb[:, col].abs().max() > 1e-8 and gg[:, col].abs().max() > 1e-8 for col in (0, 1))
    modified = batch.clone(); modified['load'].x[:, 0] += .1
    changed, _ = physics.residuals({'bus': bus, 'generator': gen}, modified)
    torch.testing.assert_close(changed[:, 0] - r[:, 0], torch.full((2,), -.1, dtype=torch.float64))


def test_multiple_generators_shunts_empty_branches_and_safe_limit_mask():
    g = graph()
    g['generator'].x = g['generator'].x.repeat(2, 1)
    g['generator'].y = g['generator'].y.repeat(2, 1)
    g['generator', 'generator_link', 'bus'].edge_index = torch.tensor([[1, 0], [0, 0]])
    g['bus', 'ac_line', 'bus'].edge_attr[:, 6] = float('inf')
    physics = FixedTopologyResiduals(g)
    batch = Batch.from_data_list([g]); pred = predictions(batch)
    r, h = physics.residuals(pred, batch)
    torch.testing.assert_close(r, torch.tensor([[.5, 0, .1, 0]], dtype=torch.float64))
    assert h.shape == (1, 0)
    g['bus', 'ac_line', 'bus'].edge_index = torch.empty(2, 0, dtype=torch.long)
    g['bus', 'ac_line', 'bus'].edge_attr = torch.empty(0, 9, dtype=torch.float64)
    g['shunt'].x[:] = torch.tensor([[.1, .2]], dtype=torch.float64)
    p = FixedTopologyResiduals(g)
    b = Batch.from_data_list([g]); r, h = p.residuals(predictions(b), b)
    torch.testing.assert_close(r, torch.tensor([[.3, 0, .2, 0]], dtype=torch.float64))
    assert h.shape == (1, 0)


@pytest.mark.parametrize('change', ['connectivity', 'rate', 'impedance', 'shunt', 'base', 'missing_load'])
def test_cache_never_reuses_changed_physical_inputs(change):
    g = graph(); physics = FixedTopologyResiduals(g); changed = g.clone()
    if change == 'connectivity': changed['bus', 'ac_line', 'bus'].edge_index = torch.tensor([[1], [0]])
    if change == 'rate': changed['bus', 'ac_line', 'bus'].edge_attr[:, 6] *= 2
    if change == 'impedance': changed['bus', 'ac_line', 'bus'].edge_attr[:, 4] *= 2
    if change == 'shunt': changed['shunt'].x += .1
    if change == 'base': changed.baseMVA *= 2
    if change == 'missing_load': del changed['load', 'load_link', 'bus']
    with pytest.raises(ValueError):
        physics.validate_batch(Batch.from_data_list([g, changed]))


def test_real_case30_ground_truth_endpoint_labels_and_perturbations():
    root = os.environ.get('LUMINA_BENCH_TEST_DATA')
    if not root: pytest.skip('Set LUMINA_BENCH_TEST_DATA')
    from lumina.dataset.opf.opf_dataset import OPFDataset
    data = OPFDataset(root=root, case_name='pglib_opf_case30_ieee', group_id=0)
    gs = [data[i] for i in (0, 19, 137, 2000)]
    physics = FixedTopologyResiduals(gs[0]); batch = Batch.from_data_list(gs)
    pred = {k: v.double() for k, v in predictions(batch).items()}
    r, h = physics.residuals(pred, batch)
    assert r.norm(dim=1).max() < 1e-3
    assert h.clamp_min(0).norm(dim=1).max() < 1e-3
    v = pred['bus'].reshape(4, 30, 2)
    sf, st = physics.terminal_powers(v[..., 1] * v[..., 0].cos(), v[..., 1] * v[..., 0].sin())
    labels = torch.cat([batch['bus', k, 'bus'].edge_label.reshape(4, -1, 4) for k in physics.branches], 1).double()
    torch.testing.assert_close(sf.real, labels[..., 2], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(sf.imag, labels[..., 3], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(st.real, labels[..., 0], atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(st.imag, labels[..., 1], atol=2e-5, rtol=1e-4)
    for kind, col in [('bus', 0), ('bus', 1), ('generator', 0), ('generator', 1)]:
        pert = {k: v.clone() for k, v in pred.items()}; pert[kind][0, col] += .01
        rr, _ = physics.residuals(pert, batch)
        assert (rr - r).abs().max() > .001


def test_raw_electrical_results_match_pinned_source(tmp_path):
    import importlib.util
    import subprocess
    path = tmp_path / 'source_al.py'
    path.write_bytes(subprocess.check_output(['git', 'show',
        'f53a7a35ec362121ae39b1c9781bd66b5b9e1b5d:lumina/model/opf/augmented_lagrangian.py']))
    spec = importlib.util.spec_from_file_location('source_al_reference', path)
    reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
    g = fixture(True); physics = FixedTopologyResiduals(g)
    batch = Batch.from_data_list([g, g]); pred = predictions(batch)
    pred['bus'][1, 0] = .04
    r, h = physics.residuals(pred, batch)
    src = reference.AugmentedLagrangianACOPF(normalize_by_size=False, normalize_by_rms=False)
    kwargs = {'Y_real_sparse': physics.y_real, 'Y_imag_sparse': physics.y_imag,
              'line_edge_index': physics.edge, 'line_limits': physics.ratings}
    for name in ('ff', 'ft', 'tf', 'tt'):
        value = getattr(physics, 'y' + name)
        kwargs['line_y_' + name + '_real'] = value.real
        kwargs['line_y_' + name + '_imag'] = value.imag
    src.set_network_parameters(**kwargs)
    balances = []
    def capture(inj, calc):
        result = inj - calc; balances.append(result); return result
    src._compute_balance = capture
    v = pred['bus'].reshape(2, 2, 2); gen = pred['generator'].reshape(2, 1, 2)
    demand = batch['load'].x.reshape(2, 1, 2)
    src.compute_power_flow_constraints(v[..., 1], v[..., 0], gen[..., 0], gen[..., 1],
        demand[..., 0], demand[..., 1], physics.links['generator'][1], physics.links['load'][1])
    torch.testing.assert_close(torch.cat(balances, 1), r, atol=1e-12, rtol=1e-10)
    source_thermal = src.compute_line_flow_constraints(v[..., 1], v[..., 0], physics.edge)
    transformed = torch.maximum(h[:, :1], h[:, 1:]).clamp_min(0).sqrt().mean(0)
    torch.testing.assert_close(source_thermal, transformed, atol=1e-12, rtol=1e-10)
