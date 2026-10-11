"""Numerical mixed derivatives, update separation, and transactional solver gate."""
from copy import deepcopy

import pytest
import torch

from distillation.DC.dcgm import (SyntheticInputs, data_gradients, feasible_step,
                                  gradient_distance, matching_loss, train_student)
from distillation.synthetic.learning import benchmark_graph, model_for, model_options
from distillation.test import test_dd_demo as reference
from torch_geometric.data import Batch

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


class SmoothRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.2, .3], [.4, .5]], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.unused = torch.nn.Parameter(torch.ones(2, 2, dtype=torch.float64))

    def forward(self, batch):
        return dict(bus=batch['bus'].x[:, 1:3] @ self.weight + self.bias,
                    generator=batch['generator'].x[:, [3, 6]] @ self.weight + self.bias)


def real_batch(graph):
    return Batch.from_data_list([benchmark_graph(graph, labels=True)])


def test_reference_distance_skips_vectors_and_detaches_real():
    p = torch.ones(2, 2)
    gs = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    gr = torch.tensor([[2., 1.], [4., 3.]], requires_grad=True)
    loss = gradient_distance([gs, torch.ones(2)], [gr, torch.zeros(2)], [p, torch.ones(2)])
    expected = sum(1 - (a*b).sum()/(a.norm()*b.norm()+1e-6) for a,b in zip(gs,gr.detach()))
    torch.testing.assert_close(loss, expected.double(), rtol=1e-6, atol=1e-7)
    loss.backward()
    assert gs.grad is not None and gr.grad is None


def test_matching_mixed_derivative_finite_difference_and_model_separation(graph):
    synthetic, model = SyntheticInputs([graph]), SmoothRegressor()
    before = deepcopy(model.state_dict())
    variable = synthetic.edits[0].delta['active']
    loss, _ = matching_loss(model, synthetic, real_batch(graph))
    exact = torch.autograd.grad(loss, variable)[0][0].item()
    values = []
    epsilon = 1e-4
    for sign in (1, -1):
        with torch.no_grad(): variable[0] = sign * epsilon
        values.append(float(matching_loss(model, synthetic, real_batch(graph))[0].detach()))
    with torch.no_grad(): variable.zero_()
    assert abs(exact) > 1e-12
    assert exact == pytest.approx((values[0]-values[1])/(2*epsilon), rel=2e-4, abs=1e-10)
    data_gradients(model, synthetic, real_batch(graph))
    assert all(p.grad is None for p in model.parameters())
    for name, value in before.items(): torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


def test_student_training_cannot_change_data_or_accumulate_data_gradients(graph):
    synthetic, model = SyntheticInputs([graph]*2), SmoothRegressor()
    before = deepcopy(synthetic.state_dict())
    labels = deepcopy(synthetic.labels)
    weights = model.weight.detach().clone()
    train_student(model, torch.optim.SGD(model.parameters(), lr=1e-4), synthetic, batch_size=1)
    assert not torch.equal(model.weight, weights)
    assert all(p.grad is None for p in synthetic.parameters())
    for name, value in before.items(): torch.testing.assert_close(value, synthetic.state_dict()[name], rtol=0, atol=0)
    for a,b in zip(labels, synthetic.labels):
        for k in a: torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def test_rejected_solver_update_restores_sgd_momentum_and_labels(graph):
    synthetic = SyntheticInputs([graph])
    optimizer = torch.optim.SGD(synthetic.parameters(), lr=.1, momentum=.5)
    for p in synthetic.parameters(): p.grad = torch.ones_like(p)
    optimizer.step()
    before, state, labels = deepcopy(synthetic.state_dict()), deepcopy(optimizer.state_dict()), deepcopy(synthetic.labels)
    ok, _, _, attempts = feasible_step(synthetic, optimizer, audit=lambda _: (False, [], []))
    assert not ok and len(attempts) == 5
    for k,v in before.items(): torch.testing.assert_close(v, synthetic.state_dict()[k], rtol=0, atol=0)
    for k,v in state['state'].items():
        torch.testing.assert_close(v['momentum_buffer'], optimizer.state_dict()['state'][k]['momentum_buffer'], rtol=0, atol=0)
    for k in labels[0]: torch.testing.assert_close(labels[0][k], synthetic.labels[0][k], rtol=0, atol=0)


def test_accepted_solve_refreshes_detached_labels(graph):
    synthetic = SyntheticInputs([graph])
    optimizer = torch.optim.SGD(synthetic.parameters(), lr=.001, momentum=.5)
    data_gradients(SmoothRegressor(), synthetic, real_batch(graph))
    ok, graphs, checks, _ = feasible_step(synthetic, optimizer)
    assert ok and checks[0]['success']
    for k in synthetic.labels[0]:
        assert not synthetic.labels[0][k].requires_grad
        torch.testing.assert_close(synthetic.labels[0][k], graphs[0][k].y)


@pytest.mark.parametrize('kind', ['GAT', 'GIN', 'HeteroGNN'])
def test_actual_models_have_finite_nonzero_physical_matching_gradients(graph, kind):
    from distillation.synthetic import generate
    graphs = generate([graph], count=2, seed=13, min_topologies=1).physical_graphs
    synthetic = SyntheticInputs(graphs)
    model = model_for(graph, dict(model=kind, model_options=model_options(kind, width=8)), 71, 'cpu')
    value, info = data_gradients(model, synthetic, real_batch(graph))
    assert info['real_gradient_norm'] > 0 and info['synthetic_gradient_norm'] > 0
    assert sum(float(p.grad.abs().sum()) for p in synthetic.parameters()) > 0
    assert all(p.grad is None for p in model.parameters())


def test_resume_preserves_next_matching_update_and_student_trajectory(graph):
    from distillation.DC.campaign import payload_for, restore
    config = dict(strength=.02, data_rate=.1)
    synthetic, model = SyntheticInputs([graph]), SmoothRegressor()
    optimizer = torch.optim.SGD(synthetic.parameters(), lr=.1, momentum=.5)
    student_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    data_gradients(model, synthetic, real_batch(graph)); optimizer.step()
    train_student(model, student_optimizer, synthetic, seed=7)
    payload = deepcopy(payload_for(synthetic, optimizer, config, {}, [graph], [], []))
    resumed, resumed_optimizer = restore(payload)
    second_model = deepcopy(model)
    for m,s,o in ((model,synthetic,optimizer),(second_model,resumed,resumed_optimizer)):
        o.zero_grad(set_to_none=True)
        data_gradients(m,s,real_batch(graph)); o.step()
    for k,v in synthetic.state_dict().items():
        torch.testing.assert_close(v,resumed.state_dict()[k],rtol=0,atol=0)
    for k in synthetic.labels[0]:
        torch.testing.assert_close(synthetic.labels[0][k],resumed.labels[0][k],rtol=0,atol=0)


def test_zero_edit_exactly_preserves_all_physical_inputs(graph):
    synthetic = SyntheticInputs([graph])
    a,b=synthetic.anchors[0].params,synthetic.matrices()[0].params
    for k,v in vars(a).items():
        if isinstance(v,torch.Tensor): torch.testing.assert_close(v,getattr(b,k),rtol=0,atol=0)


@pytest.mark.parametrize('kind', ['GAT', 'GIN', 'HeteroGNN'])
def test_precision_adapter_preserves_model_and_supports_double_derivatives(graph, kind):
    from distillation.synthetic.learning import PrecisionModel
    config = dict(model=kind, model_options=model_options(kind, width=8))
    base = model_for(graph, config, 43, 'cpu')
    converted = PrecisionModel(deepcopy(base), torch.float32).eval()
    for k,v in base(real_batch(graph)).items():
        torch.testing.assert_close(v, converted(real_batch(graph))[k], rtol=0, atol=0)
    model = model_for(graph, config | dict(model_precision='float64'), 43, 'cpu')
    synthetic = SyntheticInputs([graph])
    data_gradients(model, synthetic, real_batch(graph))
    assert next(model.parameters()).dtype == torch.float64
    assert all(torch.isfinite(p.grad).all() for p in synthetic.parameters())


def test_evaluation_overflow_is_reported_collectively_without_skipping_seeds(graph, monkeypatch):
    from distillation.DC import campaign
    class Comm:
        rank, size = 0, 1
        def allgather(self, value): return [value]
    monkeypatch.setattr(campaign, 'model_for', lambda *a, **k: SmoothRegressor())
    def overflow(*args): raise ValueError('Non-finite physical metric: thermal_l2')
    monkeypatch.setattr(campaign, 'evaluate', overflow)
    config=dict(model='smooth',model_options={},eval_batch=1,evaluation_epochs=1)
    result,_=campaign.panel(SyntheticInputs([graph]),config,.001,None,[0],Comm(),'cpu',trained=False)
    assert result['mean'] is None and result['nonfinite_initializations']==1
    assert result['per_initialization_metrics'][0]['error'].startswith('Non-finite physical')
    def invalid(*args): raise ValueError('Unexpected implementation error')
    monkeypatch.setattr(campaign,'evaluate',invalid)
    with pytest.raises(RuntimeError,match='collectively'):
        campaign.panel(SyntheticInputs([graph]),config,.001,None,[0],Comm(),'cpu',trained=False)


def test_finite_huge_calibration_scores_do_not_overflow_report():
    import json
    from distillation.synthetic.runtime import score_summary
    result = score_summary([1e300, 2e300])
    assert result['mean'] == pytest.approx(1.5e300)
    assert result['std'] == pytest.approx(.5e300)
    json.dumps(result, allow_nan=False)


def test_calibration_requires_all_panels_and_applies_fixed_safety_margin(graph, monkeypatch):
    import numpy as np
    from distillation.DC import campaign
    class Comm:
        rank, size = 0, 12
    calls=[]
    def panel(synthetic, config, rate, dataset, ids, comm, device, seed_base):
        calls.append((rate, seed_base))
        metrics={'graph_balanced_mse':2. if rate==.001 else 1.}
        rows=[dict(metrics=metrics.copy(),error='',seed=seed_base+i) for i in range(12)]
        if rate==.01 and seed_base==300012:rows[0]['metrics']={}
        return dict(per_initialization_metrics=rows),None
    monkeypatch.setattr(campaign,'panel',panel)
    config=dict(strength=.02,student_rates=[.001,.01],calibration_panels=2,
                student_rate_factor=.1,evaluation_epochs=10)
    rate, report=campaign.calibrate([graph],config,None,np.arange(200),Comm(),'cpu')
    assert rate==pytest.approx(.0001)
    assert report['calibration_initializations']==24 and len(calls)==4
    assert report['candidates'][1]['nonfinite_initializations']==1
