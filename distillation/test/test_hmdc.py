"""HMDC formulas, graph correspondence, all backbones, update separation and resume."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from distillation.HMDC.features import SpatialSemanticAlignment, forward_features, node_graph_ids
from distillation.HMDC.hmdc import (GradientBalance, alignment_step, data_gradients,
                                  gradient_distance, matching_terms, model_step,
                                  normalize_data_gradients)
from distillation.HMDC.models import build_pair
from distillation.HMDC.selection import representatives
from distillation.synthetic.inputs import SyntheticInputs
from distillation.synthetic.learning import PrecisionModel, benchmark_graph, model_for, model_options
from distillation.test import test_dd_demo as reference
from experiments.lumina_bench.models import SUPPORTED_MODELS

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def config(pair=('GAT', 'GIN')):
    return dict(pair=list(pair), models={k: model_options(k, width=8) for k in SUPPORTED_MODELS},
                strength=.02, batch_size=2, eval_batch=2, evaluation_epochs=2,
                student_rates=[1e-7], alignment_rate=.01)


def real_batch(graph):
    return Batch.from_data_list([benchmark_graph(graph, labels=True)])


def test_hmdc_distance_is_squared_sum_including_bias_and_detached_real():
    a = torch.tensor([1., 3.], requires_grad=True)
    b = torch.tensor([2., 1.], requires_grad=True)
    p = torch.ones(2)
    loss = gradient_distance([a, None], [b, torch.tensor(2.)], [p, torch.tensor(0.)])
    assert float(loss.detach()) == 9.
    loss.backward()
    torch.testing.assert_close(a.grad, torch.tensor([-2., 4.]))
    assert b.grad is None


def test_gbm_accumulates_max_not_norm_and_restores_counter():
    balance = GradientBalance(3)
    balance.observe([[torch.tensor([-2., 1.])], [torch.tensor([4., 4.])], [None]])
    expected = torch.tensor([2., 4., 0.], dtype=torch.float64) + 1e-6
    torch.testing.assert_close(balance.accumulator, expected)
    torch.testing.assert_close(balance.weights(), expected.min()/expected)
    balance.steps = 7
    other = GradientBalance(3); other.load_state_dict(balance.state_dict())
    assert other.steps == 7
    torch.testing.assert_close(balance.weights(), other.weights())


def test_graph_family_normalization_handles_empty_and_zero_gradients(graph):
    synthetic = SyntheticInputs([graph]*2)
    for p in synthetic.parameters():
        p.grad = torch.arange(p.numel(), dtype=p.dtype).reshape(p.shape)
    normalize_data_gradients(synthetic)
    for p in synthetic.parameters():
        assert torch.isfinite(p.grad).all()
        if p.numel() > 1:
            assert float(p.grad.abs().max()) == pytest.approx((p.numel()-1)/(p.numel()-1+1e-6))


@pytest.mark.parametrize('depth_a,depth_b', [(2, 3), (3, 2), (2, 2)])
def test_ssd_matches_reference_layer_softmax_and_graph_token_mse(depth_a, depth_b):
    torch.manual_seed(91)
    alignment = SpatialSemanticAlignment([4]*depth_a, [3]*depth_b).double()
    fa = [torch.randn(5, 4, dtype=torch.float64) for _ in range(depth_a)]
    fb = [torch.randn(5, 3, dtype=torch.float64) for _ in range(depth_b)]
    a, b = alignment.align(fa, fb)
    x = torch.stack([p(f) for p,f in zip(alignment.projections[0], fa)], dim=-1)
    y = torch.stack([p(f) for p,f in zip(alignment.projections[1], fb)], dim=-1)
    if depth_a < depth_b:
        y = y @ alignment.layer_logits.T.softmax(1)
    else:
        x = x @ alignment.layer_logits.softmax(1)
    torch.testing.assert_close(a, x); torch.testing.assert_close(b, y)
    ids = torch.tensor([0,0,1,1,1])
    expected = []
    for i in range(2):
        ax, bx = x[ids==i], y[ids==i]
        expected.append(torch.cat([ax, ax.mean(0,keepdim=True)]).sub(
                        torch.cat([bx, bx.mean(0,keepdim=True)])).square().mean())
    torch.testing.assert_close(alignment(fa,fb,ids,2),torch.stack(expected).mean())
    order=torch.tensor([4,0,2,1,3])
    torch.testing.assert_close(alignment(fa,fb,ids,2),
                              alignment([f[order] for f in fa],[f[order] for f in fb],ids[order],2))


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
def test_all_eight_precision_adapters_and_features_preserve_predictions(graph, kind):
    cfg=dict(model=kind,model_options=model_options(kind,width=8))
    base=model_for(graph,cfg,91,'cpu')
    converted=PrecisionModel(deepcopy(base),torch.float32).eval()
    batch=real_batch(graph)
    expected=base(batch)
    predictions,features=forward_features(converted,batch)
    for key,value in expected.items():
        torch.testing.assert_close(predictions[key],value,rtol=0,atol=0)
    assert len(features)==len(base.backbone.convs)
    assert all(f.shape[0]==batch.num_nodes for f in features)
    assert all(not conv._forward_hooks for conv in converted.backbone.convs)


@pytest.mark.parametrize('pair_names', [('GAT','GIN'),('GCN','HeteroGNN'),('Transformer','HGT'),('RGAT','HEAT')])
def test_full_hmdc_pairs_have_finite_nonzero_input_gradients_only(graph,pair_names):
    cfg=config(pair_names)
    pair=build_pair(graph,cfg,{k:1e-8 for k in pair_names},81,'cpu')
    synthetic=SyntheticInputs([graph])
    with torch.no_grad():
        for p in synthetic.parameters():p.fill_(.15)
    models=[deepcopy(m.state_dict()) for m in pair.models]
    alignment=deepcopy(pair.alignment.state_dict())
    info=data_gradients(pair.models,pair.alignment,pair.balance,synthetic,real_batch(graph))
    assert len(info['terms'])==3 and all(np.isfinite(info['terms']))
    assert sum(float(p.grad.abs().sum()) for p in synthetic.parameters()) > 0
    for before,model in zip(models,pair.models):
        assert all(p.grad is None for p in model.parameters())
        for k,v in before.items():torch.testing.assert_close(v,model.state_dict()[k],rtol=0,atol=0)
    for k,v in alignment.items():torch.testing.assert_close(v,pair.alignment.state_dict()[k],rtol=0,atol=0)
    assert all(p.grad is None for p in pair.alignment.parameters())


def test_all_three_objectives_have_correct_mixed_derivatives(graph):
    cfg=config();pair=build_pair(graph,cfg,{'GAT':1e-8,'GIN':1e-8},83,'cpu')
    synthetic=SyntheticInputs([graph]);variable=synthetic.edits[0].delta['active']
    with torch.no_grad():variable.fill_(.17)
    terms,_=matching_terms(pair.models,pair.alignment,synthetic,real_batch(graph))
    exact=[torch.autograd.grad(term,variable,retain_graph=True)[0][0].item() for term in terms]
    values=[];epsilon=1e-4
    for sign in (1,-1):
        with torch.no_grad():variable[0]=.17+sign*epsilon
        values.append([float(t.detach()) for t in matching_terms(pair.models,pair.alignment,synthetic,real_batch(graph))[0]])
    for i in range(3):
        assert abs(exact[i])>1e-12
        assert exact[i]==pytest.approx((values[0][i]-values[1][i])/(2*epsilon),rel=1e-3,abs=1e-8)


def test_real_model_training_and_affine_training_update_separate_parameters(graph):
    cfg=config();pair=build_pair(graph,cfg,{'GAT':1e-9,'GIN':1e-9},85,'cpu')
    real=real_batch(graph);alignment=deepcopy(pair.alignment.state_dict())
    before=[deepcopy(m.state_dict()) for m in pair.models]
    model_step(pair.models,pair.alignment,pair.optimizers,real)
    assert all(any(not torch.equal(v,m.state_dict()[k]) for k,v in old.items())
               for old,m in zip(before,pair.models))
    for k,v in alignment.items():torch.testing.assert_close(v,pair.alignment.state_dict()[k],rtol=0,atol=0)
    before=[deepcopy(m.state_dict()) for m in pair.models]
    alignment_step(pair.models,pair.alignment,pair.alignment_optimizer,real)
    for old,m in zip(before,pair.models):
        for k,v in old.items():torch.testing.assert_close(v,m.state_dict()[k],rtol=0,atol=0)
    assert any(not torch.equal(v,pair.alignment.state_dict()[k]) for k,v in alignment.items())


def test_resume_reproduces_next_data_model_alignment_and_balance_updates(graph):
    cfg=config();rates={'GAT':1e-9,'GIN':1e-9}
    pair=build_pair(graph,cfg,rates,89,'cpu');synthetic=SyntheticInputs([graph])
    with torch.no_grad():
        for p in synthetic.parameters():p.fill_(.1)
    real=real_batch(graph)
    data_gradients(pair.models,pair.alignment,pair.balance,synthetic,real)
    model_step(pair.models,pair.alignment,pair.optimizers,real)
    alignment_step(pair.models,pair.alignment,pair.alignment_optimizer,real)
    pair.query_ids=[7,19]
    resumed=build_pair(graph,cfg,rates,1,'cpu');resumed.load_state_dict(deepcopy(pair.state_dict()))
    duplicate=deepcopy(synthetic)
    for current,data in ((pair,synthetic),(resumed,duplicate)):
        data_gradients(current.models,current.alignment,current.balance,data,real)
        model_step(current.models,current.alignment,current.optimizers,real)
        alignment_step(current.models,current.alignment,current.alignment_optimizer,real)
    assert resumed.query_ids==pair.query_ids
    for p,q in zip(synthetic.parameters(),duplicate.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    for a,b in zip(pair.models+[pair.alignment],resumed.models+[resumed.alignment]):
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)
    torch.testing.assert_close(pair.balance.accumulator,resumed.balance.accumulator,rtol=0,atol=0)


def test_representative_selection_is_seeded_unique_and_handles_duplicate_features():
    x=np.array([[0.,0.],[0.,0.],[10.,10.],[10.,10.],[20.,20.]])
    selected=representatives(x,3,seed=7)
    assert selected==representatives(x,3,seed=7) and len(set(selected))==3
    assert set(x[selected,0])=={0.,10.,20.}
    assert len(set(representatives(np.zeros((8,4)),5)))==5
    with pytest.raises(ValueError):representatives(np.array([[np.nan]]),1)


def test_exploration_pairs_protocols_and_bus_budgets(tmp_path):
    from distillation.HMDC.launch import CROSS_PAIRS, exploration
    from distillation.HMDC.campaign import validate_config
    from distillation.synthetic.experiments import balanced_sizes
    experiments=exploration(tmp_path)
    assert len(experiments)==len({e['output'] for e in experiments})==16
    for pair in CROSS_PAIRS:
        conditions=[e['config'] for e in experiments if e['config']['pair']==list(pair)]
        assert {(c['generation_protocol'],c['synthetic_count']) for c in conditions}=={
            ('conservative',48),('random_fixed',48),('random_size',48),('random_size',24)}
        for c in conditions:
            validate_config(c)
            assert c['case']==57 and c['bus_range']==[43,71]
            sizes=balanced_sizes(57,c['batches'],c['batch_size'])
            assert min(sizes)==43 and max(sizes)==71
            assert sum(sizes)==57*c['synthetic_count']


def test_queries_restart_with_the_pair_without_reusing_candidate_pools():
    from distillation.HMDC.campaign import query_schedule
    # The recorded full campaigns use the unchanged 100/10 schedule.
    for i in range(301):
        assert query_schedule(i,100,10)==(i//10,(i//10)%2,i%10==0)
    assert [query_schedule(i,3,2) for i in range(7)]==[
        (0,0,True),(0,0,False),(1,1,True),(2,0,True),
        (2,0,False),(3,1,True),(4,0,True)]
    assert [query_schedule(i,2,10) for i in range(4)]==[
        (0,0,True),(0,0,False),(1,0,True),(1,0,False)]


def test_larger_synthetic_budget_cannot_silently_shorten_real_control(tmp_path):
    from types import SimpleNamespace
    from distillation.synthetic.experiments import initialize
    source=tmp_path/'references.pt'
    torch.save(dict(graphs=[None,None],ids=[1,2]),source)
    comm=SimpleNamespace(rank=0,bcast=lambda value,root:value)
    with pytest.raises(ValueError,match='reference file'):
        initialize(dict(reference_file=str(source),synthetic_count=4),tmp_path,comm)
