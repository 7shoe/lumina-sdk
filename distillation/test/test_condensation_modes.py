"""Target semantics, mixed derivatives and comparable downstream evaluation."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from distillation.DC import campaign as dc, dcgm
from distillation.DD import campaign as dd
from distillation.DD.mvp import SyntheticOPF
from distillation.HMDC import campaign as hm, hmdc
from distillation.HMDC.models import build_pair
from distillation.synthetic import evaluation
from distillation.synthetic.inputs import SyntheticInputs, feasible_step, optimizer_for, changes_summary
from distillation.synthetic.learning import benchmark_graph, model_for, model_options, train_student
from distillation.test.test_dcgm import SmoothRegressor, real_batch
from distillation.test import test_dd_demo as reference
from experiments.lumina_bench.models import SUPPORTED_MODELS

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_common_targets_match_dd_and_gate_preserves_policy(graph, mode):
    data = SyntheticInputs([graph], target_mode=mode)
    wang = SyntheticOPF([graph], steps=1, target_mode=mode)
    if mode == 'learned':
        with torch.no_grad():
            data.target_offsets[0]['bus'].fill_(.3)
            wang.target_offsets[0]['bus'].fill_(.3)
    torch.testing.assert_close(data.batch()[1], wang.batch(0)[1], rtol=0, atol=0)
    before = deepcopy(data.labels)
    opt = optimizer_for(data, dict(data_rate=.01, target_rate=.002))
    for p in data.parameters(): p.grad = torch.ones_like(p)
    solved = graph.clone()
    solved['bus'].y += .25
    accepted, physical, _, _ = feasible_step(data, opt, audit=lambda _: (True, [solved], []))
    assert accepted and torch.equal(physical[0]['bus'].y, solved['bus'].y)
    torch.testing.assert_close(data.labels[0]['bus'], solved['bus'].y if mode == 'solver' else before[0]['bus'])
    assert data.batch()[1]['bus'].requires_grad == (mode == 'learned')
    if mode == 'learned':
        reference_bus = data.anchors[0].params.bus_type == 3
        torch.testing.assert_close(data.batch()[1]['bus'][reference_bus, 0], before[0]['bus'][reference_bus, 0])
    before, momentum, labels = deepcopy(data.state_dict()), deepcopy(opt.state_dict()), deepcopy(data.labels)
    accepted, _, _, _ = feasible_step(data, opt, audit=lambda _: (False, [], []))
    assert not accepted
    torch.testing.assert_close(data.state_dict(), before, rtol=0, atol=0)
    torch.testing.assert_close(opt.state_dict(), momentum, rtol=0, atol=0)
    torch.testing.assert_close(data.labels, labels, rtol=0, atol=0)
    assert changes_summary(data)['target_drift_rms'] > 0 if mode != 'frozen' else changes_summary(data)['target_drift_rms'] == 0


def test_nonfinite_targets_and_audit_errors_roll_back_both_optimizer_groups(graph):
    data = SyntheticInputs([graph], target_mode='learned')
    opt = optimizer_for(data, dict(data_rate=.1, target_rate=1e308))
    before = deepcopy(data.state_dict())
    for p in data.parameters(): p.grad = torch.ones_like(p)
    for p in data.target_offsets.parameters(): p.grad.fill_(1e308)
    def no_audit(_): raise AssertionError('Nonfinite targets must be rejected before OPF audit')
    accepted, _, _, attempts = feasible_step(data, opt, audit=no_audit)
    assert not accepted and all(r['finite'] is False for r in attempts) and not opt.state
    torch.testing.assert_close(data.state_dict(), before, rtol=0, atol=0)
    opt.param_groups[1]['lr'] = .01
    for p in data.parameters(): p.grad.fill_(1.)
    def fail(_): raise RuntimeError('Audit unavailable')
    with pytest.raises(RuntimeError, match='Audit unavailable'):
        feasible_step(data, opt, audit=fail)
    torch.testing.assert_close(data.state_dict(), before, rtol=0, atol=0)
    assert not opt.state


@pytest.mark.parametrize('method', ['DCGM', 'HMDC'])
def test_learned_target_mixed_derivative_finite_difference(graph, method):
    data = SyntheticInputs([graph], target_mode='learned')
    if method == 'DCGM':
        model = SmoothRegressor()
        objective = lambda: dcgm.matching_loss(model, data, real_batch(graph))[0]
    else:
        cfg = hm.default_config(30) | {'models': {k: model_options(k, width=8) for k in ['GIN', 'GAT']}}
        pair = build_pair(graph, cfg, {'GIN': 1e-7, 'GAT': 1e-7}, 42, 'cpu')
        objective = lambda: sum(hmdc.matching_terms(pair.models, pair.alignment, data, real_batch(graph))[0])
    variable = data.target_offsets[0]['bus']
    with torch.no_grad(): variable[0, 1] = .03
    exact = torch.autograd.grad(objective(), variable)[0][0, 1].item()
    values = []
    for offset in (1e-5, -1e-5):
        with torch.no_grad(): variable[0, 1] = .03 + offset
        values.append(float(objective().detach()))
    assert abs(exact) > 1e-10
    assert exact == pytest.approx((values[0]-values[1])/2e-5, rel=2e-4, abs=1e-8)


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
def test_all_models_differentiate_learned_targets_without_student_data_leak(graph, kind):
    cfg = dc.default_config(30, kind) | {'model_options': model_options(kind, width=8)}
    model = model_for(graph, cfg, 71, 'cpu')
    data = SyntheticInputs([graph], target_mode='learned')
    value, _ = dcgm.data_gradients(model, data, real_batch(graph))
    assert np.isfinite(value)
    assert sum(float(p.grad.abs().sum()) for p in data.target_offsets.parameters()) > 0
    for p in data.parameters(): p.grad = None
    before = deepcopy(data.state_dict())
    train_student(model, torch.optim.SGD(model.parameters(), lr=1e-7), data)
    torch.testing.assert_close(data.state_dict(), before, rtol=0, atol=0)
    assert all(p.grad is None for p in data.parameters())


def test_common_defaults_and_100_update_evaluation(graph):
    configs = [dd.default_config(30, 'GIN'), dc.default_config(30, 'GIN'), hm.default_config(30)]
    keys = ['synthetic_count', 'batches', 'batch_size', 'strength', 'model_precision', 'bus_range',
            'evaluation_epochs', 'split_seed', 'real_batch', 'validation_samples', 'final_samples']
    for key in keys: assert configs[0][key] == configs[1][key] == configs[2][key]
    cfg = configs[1]
    assert cfg['synthetic_count'] == 100 and (cfg['restart_rounds']-1)*cfg['batches']*cfg['student_epochs'] == 100
    data, model = SyntheticInputs([graph]*100), SmoothRegressor()
    observed = []
    train_student(model, torch.optim.SGD(model.parameters(), lr=1e-7), data,
        epochs=cfg['evaluation_epochs'], batch_size=cfg['batch_size'], observe=lambda a,b: observed.append((a,b)))
    assert observed == [(i, i*10) for i in range(101)]


def test_manifest_pairs_methods_with_identical_generation_and_downstream_settings(tmp_path):
    from distillation.experiments.comparison import manifest
    result = manifest(tmp_path, tmp_path/'references', cases=[30], models=['GIN','GAT'],
                      pairs=[('GIN','GAT')], protocols=['random_size'])
    experiments = result['experiments']
    assert len(experiments) == 15 and len({r['output'] for r in experiments}) == 15
    for key in ['initial_cache','reference_file','model_precision','strength','synthetic_count','bus_range',
                'student_rates','calibration_seed','calibration_model_seed','evaluation_epochs']:
        assert len({json.dumps(r['config'][key]) for r in experiments}) == 1
    assert {r['config']['target_mode'] for r in experiments} == {'solver','frozen','learned'}


def test_dd_export_preserves_learned_targets_for_common_evaluation(graph, tmp_path):
    cfg = dd.default_config(30, 'GIN') | dict(batches=1, batch_size=1, synthetic_count=1, passes=1,
        target_mode='learned', model_options=model_options('GIN', width=8))
    data = dd.Schedule([graph], cfg)
    with torch.no_grad():
        data.target_offsets[0]['generator'].fill_(.5)
        data.edits[0].delta['load'].fill_(.1)
    dd.export(tmp_path, data, cfg, {}, [graph], [], {}, name='best.pt')
    artifact = torch.load(tmp_path/'best.pt', weights_only=False)
    restored = SyntheticInputs.from_export(artifact)
    x, y = data.batch(0)
    bx, by = restored.batch()
    torch.testing.assert_close(y, by, rtol=0, atol=0)
    for kind in x.node_types: torch.testing.assert_close(x[kind].x, bx[kind].x, rtol=0, atol=0)


def test_fixed_evaluation_resumes_by_arm_and_uses_saved_teaching_targets(graph, tmp_path, monkeypatch):
    from mpi4py import MPI
    cfg = dc.default_config(30, 'GIN') | dict(batches=1, batch_size=1, synthetic_count=1,
        target_mode='learned', initialization='fixed', evaluation_epochs=2, trace_every=1, trace_samples=1,
        model_options=model_options('GIN', width=8))
    data = SyntheticInputs([graph], target_mode='learned')
    with torch.no_grad(): data.target_offsets[0]['generator'].fill_(.5)
    artifact = dict(config=cfg, physical_solved_graphs=[graph], **data.snapshot())
    variants = evaluation.controls(artifact, [graph], [graph], cfg)
    assert not torch.equal(variants['best'].batch()[1]['generator'], graph['generator'].y)
    dataset = [benchmark_graph(graph, labels=True)]*2
    cached, done = evaluation.evaluate_controls(variants, cfg, 1e-7, dataset, [0,1], MPI.COMM_SELF, 'cpu', deadline=0)
    assert not done and not cached
    cached, done = evaluation.evaluate_controls({'best': variants['best']}, cfg, 1e-7, dataset, [0,1],
        MPI.COMM_SELF, 'cpu', student_path=tmp_path/'student.pt')
    assert done and cached['best']['student_updates'] == 2
    assert [r['step'] for r in cached['best']['per_initialization_metrics'][0]['learning_curve']] == [0,1,2]
    saved = deepcopy(cached['best'])
    calls, panel = [], evaluation.panel
    def capture(*args, **kwargs):
        calls.append(args[1]['initialization'])
        return panel(*args, **kwargs)
    monkeypatch.setattr(evaluation, 'panel', capture)
    results, done = evaluation.evaluate_controls(variants, cfg, 1e-7, dataset, [0,1], MPI.COMM_SELF, 'cpu', results=cached)
    assert done and results['best'] == saved and len(calls) == len(variants)-1
    assert calls.count('random') == 2
    student = torch.load(tmp_path/'student.pt', weights_only=False)
    artifact.update(model_checkpoint=student, student_rate=1e-7)
    replayed = evaluation.fit_export(artifact)
    torch.testing.assert_close(replayed.state_dict(), student['trained_state'], rtol=0, atol=0)


def test_source_retention_is_final_only_and_excludes_selection_and_test(graph, tmp_path, monkeypatch):
    from mpi4py import MPI
    from distillation.test.test_dd_pretrained import checkpoint
    from distillation.experiments.study import retention_inputs
    from distillation.synthetic.learning import pretrained_config
    source = tmp_path/'pglib_opf_case30_ieee'
    source.mkdir(); (source/'manifest.json').write_text('{}')
    path = tmp_path/'best.pt'; checkpoint(graph, path)
    cfg = pretrained_config(dc.default_config(57, 'GIN') | dict(initialization='pretrained',
        pretrained_checkpoint=str(path), synthetic_count=1, batches=1, batch_size=1, evaluation_epochs=1,
        validation_samples=2, retention_samples=3, data_root=str(tmp_path), trace_every=0,
        student_rates=[1e-7], calibration_samples=2, calibration_panels=1))
    cfg = retention_inputs(cfg)
    reads = []
    class Source:
        folder = source
        def splits(self, seed):
            assert seed == 42
            return dict(train=np.arange(2), val=np.arange(2, 10), test=np.arange(10, 12))
        def __getitem__(self, index):
            reads.append(index)
            assert index in (4, 5, 6)
            return benchmark_graph(graph, labels=True)
    monkeypatch.setattr(evaluation, 'CaseDataset', lambda *args: Source())
    target = [benchmark_graph(graph, labels=True)]*2
    data = SyntheticInputs([graph])
    evaluation.calibrate([graph], cfg, target, np.arange(2), MPI.COMM_SELF, 'cpu')
    selection, _ = evaluation.panel(data, cfg, 1e-7, target, [0, 1], MPI.COMM_SELF, 'cpu')
    assert not reads and 'source_retention' not in selection
    results, complete = evaluation.evaluate_controls(dict(best=data, pretrained_baseline=data), cfg,
        1e-7, target, [0, 1], MPI.COMM_SELF, 'cpu')
    assert complete and reads == [4, 5, 6]*2
    for summary in results.values():
        report = summary['source_retention'][source.name]
        assert report['validation_ids'] == [4, 5, 6] and not report['test_accessed']
        assert np.isfinite(report['mean']) and report['data_sha256'] == cfg['retention_data'][source.name]


@pytest.mark.parametrize('runner', [dc, hm], ids=['DCGM','HMDC'])
def test_evaluation_only_resume_keeps_selected_export_frozen(graph, tmp_path, monkeypatch, runner):
    from mpi4py import MPI
    class Dataset(list):
        def splits(self, seed):
            return dict(train=np.arange(8), val=np.arange(8,12), test=np.arange(12,14))
    dataset=Dataset([benchmark_graph(graph, labels=True)]*14)
    monkeypatch.setattr(runner,'CaseDataset',lambda *a: dataset)
    def initialize(config, folder, comm):
        torch.save(dict(graphs=[graph],ids=[0]),folder/'original.pt')
        torch.save(dict(graphs=[graph]),folder/'initial.pt')
        return [graph],[0],[graph],[{}]
    monkeypatch.setattr(runner,'initialize',initialize)
    cfg=(dc.default_config(30,'GIN') if runner is dc else hm.default_config(30)) | dict(
        target_mode='learned', initialization='fixed', batches=1,batch_size=1,synthetic_count=1,
        evaluation_epochs=1,student_epochs=1,student_rates=[1e-7],calibration_panels=1,calibration_samples=4,
        restart_rounds=2,max_updates=1,real_steps=1,candidate_pool=2,real_batch=1,
        validation_samples=2,final_samples=2,eval_batch=2,trace_every=0)
    cfg.update(model_options=model_options('GIN',width=8),models={k:model_options(k,width=8) for k in ['GIN','GAT']})
    finish=runner.final_evaluation
    def interrupt(*args,**kwargs):
        return finish(*args,**(kwargs | {'deadline':0}))
    monkeypatch.setattr(runner,'final_evaluation',interrupt)
    folder=tmp_path/'run'
    assert runner.run(cfg,folder,MPI.COMM_SELF,'cpu')['status']=='FINALIZING'
    saved=(folder/'best.pt').read_bytes()
    monkeypatch.setattr(runner,'final_evaluation',finish)
    def no_reselection(*a,**k): raise AssertionError('Selection reran during final evaluation resume')
    monkeypatch.setattr(runner,'panel' if runner is dc else 'selection_panel',no_reselection)
    assert runner.run(cfg,folder,MPI.COMM_SELF,'cpu')['status']=='COMPLETE'
    assert (folder/'best.pt').read_bytes()==saved
