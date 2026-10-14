"""Checkpoint-conditioned DD: reset, meta-gradients, controls, replay and resume."""
from copy import deepcopy
import json
import runpy
import sys

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from distillation.DD import campaign as dd
from distillation.synthetic.learning import benchmark_graph, model_options
from distillation.test import test_dd_demo as reference
from experiments.lumina_bench.models import SUPPORTED_MODELS

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def checkpoint(graph, path, kind='GIN', family='t1', variant='final'):
    options = model_options(kind, width=8)
    config = dd.default_config(30, kind) | {'model_options': options}
    model = dd.model_for(graph, config, 42, 'cpu')
    # One real supervised update makes this distinct from the saved random seed.
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    opt = torch.optim.SGD(model.parameters(), lr=1e-6)
    dd.mse(model(real), {k: real[k].y for k in ('bus', 'generator')}, real).backward()
    opt.step()
    spec = dict(format=f'lumina-{family}-v1', model=kind, model_options=options, seed=42, split_seed=42)
    spec.update({'case': 'pglib_opf_case30_ieee'} if family == 't1' else
                {'cases': ['pglib_opf_case30_ieee', 'pglib_opf_case57_ieee']})
    payload = dict(specification=spec, model=deepcopy(model.state_dict()))
    payload.update({'samples': 1} if variant == 'best' else {'state': {'samples': 1}, 'optimizer': opt.state_dict()})
    torch.save(payload, path)
    return payload


def config(path, kind='GIN', mode='learned'):
    return dd.default_config(118, kind) | dict(initialization='pretrained', pretrained_checkpoint=str(path),
        batches=1, batch_size=1, passes=2, synthetic_count=1, target_mode=mode, inner_rate=1e-7)


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
@pytest.mark.parametrize('family', ['t1', 't2'])
def test_all_backbones_reset_pretrained_weights_and_differentiate(graph, tmp_path, kind, family):
    path = tmp_path / 'final.pt'
    source = checkpoint(graph, path, kind, family)
    cfg = dd.pretrained_config(config(path, kind))
    assert cfg['case'] == 118 and cfg['bus_range'] == [89, 148]
    assert cfg['model_options'] == source['specification']['model_options']
    assert cfg['pretrained']['samples'] == 1
    assert dd.student_seed(cfg, 0, 0, 12) == dd.student_seed(cfg, 8, 11, 12) == 42
    data = dd.Schedule([graph], cfg)
    model = dd.model_for(graph, cfg, 999, 'cpu')
    for key, value in source['model'].items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    loss = dd.outer_loss(model, data, real, create_graph=True)
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in data.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert data.rate_logits.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in data.target_offsets.parameters()) > 0
    # Neither inner training nor a later mutated student may contaminate resets.
    for key, value in source['model'].items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        next(model.parameters()).add_(1)
    again = dd.model_for(graph, cfg, 123, 'cpu')
    for key, value in source['model'].items():
        torch.testing.assert_close(again.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['solver', 'frozen', 'learned'])
def test_pretrained_panels_and_portable_export(graph, tmp_path, monkeypatch, mode):
    class Panel:
        rank = 0
        def allgather(self, value):
            return [value, value]
    path = tmp_path / 'best_source.pt'
    source = checkpoint(graph, path, family='t2', variant='best')
    cfg = dd.pretrained_config(config(path, mode=mode))
    data = dd.Schedule([graph], cfg)
    dataset = [benchmark_graph(graph, labels=True)]
    metrics, student = dd.panel(data, cfg, dataset, [0], Panel(), 'cpu')
    assert metrics['model_seeds'] == [42, 42] and metrics['unique_initializations'] == 1
    assert metrics['pretrained_sha256'] == cfg['pretrained']['sha256']
    for key, value in source['model'].items():
        torch.testing.assert_close(student['initial_state'][key], value, rtol=0, atol=0)
    _, untouched = dd.panel(data, cfg, dataset, [0], Panel(), 'cpu', trained=False)
    for key, value in source['model'].items():
        torch.testing.assert_close(untouched['trained_state'][key], value, rtol=0, atol=0)
    dd.export(tmp_path, data, cfg, {}, [graph], [], student)
    artifact = torch.load(tmp_path / 'distilled.pt', weights_only=False)
    path.unlink()  # An exported curriculum must not depend on the source file.
    model = dd.model_from_export(artifact, 'cpu', seed=333)
    assert dd.replay_export(model, artifact, dataset, [0], 1) == pytest.approx(metrics['mean'], abs=1e-9)
    if mode == 'learned':
        output = tmp_path / 'replayed.pt'
        monkeypatch.setattr(sys, 'argv', ['replay_campaign', str(tmp_path / 'distilled.pt'),
                                         '--seed', '333', '--output', str(output)])
        runpy.run_module('distillation.DD.aux.replay_campaign', run_name='__main__')
        replayed = torch.load(output, weights_only=False)
        assert replayed['seed'] == 42
        for key, value in student['trained_state'].items():
            torch.testing.assert_close(replayed['model_state'][key], value, rtol=0, atol=0)


def test_reject_incompatible_or_changed_checkpoint(graph, tmp_path):
    path = tmp_path / 'source.pt'
    source = checkpoint(graph, path)
    cfg = dd.pretrained_config(config(path))
    with pytest.raises(ValueError, match='does not match'):
        dd.pretrained_config(config(path, 'GAT'))
    with pytest.raises(ValueError, match='requires pretrained_checkpoint'):
        dd.validate_config(dd.default_config(118, 'GIN') | {'initialization': 'pretrained'})
    with pytest.raises(ValueError, match='Set initialization=pretrained'):
        dd.validate_config(config(path) | {'initialization': 'random'})
    source['model'].pop(next(iter(source['model'])))
    torch.save(source, path)
    with pytest.raises(ValueError, match='checkpoint changed'):
        dd.pretrained_config(cfg)
    with pytest.raises(RuntimeError, match='Missing key'):
        dd.model_for(graph, config(path), 0, 'cpu')
    source['specification']['split_seed'] = 43
    torch.save(source, path)
    with pytest.raises(ValueError, match='split_seed=42'):
        dd.pretrained_config(config(path))


@pytest.mark.parametrize('calibrated', [False, True])
def test_run_resume_controls_and_source_identity(graph, tmp_path, monkeypatch, calibrated):
    from mpi4py import MPI
    from distillation.synthetic import experiments
    class Dataset(list):
        def splits(self, seed):
            assert seed == 42
            return dict(train=np.arange(8), val=np.arange(8, 12), test=np.arange(12, 14))
    dataset = Dataset([benchmark_graph(graph, labels=True)] * 14)
    monkeypatch.setattr(dd, 'CaseDataset', lambda root, case: dataset)
    monkeypatch.setattr(experiments, 'initialize', lambda *args: ([graph], [0], [graph], [{}]))
    path = tmp_path / 'source.pt'
    source = checkpoint(graph, path)
    source_bytes = path.read_bytes()
    cfg = config(path) | dict(generation_protocol='conservative', real_batch=1, eval_batch=1,
        validation_samples=2, final_samples=2, calibration_samples=2, real_control_rates=[1e-7],
        max_updates=2, reserve_seconds=0, trace_every=1, trace_samples=1)
    calls = []
    if calibrated:
        from distillation.synthetic import evaluation
        original_calibrate = evaluation.calibrate
        def calibrate(*args, **kwargs):
            calls.append(True)
            return original_calibrate(*args, **kwargs)
        monkeypatch.setattr(evaluation, 'calibrate', calibrate)
        cfg.update(calibrate_inner_rate=True, checkpoint_inner=True, student_rates=[1e-5],
                   student_rate_factor=.1, calibration_panels=1, evaluation_epochs=2, inner_rate_cap=5e-7)
    folder = tmp_path / 'run'
    state = dd.run(cfg, folder, MPI.COMM_SELF, 'cpu', stop_after_updates=1)
    assert state['status'] == 'RESUMABLE' and state['iteration'] == 1
    if calibrated:
        assert state['initial_inner_rate'] == pytest.approx(5e-7)
    before = (folder / 'config.json').read_bytes()
    source['model'][next(iter(source['model']))].add_(.1)
    torch.save(source, path)
    with pytest.raises(ValueError, match='pretrained checkpoint differs'):
        dd.run(cfg, folder, MPI.COMM_SELF, 'cpu')
    assert (folder / 'config.json').read_bytes() == before
    path.write_bytes(source_bytes)
    state = dd.run(cfg, folder, MPI.COMM_SELF, 'cpu')
    assert state['status'] == 'COMPLETE' and state['iteration'] == 2 and state['resume_count'] == 2
    report = json.loads((folder / 'evaluation.json').read_text())
    assert 'pretrained_baseline' in report and 'untrained' not in report
    assert 'best_distilled_unseen' not in report
    for name in ('best_distilled', 'real_subset_tuned_rates', 'pretrained_baseline'):
        assert report[name]['unique_initializations'] == 1
        assert report[name]['pretrained_sha256'] == report['pretrained']['sha256']
    assert report['case'] == 118 and not report['test_accessed']
    if calibrated:
        assert len(calls) == 1  # Resume and ordinary-SGD controls reuse the train-only calibration.
        common = json.loads((folder/'common_evaluation.json').read_text())
        assert common['calibration']['chosen_rate'] == pytest.approx(1e-6)
