"""Shared initialization, DCGM/HMDC derivatives, and resumable adaptation."""
from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from distillation.DC import campaign as dc
from distillation.DC import dcgm
from distillation.HMDC import campaign as hm, hmdc
from distillation.HMDC.models import build_pair, model_config, pretrained_pair_config
from distillation.synthetic.inputs import SyntheticInputs
from distillation.synthetic.learning import (benchmark_graph, model_for, model_from_export,
                                             model_options, pretrained_config)
from distillation.test import test_dd_demo as reference
from distillation.test.test_dd_pretrained import checkpoint
from experiments.lumina_bench.models import SUPPORTED_MODELS

cpu_threads, record, graph = reference.cpu_threads, reference.record, reference.graph


def configuration(paths, *, pair=False):
    cfg = dict(case=118, initialization='pretrained', model_precision='float64', split_seed=42,
        strength=.02, batch_size=1, batches=1, synthetic_count=1, eval_batch=2,
        evaluation_epochs=1, student_epochs=1, student_rates=[1e-7], student_rate_factor=1.,
        calibration_panels=1, data_rate=1e-9, restart_rounds=2, max_updates=3,
        training_seconds=1000, reserve_seconds=0, checkpoint_seconds=9999, eval_seconds=9999,
        validation_samples=2, final_samples=2, data_root='mock', real_batch=1)
    if pair:
        cfg.update(pair=list(paths), evaluation_models=list(paths), pretrained_checkpoints=paths,
                   models={}, real_steps=1, query_interval=1, candidate_pool=2, alignment_rate=.001,
                   mutual=True, balanced=True, balance_interval=2)
    else:
        kind, path = next(iter(paths.items()))
        cfg.update(model=kind, model_options=model_options(kind, width=32), pretrained_checkpoint=path)
    return cfg


def sources(graph, folder, kinds):
    paths, payloads = {}, {}
    for i, kind in enumerate(kinds):
        path = folder / f'{kind}.pt'
        payloads[kind] = checkpoint(graph, path, kind, 't1' if i == 0 else 't2')
        paths[kind] = str(path)
    return paths, payloads


def assert_weights(model, source):
    for name, value in source.items():
        torch.testing.assert_close(model.state_dict()[name], value.to(model.state_dict()[name]), rtol=0, atol=0)


@pytest.mark.parametrize('kind', SUPPORTED_MODELS)
def test_shared_pretrained_dcgm_derivatives_and_reset(graph, tmp_path, kind):
    paths, states = sources(graph, tmp_path, [kind])
    cfg = pretrained_config(configuration(paths))
    model = model_for(graph, cfg, 99, 'cpu')
    assert next(model.parameters()).dtype == torch.float64
    assert cfg['case'] == 118 and cfg['model_options'] == states[kind]['specification']['model_options']
    assert_weights(model, states[kind]['model'])
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    real['load'].x *= 1.05
    synthetic = SyntheticInputs([graph], cfg['strength'])
    loss, _ = dcgm.data_gradients(model, synthetic, real)
    assert np.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in synthetic.parameters())
    assert sum(p.grad.abs().sum() for p in synthetic.parameters()) > 0
    dcgm.train_student(model, torch.optim.SGD(model.parameters(), lr=1e-7), synthetic, batch_size=1)
    assert any(not torch.equal(v.to(model.state_dict()[k]), model.state_dict()[k])
               for k, v in states[kind]['model'].items())
    assert_weights(model_for(graph, cfg, 100, 'cpu'), states[kind]['model'])


@pytest.mark.parametrize('kinds', [('GAT', 'GIN'), ('HeteroGNN', 'Transformer'), ('HGT', 'HEAT'), ('GCN', 'RGAT'), ('GIN', 'HEAT')])
def test_pretrained_hmdc_pairs_differentiate_resume_and_restart(graph, tmp_path, kinds):
    paths, states = sources(graph, tmp_path, kinds)
    cfg = pretrained_pair_config(configuration(paths, pair=True))
    rates = {k: 1e-7 for k in kinds}
    pair = build_pair(graph, cfg, rates, 50, 'cpu')
    for kind, model in zip(kinds, pair.models):
        assert_weights(model, states[kind]['model'])
    real = Batch.from_data_list([benchmark_graph(graph, labels=True)])
    real['load'].x *= 1.05
    synthetic = SyntheticInputs([graph])
    info = hmdc.data_gradients(pair.models, pair.alignment, pair.balance, synthetic, real)
    assert all(np.isfinite(info['terms']))
    assert all(torch.isfinite(p.grad).all() for p in synthetic.parameters())
    assert sum(p.grad.abs().sum() for p in synthetic.parameters()) > 0
    hmdc.model_step(pair.models, pair.alignment, pair.optimizers, real)
    hmdc.alignment_step(pair.models, pair.alignment, pair.alignment_optimizer, real)
    pair.query_ids = [17]
    resumed = build_pair(graph, cfg, rates, 51, 'cpu')
    for kind, model in zip(kinds, resumed.models):
        assert_weights(model, states[kind]['model'])
    assert not resumed.alignment_optimizer.state and all(not o.state for o in resumed.optimizers)
    resumed.load_state_dict(deepcopy(pair.state_dict()))
    for current in (pair, resumed):
        hmdc.data_gradients(current.models, current.alignment, current.balance, synthetic, real)
        hmdc.model_step(current.models, current.alignment, current.optimizers, real)
        hmdc.alignment_step(current.models, current.alignment, current.alignment_optimizer, real)
    torch.testing.assert_close(pair.state_dict(), resumed.state_dict(), rtol=0, atol=0)


def test_hmdc_requires_explicit_matching_checkpoints(graph, tmp_path):
    paths, _ = sources(graph, tmp_path, ['GIN', 'GAT'])
    cfg = configuration(paths, pair=True)
    with pytest.raises(ValueError, match='exactly evaluation_models'):
        pretrained_pair_config(cfg | {'evaluation_models': ['GIN', 'GAT', 'GCN']})
    with pytest.raises(ValueError, match='Missing pretrained_checkpoints'):
        model_config(cfg | {'pretrained_checkpoints': {}}, 'GIN')
    with pytest.raises(ValueError, match='Set initialization=pretrained'):
        pretrained_pair_config(cfg | {'initialization': 'random'})
    with pytest.raises(ValueError, match='does not match'):
        pretrained_pair_config(cfg | {'pretrained_checkpoints': {'GIN': paths['GAT'], 'GAT': paths['GIN']}})


@pytest.mark.parametrize('runner', [dc, hm], ids=['DCGM', 'HMDC'])
@pytest.mark.parametrize('target_mode', ['solver', 'frozen', 'learned'])
def test_campaign_resumes_in_progress_students_then_restarts_from_source(graph, tmp_path, monkeypatch, runner, target_mode):
    from mpi4py import MPI
    class Dataset(list):
        def splits(self, seed):
            assert seed == 42
            return dict(train=np.arange(8), val=np.arange(8, 12), test=np.arange(12, 14))
    dataset = Dataset([benchmark_graph(graph, labels=True)] * 14)
    monkeypatch.setattr(runner, 'CaseDataset', lambda *args: dataset)
    def initialize(config, folder, comm):
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(dict(graphs=[graph], ids=[0]), folder/'original.pt')
        torch.save(dict(graphs=[graph]), folder/'initial.pt')
        return [graph], [0], [graph], [{}]
    monkeypatch.setattr(runner, 'initialize', initialize)
    kinds = ['GIN'] if runner is dc else ['GIN', 'GAT']
    paths, sources_ = sources(graph, tmp_path, kinds)
    cfg = configuration(paths, pair=runner is hm)
    cfg.update(target_mode=target_mode, target_rate=1e-5, trace_every=1, trace_samples=1)
    recorded = []
    matching = runner.data_gradients
    def capture(models, *args, **kwargs):
        recorded.append([dc.cpu_state(m) for m in (models if isinstance(models, list) else [models])])
        return matching(models, *args, **kwargs)
    monkeypatch.setattr(runner, 'data_gradients', capture)
    folder = tmp_path/'resumed'
    assert runner.run(cfg, folder, MPI.COMM_SELF, 'cpu', stop_after_updates=1)['status'] == 'RESUMABLE'
    saved = torch.load(folder/'latest.pt', weights_only=False)
    mid = [saved['students'][0]['model']] if runner is dc else saved['pairs'][0]['models']
    assert any(not torch.equal(v, recorded[0][i][k]) for i, state in enumerate(mid) for k, v in state.items())
    before = (folder/'config.json').read_bytes()
    source_path = tmp_path/f'{kinds[0]}.pt'
    original_bytes = source_path.read_bytes()
    original_stat = source_path.stat()
    changed = sources_[kinds[0]]
    changed['model'][next(iter(changed['model']))].add_(.1)
    torch.save(changed, source_path)
    # Same-size replacements can retain timestamps; resume must reread bytes.
    os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    with pytest.raises(RuntimeError, match='configuration|config|checkpoint'):
        runner.run(cfg, folder, MPI.COMM_SELF, 'cpu')
    assert (folder/'config.json').read_bytes() == before
    source_path.write_bytes(original_bytes)
    os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    state = runner.run(cfg, folder, MPI.COMM_SELF, 'cpu')
    assert state['status'] == 'COMPLETE' and state['iteration'] == 3 and state['resume_count'] == 2
    torch.testing.assert_close(recorded[1], mid, rtol=0, atol=0)
    torch.testing.assert_close(recorded[2], recorded[0], rtol=0, atol=0)
    # The same ordinary run must end at the same parameters after interruption.
    full = tmp_path/'continuous'
    assert runner.run(cfg, full, MPI.COMM_SELF, 'cpu')['status'] == 'COMPLETE'
    resumed, continuous = [torch.load(p/'latest.pt', weights_only=False) for p in (folder, full)]
    for key in ('synthetic_state', 'optimizer', 'students' if runner is dc else 'pairs'):
        torch.testing.assert_close(resumed[key], continuous[key], rtol=0, atol=0)
    report = json.loads((folder/'evaluation.json').read_text())
    assert not report['test_accessed'] and report['initialization'] == 'pretrained'
    assert report['pretrained'] == resumed['config']['pretrained']
    results = [report['results']] if runner is dc else list(report['results'].values())
    for metrics in results:
        assert 'pretrained_baseline' in metrics and 'untrained' not in metrics
        assert all(m['unique_initializations'] == 1 for m in metrics.values())
    artifact = torch.load(folder/'best.pt', weights_only=False)
    for path in paths.values():
        Path(path).unlink()
    for kind in kinds:
        student = artifact['model_checkpoint'] if runner is dc else artifact['model_checkpoints'][kind]
        model = model_from_export(artifact, 'cpu', model=kind)
        assert_weights(model, student['initial_state'])
        # Reproduce selection-time student training after source files are gone.
        synthetic = SyntheticInputs.from_export(artifact)
        rate = artifact['student_rate'] if runner is dc else artifact['student_rates'][kind]
        dcgm.train_student(model, torch.optim.SGD(model.parameters(), lr=rate), synthetic,
                           epochs=cfg['evaluation_epochs'], batch_size=cfg['batch_size'], seed=student['training_seed'])
        for name, value in student['trained_state'].items():
            torch.testing.assert_close(model.state_dict()[name], value, rtol=1e-10, atol=1e-12)
