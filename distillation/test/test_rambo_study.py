"""RAMBO study boundaries: one generation, matched evaluation and exact physical labels."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from distillation.experiments import study, rambo
from distillation.experiments.smoke import run_rambo
from distillation.selection import ids_hash, input_view
from distillation.synthetic import artifacts
from distillation.synthetic.experiments import common_config
from distillation.synthetic.learning import benchmark_graph, model_options
from distillation.training.T1_single_topo.common import digest, file_hash
from distillation.test.test_experiment_study import spec, cpu_threads, record, graph
from distillation.test.test_rambo import fake_backend


class Serial:
    rank, size = 0, 1
    def allgather(self, value): return [value]
    def bcast(self, value, root=0): return value
    def Barrier(self): pass


def test_rambo_axes_do_not_multiply_topology_or_target_modes(spec):
    spec['methods'] = {'RAMBO': study.load()['methods']['RAMBO']}
    rows = study.expand(spec, {'runtime': 'v1'})['experiments']
    assert len(rows) == 18  # 3 cases, 2 samplers, 3 evaluators; generation itself has six libraries.
    assert len({r['config']['generation_id'] for r in rows}) == 6
    assert all(r['config']['target_mode']=='solver' and r['config']['bus_range']==[r['config']['case']]*2 for r in rows)
    spec['matrix'].update(protocols=['random_size'], target_modes=['learned'])
    assert study.expand(spec, {'runtime': 'v1'})['experiments'] == rows
    spec['matrix']['initializations'] = ['random','fixed']
    expanded = study.expand(spec, {'runtime': 'v1'})['experiments']
    assert len(expanded) == 36 and len({r['config']['generation_output'] for r in expanded}) == 6
    spec['methods']['RAMBO']['settings']['strength'] = .2
    with pytest.raises(ValueError, match='Unknown fields'): study.expand(spec,{})


def test_snapshot_contains_julia_environment_cases_and_rambo_defaults():
    files = study.source_files()
    for name in ('config.yaml','julia/Project.toml','julia/Manifest.toml','julia/worker.jl',
                 'cases/pglib_opf_case30_ieee.m','cases/pglib_opf_case14_ieee.m','cases/pglib_opf_case500_goc.m'):
        assert 'distillation/RAMBO/'+name in files
    assert not any('/aux/' in name for name in files)


@pytest.mark.parametrize('count', [100, 800])
def test_new_cases_share_generation_across_evaluators_and_keep_count_in_identity(spec, tmp_path, count):
    spec['methods'] = {'RAMBO': study.load()['methods']['RAMBO']}
    spec['shared']['synthetic_count'] = count
    group = spec['matrix'] | {'cases': [14, 500], 'initializations': ['random', 'fixed']}
    rows = rambo.expand(spec, tmp_path, {'source': 'test'}, group,
                        lambda case: dict(data_sha256=f'data{case}', reference_sha256=f'ref{case}'))
    assert len(rows) == 24 and len({r['config']['generation_id'] for r in rows}) == 4
    for row in rows:
        cfg = row['config']
        assert cfg['generator']['count'] == cfg['synthetic_count'] == count
        assert cfg['split_seed'] == 42 and cfg['target_mode'] == 'solver'
        assert cfg['bus_range'] == [cfg['case']] * 2
    spec['shared']['synthetic_count'] = 800 if count == 100 else 100
    other = rambo.expand(spec, tmp_path, {'source': 'test'}, group,
                         lambda case: dict(data_sha256=f'data{case}', reference_sha256=f'ref{case}'))
    assert not {r['config']['generation_id'] for r in rows} & {r['config']['generation_id'] for r in other}


def test_delayed_julia_exit_does_not_invalidate_saved_pause(monkeypatch):
    from distillation.RAMBO import backend
    from threading import Event
    reaped = Event()
    class Process:
        pid = 123
        def poll(self): return None
        def wait(self, timeout=None):
            if timeout is not None: raise backend.subprocess.TimeoutExpired('julia', timeout)
            reaped.set()
    class Reader:
        def join(self, timeout): pass
    signals = []
    monkeypatch.setattr(backend.os, 'killpg', lambda pid, sig: signals.append((pid, sig)))
    worker = backend.JuliaBackend.__new__(backend.JuliaBackend)
    worker.process, worker.thread = Process(), Reader()
    worker.close()
    assert reaped.wait(1)
    assert signals == [(123, backend.signal.SIGTERM), (123, backend.signal.SIGKILL)]


@pytest.fixture
def experiment(tmp_path, monkeypatch, graph, fake_backend):
    splits = dict(train=np.arange(8), val=np.arange(8,14), test=np.arange(14,16))
    class Dataset:
        manifest = {'fixture': 'case9'}
        reads = []
        def splits(self, seed):
            assert seed == 42
            return splits
        def __getitem__(self, index):
            assert index not in splits['test']
            self.reads.append(index)
            return benchmark_graph(graph, labels=True)
    dataset = Dataset()
    monkeypatch.setattr(rambo, 'CaseDataset', lambda *args: dataset)
    monkeypatch.setattr(rambo.generator, 'prepare_template', lambda *args: (input_view(graph), dict(
        data_identity=digest(dataset.manifest), split_hashes={k:ids_hash(v) for k,v in splits.items()})))
    original_run = rambo.generator.run
    monkeypatch.setattr(rambo.generator, 'run', lambda *args,**kwargs:
        original_run(*args, **kwargs, backend_factory=fake_backend, dataset=dataset))
    reference = tmp_path/'references.pt'
    torch.save(dict(graphs=[graph,graph], ids=[0,1]), reference)
    cfg = common_config(9) | dict(model='GIN',model_options=model_options('GIN',width=8,layers=2),
        synthetic_count=2,batch_size=1,batches=2,evaluation_epochs=2,student_rates=[1e-7],
        calibration_samples=2,calibration_panels=1,validation_samples=2,final_samples=2,eval_batch=2,
        trace_samples=2,evaluation_reserve_seconds=0,reference_file=str(reference),
        generator=rambo.generator.configuration(count=2),generation_id='RAMBO-test',
        generation_output=str(tmp_path/'generation'),generation_domain='fixture',generation_protocol='rambo',
        artifact_identity=dict(dataset_id='RAMBO-eval-test',method='RAMBO',initialization_name='random'))
    return dict(method='RAMBO',dataset_id='RAMBO-eval-test',config=cfg,output=str(tmp_path/'evaluation'),smoke=True)


def test_generation_evaluation_resume_physical_labels_and_portable_replay(experiment, fake_backend):
    cfg,folder=experiment['config'],Path(experiment['output'])
    result=run_rambo(rambo,experiment,Serial(),'cpu',3300)
    assert result['status']=='COMPLETE'
    smoke=json.loads((folder/'smoke.json').read_text())
    assert smoke['status']=='PASS' and smoke['evaluation_resume'] and smoke['generation_resume']
    loaded=artifacts.load_dataset(folder)
    assert loaded['metadata']['kind']=='generated_operating_points'
    assert loaded['metadata']['dataset_selection'].startswith('entire')
    for physical,actual in zip(loaded['physical_solved_graphs'],loaded['graphs']):
        expected=benchmark_graph(physical,labels=True)
        for kind in actual.node_types:
            torch.testing.assert_close(actual[kind].x,expected[kind].x,rtol=0,atol=0)
        for edge in actual.edge_types:
            torch.testing.assert_close(actual[edge].edge_index,expected[edge].edge_index,rtol=0,atol=0)
            if 'edge_attr' in actual[edge]: torch.testing.assert_close(actual[edge].edge_attr,expected[edge].edge_attr,rtol=0,atol=0)
    before=file_hash(folder/'dataset.pt');calls=len(fake_backend.requests)
    assert rambo.run(cfg,folder,Serial(),'cpu')['status']=='COMPLETE'
    assert len(fake_backend.requests)==calls and file_hash(folder/'dataset.pt')==before
    # The next evaluation uses the SAME physical library, without new solver calls.
    other=deepcopy(experiment);other['output']=str(folder.with_name('fixed'))
    other['config'].update(initialization='fixed',model_seed=71)
    assert run_rambo(rambo,other,Serial(),'cpu',3300)['status']=='COMPLETE'
    assert len(fake_backend.requests)==calls
    report=json.loads((folder/'evaluation.json').read_text())
    assert report['student_updates']==4 and report['validation_ids']==[10,11]
    assert set(report['calibration']['training_ids']) <= set(range(2,8))
    # Export is self-contained: no generation files, OPF solve or original references needed for reuse.
    Path(cfg['generation_output']).rename(folder.parent/'generation_moved')
    _, provenance=artifacts.train_from_artifact(folder,initialization='scratch',seed=17)
    assert provenance['student_updates']==4
    with (folder/'dataset.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='checksum'): artifacts.load_dataset(folder)


def test_exhausted_generation_never_becomes_complete_dataset(experiment, fake_backend):
    fake_backend.fail_stage='bilevel'
    experiment['config']['generator']['max_attempts']=3
    result=rambo.run(experiment['config'],experiment['output'],Serial(),'cpu')
    assert result['status']=='EXHAUSTED'
    record=artifacts.publish(experiment)
    assert not record['complete'] and record['dataset'] is None
    with pytest.raises(ValueError,match='incomplete'): artifacts.load_dataset(experiment['output'])
