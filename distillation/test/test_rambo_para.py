"""RAMBO* preserves physical labels, frozen batches, restart and baseline isolation."""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import threading
import time

import numpy as np
import pytest
import torch

from distillation.RAMBO import run as sequential
from distillation.RAMBO.data import network_record
from distillation.RAMBO_para import run as runner
from distillation.RAMBO_para import launch
from distillation.conversion.matrix_form import mat2opf, opf2mat
from distillation.experiments.datasets import load_dataset
from distillation.selection import input_view
from distillation.solver import ACOPF, PyPowerSolver, Solution
from distillation.synthetic import artifacts
from distillation.training.T1_single_topo.common import file_hash, write_json
from distillation.training.T1_single_topo.state import Stop
from distillation.test.test_dd_demo import cpu_threads, record, graph


def point_of(graph):
    solution, p = Solution.from_label(graph), opf2mat(graph).params
    return dict(theta=solution.theta.tolist(), vm=solution.vm.tolist(), pg=solution.pg.tolist(),
                qg=solution.qg.tolist(), pd=p.pd.tolist(), qd=p.qd.tolist(), cost=solution.objective)


@pytest.fixture
def backend(monkeypatch, graph):
    template = input_view(graph)
    provenance = dict(data_identity='fixture', split_hashes={'train': 'train-only'},
                      nominal_case_sha256='fixture-case')
    monkeypatch.setattr(runner, 'prepare_template', lambda *args: (template, provenance))
    monkeypatch.setattr(sequential, 'prepare_template', lambda *args: (template, provenance))

    class Backend:
        versions = dict(julia='test')
        lock = threading.Lock()
        requests = []
        active = 0
        peak = 0
        fail = None
        duplicate = False
        pause = None

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, request, timeout):
            cls = type(self)
            stage, seed = request['stage'], request['seed']
            with cls.lock:
                cls.requests.append(deepcopy(request))
                cls.active += 1
                cls.peak = max(cls.peak, cls.active)
            try:
                time.sleep(.015 * (1 + seed % 3))
                if cls.fail == (stage, seed):
                    raise RuntimeError('scripted worker failure')
                if stage == 'opf':
                    mats = opf2mat(template)
                    params = replace(mats.params, pd=torch.tensor(request['pd'], dtype=torch.float64),
                                     qd=torch.tensor(request['qd'], dtype=torch.float64))
                    problem = ACOPF(mat2opf(replace(mats, params=params), include_solution=False))
                    solution = PyPowerSolver().solve(problem)
                    assert solution.success
                    point = point_of(solution.to_hetero(problem))
                else:
                    point = deepcopy(request['history'][-1])
                    factor = 1. if cls.duplicate else .90 + .2 * (seed % 10007) / 10007
                    point['pd'] = (np.asarray(request['network']['pd']) * factor).tolist()
                    point['qd'] = (np.asarray(request['network']['qd']) * factor).tolist()
                    point['qg'] = [123.] * len(point['qg'])  # Never use these relaxed labels.
                if cls.pause and cls.pause[0] == stage:
                    cls.pause[1].signalled = True
                return dict(success=True, status='LOCALLY_SOLVED', point=point, objective=1., wall_seconds=.01)
            finally:
                with cls.lock:
                    cls.active -= 1
    return Backend


def execute(config, folder, backend, **kwargs):
    return runner.run(config, folder, dataset=object(), backend_factory=backend, bind=False, **kwargs)


def state(folder):
    return json.loads((folder / 'state.json').read_text())


@pytest.mark.parametrize('case', [14, 30, 57, 118, 500])
def test_separate_identity_and_same_physical_configuration(case):
    old = sequential.configuration(case=f'case{case}')
    cfg = runner.configuration(case=f'case{case}')
    assert cfg['method'] == 'RAMBO_para' and old['method'] == 'rambo'
    assert cfg['proposal_batch_size'] == 32
    assert all(cfg[k] == v for k, v in old.items() if k != 'method')
    with pytest.raises(ValueError, match='identity'):
        runner.configuration(method='rambo')
    with pytest.raises(ValueError, match='positive integer'):
        runner.configuration(proposal_batch_size=0)


def test_workers_preserve_batch_order_history_and_export(tmp_path, backend):
    cfg = runner.configuration(count=5, proposal_batch_size=3)
    serial, parallel = tmp_path / 'serial', tmp_path / 'parallel'
    assert execute(cfg, serial, backend, workers=1)['status'] == 'complete'
    backend.requests.clear()
    assert execute(cfg, parallel, backend, workers=3)['status'] == 'complete'
    assert backend.peak >= 2, 'Must actually execute candidates concurrently'
    a, b = state(serial), state(parallel)
    assert a['library'] == b['library']
    assert [r['input_sha256'] for r in a['records']] == [r['input_sha256'] for r in b['records']]
    assert [r.get('history_count') for r in b['records']] == [0, 1, 1, 1, 4]
    assert all(max(abs(v) for v in p['qg']) < 10 for p in b['library'])
    loaded = artifacts.load_dataset(parallel)
    assert loaded['metadata']['method'] == 'RAMBO_para'
    assert loaded['metadata']['display_name'] == 'RAMBO*'
    assert loaded['metadata']['dataset_id'].startswith('RAMBO_para-')
    assert not loaded['metadata']['split']['test_accessed']
    assert not loaded['metadata']['split']['validation_accessed']
    for physical, converted in zip(loaded['physical_solved_graphs'], load_dataset(parallel)['graphs']):
        for kind in ('bus', 'generator'):
            torch.testing.assert_close(converted[kind].y, physical[kind].y, rtol=0, atol=0)
    # Publication is independent of the native stage files / original training data.
    (parallel / 'samples').rename(parallel / 'samples_hidden')
    assert len(load_dataset(parallel)['graphs']) == 5


def test_resume_reuses_stages_and_can_change_workers(tmp_path, backend):
    cfg = runner.configuration(count=4, proposal_batch_size=3)
    assert execute(cfg, tmp_path, backend, workers=2, session_batches=1)['accepted'] == 1
    stop = Stop(3300)
    backend.pause = ('feasible', stop)
    result = execute(cfg, tmp_path, backend, workers=2, stop=stop)
    assert result['status'] == 'paused' and result['accepted'] == 1
    saved = Counter(r['seed'] for r in backend.requests if r['stage'] == 'feasible')
    assert saved
    backend.pause = None
    result = execute(cfg, tmp_path, backend, workers=3)
    assert result['status'] == 'complete'
    after = Counter(r['seed'] for r in backend.requests if r['stage'] == 'feasible')
    assert all(after[seed] == calls for seed, calls in saved.items())
    before, calls = file_hash(tmp_path / 'dataset.pt'), len(backend.requests)
    assert execute(cfg, tmp_path, backend, workers=1) == result
    assert before == file_hash(tmp_path / 'dataset.pt') and len(backend.requests) == calls
    with pytest.raises(ValueError, match='changed'):
        execute(cfg | dict(proposal_batch_size=2), tmp_path, backend)


def test_worker_failure_keeps_other_results_and_retry_order(tmp_path, backend):
    cfg = runner.configuration(count=4, proposal_batch_size=3)
    execute(cfg, tmp_path, backend, workers=2, session_batches=1)
    seed = int(np.random.default_rng(np.random.SeedSequence([42, 1])).integers(0, 2**31))
    backend.fail = ('warm_opf', seed)
    result = execute(cfg, tmp_path, backend, workers=3)
    assert result['status'] == 'failed' and result['accepted'] == 1
    assert state(tmp_path)['batch']['history_count'] == 1
    completed = Counter((r['seed'], r['stage']) for r in backend.requests if r['seed'] != seed)
    backend.fail = None
    assert execute(cfg, tmp_path, backend, workers=1)['status'] == 'complete'
    assert Counter((r['seed'], r['stage']) for r in backend.requests if r['seed'] != seed) == completed
    assert [r['attempt'] for r in state(tmp_path)['records']] == [0, 1, 2, 3]


def test_duplicates_rejected_against_other_members_of_same_batch(tmp_path, backend):
    backend.duplicate = True
    cfg = runner.configuration(count=4, proposal_batch_size=3, max_attempts=7)
    result = execute(cfg, tmp_path, backend, workers=3)
    assert result['status'] == 'exhausted' and result['accepted'] == 2
    assert result['rejections'] == 5
    assert not (tmp_path / 'dataset.pt').exists()
    decisions = json.loads((tmp_path / 'batches/000001.json').read_text())['decisions']
    assert [d['status'] for d in decisions] == ['accepted', 'rejected', 'rejected']


def test_cached_stage_and_accepted_graph_corruption_are_detected(tmp_path, backend):
    cfg = runner.configuration(count=2, proposal_batch_size=1)
    execute(cfg, tmp_path, backend, workers=1, session_batches=1)
    stop = Stop(3300)
    backend.pause = ('feasible', stop)
    execute(cfg, tmp_path, backend, workers=1, stop=stop)
    path = tmp_path / 'attempts/000001/feasible.json'
    cached = json.loads(path.read_text())
    cached['result']['objective'] = 999.
    write_json(path, cached)
    backend.pause = None
    with pytest.raises(ValueError, match='Cached proposal changed'):
        execute(cfg, tmp_path, backend)
    (tmp_path / 'samples/000000.pt').write_bytes(b'bad')
    with pytest.raises(ValueError, match='changed'):
        execute(cfg, tmp_path, backend)


def test_legacy_directory_is_rejected_before_any_write(tmp_path, backend):
    path = tmp_path / 'config.json'
    write_json(path, dict(format='lumina-rambo-v1'))
    before = {p.name: file_hash(p) for p in tmp_path.iterdir()}
    with pytest.raises(ValueError, match='read-only'):
        execute(runner.configuration(), tmp_path, backend)
    assert {p.name: file_hash(p) for p in tmp_path.iterdir()} == before
    with pytest.raises(ValueError, match='read-only'):
        execute(runner.configuration(), tmp_path / 'nested', backend)
    assert not (tmp_path / 'nested').exists()
    with pytest.raises(ValueError, match='read-only'):
        launch.prepare(tmp_path / 'nested', runner.configuration())
    assert not (tmp_path / 'nested').exists()


def test_batch_one_reproduces_sequential_proposals(tmp_path, backend):
    old, new = tmp_path / 'old', tmp_path / 'new'
    sequential.run(sequential.configuration(count=3), old, backend_factory=backend, dataset=object())
    execute(runner.configuration(count=3, proposal_batch_size=1), new, backend, workers=2)
    assert state(old)['library'] == state(new)['library']


def test_portable_legacy_artifact_and_foreign_directory_are_not_overwritten(tmp_path, backend):
    published, unrelated = tmp_path / 'published', tmp_path / 'unrelated'
    write_json(published / 'artifact.json', dict(method='RAMBO', format='lumina-synthetic-dataset-v1'))
    before = file_hash(published / 'artifact.json')
    with pytest.raises(ValueError, match='read-only'):
        execute(runner.configuration(), published, backend)
    assert file_hash(published / 'artifact.json') == before
    assert not (published / 'run.lock').exists()
    unrelated.mkdir()
    (unrelated / 'dataset.pt').write_bytes(b'previous dataset')
    with pytest.raises(FileExistsError, match='fresh directory'):
        execute(runner.configuration(), unrelated, backend)
    assert list(unrelated.iterdir()) == [unrelated / 'dataset.pt']


def test_bootstrap_is_a_separate_verified_copy_and_needs_no_old_files_on_resume(tmp_path, backend):
    old, new = tmp_path / 'old', tmp_path / 'new'
    sequential.run(sequential.configuration(count=2), old, backend_factory=backend, dataset=object())
    before = {str(p.relative_to(old)): file_hash(p) for p in old.rglob('*') if p.is_file()}
    cfg = runner.configuration(count=4, proposal_batch_size=2, bootstrap=old / 'dataset.json')
    result = execute(cfg, new, backend, workers=2, session_batches=1)
    assert result['status'] == 'paused' and result['accepted'] == 3
    assert {str(p.relative_to(old)): file_hash(p) for p in old.rglob('*') if p.is_file()} == before
    original = state(old)
    assert state(new)['library'][:2] == original['library']
    old.rename(tmp_path / 'old_moved')
    result = execute(cfg, new, backend, workers=1)
    assert result['status'] == 'complete'
    meta = artifacts.load_dataset(new)['metadata']
    assert meta['source_generation']['bootstrap']['count'] == 2
    assert meta['method'] == 'RAMBO_para'
    assert execute(cfg, new, backend, workers=1) == result


def test_physical_failure_never_publishes(tmp_path, backend, monkeypatch):
    original = runner.attach_solution
    calls = 0
    def reject_after_bootstrap(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError('physical verification failed')
        return original(*args, **kwargs)
    monkeypatch.setattr(runner, 'attach_solution', reject_after_bootstrap)
    result = execute(runner.configuration(count=2, max_attempts=3), tmp_path, backend, workers=2)
    assert result['status'] == 'exhausted' and result['accepted'] == 1
    assert not (tmp_path / 'artifact.json').exists()


def test_launch_frozen_manifest_and_no_overwrite(tmp_path, monkeypatch):
    source = 'distillation/RAMBO_para/run.py'
    monkeypatch.setattr(launch, 'source_files', lambda: {source: file_hash(launch.ROOT / source)})
    runtime = tmp_path / 'runtime.tar'
    runtime.write_bytes(b'fixture runtime')
    root = tmp_path / 'campaign'
    cfg = runner.configuration(case='case500')
    manifest = launch.prepare(root, cfg, runtime)
    assert manifest['method'] == 'RAMBO_para'
    assert launch.verify(root) == manifest
    with pytest.raises(FileExistsError):
        launch.prepare(root, cfg, runtime)
    (root / 'source' / source).write_text('changed')
    with pytest.raises(ValueError, match='Frozen source changed'):
        launch.verify(root)
