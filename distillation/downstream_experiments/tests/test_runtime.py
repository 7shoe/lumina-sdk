"""Runtime changes must preserve scientific identity and isolate failed students."""
import json
from pathlib import Path
import time

import pytest

from distillation.downstream_experiments import runtime
from distillation.downstream_experiments.tests.test_downstream import config, prepared
from distillation.training.T1_single_topo.common import read_json, write_json


def bundle(config):
    config['seeds'] = [42, 43]
    manifest, _ = prepared(config)
    path = Path(manifest['paths']['models'])/'manifest.json'
    value = dict(run_ids=[r['run_id'] for r in manifest['runs']], devices=[0], output=str(path.parent))
    return path, value


def test_late_worker_does_not_start_another_python(config, monkeypatch):
    path, value = bundle(config)
    monkeypatch.setattr(runtime, 'run_child', lambda *a: (_ for _ in ()).throw(AssertionError('Too late')))
    assert runtime.worker(path, value, 0, 'cpu', time.time()+20) == {}


def test_late_retry_preserves_previous_failure_and_completion(config, monkeypatch):
    path, value = bundle(config)
    previous = dict(zip(value['run_ids'], ['failed', 'complete']))
    report = Path(value['output'])/'worker-0.json'
    write_json(report, previous)
    monkeypatch.setattr(runtime, 'run_child', lambda *a: pytest.fail('Too late'))
    assert runtime.worker(path, value, 0, 'cpu', time.time()+20) == previous
    assert read_json(report) == previous


def test_pause_stops_worker_without_launching_next_run(config, monkeypatch):
    path, value = bundle(config)
    calls = []
    monkeypatch.setattr(runtime, 'run_child', lambda *a: calls.append(a) or 0)
    result = runtime.worker(path, value, 0, 'cpu', time.time()+300)
    assert len(calls) == 1 and list(result.values()) == ['paused']


def test_failed_student_does_not_cancel_next(config, monkeypatch):
    path, value = bundle(config)
    codes = iter([17, 0])
    monkeypatch.setattr(runtime, 'run_child', lambda *a: next(codes))
    result = runtime.worker(path, value, 0, 'cpu', time.time()+300)
    assert list(result.values()) == ['failed', 'paused']


def test_committed_result_wins_over_shutdown_exit_code(config, monkeypatch):
    path, value = bundle(config)
    value['run_ids'] = value['run_ids'][:1]
    checks = iter([False, True])
    monkeypatch.setattr(runtime, 'completed', lambda *a: next(checks))
    monkeypatch.setattr(runtime, 'run_child', lambda *a: -15)
    assert list(runtime.worker(path, value, 0, 'cpu', time.time()+300).values()) == ['complete']


def test_busy_student_returns_retry_code_and_expired_student_does_not_start(config, monkeypatch):
    path, value = bundle(config)
    monkeypatch.setattr(runtime, 'train_run', lambda *a, **k: 'busy')
    assert runtime.student(path, value['run_ids'][0], 'cpu', time.time()+10) == 75
    monkeypatch.setattr(runtime, 'train_run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Expired')))
    assert runtime.student(path, value['run_ids'][0], 'cpu', time.time()-1) == 0


def test_heartbeat_includes_staging_and_final_state(tmp_path):
    state = dict(status='running', phase='staging')
    with runtime.heartbeat(tmp_path, state):
        assert read_json(tmp_path/'status.json')['phase'] == 'staging'
        state.update(status='resumable', phase='finished')
    assert read_json(tmp_path/'status.json')['status'] == 'resumable'


def test_compact_node_catalog_preserves_exact_run_ids_and_settings(config, tmp_path, monkeypatch):
    path, _ = bundle(config)
    original = read_json(path)
    hostfile = tmp_path/'hosts'; hostfile.write_text('node-a\nnode-b\n')
    monkeypatch.setenv('PBS_NODEFILE', str(hostfile))
    monkeypatch.delenv('DOWNSTREAM_STARTED', raising=False)
    async def inspect(wrapper, hosts, directory, deadline, **kwargs):
        assert len(wrapper['experiments']) == 2
        seen = []
        for row in wrapper['experiments']:
            subset = read_json(row['manifest'])
            assert subset['sources'] == original['sources']
            assert subset['config'] == original['config']
            assert subset['paths'] == original['paths']
            seen.extend(subset['runs'])
        assert sorted(seen, key=lambda r:r['run_id']) == sorted(original['runs'], key=lambda r:r['run_id'])
        return dict(counts={'RESUMABLE': 2})
    monkeypatch.setattr(runtime, 'supervise', inspect)
    assert runtime.launch(path, 300) == 0
    # PBS returns without scanning every per-run report on Lustre.
    assert not (path.parent/'results.json').exists()
    collection, = (path.parent/'jobs').glob('*/collection.json')
    request = read_json(collection)
    assert request['status'] == 'deferred' and request['manifest'] == str(path)
    assert request['command'][-3:] == ['collect', '--manifest', str(path)]


def test_pending_runs_checks_artifacts_and_preserves_order(config, monkeypatch):
    path, _ = bundle(config)
    manifest = read_json(path)
    finished = manifest['runs'][0]['run_id']
    seen = []
    def verify(folder, spec):
        seen.append(folder.name)
        return folder.name == finished
    monkeypatch.setattr(runtime, 'completed', verify)
    assert runtime.pending_runs(manifest) == manifest['runs'][1:]
    assert set(seen) == {r['run_id'] for r in manifest['runs']}
    def corrupt(*args):
        raise ValueError('Completed result changed')
    monkeypatch.setattr(runtime, 'completed', corrupt)
    with pytest.raises(ValueError, match='Completed result changed'):
        runtime.pending_runs(manifest)


def test_mpi_launch_has_exactly_one_rank_and_one_host(tmp_path):
    folder = tmp_path/'node'; folder.mkdir()
    subset = tmp_path/'subset.json'
    write_json(subset, dict(experiments=[dict(output=str(folder), manifest=str(folder/'manifest.json'))]))
    argv = runtime.mpi_command(tmp_path/'hosts', subset, 100, tmp_path/'result.json')
    assert argv[argv.index('-n')+1] == argv[argv.index('--ppn')+1] == '1'
    assert argv[argv.index('--hostfile')+1] == str(tmp_path/'hosts')
    assert argv[argv.index('-m')+1] == runtime.MODULE
