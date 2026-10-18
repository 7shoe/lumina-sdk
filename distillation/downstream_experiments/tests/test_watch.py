"""Continuation requires confirmed PBS completion and preserves the frozen campaign."""
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from distillation.downstream_experiments import watch
from distillation.downstream_experiments.tests.test_downstream import config, prepared
from distillation.training.T1_single_topo.common import read_json


def test_waits_for_finished_job_not_running_or_transient_scheduler_error(monkeypatch):
    responses = iter([CompletedProcess([], 1, '', 'PBS unavailable'),
                      CompletedProcess([], 0, 'job_state = R', ''),
                      CompletedProcess([], 0, 'job_state = F', '')])
    monkeypatch.setattr(watch.subprocess, 'run', lambda *a, **k: next(responses))
    sleeps = []; reports = []
    monkeypatch.setattr(watch.time, 'sleep', sleeps.append)
    watch.wait_job('job-1', lambda **kw: reports.append(kw), 60)
    assert sleeps == [60, 60]
    assert 'scheduler_error' in reports[0]
    assert reports[-1]['pbs_state'] == 'F'


def test_resumes_remaining_work_then_collects_once(config, monkeypatch):
    manifest, run = prepared(config)
    path = Path(manifest['paths']['models'])/'manifest.json'
    remaining = iter([[run], []]); waited = []; submitted = []; collected = []
    monkeypatch.setattr(watch, 'wait_job', lambda job, *a: waited.append(job))
    monkeypatch.setattr(watch, 'pending_runs', lambda m: next(remaining))
    monkeypatch.setattr(watch, 'submit', lambda *a: submitted.append(a) or 'job-2')
    monkeypatch.setattr(watch, 'collect', lambda p: collected.append(p))
    assert watch.watch(path, 'job-1', max_followups=1) == 0
    assert waited == ['job-1', 'job-2']
    assert submitted == [(path, 1, 'job-1')]
    assert collected == [path]
    assert read_json(path.parent/'continuation.json')['status'] == 'complete'
    assert read_json(path) == manifest


def test_followup_limit_prevents_endless_resubmission(config, monkeypatch):
    manifest, run = prepared(config)
    path = Path(manifest['paths']['models'])/'manifest.json'
    submitted = []
    monkeypatch.setattr(watch, 'wait_job', lambda *a: None)
    monkeypatch.setattr(watch, 'pending_runs', lambda m: [run])
    monkeypatch.setattr(watch, 'submit', lambda *a: submitted.append(a) or 'job-2')
    monkeypatch.setattr(watch, 'collect', lambda p: None)
    assert watch.watch(path, 'job-1', max_followups=1) == 1
    assert len(submitted) == 1
    assert read_json(path.parent/'continuation.json')['status'] == 'incomplete'


def test_changed_scientific_sources_stop_continuation(config, monkeypatch):
    manifest, _ = prepared(config)
    path = Path(manifest['paths']['models'])/'manifest.json'
    monkeypatch.setattr(watch, 'wait_job', lambda *a: None)
    monkeypatch.setattr(watch, 'source_identity', lambda: {})
    monkeypatch.setattr(watch, 'submit', lambda *a: pytest.fail('Changed sources'))
    with pytest.raises(ValueError, match='sources changed'):
        watch.watch(path, 'job-1')
    assert read_json(path.parent/'continuation.json')['status'] == 'error'
