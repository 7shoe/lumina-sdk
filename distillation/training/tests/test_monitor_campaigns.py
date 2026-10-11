"""Check resubmission, failure handling, and the T1-to-T2 dependency."""
from types import SimpleNamespace
import subprocess

import pytest

from distillation.training import monitor_campaigns as monitor


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    config_path = tmp_path / 'default.yaml'
    config_path.write_text('test')
    job_dir = tmp_path / 'T1' / 'jobs'
    job_dir.mkdir(parents=True)
    monitor.write_json(job_dir / 'submissions.json', [{'id': '101', 'kind': 'train'}])
    common = SimpleNamespace(DEFAULT_CONFIG=config_path, load_config=lambda path: {})
    campaign = SimpleNamespace(task_list=lambda config: [('HGT', 'case14', 42)])
    monkeypatch.setattr(monitor, 'BASE', tmp_path)
    monkeypatch.setattr(monitor, 'modules', lambda name: (common, campaign))
    monkeypatch.setattr(monitor, 'snapshot', lambda *args: {'statuses': {'paused': 1}})
    monkeypatch.setattr(monitor.time, 'sleep', lambda seconds: None)
    return campaign


def test_resubmit_only_when_previous_job_finished(campaign, monkeypatch):
    statuses = iter([{'id': '101', 'state': 'R'}, {'id': '101', 'state': 'F', 'exit': 0},
                     {'id': '102', 'state': 'F', 'exit': 0}])
    monkeypatch.setattr(monitor, 'scheduler', lambda job_id: next(statuses))
    complete = False
    campaign.is_complete = lambda *args: complete
    calls = []
    def submit(*args):
        nonlocal complete
        complete = True
        calls.append('submit')
    monkeypatch.setattr(monitor, 'submit', submit)
    monkeypatch.setattr(monitor, 'smoke_passed', lambda *args: True)
    monkeypatch.setattr(monitor, 'audit', lambda *args: calls.append('audit'))
    monitor.monitor('T1', 0)
    assert calls == ['submit', 'audit']


def test_failed_job_requires_inspection(campaign, monkeypatch):
    monkeypatch.setattr(monitor, 'scheduler', lambda job_id: {'state': 'F', 'exit': 1})
    campaign.is_complete = lambda *args: False
    monkeypatch.setattr(monitor, 'submit', lambda *args: pytest.fail('Failed job was resubmitted'))
    with pytest.raises(RuntimeError, match='inspect logs'):
        monitor.monitor('T1', 0)


def test_t2_waits_for_t1_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(monitor, 'BASE', tmp_path)
    monkeypatch.setattr('sys.argv', ['monitor'])
    calls = []
    def run(name, interval):
        calls.append(name)
        if name == 'T1':
            raise RuntimeError('T1 is unfinished')
    monkeypatch.setattr(monitor, 'monitor', run)
    with pytest.raises(RuntimeError, match='unfinished'):
        monitor.main()
    assert calls == ['T1']


def test_scheduler_timeout_keeps_job_state_unknown(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('qstat', 30)
    monkeypatch.setattr(subprocess, 'check_output', timeout)
    status = monitor.scheduler('101')
    assert status['state'] == 'unknown' and status['exit'] is None
