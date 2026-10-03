"""Timing must also work in accelerator builds with no visible devices."""
import pytest
import torch

from lumina.trainer.opf.trainer import BaseOPFTrainer
from lumina.utils.throughput import ThroughputTracker


@pytest.mark.parametrize("available", [False, True])
def test_throughput_synchronizes_only_available_accelerator(monkeypatch, available):
    calls = []
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: available)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: calls.append("sync"))
    ThroughputTracker({}, 1, 0, None).accelerator_synchronize()
    assert calls == (["sync"] if available else [])


@pytest.mark.parametrize("kind", ["cpu", "cuda", "xpu"])
def test_validation_timing_uses_selected_device(monkeypatch, kind):
    trainer = BaseOPFTrainer.__new__(BaseOPFTrainer)
    trainer.device = torch.device(kind)
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(device))
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: calls.append("accelerator"))
    trainer._sync_for_timing()
    expected = {"cpu": [], "cuda": [trainer.device], "xpu": ["accelerator"]}
    assert calls == expected[kind]
