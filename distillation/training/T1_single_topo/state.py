"""Training state and cooperative shutdown."""
import random
import signal
import time

import numpy as np
import torch

from .common import atomic_file


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def save_torch(path, value):
    with atomic_file(path) as stream:
        torch.save(cpu_tree(value), stream)


def capture_rng(device):
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state(), "device_type": device.type}
    if device.type in {"xpu", "cuda"}:
        state["accelerator"] = getattr(torch, device.type).get_rng_state(device)
    return state


def restore_rng(state, device):
    if state["device_type"] != device.type:
        raise ValueError("Resume training on the same device type as its checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.type in {"xpu", "cuda"}:
        getattr(torch, device.type).set_rng_state(state["accelerator"], device)


class Stop:
    def __init__(self, seconds=3300):
        self.deadline = time.monotonic() + seconds
        self.signalled = False
        self.handlers = {}

    def __call__(self):
        return self.signalled or time.monotonic() >= self.deadline

    def __enter__(self):
        def request_stop(signum, frame):
            self.signalled = True
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
            self.handlers[signum] = signal.signal(signum, request_stop)
        return self

    def __exit__(self, *args):
        for signum, handler in self.handlers.items():
            signal.signal(signum, handler)
