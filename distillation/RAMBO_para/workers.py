"""Persistent, single-threaded Julia workers with independent stage checkpoints."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import time

from distillation.RAMBO.backend import JuliaBackend
from distillation.training.T1_single_topo.common import digest, write_json


def physical_cpus():
    """Choose one allowed hardware thread per physical core (Linux/Aurora)."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = tuple((topology / name).read_text().strip()
                    for name in ('physical_package_id', 'core_id'))
        cores.setdefault(key, cpu)
    return list(cores.values())


class ProposalPool:
    def __init__(self, folder, config, network, stop, workers, *, backend_factory=JuliaBackend, bind=True):
        self.folder, self.config, self.network, self.stop = Path(folder), config, network, stop
        self.factory, self.local, self.lock = backend_factory, threading.local(), threading.Lock()
        self.backends = []
        self.thread_count = 0
        self.cpus = physical_cpus() if bind else []
        if bind and workers > len(self.cpus):
            raise ValueError(f'{workers} workers requested but only {len(self.cpus)} physical cores allowed')
        self.executor = ThreadPoolExecutor(max_workers=workers)

    def backend(self):
        if getattr(self.local, 'backend', None) is None:
            with self.lock:
                if not hasattr(self.local, 'core_index'):
                    self.local.core_index = self.thread_count
                    self.thread_count += 1
                index = len(self.backends)
                self.backends.append(None)
            if self.cpus:
                os.sched_setaffinity(0, {self.cpus[self.local.core_index]})
            log = self.folder / 'workers' / f'{os.getpid()}-{index}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            backend = self.factory(log, startup_seconds=max(1., min(600., self.stop.deadline-time.monotonic()-5)))
            self.local.backend = backend
            with self.lock:
                self.backends[index] = backend
                path = self.folder / 'runtime.json'
                if path.exists() and json.loads(path.read_text()) != backend.versions:
                    raise ValueError('Julia solver environment changed')
                write_json(path, backend.versions)
        return self.local.backend

    def proposal(self, pending, history):
        """One full candidate. Resume completed stages without another solver call."""
        cfg = self.config
        folder = self.folder / 'attempts' / f'{pending["index"]:06d}'
        stage, warm, relaxed = pending['stage'], None, None
        pd, qd = pending.get('pd'), pending.get('qd')
        while True:
            request = dict(network=self.network, stage=stage, seed=pending['seed'], history=history,
                           seconds=cfg['stage_seconds'], epsilon=cfg['epsilon'], log_delta=cfg['log_delta'],
                           tolerance=cfg['tolerance'], casefile=str((self.folder / 'case.m').resolve()))
            if 'subsets' in pending:
                request['subsets'] = pending['subsets']
            if pd is not None:
                request.update(pd=pd, qd=qd)
            if warm is not None:
                request['warm'] = warm
            path, identity = folder / f'{stage}.json', digest(request)
            if path.exists():
                cached = json.loads(path.read_text())
                if cached['request_sha256'] != identity or digest(cached['result']) != cached['result_sha256']:
                    raise ValueError(f'Cached proposal changed: {path}')
                result = cached['result']
            else:
                if self.stop() or time.monotonic() + cfg['request_seconds'] + 5 >= self.stop.deadline:
                    return dict(status='paused')
                journal = folder / f'{stage}.calls.json'
                calls = json.loads(journal.read_text()) if journal.exists() else []
                try:
                    backend = self.backend()
                    if self.stop() or time.monotonic() + cfg['request_seconds'] + 5 >= self.stop.deadline:
                        return dict(status='paused')
                    calls.append(dict(request_sha256=identity, status='started'))
                    write_json(journal, calls)
                    result = backend.request(request, timeout=cfg['request_seconds'])
                except (TimeoutError, RuntimeError, OSError) as error:
                    if calls and calls[-1]['status'] == 'started':
                        calls[-1].update(status=type(error).__name__, error=str(error))
                        write_json(journal, calls)
                    backend = getattr(self.local, 'backend', None)
                    if backend is not None:
                        backend.__exit__(None, None, None)
                        self.local.backend = None
                    write_json(folder / 'error.json', dict(stage=stage, error=repr(error)))
                    return dict(status='error', error=repr(error))
                calls[-1].update(status=result['status'], wall_seconds=result.get('wall_seconds'))
                write_json(journal, calls)
                if result.get('status') == 'ERROR':
                    return dict(status='error', error=result.get('error', 'Julia model error'))
                write_json(path, dict(request_sha256=identity, result=result, result_sha256=digest(result)))
            if not result.get('success'):
                return dict(status='rejected', stage=stage, reason=result['status'])
            if stage == 'feasible':
                stage, warm = ('warm_opf' if cfg['warm_start'] == 'opf' else 'bilevel'), result
            elif stage == 'warm_opf':
                stage, warm = 'bilevel', result
            elif stage == 'bilevel':
                relaxed, stage, warm = result, 'opf', None
                pd, qd = result['point']['pd'], result['point']['qd']
            else:
                return dict(status='ready', result=result, relaxed=relaxed)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for backend in self.backends:
            if backend is not None:
                backend.__exit__(*args)
