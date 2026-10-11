"""Persistent, bounded CPU workers for identified AC-OPF training labels."""
from collections import OrderedDict
from copy import deepcopy
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ..problem import as_ppc
from ..solution import Solution
from .solver import SparseProblem, SparseSolver, structural_signature


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.copy()
        value.setflags(write=False)
        return value
    return deepcopy(value)


@dataclass(frozen=True, init=False)
class Topology:
    """Immutable CPU template; scenarios replace complete PPC parameter arrays."""
    topology_id: str | int
    _items: tuple = field(repr=False)
    label_columns: tuple

    def __init__(self, topology_id, network):
        if not isinstance(topology_id, (str, int)):
            raise TypeError("topology_id must be a string or integer")
        columns = getattr(network, "label_columns", ("pt", "qt", "pf", "qf"))
        if isinstance(network, SparseProblem) or hasattr(network, "ppc"):
            network = network.ppc
        try:
            ppc, _, inferred = as_ppc(network)
            if not isinstance(network, dict):
                columns = inferred
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            ppc, columns = {"_input_error": str(exc)}, ("pt", "qt", "pf", "qf")
            
        # override attr
        object.__setattr__(self, "topology_id", topology_id)
        object.__setattr__(self, "_items", tuple((k, _snapshot(v)) for k, v in ppc.items()))
        object.__setattr__(self, "label_columns", columns)


@dataclass(frozen=True, init=False)
class Scenario:
    """Snapshot at construction, including detached generator-produced parameters."""
    topology: Topology
    scenario_id: str | int
    _parameters: tuple = field(repr=False)
    x0: object = field(repr=False)
    snapshot_s: float

    def __init__(self, topology, scenario_id, parameters=None, *, x0=None):
        start = time.perf_counter()
        if not isinstance(topology, Topology):
            raise TypeError("Expected a Topology")
        if not isinstance(scenario_id, (str, int)):
            raise TypeError("scenario_id must be a string or integer")
        if isinstance(x0, Solution):
            x0 = x0.x()
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "_parameters", tuple((k, _snapshot(v)) for k, v in (parameters or {}).items()))
        object.__setattr__(self, "x0", _snapshot(x0))
        object.__setattr__(self, "snapshot_s", time.perf_counter()-start)

    def inputs(self):
        """Actual scenario values in original order, returned as a fresh PPC copy."""
        return {k: _snapshot(v) for k, v in dict(self.topology._items + self._parameters).items()}


@dataclass
class Result:
    request_id: str | int
    index: int
    scenario: Scenario
    generator_version: str | int | None
    status: str
    solution: Solution | None
    y: dict | None
    failure_reason: str | None
    timing: dict
    worker: dict

    @property
    def topology_id(self):
        return self.scenario.topology.topology_id

    @property
    def scenario_id(self):
        return self.scenario.scenario_id

    @property
    def ids(self):
        return self.request_id, self.topology_id, self.scenario_id

    @property
    def success(self):
        return self.status == "success"


def _size(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return value.untyped_storage().nbytes()
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, dict):
        return sum(_size(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_size(v, seen) for v in value)
    if hasattr(value, "__dict__"):
        return _size(vars(value), seen)
    return 0


class _Cache:
    def __init__(self, entries, max_bytes):
        self.entries, self.max_bytes = entries, max_bytes
        self.data, self.bytes = OrderedDict(), 0
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        entry = self.data.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self.data.move_to_end(key)
        return entry[0]

    def put(self, key, value):
        size = _size(value)
        if size > self.max_bytes or self.entries == 0:
            return
        if key in self.data:
            self.bytes -= self.data.pop(key)[1]
        while self.data and (len(self.data) >= self.entries or self.bytes+size > self.max_bytes):
            self.bytes -= self.data.popitem(last=False)[1][1]
            self.evictions += 1
        self.data[key] = value, size
        self.bytes += size


def physical_cpus():
    """One allowed logical CPU per physical core, respecting scheduler affinity."""
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((path/"physical_package_id").read_text(), (path/"core_id").read_text())
        except OSError:
            key = (cpu,)
        cores.setdefault(key, cpu)
    return list(cores.values())


def _init_worker(slots, counter, entries, cache_bytes, options):
    global _CACHE, _SOLVER, _LIMITS, _WORKER
    with counter.get_lock():
        index = counter.value
        counter.value += 1
    cpus = slots[index]
    for task in Path("/proc/self/task").iterdir():
        try:
            os.sched_setaffinity(int(task.name), cpus)
        except ProcessLookupError:
            pass
    threads = len(cpus)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    _LIMITS = threadpool_limits(limits=threads)
    _CACHE = _Cache(entries, cache_bytes)
    _SOLVER = SparseSolver(cache=_CACHE, **options)
    _WORKER = dict(pid=os.getpid(), cpus=cpus, threads=threads)


def _work(packets, submitted, branch_labels):
    """Only expected input errors become records; infrastructure exceptions propagate."""
    started = time.perf_counter()
    groups, output = OrderedDict(), []
    for index, ppc, x0, columns in packets:
        preparation = time.perf_counter()
        try:
            if "_input_error" in ppc:
                raise ValueError(ppc["_input_error"])
            model = SparseProblem(ppc, _CACHE)
            if x0 is not None:
                x0 = np.asarray(x0, dtype=float)
                if x0.shape != tuple(model.lo.shape) or not np.isfinite(x0).all():
                    raise ValueError("Invalid initial point shape or values")
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            output.append((index, "invalid_input", None, None, str(exc),
                           dict(preparation_s=time.perf_counter()-preparation)))
            continue
        groups.setdefault(model.signature, []).append((index, model, x0, columns,
                                                       time.perf_counter()-preparation))
    for group in groups.values():
        solutions = _SOLVER.solve_batch([r[1] for r in group], [r[2] for r in group])
        for (index, model, _, columns, preparation), solution in zip(group, solutions):
            message = solution.info["message"]
            if solution.success:
                status = "success"
            elif message == "elapsed time limit":
                status = "timeout"
            elif solution.converged:
                status = "failed_feasibility"
            elif message == "iteration limit":
                status = "not_converged"
            else:
                status = "numerical_failure"
            labels = None
            if solution.success:
                labels = dict(bus=np.column_stack((solution.theta, solution.vm)),
                              generator=np.column_stack((solution.pg, solution.qg)))
                if branch_labels:
                    sf, st = model.branch_flows(solution.x())
                    values = dict(pf=sf.real, qf=sf.imag, pt=st.real, qt=st.imag)
                    labels["branch"] = np.column_stack([values[k].numpy() for k in columns])
            timing = dict(preparation_s=preparation, setup_batch_s=solution.info["setup_batch_s"],
                          solve_batch_s=solution.info["solve_batch_s"], solve_s=solution.info["solve_s"],
                          validation_s=solution.info["validation_s"])
            output.append((index, status, solution, labels, None if solution.success else message, timing))
    worker = _WORKER | dict(cache_entries=len(_CACHE.data), cache_bytes=_CACHE.bytes,
                            cache_hits=_CACHE.hits, cache_misses=_CACHE.misses,
                            cache_evictions=_CACHE.evictions, xpu_initialized=torch.xpu.is_initialized())
    for record in output:
        record[-1].update(queue_s=started-submitted, task_s=time.perf_counter()-started)
    return output, worker


class SolverPool:
    """One pool per node/rank allocation; one active stream, reused for many requests."""
    def __init__(self, *, workers=None, cpu_budget=16, threads_per_worker=1,
                 local_rank=None, local_world_size=None, task_size=None, max_pending=None,
                 cache_entries=32, cache_mib=256, solver_options=None):
        if mp.current_process().daemon or torch.utils.data.get_worker_info() is not None:
            raise RuntimeError("Create SolverPool in the training entry point, not a DataLoader worker")
        if local_rank is None:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", "0")))
        if local_world_size is None:
            size = os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("MPI_LOCALNRANKS"))
            if size is None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
                raise ValueError("Distributed callers must provide the node's local_world_size")
            local_world_size = int(size or "1")
        if local_world_size < 1 or not 0 <= local_rank < local_world_size:
            raise ValueError("Invalid local rank allocation")
        available = physical_cpus()
        if not isinstance(cpu_budget, int) or not 1 <= cpu_budget <= len(available):
            raise ValueError(f"cpu_budget must be between 1 and {len(available)} physical cores")
        if not isinstance(threads_per_worker, int) or threads_per_worker < 1:
            raise ValueError("threads_per_worker must be positive")
        cpus = available[:cpu_budget][local_rank::local_world_size]
        capacity = len(cpus)//threads_per_worker
        workers = capacity if workers is None else workers
        if not isinstance(workers, int) or not 1 <= workers <= capacity:
            raise ValueError(f"This rank's CPU allocation supports at most {capacity} workers")
        if task_size is not None and (not isinstance(task_size, int) or task_size < 1):
            raise ValueError("task_size must be positive or None for automatic grouping")
        max_pending = 2*workers if max_pending is None else max_pending
        if not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be positive")
        if not isinstance(cache_entries, int) or cache_entries < 0 or cache_mib < 0 or not np.isfinite(cache_mib):
            raise ValueError("Cache limits must be nonnegative and finite")
        options = dict(time_limit=600) | (solver_options or {})
        SparseSolver(**options)
        slots = [cpus[i*threads_per_worker:(i+1)*threads_per_worker] for i in range(workers)]
        context = mp.get_context("spawn")
        self._executor = ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker,
                                             initargs=(slots, context.Value("i", 0), cache_entries,
                                                       int(cache_mib*1024**2), options))
        self.workers, self.cpus, self.task_size = workers, cpus, task_size
        self.max_pending, self.closed = max_pending, False
        self._request_lock = threading.Lock()
        self.stats = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        """Cancel unstarted work and wait for running cooperative solves to finish."""
        if not self.closed:
            self.closed = True
            self._executor.shutdown(wait=True, cancel_futures=True)

    def _tasks(self, scenarios):
        source, seen, index = iter(scenarios), set(), 0
        width = self.task_size or 8
        while True:
            groups = OrderedDict()
            for _ in range(width):
                try:
                    scenario = next(source)
                except StopIteration:
                    break
                if not isinstance(scenario, Scenario):
                    raise TypeError("Each input must be a Scenario snapshot")
                ids = scenario.topology.topology_id, scenario.scenario_id
                if ids in seen:
                    raise ValueError(f"Duplicate scenario IDs: {ids}")
                seen.add(ids)
                ppc = scenario.inputs()
                try:
                    key = structural_signature(ppc)
                    buses = len(ppc["bus"])
                except (ValueError, TypeError, KeyError, IndexError):
                    key, buses = ("invalid", index), 0
                size = self.task_size or (1 if buses >= 500 else 4 if buses >= 118 else 8)
                groups.setdefault(key, (size, []))[1].append((index, scenario, ppc))
                index += 1
            if not groups:
                return
            for size, items in groups.values():
                for offset in range(0, len(items), size):
                    yield items[offset:offset+size]

    def stream(self, scenarios, *, request_id=None, generator_version=None, branch_labels=False):
        """Yield completed task members with IDs; task_size=1 gives per-solve latency."""
        if self.closed:
            raise RuntimeError("SolverPool is closed")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeError("Consume or close the previous stream before starting another request")
        pending, tasks = {}, iter(self._tasks(scenarios))
        request_id = uuid.uuid4().hex if request_id is None else request_id
        if not isinstance(request_id, (str, int)):
            self._request_lock.release()
            raise TypeError("request_id must be a string or integer")
        self.stats = dict(submitted=0, completed=0, peak_pending_tasks=0, peak_buffered_results=0)
        exhausted = False
        try:
            while pending or not exhausted:
                while not exhausted and len(pending) < self.max_pending:
                    try:
                        task = next(tasks)
                    except StopIteration:
                        exhausted = True
                        break
                    submitted = time.perf_counter()
                    packets = [(i, ppc, s.x0, s.topology.label_columns) for i, s, ppc in task]
                    future = self._executor.submit(_work, packets, submitted, branch_labels)
                    pending[future] = {i: (s, submitted) for i, s, _ in task}
                    self.stats["submitted"] += len(task)
                    self.stats["peak_pending_tasks"] = max(self.stats["peak_pending_tasks"], len(pending))
                if not pending:
                    break
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                buffered = sum(len(pending[f]) for f in pending if f.done())
                self.stats["peak_buffered_results"] = max(self.stats["peak_buffered_results"], buffered)
                for future in done:
                    matches = pending.pop(future)
                    records, worker = future.result()
                    if {row[0] for row in records} != set(matches) or len(records) != len(matches):
                        raise RuntimeError("Worker returned mismatched scenario IDs")
                    for index, status, solution, labels, reason, timing in records:
                        scenario, submitted = matches[index]
                        timing.update(snapshot_s=scenario.snapshot_s, latency_s=time.perf_counter()-submitted)
                        self.stats["completed"] += 1
                        yield Result(request_id, index, scenario, generator_version,
                                     status, solution, labels, reason, timing, worker)
        finally:
            try:
                for future in pending:
                    future.cancel()
                # Drain running tasks before releasing the bounded scheduling window.
                if pending:
                    wait(pending)
                    for future in pending:
                        if not future.cancelled():
                            future.result()
            finally:
                self._request_lock.release()

    def solve(self, scenarios, **options):
        """Return every result in input order; failures retain their positions and IDs."""
        return sorted(self.stream(scenarios, **options), key=lambda result: result.index)
