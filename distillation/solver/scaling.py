"""Measure CPU AC-OPF latency, throughput, memory and paired warm starts.

Activate the Aurora environment, then run this file. Results, input JSONs and
plots go to reports/scaling_<UTC>. Use --resume DIR after an interruption or
--report-only DIR to rebuild the report without solving anything.

Add a SolverSpec to SOLVERS for another solve(model, x0=None) -> Solution backend.
Workers use disjoint CPU affinities and bounded BLAS/PyTorch thread pools.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import statistics as stats
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RAW_ROOT = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/raw/OPFData/dataset_release_1")
REPORT_ROOT = Path(__file__).resolve().parent / "reports"
CASES = {n: f"pglib_opf_case{n}_{'goc' if n >= 500 else 'ieee'}"
         for n in (14, 30, 57, 118, 500, 2000)}
GIB = 1024 ** 3


@dataclass(frozen=True)
class SolverSpec:
    module: str
    factory: str
    warm_start: bool = False
    max_buses: int | None = None


SOLVERS = {
    "pypower": SolverSpec("distillation.solver", "PyPowerSolver"),
    "scipy": SolverSpec("distillation.solver", "ScipySolver", True, 118),
}


def stamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_ready(x):
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def write_json(path, value):
    """Replace a JSON file atomically; nonfinite observations become null."""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_ready(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def cpu_order():
    """Prefer distinct physical cores before SMT siblings, within our affinity."""
    allowed = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in allowed:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = tuple((topology / name).read_text().strip()
                        for name in ("physical_package_id", "core_id"))
        except OSError:
            key = (cpu,)
        cores.setdefault(key, []).append(cpu)
    return [group[i] for i in range(max(map(len, cores.values())))
            for group in cores.values() if i < len(group)]


def prepare_inputs(config, output):
    """Stream just the requested records; preserve exact bytes and SHA-256."""
    manifest = []
    folder = output / "inputs"
    folder.mkdir(exist_ok=True)
    for case in config["cases"]:
        archive_path = Path(config["raw_root"]) / f"{CASES[case]}_0.tar.gz"
        records = []
        with tarfile.open(archive_path, "r|gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                payload = archive.extractfile(member).read()
                index = len(records)
                path = folder / f"case{case}_{index:03d}.json"
                path.write_bytes(payload)
                obj = json.loads(payload)
                grid = obj["grid"]
                edges = sum(len(grid["edges"][kind]["senders"])
                            for kind in ("ac_line", "transformer"))
                records.append(dict(case=case, index=index, path=str(path),
                                    archive=str(archive_path), member=member.name,
                                    sha256=hashlib.sha256(payload).hexdigest(),
                                    buses=len(grid["nodes"]["bus"]), branches=edges,
                                    generators=len(grid["nodes"]["generator"]),
                                    reference_cost=float(obj["metadata"]["objective"])))
                if len(records) == config["batch"] + 1:
                    break
        if len(records) != config["batch"] + 1:
            raise ValueError(f"Too few records in {archive_path}")
        manifest.extend(records)
        print(f"{stamp()} read case {case}: {len(records)} records", flush=True)
    write_json(output / "inputs.json", manifest)
    return manifest


def load_network(path):
    """Read native OPFData inputs only; target solutions are never model inputs."""
    import torch
    from torch_geometric.data import HeteroData
    obj = json.loads(Path(path).read_text())
    grid = obj["grid"]
    data = HeteroData()
    data.x = torch.tensor(grid["context"], dtype=torch.float64).reshape(-1)
    for node, width in (("bus", 4), ("generator", 11), ("load", 2), ("shunt", 2)):
        data[node].x = torch.tensor(grid["nodes"][node], dtype=torch.float64).reshape(-1, width)
    for kind, edge in grid["edges"].items():
        source = "bus" if kind in ("ac_line", "transformer") else kind.removesuffix("_link")
        store = data[source, kind, "bus"]
        store.edge_index = torch.tensor([edge["senders"], edge["receivers"]], dtype=torch.long).reshape(2, -1)
        if kind in ("ac_line", "transformer"):
            width = 9 if kind == "ac_line" else 11
            store.edge_attr = torch.tensor(edge["features"], dtype=torch.float64).reshape(-1, width)
    return data, float(obj["metadata"]["objective"])


def structure(model):
    return dict(base=model.base, bus_ids=model.ppc["bus"][:, 0].tolist(),
                generator_buses=model.ppc["gen"][:, 0].tolist(),
                branch_buses=model.ppc["branch"][:, :2].tolist())


def run_task(task):
    """Time setup separately; solve time includes the backend's feasibility check."""
    import numpy as np
    from distillation.solver import ACOPF, Solution
    start, cpu = time.perf_counter(), time.process_time()
    network, reference_cost = load_network(task["input"])
    model = ACOPF(network)
    spec = SOLVERS[task["solver"]]
    backend = getattr(importlib.import_module(spec.module), spec.factory)()
    seed = None
    demand_delta = None
    if task.get("seed_file"):
        previous = json.loads(Path(task["seed_file"]).read_text())
        if previous["structure"] != structure(model):
            raise ValueError("Warm-start source has different equipment ordering/topology")
        seed = Solution(*(np.asarray(previous[k]) for k in ("theta", "vm", "pg", "qg")),
                        previous["objective"], True, "previous_problem")
        demand = model.ppc["bus"][:, 2:4]
        demand_delta = float(np.linalg.norm(demand - previous["demand"]) /
                             max(np.linalg.norm(demand), 1e-12))
    initial = model.flat_start() if seed is None else seed.x(model)
    initial_balance = float(model.balance(initial).abs().max())
    setup_s = time.perf_counter() - start
    solve_start, solve_cpu = time.perf_counter(), time.process_time()
    result = backend.solve(model, x0=seed)
    solve_s, solve_cpu_s = time.perf_counter() - solve_start, time.process_time() - solve_cpu
    iterations = result.info.get("iterations", result.info.get("output", {}).get("iterations"))
    observation = dict(status="ok" if result.success else "failed", setup_s=setup_s,
                       solve_s=solve_s, solve_cpu_s=solve_cpu_s,
                       task_s=time.perf_counter() - start, task_cpu_s=time.process_time() - cpu,
                       success=bool(result.success), converged=bool(result.converged),
                       feasible=bool(result.feasible), iterations=iterations,
                       objective=result.objective, reference_cost=reference_cost,
                       cost_gap_pct=100 * (result.objective - reference_cost) / max(abs(reference_cost), 1e-12),
                       initial_balance_pu=initial_balance, demand_change_relative=demand_delta,
                       checks=result.checks, message=str(result.info.get("message", "")))
    if result.success and task.get("save_solution"):
        write_json(task["save_solution"], dict(
            **{k: getattr(result, k).tolist() for k in ("theta", "vm", "pg", "qg")},
            objective=result.objective, structure=structure(model),
            demand=model.ppc["bus"][:, 2:4].tolist(), source_task=task["id"], solve_s=solve_s))
    return observation


def worker(connection, affinity, log_path, solvers):
    """One persistent CPU worker; the parent can terminate a stalled solve."""
    os.sched_setaffinity(0, affinity[:1])
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    with open(log_path, "a", buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        start = time.perf_counter()
        try:
            import gc
            import psutil
            import torch
            import scipy
            import numpy
            from importlib.metadata import version
            from threadpoolctl import threadpool_limits, threadpool_info
            from pypower.api import case9
            from distillation.solver import ACOPF
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            with threadpool_limits(limits=1):
                # Prime imports and derivative tracing outside measured solves.
                model = ACOPF(case9())
                for name in solvers:
                    spec = SOLVERS[name]
                    getattr(importlib.import_module(spec.module), spec.factory)().solve(model)
                del model
                gc.collect()
            connection.send(dict(event="ready", startup_s=time.perf_counter() - start,
                                 versions=dict(python=platform.python_version(), torch=torch.__version__,
                                               scipy=scipy.__version__, numpy=numpy.__version__,
                                               pypower=version("pypower"), psutil=psutil.__version__),
                                 libraries=threadpool_info(), rss=psutil.Process().memory_info().rss))
            while True:
                task = connection.recv()
                if task is None:
                    break
                threads = task["threads"]
                os.sched_setaffinity(0, affinity[:threads])
                torch.set_num_threads(threads)
                with threadpool_limits(limits=threads):
                    task_start, task_cpu = time.perf_counter(), time.process_time()
                    try:
                        observation = run_task(task)
                    except Exception:
                        observation = dict(status="error", success=False, error=traceback.format_exc())
                    cleanup = time.perf_counter()
                    gc.collect()
                    observation.update(cleanup_s=time.perf_counter() - cleanup,
                                       task_s=time.perf_counter() - task_start,
                                       task_cpu_s=time.process_time() - task_cpu)
                connection.send(dict(event="result", result=observation))
        except (EOFError, BrokenPipeError):
            pass
        except Exception:
            connection.send(dict(event="startup_error", error=traceback.format_exc()))


class WorkerPool:
    """Bounded workers with sampled RSS/thread counts and per-task deadlines."""
    def __init__(self, config, output, cpus):
        self.config, self.output, self.cpus = config, output, cpus
        self.slots = []
        path = output / "worker_startups.json"
        self.startups = json.loads(path.read_text()) if path.exists() else []

    def close(self):
        for slot in self.slots:
            if slot["process"].is_alive():
                try:
                    slot["pipe"].send(None)
                except (BrokenPipeError, EOFError):
                    pass
        for slot in self.slots:
            slot["process"].join(timeout=2)
            if slot["process"].is_alive():
                slot["process"].terminate()
                slot["process"].join(timeout=2)
            slot["pipe"].close()

    def add_worker(self, index):
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        stride = max(self.config["thread_counts"])
        cpus = self.cpus[index * stride:(index + 1) * stride]
        process = context.Process(target=worker, args=(child, cpus, str(self.output / f"worker_{index}.log"),
                                                       self.config["solvers"]))
        process.start()
        child.close()
        start = time.monotonic()
        while not parent.poll(0.1):
            if not process.is_alive() or time.monotonic() - start > 180:
                process.terminate()
                process.join(timeout=2)
                raise RuntimeError(f"Worker {index} did not start; see its log")
        ready = parent.recv()
        if ready["event"] != "ready":
            raise RuntimeError(ready)
        self.startups.append(dict(worker=index, pid=process.pid, **ready))
        slot = dict(process=process, pipe=parent, cpus=cpus)
        if index < len(self.slots):
            self.slots[index]["pipe"].close()
            self.slots[index] = slot
        else:
            self.slots.append(slot)
        write_json(self.output / "worker_startups.json", self.startups)

    def run(self, tasks, workers, on_result):
        import psutil
        from collections import deque
        while len(self.slots) < workers:
            self.add_worker(len(self.slots))
        for index, slot in enumerate(self.slots[:workers]):
            if not slot["process"].is_alive():
                self.add_worker(index)
        queue, active, results, trace = deque(tasks), {}, [], []
        start = time.monotonic()
        controller = psutil.Process()
        controller_cpu = sum(controller.cpu_times()[:2])
        peak_rss = peak_pool_rss = peak_controller_rss = peak_total_rss = peak_threads = peak_active = 0
        total_cpu = 0.0
        aborted = None
        last_notice = start
        while queue or active:
            for index, slot in enumerate(self.slots[:workers]):
                if index not in active and queue and not aborted:
                    task = queue.popleft()
                    proc = psutil.Process(slot["process"].pid)
                    active[index] = dict(task=task, start=time.monotonic(), rss=0, threads=0,
                                         process=proc, cpu_start=sum(proc.cpu_times()[:2]), cpu_s=0.)
                    slot["pipe"].send(task)
            live_rss = live_threads = 0
            peak_active = max(peak_active, len(active))
            for index, job in list(active.items()):
                slot = self.slots[index]
                observation = None
                try:
                    proc = job["process"]
                    rss, threads = proc.memory_info().rss, proc.num_threads()
                    job["rss"] = max(job["rss"], rss)
                    job["threads"] = max(job["threads"], threads)
                    job["cpu_s"] = max(job["cpu_s"], sum(proc.cpu_times()[:2]) - job["cpu_start"])
                    live_rss += rss
                    live_threads += threads
                except psutil.Error:
                    pass
                elapsed = time.monotonic() - job["start"]
                if slot["pipe"].poll():
                    try:
                        observation = slot["pipe"].recv()["result"]
                    except (EOFError, KeyError):
                        observation = dict(status="error", success=False, error="Worker exited without a result")
                elif elapsed >= self.config["timeout"]:
                    observation = dict(status="timeout", success=False, elapsed_s=elapsed)
                elif job["rss"] > self.config["memory_gib"] * GIB:
                    observation = dict(status="memory_limit", success=False, elapsed_s=elapsed)
                elif not slot["process"].is_alive():
                    observation = dict(status="error", success=False, error="Worker process died")
                if observation is not None:
                    if observation["status"] in ("timeout", "memory_limit"):
                        slot["process"].terminate()
                        slot["process"].join(timeout=2)
                        aborted = observation["status"]
                    observation.update(job["task"], measured_at=stamp(), worker_pid=slot["process"].pid,
                                       peak_rss_gib=job["rss"] / GIB, peak_os_threads=job["threads"],
                                       observed_cpu_s=job["cpu_s"], dispatch_s=elapsed)
                    total_cpu += max(job["cpu_s"], observation.get("task_cpu_s", 0))
                    results.append(observation)
                    on_result(observation)
                    del active[index]
            peak_rss = max(peak_rss, live_rss)
            peak_threads = max(peak_threads, live_threads)
            pool_rss = 0
            for slot in self.slots:
                try:
                    pool_rss += psutil.Process(slot["process"].pid).memory_info().rss
                except psutil.Error:
                    pass
            peak_pool_rss = max(peak_pool_rss, pool_rss)
            controller_rss = controller.memory_info().rss
            peak_controller_rss = max(peak_controller_rss, controller_rss)
            peak_total_rss = max(peak_total_rss, pool_rss + controller_rss)
            trace.append(dict(elapsed_s=time.monotonic() - start, active=len(active),
                              rss_gib=live_rss / GIB, pool_rss_gib=pool_rss / GIB,
                              controller_rss_gib=controller_rss / GIB, os_threads=live_threads))
            if aborted:
                # Do not repeat an out-of-scope case after its first limit hit.
                for task in queue:
                    observation = dict(**task, status="skipped", success=False, reason=f"Batch stopped: {aborted}")
                    results.append(observation)
                    on_result(observation)
                queue.clear()
                for index, job in list(active.items()):
                    slot = self.slots[index]
                    slot["process"].terminate()
                    slot["process"].join(timeout=2)
                    observation = dict(**job["task"], status="cancelled", success=False,
                                       reason=f"Another task reached {aborted}",
                                       peak_rss_gib=job["rss"] / GIB, observed_cpu_s=job["cpu_s"])
                    total_cpu += job["cpu_s"]
                    results.append(observation)
                    on_result(observation)
                    del active[index]
            if time.monotonic() - last_notice >= 30:
                print(f"{stamp()} running: {len(results)}/{len(tasks)}, RSS {live_rss / GIB:.2f} GiB", flush=True)
                last_notice = time.monotonic()
            if active:
                time.sleep(self.config["poll_seconds"])
        elapsed = time.monotonic() - start
        return dict(wall_s=elapsed, cpu_s=total_cpu, cpu_cores=total_cpu / elapsed,
                    peak_rss_gib=peak_rss / GIB, peak_pool_rss_gib=peak_pool_rss / GIB, peak_os_threads=peak_threads,
                    controller_cpu_s=sum(controller.cpu_times()[:2]) - controller_cpu,
                    controller_rss_gib=peak_controller_rss / GIB, peak_total_rss_gib=peak_total_rss / GIB,
                    peak_active=peak_active, success_count=sum(r["success"] for r in results),
                    attempted=len(tasks), trace=trace)


def make_task(output, record, solver, phase, threads=1, workers=1, mode="cold", seed_file=None):
    task_id = f"{solver}_{record['case']}_{phase}_{mode}_t{threads}_w{workers}_i{record['index']}"
    return dict(id=task_id, solver=solver, case=record["case"], index=record["index"],
                phase=phase, threads=threads, workers=workers, mode=mode, input=record["path"],
                seed_file=str(seed_file) if seed_file else None,
                save_solution=str(output / "solutions" / f"{task_id}.json"))


def campaign(config, output, manifest, cpus):
    """Pilot each size, then reuse exactly the same targets at each concurrency."""
    journal = output / "observations.jsonl"
    observations = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    done = {row["id"]: row for row in observations}
    batch_file = output / "batches.json"
    batches = json.loads(batch_file.read_text()) if batch_file.exists() else []
    exclusions_file = output / "exclusions.json"
    exclusions = json.loads(exclusions_file.read_text()) if exclusions_file.exists() else []
    (output / "solutions").mkdir(exist_ok=True)
    pool = WorkerPool(config, output, cpus)

    def record(row):
        row = json_ready(row)
        with journal.open("a") as handle:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        done[row["id"]] = row
        print(f"{stamp()} {row['id']}: {row['status']} {row.get('solve_s', row.get('elapsed_s', 0)):.3f}s", flush=True)

    def execute(tasks, workers=1):
        key = tasks[0]["id"] if tasks[0]["phase"] in ("paired", "pilot") else tasks[0]["id"].rsplit("_i", 1)[0]
        if any(b["id"] == key for b in batches):
            return [done[t["id"]] for t in tasks]
        missing = [t for t in tasks if t["id"] not in done]
        if missing:
            print(f"{stamp()} starting {key}: {len(missing)} tasks, {workers} workers", flush=True)
            summary = pool.run(missing, workers, record)
            trace = summary.pop("trace")
            write_json(output / f"trace_{key}.json", trace)
        else:
            summary = dict(wall_s=None, cpu_s=None, cpu_cores=None, peak_rss_gib=None,
                           peak_os_threads=None, success_count=sum(done[t["id"]]["success"] for t in tasks))
        summary.update(id=key, solver=tasks[0]["solver"], case=tasks[0]["case"], phase=tasks[0]["phase"],
                       mode=tasks[0]["mode"], threads=tasks[0]["threads"], workers=workers,
                       requested=len(tasks), complete_timing=len(missing) == len(tasks))
        batches.append(summary)
        write_json(batch_file, batches)
        return [done[t["id"]] for t in tasks]

    def exclude(solver, case, reason):
        exclusions.append(dict(solver=solver, case=case, reason=reason))
        write_json(exclusions_file, exclusions)

    try:
        for solver in config["solvers"]:
            for case in config["cases"]:
                if any(e["solver"] == solver and e["case"] == case for e in exclusions):
                    continue
                limit = config["scipy_max_buses"] if solver == "scipy" else SOLVERS[solver].max_buses
                if limit and case > limit:
                    exclude(solver, case, f"Dense derivative size guard: maximum {limit} buses; not timed")
                    continue
                samples = [r for r in manifest if r["case"] == case]
                pilot = execute([make_task(output, samples[0], solver, "pilot")])[0]
                if not pilot["success"]:
                    exclude(solver, case, f"Pilot {pilot['status']}; no batch repeats")
                    continue
                if config["pilot_only"]:
                    continue
                targets = samples[1:]
                serial = execute([make_task(output, r, solver, "batch") for r in targets])
                if any(r["status"] in ("timeout", "memory_limit") for r in serial):
                    exclude(solver, case, "Serial batch reached a time/memory limit; no further repeats")
                    continue
                # Sparse reference backend provides the main concurrency sweep.
                levels = config["workers"] if solver == "pypower" else [1, min(2, max(config["workers"]))]
                for count in sorted(set(levels) - {1}):
                    rows = execute([make_task(output, r, solver, "batch", workers=count) for r in targets], count)
                    if any(r["status"] in ("timeout", "memory_limit") for r in rows):
                        exclude(solver, case, "Parallel batch reached a time/memory limit")
                        break
                else:
                    limited = False
                    for threads in config["thread_counts"]:
                        if threads > 1:
                            rows = execute([make_task(output, r, solver, "threads", threads=threads) for r in targets[:3]])
                            if any(r["status"] in ("timeout", "memory_limit") for r in rows):
                                exclude(solver, case, "Thread study reached a time/memory limit")
                                limited = True
                                break
                    if SOLVERS[solver].warm_start and not limited:
                        sources = [pilot] + serial[:-1]
                        for target, previous in zip(targets, sources):
                            if not previous["success"]:
                                continue
                            tasks = [make_task(output, target, solver, "paired", mode=mode,
                                               seed_file=previous["save_solution"] if mode == "previous" else None)
                                     for mode in ("cold", "previous")]
                            random.Random(42 + case + target["index"]).shuffle(tasks)
                            for task in tasks:
                                rows = execute([task])
                                if rows[0]["status"] in ("timeout", "memory_limit"):
                                    break
                            else:
                                continue
                            exclude(solver, case, "Warm-start study reached a time/memory limit")
                            break
    finally:
        pool.close()
    return observations


def paired_results(rows):
    """Compare successful, feasible pairs reaching costs within 0.01% of each other."""
    groups = {}
    for row in rows:
        if row["phase"] == "paired":
            groups.setdefault((row["solver"], row["case"], row["index"]), {})[row["mode"]] = row
    pairs = []
    for (solver, case, index), modes in groups.items():
        cold, warm = modes.get("cold"), modes.get("previous")
        if not cold or not warm or not cold["success"] or not warm["success"]:
            continue
        gap = abs(cold["objective"] - warm["objective"]) / max(abs(cold["objective"]), 1e-12)
        if gap <= 1e-4:
            pairs.append(dict(solver=solver, case=case, index=index, cold=cold, warm=warm,
                              speedup=cold["solve_s"] / warm["solve_s"]))
    return pairs


def average(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return stats.mean(values) if values else None


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(cell(x) for x in row) + " |" for row in rows]
    return "\n".join(lines)


def build_report(output):
    """Derive every reported metric from saved observations; no solver calls."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metadata = json.loads((output / "metadata.json").read_text())
    report_source = output / "report_source" / "scaling.py"
    report_source.parent.mkdir(exist_ok=True)
    report_source.write_bytes(Path(__file__).read_bytes())
    write_json(output / "report_generator.json", dict(
        generated_at=stamp(), script_sha256=digest(report_source), snapshot=str(report_source),
        measurement_script_sha256=metadata["source_files"]["distillation/solver/scaling.py"],
        python=platform.python_version(), numpy=np.__version__, matplotlib=matplotlib.__version__))
    config = metadata["config"]
    inputs = json.loads((output / "inputs.json").read_text())
    journal = output / "observations.jsonl"
    observations = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    rows = list({r["id"]: r for r in observations}.values())
    batches = json.loads((output / "batches.json").read_text()) if (output / "batches.json").exists() else []
    exclusions = json.loads((output / "exclusions.json").read_text()) if (output / "exclusions.json").exists() else []
    cold = [r for r in rows if r["phase"] == "batch" and r["workers"] == 1 and r["threads"] == 1]
    pairs = paired_results(rows)
    plot_files = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=170)
            plot_files.append(f"{name}.{ext}")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for solver in config["solvers"]:
        x, y, err = [], [], []
        for case in config["cases"]:
            values = [r["solve_s"] for r in cold if r["solver"] == solver and r["case"] == case and r["success"]]
            if values:
                x.append(case)
                y.append(stats.mean(values))
                err.append(stats.stdev(values) if len(values) > 1 else 0)
        if x:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{solver}, cold mean ± SD")
    ax.set(xscale="log", yscale="log", xlabel="Buses", ylabel="Solver wall time (s)",
           title="AC-OPF latency; one CPU thread per solve")
    ax.set_xticks(config["cases"], [str(c) for c in config["cases"]])
    ax.grid(True, which="both", alpha=.2)
    if ax.lines:
        ax.legend(fontsize=8)
    save(fig, "runtime_by_size")

    parallel = [b for b in batches if b["phase"] == "batch" and b["complete_timing"] and b.get("wall_s")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for solver in config["solvers"]:
        for case in config["cases"]:
            group = sorted([b for b in parallel if b["solver"] == solver and b["case"] == case], key=lambda b: b["workers"])
            if not group:
                continue
            x = [b["workers"] for b in group]
            style = "o-" if solver == "pypower" else "s--"
            label = f"{solver} {case}"
            for ax, y in zip(axes, ([b["success_count"] / b["wall_s"] for b in group],
                                     [b["cpu_cores"] for b in group], [b["peak_rss_gib"] for b in group])):
                ax.plot(x, y, style, label=label, markersize=4)
    for ax, ylabel in zip(axes, ("Successful solves / s", "Mean CPU cores occupied", "Peak active-worker RSS sum (GiB)")):
        ax.set(xlabel="Concurrent workers (one thread each)", ylabel=ylabel)
        ax.grid(True, alpha=.2)
    axes[0].set_yscale("log")
    if axes[2].lines:
        axes[2].legend(fontsize=7, loc="best")
    save(fig, "parallel_resources")

    if pairs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sizes = sorted({p["case"] for p in pairs})
        speedups = [[p["speedup"] for p in pairs if p["case"] == c] for c in sizes]
        axes[0].boxplot(speedups, tick_labels=[str(c) for c in sizes])
        axes[0].axhline(1, color="gray", linestyle="--")
        axes[0].set(xlabel="Buses", ylabel="Paired cold / warm time", title="Above 1 means faster with a warm start")
        for case in sizes:
            group = [p for p in pairs if p["case"] == case]
            axes[1].scatter([100 * p["warm"]["demand_change_relative"] for p in group],
                            [p["speedup"] for p in group], label=str(case))
        axes[1].axhline(1, color="gray", linestyle="--")
        axes[1].set(xlabel="Demand difference / target demand norm (%)", ylabel="Paired cold / warm time")
        axes[1].legend(title="Buses")
        save(fig, "warm_start")

    complete = output / "completed.json"
    status = (f"Measurements completed {json.loads(complete.read_text())['completed_at']}." if complete.exists()
              else "Partial report: the measurement campaign has not completed.")
    lines = ["# CPU AC-OPF scaling", "", f"Generated {stamp()}. Run started {metadata['started_at']}. {status}", "",
             f"Host `{metadata['host']}`; {metadata['cpu_model']}. Available affinity: {len(metadata['available_cpus'])} logical CPUs; "
             f"benchmark budget: {len(metadata['benchmark_cpus'])}. RAM: {metadata['ram_gib']:.1f} GiB total, "
             f"{metadata['available_ram_gib']:.1f} GiB available at launch.", "",
             f"Each case uses {config['batch']} distinct targets plus one pilot/source record, streamed from archive group 0. "
             "These are deterministic first records, not a random test-split estimate or an exhaustive capacity test. "
             "Every concurrency level reuses the same targets. All inputs and optimization use CPU float64.", "",
             f"Each task has a {config['timeout']:.0f} s deadline including input/model setup and a {config['memory_gib']:.1f} GiB RSS guard. "
             "A limit hit cancels that batch and excludes further work for the solver/case. "
             "PYPOWER uses sparse PIPS; SciPy uses dense trust-constr with automatic derivatives. "
             f"SciPy is pre-screened above {config['scipy_max_buses']} buses; this is a memory/compute policy, not a measured timeout.", "",
             "## Runtime", "",
             "Solver time includes independent feasibility validation, but excludes file loading/model setup, worker startup and priming. "
             "Means, percentiles and resource averages below use converged, feasible serial solves only; success counts show exclusions. "
             "Failed attempts are retained in the raw journal and batch wall times.", ""]
    runtime_table = []
    for case in config["cases"]:
        description = next(r for r in inputs if r["case"] == case)
        for solver in config["solvers"]:
            group = [r for r in cold if r["case"] == case and r["solver"] == solver]
            valid = [r for r in group if r["success"]]
            pilot = next((r for r in rows if r["case"] == case and r["solver"] == solver and r["phase"] == "pilot"), {})
            pilot_time = (f"≥{config['timeout']:g} (task timeout)" if pilot.get("status") == "timeout"
                          else number(pilot.get("solve_s", pilot.get("elapsed_s"))))
            values = [r["solve_s"] for r in valid]
            runtime_table.append([case, description["branches"], description["generators"], solver,
                                  f"{len(valid)}/{len(group)}", pilot_time, number(average(valid, "solve_s")),
                                  number(float(np.percentile(values, 95))) if values else "—",
                                  number(average(valid, "setup_s")), number(average(valid, "solve_cpu_s")),
                                  number(max((r["peak_rss_gib"] for r in valid), default=None))])
    lines += [table(["Buses", "Branches", "Generators", "Solver", "Valid / tried", "Pilot (s)", "Mean (s)",
                     "P95 (s)", "Mean setup (s)", "CPU (s/solve)", "Peak worker RSS (GiB)"], runtime_table), "",
              "![Runtime by network size](runtime_by_size.png)", "", "## Parallel solves", "",
              "Batch wall time includes per-task loading/setup, result writing, garbage collection, IPC and the monitoring loop, and excludes worker startup. "
              "Throughput counts successful solutions only. Speedup is serial batch wall time divided by parallel batch wall time. "
              "Mean occupied cores = summed worker CPU seconds / batch wall seconds; 1 core is 100% in the usual per-process CPU display. "
              "OS thread counts also include idle runtime threads; they do not measure CPU utilization.", ""]
    parallel_table = []
    for b in sorted(parallel, key=lambda b: (b["solver"], b["case"], b["workers"])):
        serial = next((a for a in parallel if a["solver"] == b["solver"] and a["case"] == b["case"] and a["workers"] == 1), None)
        parallel_table.append([b["solver"], b["case"], b["workers"], f"{b['success_count']}/{b['requested']}",
                               number(b["wall_s"]), number(b["success_count"] / b["wall_s"]),
                               number(serial["wall_s"] / b["wall_s"]) if serial else "—",
                               number(b["cpu_cores"]), number(b["peak_rss_gib"]), b["peak_os_threads"]])
    lines += [table(["Solver", "Buses", "Workers", "Valid / batch", "Batch (s)", "Solves/s", "Speedup (×)",
                     "CPU cores", "Peak RSS sum (GiB)", "OS threads"], parallel_table), "",
              "![Parallel throughput and CPU/memory consumption](parallel_resources.png)", ""]
    pypower = [b for b in parallel if b["solver"] == "pypower" and b["success_count"] == b["requested"]]
    for case in config["cases"]:
        choices = [b for b in pypower if b["case"] == case]
        if choices:
            best = min(choices, key=lambda b: b["wall_s"])
            maximum = max(b["workers"] for b in choices)
            lines.append(f"- Case {case}: up to **{maximum}** simultaneous workers completed; fastest tested batch used "
                         f"**{best['workers']}** workers ({best['wall_s']:.3f} s for {best['requested']} problems, "
                         f"{best['cpu_cores']:.2f} mean CPU cores, {best['peak_rss_gib']:.2f} GiB peak RSS sum).")
    lines += ["", f"RSS is sampled every {config['poll_seconds']:.3f} s and includes interpreter/libraries. "
              "Summed RSS double-counts shared pages and may miss brief peaks; treat it as a conservative resident-memory planning estimate, "
              "not private allocations. The persistent pool can retain idle workers; its full RSS is saved separately in batches.json. "
              "The experiment establishes tested concurrency, not maximum sustainable capacity on this shared login host.", "",
              "## Warm starts", "",
              "PYPOWER/PIPS constructs its own interior point and rejects supplied x0 in this wrapper, so no PYPOWER warm-start speedup is claimed. "
              "SciPy receives only the primal solution of the preceding distinct problem in archive order (same equipment ordering/topology). "
              "This is a fresh optimization, not continuation of multipliers or a barrier state. Target labels are used only for objective checks. "
              "The paired cold/warm order is deterministically randomized. Speedups use adjacent cold remeasurements to reduce run-order effects. "
              "Both must converge, pass independent feasibility checks, and reach costs within 0.01% of each other to enter speedup statistics. "
              "Demand change is the norm of target minus source demand divided by the target demand norm, including active and reactive bus demands.", ""]
    totals = [b["peak_total_rss_gib"] for b in batches if b.get("peak_total_rss_gib") is not None]
    if totals:
        lines.insert(lines.index("## Warm starts") - 1,
                     f"\nPeak total RSS including the controller and idle workers was {max(totals):.2f} GiB.")
    startup_path = output / "worker_startups.json"
    if startup_path.exists():
        startups = json.loads(startup_path.read_text())
        lines.insert(lines.index("## Warm starts") - 1,
                     f"\nWorker import/priming time averaged {stats.mean(s['startup_s'] for s in startups):.2f} s "
                     f"across {len(startups)} process launches; mean idle RSS after priming was "
                     f"{stats.mean(s['rss'] for s in startups) / GIB:.2f} GiB. Reuse workers to amortize this startup cost.")
    warm_table = []
    for solver, case in sorted({(p["solver"], p["case"]) for p in pairs}):
        group = [p for p in pairs if p["solver"] == solver and p["case"] == case]
        cold_mean = stats.mean(p["cold"]["solve_s"] for p in group)
        warm_mean = stats.mean(p["warm"]["solve_s"] for p in group)
        attempted = len({r["index"] for r in rows if r["phase"] == "paired" and r["solver"] == solver and r["case"] == case})
        warm_table.append([solver, case, f"{len(group)}/{attempted}", number(cold_mean), number(warm_mean),
                           number(cold_mean / warm_mean), number(stats.geometric_mean(p["speedup"] for p in group)),
                           number(average([p["cold"] for p in group], "iterations"), 1),
                           number(average([p["warm"] for p in group], "iterations"), 1),
                           number(100 * stats.mean(p["warm"]["demand_change_relative"] for p in group), 1)])
    if warm_table:
        lines += [table(["Solver", "Buses", "Valid pairs / tried", "Cold mean (s)", "Warm mean (s)",
                         "Ratio of means (×)", "Geometric speedup (×)", "Cold iterations", "Warm iterations", "Demand change (%)"], warm_table), ""]
        total_cold = sum(p["cold"]["solve_s"] for p in pairs)
        total_warm = sum(p["warm"]["solve_s"] for p in pairs)
        geo = stats.geometric_mean(p["speedup"] for p in pairs)
        lines += [f"Across {len(pairs)} accepted pairs, total cold time / total warm time is **{total_cold / total_warm:.3f}×**; "
                  f"the geometric mean of individual speedups is **{geo:.3f}×**. Values below 1 mean a slowdown. "
                  "The source solve is excluded from warm latency: it must already exist for reuse to help. "
                  "Solving an extra problem solely to obtain a start has an additional cost recorded in solutions/*.json. "
                  "These independently perturbed OPFData examples are not a smooth time series; a closer prediction may behave differently.", "",
                  "![Warm-start comparison](warm_start.png)", ""]
    else:
        lines += ["No qualifying warm-start pairs were measured.", ""]
    lines += ["## Thread sensitivity", ""]
    thread_table = []
    for solver in config["solvers"]:
        for case in config["cases"]:
            for threads in config["thread_counts"]:
                group = [r for r in rows if r["solver"] == solver and r["case"] == case and r["threads"] == threads
                         and r["workers"] == 1 and r["index"] in range(1, 4)
                         and r["phase"] == ("batch" if threads == 1 else "threads")]
                valid = [r for r in group if r["success"]]
                if group:
                    thread_table.append([solver, case, threads, f"{len(valid)}/{len(group)}", number(average(valid, "solve_s")),
                                         number(stats.mean(r["solve_cpu_s"] / r["solve_s"] for r in valid)) if valid else "—"])
    lines += [table(["Solver", "Buses", "Threads/solve", "Valid / tried", "Mean solver time (s)", "Mean occupied cores"], thread_table), "",
              "Thread settings use the same first three targets, one process at a time. Thread affinity and numerical-library limits are both enforced.", "",
              "## Size sensitivity", ""]
    for solver in config["solvers"]:
        for factor in ("buses", "branches"):
            points = []
            for case in config["cases"]:
                valid = [r for r in cold if r["solver"] == solver and r["case"] == case and r["success"]]
                if valid:
                    size = stats.mean(i[factor] for i in inputs if i["case"] == case and i["index"] > 0)
                    points.append((size, stats.mean(r["solve_s"] for r in valid)))
            if len(points) >= 3:
                x, y = np.log(np.asarray(points)).T
                slope, intercept = np.polyfit(x, y, 1)
                residual, total = np.square(y - (intercept + slope * x)).sum(), np.square(y - y.mean()).sum()
                lines.append(f"- {solver}, {len(points)} case means: time ≈ {math.exp(intercept):.4g} × {factor}^{slope:.3f} seconds; "
                             f"log-space R² = {1 - residual / total:.3f}.")
    lines += ["", "These are descriptive one-factor log–log fits, not complexity guarantees. Bus/branch counts are correlated, "
              "and topology, operating point, generator count and iteration count also change. No independent causal edge-count effect is identifiable here.", "",
              "## Numerical checks and exclusions", ""]
    valid = [r for r in rows if r["success"]]
    if valid:
        max_balance = max(r["checks"]["balance"] for r in valid)
        max_gap = max(abs(r["cost_gap_pct"]) for r in valid)
        lines.append(f"{len(valid)}/{len(rows)} recorded tasks returned converged, feasible solutions. Maximum power-balance residual "
                     f"among accepted solves: {max_balance:.3g} p.u.; maximum absolute generation-cost difference from the stored "
                     f"OPFData solution: {max_gap:.6g}%. Feasibility uses the solver package's 1e-6 tolerance for power/voltage/angle checks. "
                     "Matching costs is a consistency check, not a certificate of global optimality for nonconvex AC-OPF.")
    failures = [r for r in rows if not r["success"]]
    if failures:
        lines += ["", table(["Task", "Status", "Task elapsed (s)", "Detail"], [[r["id"], r["status"],
                   number(r.get("dispatch_s", r.get("elapsed_s"))),
                   (r.get("reason") or r.get("message") or r.get("error", "").strip().split("\n")[-1] or "See journal")]
                   for r in failures])]
    lines += ["", table(["Solver", "Buses", "Exclusion"], [[e["solver"], e["case"], e["reason"]] for e in exclusions])
              if exclusions else "No solver/case exclusions.", "", "## Reproduction and sources", "",
              "```bash", "source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh",
              "python distillation/solver/scaling.py --report-only " + str(output), "```", "",
              "Run configuration, CPU affinity, source hashes and invocation: [metadata.json](metadata.json). "
              "Exact archived member names and input SHA-256 hashes: [inputs.json](inputs.json). "
              "Per-solve timings, checks and starts: [observations.jsonl](observations.jsonl). "
              "Batch wall/CPU/RSS measurements: [batches.json](batches.json). "
              "Worker startup timings and library versions: [worker_startups.json](worker_startups.json). "
              "Report-generator hash, snapshot and plotting versions: [report_generator.json](report_generator.json). "
              "Resource traces are saved as trace_*.json; feasible source solutions as solutions/*.json. "
              "Original measurement code is preserved in source/; the report generator is preserved separately in report_source/. "
              "Copied inputs and source files allow inspection without rereading the large archives. "
              "This was a direct login-node run; no PBS job created these results.", "",
              f"Source dataset directory: `{config['raw_root']}`. Plots are saved as PNG and SVG."]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    write_json(output / "summary.json", dict(rows=len(rows), successes=len(valid), warm_pairs=len(pairs), plots=plot_files))
    print(f"Report: {output / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=int, nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--solvers", nargs="+", choices=SOLVERS, default=list(SOLVERS))
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--thread-counts", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--memory-gib", type=float, default=16)
    parser.add_argument("--scipy-max-buses", type=int, default=118)
    parser.add_argument("--poll-seconds", type=float, default=.02)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--report-only", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    # Also bound the controller's reporting libraries.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    if args.report_only:
        build_report(args.report_only.resolve())
        return
    import psutil
    available = cpu_order()
    budget = max(1, len(available) // 2)
    cpus = available[:budget]
    output = (args.resume or args.output or REPORT_ROOT / datetime.now(timezone.utc).strftime("scaling_%Y%m%d_%H%M%S")).resolve()
    if args.resume and (output / "completed.json").exists():
        build_report(output)
        return
    if args.resume:
        metadata = json.loads((output / "metadata.json").read_text())
        config = metadata["config"]
        manifest = json.loads((output / "inputs.json").read_text())
        for path, sha256 in metadata["source_files"].items():
            if digest(ROOT / path) != sha256:
                raise ValueError(f"Source changed since this run: {path}; use a new output directory")
        for record in manifest:
            if digest(record["path"]) != record["sha256"]:
                raise ValueError(f"Input changed: {record['path']}")
    else:
        config = vars(args).copy()
        for key in ("output", "resume", "report_only"):
            config.pop(key)
        config["raw_root"] = str(args.raw_root.resolve())
        config["workers"] = sorted(set([1] + [min(w, args.batch) for w in args.workers]))
        config["thread_counts"] = sorted(set([1] + args.thread_counts))
        output.mkdir(parents=True, exist_ok=False)
        cpuinfo = Path("/proc/cpuinfo").read_text()
        model = next((line.split(":", 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith("model name")), "unknown")
        memory = psutil.virtual_memory()
        metadata = dict(started_at=stamp(), host=platform.node(), cpu_model=model,
                        available_cpus=available, benchmark_cpus=cpus, ram_gib=memory.total / GIB,
                        available_ram_gib=memory.available / GIB, config=config, argv=sys.argv,
                        load_average=os.getloadavg(), source_files={})
        sources = sorted(set(Path(__file__).parent.glob("*.py")) |
                         set((Path(__file__).parent / "backends").glob("*.py")) |
                         set((ROOT / "distillation/conversion").glob("*.py")))
        for path in sources:
            relative = path.relative_to(ROOT)
            target = output / "source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            metadata["source_files"][str(relative)] = digest(path)
        write_json(output / "metadata.json", metadata)
        manifest = prepare_inputs(config, output)
    if min(config["workers"] + config["thread_counts"] + [config["batch"], config["timeout"], config["memory_gib"], config["poll_seconds"]]) <= 0:
        parser.error("Batch, worker/thread counts and limits must be positive")
    if max(config["workers"]) * max(config["thread_counts"]) > budget:
        parser.error(f"Worker slots × maximum threads must be <= half the available CPUs ({budget})")
    os.sched_setaffinity(0, cpus)
    print(f"{stamp()} output {output}; CPU budget {budget}, peak configured slots {max(config['workers']) * max(config['thread_counts'])}", flush=True)
    try:
        campaign(config, output, manifest, cpus)
        write_json(output / "completed.json", dict(completed_at=stamp()))
    finally:
        build_report(output)


if __name__ == "__main__":
    main()
