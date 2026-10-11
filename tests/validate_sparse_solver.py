"""Validate CPU/XPU AC-OPF against PYPOWER on reproducibly selected OPFData.

Defaults: 32 samples each for 14/30/57/118, three each for 500/2000. Samples
come from the first 128 JSON records of shard 0, selected with seed 42.
Labels are used only for reporting. Exit status is nonzero on any failure.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW_ROOT = Path("/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/raw/OPFData/dataset_release_1")


def input_network(obj):
    import torch
    from torch_geometric.data import HeteroData
    grid, data = obj["grid"], HeteroData()
    data.x = torch.tensor(grid["context"], dtype=torch.float64).reshape(-1)
    for kind, width in (("bus", 4), ("generator", 11), ("load", 2), ("shunt", 2)):
        data[kind].x = torch.tensor(grid["nodes"][kind], dtype=torch.float64).reshape(-1, width)
    for kind, edge in grid["edges"].items():
        source = "bus" if kind in ("ac_line", "transformer") else kind.removesuffix("_link")
        store = data[source, kind, "bus"]
        store.edge_index = torch.tensor([edge["senders"], edge["receivers"]], dtype=torch.long).reshape(2, -1)
        if kind in ("ac_line", "transformer"):
            width = 9 if kind == "ac_line" else 11
            store.edge_attr = torch.tensor(edge["features"], dtype=torch.float64).reshape(-1, width)
    return data


def samples(root, case, count, pool, seed):
    """Stream selected members without extracting or modifying the dataset."""
    import numpy as np
    name = f"pglib_opf_case{case}_{'goc' if case >= 500 else 'ieee'}"
    archive = root / f"{name}_0.tar.gz"
    wanted = set(np.random.default_rng(seed+case).choice(pool, count, replace=False).tolist())
    found, index = [], -1
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            index += 1
            if index in wanted:
                raw = tar.extractfile(member).read()
                obj = json.loads(raw)
                source = dict(case=case, index=index, archive=str(archive), member=member.name,
                              sha256=hashlib.sha256(raw).hexdigest(), dataset_objective=obj["metadata"]["objective"])
                found.append((input_network(obj), source))
            if len(found) == count:
                break
    if len(found) != count:
        raise ValueError(f"Only found {len(found)}/{count} selected records in {archive}")
    return found


def numerical_checks(device):
    """Check derivatives, device LU, bounds, batching, warm starts and failures."""
    import numpy as np
    import torch
    from torch.func import hessian, jacrev
    from pypower.api import case9, case14
    from distillation.solver import ACOPF, PyPowerSolver
    from distillation.solver.sparse_solver import XPUSolver
    from distillation.solver.sparse_solver.solver import KKT
    from distillation.solver.sparse_solver.solver import NetworkBatch

    torch.manual_seed(42)
    ppc = case9()
    ppc["branch"][0, 8:10] = (1.03, 7.)
    ppc["bus"][3, 4:6] = (2., -3.)
    second = deepcopy(ppc)
    second["bus"][:, 2:4] *= 1.04
    second["branch"][:, 2:4] *= 1.02
    models = [ACOPF(p) for p in (ppc, second)]
    net = NetworkBatch(models, device)
    x = net.initial().cpu()+0.01*torch.randn((2, net.nx), dtype=torch.float64)
    data = net.evaluate(x.to(device))
    lam = torch.randn((2, net.neq), dtype=torch.float64)
    mu = torch.rand((2, net.niq), dtype=torch.float64)+.1
    slack = torch.rand_like(mu)+.2
    hv, gram = net.curvature(data, lam.to(device), mu.to(device), slack.to(device))
    measured_h = net.hessian.dense(hv).cpu()
    measured_g = net.jg.dense(data["jg"]).cpu()
    measured_i = net.jh.dense(data["jh"]).cpu()
    for i, model in enumerate(models):
        ae, ai = net.ae.dense(net.aev)[i].cpu(), net.ai.dense(net.aiv)[i].cpu()
        def g(z):
            return torch.cat((-model.balance(z), ae@z-net.be[i].cpu()))
        def h(z):
            sf, st = model.branch_flows(z)
            s = torch.cat((sf[model.monitored], st[model.monitored]))
            # real²+imag² has a well-defined Hessian at zero flow.
            flows = s.real.square()+s.imag.square()-model.rate[model.monitored].repeat(2).square()
            return torch.cat((flows, ai@z-net.bi[i].cpu()))
        torch.testing.assert_close(data["g"][i].cpu(), g(x[i]), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(data["h"][i].cpu(), h(x[i]), atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(measured_g[i], jacrev(g)(x[i]), atol=1e-8, rtol=1e-9)
        torch.testing.assert_close(measured_i[i], jacrev(h)(x[i]), atol=1e-8, rtol=1e-9)
        expected = hessian(lambda z: model.cost(z)*1e-4+g(z)@lam[i]+h(z)@mu[i])(x[i])
        torch.testing.assert_close(measured_h[i], expected, atol=1e-7, rtol=1e-9)
    expected_gram = measured_i.transpose(1, 2)@((mu/slack)[..., None]*measured_i)
    torch.testing.assert_close(net.gram.dense(gram).cpu(), expected_gram, atol=1e-7, rtol=1e-9)

    references = [PyPowerSolver().solve(m) for m in models]
    for backend in (["sparse", "dense"] if device.type == "cpu" else ["dense"]):
        kkt = KKT(net, backend, 4., 1e-7)
        values = kkt.assemble(hv, gram, data["jg"])
        values[1] = 0.
        active = torch.ones(2, dtype=torch.bool, device=device)
        _, valid, _ = kkt.solve(values, values.new_ones((2, kkt.n)), active)
        assert valid.tolist() == [True, False], "Singular batch member was not isolated"
        solver = XPUSolver(device=device, linear_solver=backend)
        results = solver.solve_batch(models)
        for result, reference in zip(results, references):
            assert result.success and reference.success, (result.info, result.checks)
            assert abs(result.objective/reference.objective-1) < 1e-6
        repeated = solver.solve(models[0])
        np.testing.assert_allclose(repeated.x(), results[0].x(), rtol=1e-7, atol=1e-7)
        warm = solver.solve(models[0], x0=results[0])
        assert warm.success and warm.info["warm_start"]
        assert abs(warm.objective/references[0].objective-1) < 1e-6
    solver = XPUSolver(device=device)
    ppc = case14()
    ppc["branch"][:, 5] = 0
    m = ACOPF(ppc)
    r, p = solver.solve(m), PyPowerSolver().solve(m)
    assert r.success and p.success and abs(r.objective/p.objective-1) < 1e-6
    ppc = case9()
    ref = PyPowerSolver().solve(ACOPF(ppc))
    ppc["gen"] = ppc["gen"].astype(float)
    ppc["gen"][1, 3:5] = ref.qg[1]*ppc["baseMVA"]
    ppc["branch"][:, 11:13] = (-35, 35)
    m = ACOPF(ppc)
    r, p = solver.solve(m), PyPowerSolver().solve(m)
    assert r.success and p.success and abs(r.objective/p.objective-1) < 1e-6
    ids = {value: 100+7*i for i, value in enumerate(ppc["bus"][:, 0])}
    for array, cols in ((ppc["bus"], [0]), (ppc["gen"], [0]), (ppc["branch"], [0, 1])):
        for col in cols:
            array[:, col] = [ids[value] for value in array[:, col]]
    ppc["gen"], ppc["gencost"] = ppc["gen"][::-1].copy(), ppc["gencost"][::-1].copy()
    reordered = solver.solve(ppc)
    assert reordered.success
    np.testing.assert_allclose(reordered.pg[::-1], r.pg, atol=1e-6, rtol=1e-6)
    assert not XPUSolver(device=device, max_iter=0).solve(case9()).success
    bad = case9()
    bad["gen"] = bad["gen"].astype(float)
    bad["gen"][:, 9], bad["gen"][:, 8] = 0, .1
    good, failed = XPUSolver(device=device, max_iter=50).solve_batch([case9(), bad])
    assert good.success and not failed.success
    try:
        solver.solve(case9(), x0=np.full(24, np.nan))
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite warm start was accepted")
    print("Analytic derivatives, sparse/dense agreement and solver edge cases: PASS", flush=True)


def pipeline_checks():
    """Mixed requests, immutable snapshots, cache refresh, failures and pairing."""
    import numpy as np
    import torch
    from unittest.mock import patch
    from pypower.api import case9, case14, case30
    from distillation.solver import ACOPF, PyPowerSolver
    from distillation.solver.sparse_solver import SparseProblem, SparseSolver, SolverPool, Topology, Scenario
    from distillation.solver.sparse_solver.batch import _Cache

    base = case14()
    topology = Topology("base14", base)
    params = base["bus"].copy()
    original = params.copy()
    snapshot = Scenario(topology, "immutable", {"bus": params})
    params[:, 2] *= 100
    np.testing.assert_array_equal(snapshot.inputs()["bus"], original)
    with patch.object(ACOPF, "__init__", side_effect=AssertionError("Dense model constructed")):
        result = SparseSolver().solve(case14())
        assert result.success
    cache = _Cache(2, 16*1024**2)
    solver = SparseSolver(cache=cache)
    scenarios = [snapshot]
    for i in range(7):
        data = deepcopy(base)
        data["bus"][:, 2:4] *= 1+.01*i
        data["branch"][:, 2:4] *= 1+.015*i
        data["bus"][4, 4:6] += (i*.05, i*.1)
        if i == 1:
            data["branch"][0, 8:10] = (1.02, 2.)
        if i == 2:
            data["branch"][0, 5] = 0
        if i == 3:
            data["branch"][:, 11:13] = (-35, 35)
        if i == 4:
            data["gen"], data["gencost"] = data["gen"][::-1].copy(), data["gencost"][::-1].copy()
        if i == 5:
            data["branch"] = data["branch"][::-1].copy()
        if i == 6:
            data["branch"] = data["branch"][:-1].copy()
        scenarios.append(Scenario(topology, f"changed-{i}", data))
    generated = case9()
    extra = generated["bus"][4].copy()
    extra[0], extra[1], extra[2], extra[3] = 10, 1, 5, 2
    generated["bus"] = np.vstack((generated["bus"], extra))
    edge = generated["branch"][1].copy()
    edge[0], edge[1] = 9, 10
    generated["branch"] = np.vstack((generated["branch"], edge))
    scenarios.append(Scenario(Topology("generated10", generated), 0))
    for kind, builder, count in (("nine", case9, 2), ("thirty", case30, 3)):
        template = Topology(kind, builder())
        scenarios.extend(Scenario(template, i) for i in range(count))
    expected = []
    for scenario in scenarios:
        candidate = solver.solve(scenario.inputs())
        reference = PyPowerSolver().solve(ACOPF(scenario.inputs()))
        assert candidate.success and reference.success, (scenario.scenario_id, candidate.info)
        assert abs(candidate.objective-reference.objective)/max(1., abs(reference.objective)) < 1e-5
        expected.append(candidate)
    cached = solver.solve(scenarios[-1].inputs())
    assert cached.info["layout_cache_hit"] and cached.info["admittance_cache_hit"]
    invalid = deepcopy(base)
    invalid["bus"][0, 2] = np.nan
    scenarios.append(Scenario(topology, "nan", invalid))
    invalid = deepcopy(base)
    invalid["branch"][:, 10] = 0
    scenarios.append(Scenario(topology, "offline", invalid))
    scenarios.append(Scenario(topology, "bad-start", x0=np.full(3, np.nan)))
    invalid = deepcopy(base)
    invalid["branch"] = invalid["branch"][:0]
    scenarios.append(Scenario(topology, "island", invalid))
    from torch_geometric.data import HeteroData
    scenarios.append(Scenario(Topology("malformed-native", HeteroData()), "missing-features"))
    impossible = deepcopy(base)
    impossible["gen"][:, 8:10] = (.01, 0)
    scenarios.append(Scenario(topology, "capacity", impossible))
    parent_threads, parent_affinity = torch.get_num_threads(), os.sched_getaffinity(0)
    with SolverPool(workers=4, cpu_budget=4, task_size=4, max_pending=3,
                    cache_entries=2, cache_mib=16) as pool:
        stream = list(pool.stream(scenarios, request_id="stream", generator_version=7, branch_labels=True))
        assert len(stream) == len(scenarios)
        by_index = {r.index: r for r in stream}
        for i, reference in enumerate(expected):
            result = by_index[i]
            assert result.success and result.generator_version == 7
            assert result.ids == ("stream", scenarios[i].topology.topology_id, scenarios[i].scenario_id)
            np.testing.assert_allclose(result.solution.x(), reference.x(), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(result.y["bus"][:, 0], reference.theta, atol=1e-7)
            assert result.y["branch"].shape[0] == len(scenarios[i].inputs()["branch"])
            assert not result.worker["xpu_initialized"]
            assert result.worker["cache_entries"] <= 2 and result.worker["cache_bytes"] <= 16*1024**2
        for i in range(len(expected), len(scenarios)):
            assert not by_index[i].success and by_index[i].y is None
        assert [by_index[i].status for i in range(len(expected), len(expected)+5)] == ["invalid_input"]*5
        assert pool.stats["peak_pending_tasks"] <= 3 and pool.stats["peak_buffered_results"] <= 12
        pids = {r.worker["pid"] for r in stream}
        repeated = pool.solve(scenarios[:len(expected)], request_id="blocking")
        assert [r.scenario_id for r in repeated] == [s.scenario_id for s in scenarios[:len(expected)]]
        assert all(r.success for r in repeated)
        assert {r.worker["pid"] for r in repeated} & pids
        stream = pool.stream(Scenario(topology, i) for i in range(100))
        next(stream)
        stream.close()
        assert pool.solve([Scenario(topology, "after-close")])[0].success
        try:
            pool.solve([snapshot, snapshot])
        except ValueError as exc:
            assert "Duplicate" in str(exc)
        else:
            raise AssertionError("Duplicate IDs accepted")
        assert pool.solve([snapshot])[0].success
    assert parent_threads == torch.get_num_threads() and parent_affinity == os.sched_getaffinity(0)
    import psutil
    assert all(not psutil.pid_exists(pid) for pid in pids)
    with SolverPool(workers=1, cpu_budget=1, solver_options={"max_iter": 0}) as pool:
        failed = pool.solve([snapshot])[0]
        assert failed.status == "not_converged" and failed.y is None
    with SolverPool(workers=1, cpu_budget=1, solver_options={"time_limit": 1e-9}) as pool:
        failed = pool.solve([snapshot])[0]
        assert failed.status == "timeout" and failed.y is None
    from concurrent.futures.process import BrokenProcessPool
    with SolverPool(workers=1, cpu_budget=1) as pool:
        pool.solve([snapshot])
        try:
            pool._executor.submit(os._exit, 23).result()
        except BrokenProcessPool:
            pass
        else:
            raise AssertionError("Worker crash did not surface")
    assert cache.evictions > 0
    print("Persistent pool, generated topologies, caches, pairing, bounds and failures: PASS", flush=True)
    return dict(passed=True, reference_comparisons=len(expected), failure_scenarios=8,
                parent_threads_unchanged=True, worker_xpu_initialized=False, clean_shutdown=True)


class ResourceMonitor:
    """Sample the controller and children; summed RSS includes shared pages."""
    def __init__(self):
        import psutil
        import threading
        self.root, self.psutil = psutil.Process(), psutil
        self.stop_event = threading.Event()
        self.baseline, self.latest = {}, {}
        self.peak_rss = self.peak_threads = 0
        self.poll()
        self.baseline = self.latest.copy()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def poll(self):
        rss = threads = 0
        for proc in [self.root]+self.root.children(recursive=True):
            try:
                with proc.oneshot():
                    cpu = proc.cpu_times()
                    self.latest[proc.pid] = cpu.user+cpu.system
                    rss += proc.memory_info().rss
                    threads += proc.num_threads()
            except self.psutil.NoSuchProcess:
                continue
        self.peak_rss = max(self.peak_rss, rss)
        self.peak_threads = max(self.peak_threads, threads)

    def run(self):
        while not self.stop_event.wait(.05):
            self.poll()

    def finish(self, elapsed):
        self.stop_event.set()
        self.thread.join()
        self.poll()
        cpu = sum(value-self.baseline.get(pid, 0) for pid, value in self.latest.items())
        return dict(cpu_s=cpu, occupied_cores=cpu/elapsed, peak_rss_gib=self.peak_rss/1024**3,
                    peak_os_threads=self.peak_threads)


def measured_request(pool, scenarios, count, request_id):
    import numpy as np
    monitor = ResourceMonitor()
    start = time.perf_counter()
    first = first_batch = None
    successes, identities, errors, worker_ids = 0, set(), [], set()
    metrics = {k: [] for k in ("preparation_s", "queue_s", "solve_s", "validation_s", "latency_s", "snapshot_s")}
    cache_hits = cache_total = 0
    max_balance = max_thermal = max_bounds = 0.
    for result in pool.stream(scenarios, request_id=request_id, generator_version=request_id):
        elapsed = time.perf_counter()-start
        first = elapsed if first is None else first
        assert result.ids not in identities and result.generator_version == request_id
        identities.add(result.ids)
        worker_ids.add(result.worker["pid"])
        assert not result.worker["xpu_initialized"]
        if result.success:
            successes += 1
            assert all(a.dtype == np.float64 and np.isfinite(a).all() for a in result.y.values())
            if successes == min(32, count):
                first_batch = elapsed
            max_balance = max(max_balance, result.solution.checks["balance"])
            max_thermal = max(max_thermal, result.solution.checks["thermal"])
            max_bounds = max(max_bounds, result.solution.checks["bounds"])
            cache_hits += int(result.solution.info["layout_cache_hit"])
            cache_total += 1
        elif len(errors) < 20:
            errors.append(dict(ids=result.ids, status=result.status, reason=result.failure_reason))
        for key, values in metrics.items():
            if key in result.timing:
                values.append(result.timing[key])
    elapsed = time.perf_counter()-start
    resources = monitor.finish(elapsed)
    assert len(identities) == count
    row = dict(count=count, succeeded=successes, wall_s=elapsed, labels_per_s=successes/elapsed,
               first_result_s=first, first_minibatch_s=first_batch, minibatch=min(32, count),
               workers_used=len(worker_ids), layout_hit_fraction=cache_hits/max(1, cache_total),
               max_balance_pu=max_balance, max_thermal_pu=max_thermal, max_bounds_pu=max_bounds,
               errors=errors, scheduling=pool.stats.copy(), **resources)
    row["timing"] = {key: dict(mean=float(np.mean(a)), p95=float(np.percentile(a, 95)),
                               max=float(max(a))) for key, a in metrics.items() if a}
    return row


def benchmark(args):
    """Measured request sizes and cache reuse; no ideal-scaling extrapolations."""
    import numpy as np
    from pypower.api import case9
    from distillation.solver.problem import as_ppc
    from distillation.solver.sparse_solver import SolverPool, Topology, Scenario
    sources, problems = [], {}
    for case in (14, 30, 57, 118, 500, 2000):
        data, source = samples(args.raw_root, case, 1, 8, args.seed)[0]
        problems[case] = as_ppc(data)[0]
        sources.append(source)
    templates = []
    rng = np.random.default_rng(args.seed)
    for i in range(100):
        ppc = deepcopy(problems[14])
        ppc["branch"] = ppc["branch"][rng.permutation(len(ppc["branch"]))]
        templates.append(Topology(f"permutation-{i}", ppc))
    mixed = [Topology(str(case), ppc) for case, ppc in problems.items()]
    large = []
    for i in range(5):
        ppc = deepcopy(problems[2000])
        ppc["branch"] = ppc["branch"][rng.permutation(len(ppc["branch"]))]
        large.append(Topology(f"large-{i}", ppc))
    def scenarios(count):
        if count == 10:
            for i, template in enumerate((mixed*2)[:count]):
                yield Scenario(template, i)
        else:
            n = min(100, max(1, count//10))
            base, extra = divmod(count, n)
            for i in range(n):
                for j in range(base+(i < extra)):
                    bus = problems[14]["bus"].copy()
                    bus[:, 2:4] *= 1+.001*((j % 5)-2)
                    yield Scenario(templates[i], j, {"bus": bus})
    import scipy
    import torch
    report = dict(started=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  pbs_job=os.environ.get("PBS_JOBID"), kind="CPU pipeline benchmark", sources=sources,
                  torch=torch.__version__, scipy=scipy.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  source_sha256=source_hashes(), config=dict(workers=args.workers, task_sizes=args.task_sizes,
                  requests=args.requests, cache_entries=32, cache_mib=256), rows=[], startup=[])
    for workers in args.workers:
        for size in args.task_sizes:
            with SolverPool(workers=workers, cpu_budget=max(args.workers), task_size=size,
                            local_rank=0, local_world_size=1) as pool:
                start = time.perf_counter()
                warm = Topology("warmup", case9())
                pool.solve(Scenario(warm, i) for i in range(max(64, 2*workers*size)))
                report["startup"].append(dict(workers=workers, task_size=size, wall_s=time.perf_counter()-start))
                for count in args.requests:
                    for reuse in (False, True):
                        name = f"w{workers}-g{size}-n{count}-reuse{int(reuse)}"
                        # Large networks stay in individual tasks in this workload.
                        pool.task_size = 1 if count == 10 else size
                        row = measured_request(pool, scenarios(count), count, name)
                        row.update(workers=workers, task_size=pool.task_size, configured_task_size=size,
                                   reused_request=reuse, topology_count=min(100, max(1, count//10)) if count != 10 else 6)
                        report["rows"].append(row)
                        print(f"{name}: {row['succeeded']}/{count}, {row['wall_s']:.3f}s, "
                              f"{row['labels_per_s']:.1f} labels/s, {row['occupied_cores']:.1f} CPU, "
                              f"{row['peak_rss_gib']:.2f} GiB", flush=True)
                        save(args.output, report)
                        if row["succeeded"] != count:
                            raise AssertionError(row["errors"])
                if size == args.task_sizes[0] and 10 in args.requests:
                    pool.task_size = 1
                    name = f"w{workers}-large10"
                    tasks = (Scenario(template, j) for template in large for j in range(2))
                    row = measured_request(pool, tasks, 10, name)
                    row.update(workers=workers, task_size=1, workload="case2000", topology_count=5,
                               configured_task_size=size, reused_request=False)
                    report["rows"].append(row)
                    save(args.output, report)
                    print(f"{name}: {row['succeeded']}/10, {row['wall_s']:.3f}s, "
                          f"{row['occupied_cores']:.1f} CPU, {row['peak_rss_gib']:.2f} GiB", flush=True)
                    assert row["succeeded"] == 10, row["errors"]
    report.update(completed=datetime.now(timezone.utc).isoformat(), passed=True)
    save(args.output, report)
    return 0


def training_smoke(args):
    """Two generator versions, CPU labels and concurrent predictor updates on XPU."""
    import numpy as np
    import torch
    from torch import nn
    from distillation.solver.problem import as_ppc
    from distillation.solver.sparse_solver import SolverPool, Topology, Scenario
    if not torch.xpu.is_available():
        raise RuntimeError("Training integration requires an Aurora compute node")
    torch.manual_seed(args.seed)
    device = torch.device("xpu:0")
    graph, source = samples(args.raw_root, 118, 1, 8, args.seed)[0]
    ppc = as_ppc(graph)[0]
    nb, ng = len(ppc["bus"]), len(ppc["gen"])
    topology = Topology("training118", ppc)
    generator = nn.Sequential(nn.Linear(16, 64), nn.Tanh(), nn.Linear(64, 2*nb), nn.Tanh()).to(device)
    predictor = nn.Sequential(nn.Linear(2*nb, 256), nn.ReLU(), nn.Linear(256, 2*nb+2*ng)).to(device)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3)
    initial = [p.detach().clone() for p in predictor.parameters()]
    demand = torch.tensor(ppc["bus"][:, 2:4], dtype=torch.float32, device=device)
    transfer_s = generation_s = training_s = wait_s = 0.
    all_results, losses, versions = 0, [], []
    worker_pids, max_cache_bytes = set(), 0
    node_workers = min(16, max(args.workers))
    startup = time.perf_counter()
    with SolverPool(workers=node_workers, cpu_budget=node_workers, task_size=4,
                    local_rank=0, local_world_size=1) as pool:
        pool.solve(Scenario(topology, f"warm-{i}") for i in range(4*node_workers))
        startup = time.perf_counter()-startup
        monitor = ResourceMonitor()
        started = time.perf_counter()
        for version in range(2):
            generation_start = time.perf_counter()
            with torch.no_grad():
                proposals = generator(torch.randn(256, 16, device=device)).reshape(256, nb, 2)
                demands = demand[None]*(1+.002*proposals)
            torch.xpu.synchronize()
            generation_s += time.perf_counter()-generation_start
            transfer = time.perf_counter()
            cpu_demands = demands.detach().cpu().numpy()
            transfer_s += time.perf_counter()-transfer
            def scenarios():
                for i, loads in enumerate(cpu_demands):
                    bus = ppc["bus"].copy()
                    bus[:, 2:4] = loads
                    yield Scenario(topology, i, {"bus": bus})
            stream = pool.stream(scenarios(), request_id=f"train-{version}", generator_version=version)
            features, labels = [], []
            while True:
                waiting = time.perf_counter()
                try:
                    result = next(stream)
                except StopIteration:
                    wait_s += time.perf_counter()-waiting
                    break
                wait_s += time.perf_counter()-waiting
                assert result.success and result.generator_version == version
                assert not result.worker["xpu_initialized"]
                worker_pids.add(result.worker["pid"])
                max_cache_bytes = max(max_cache_bytes, result.worker["cache_bytes"])
                actual = result.scenario.inputs()["bus"][:, 2:4]
                np.testing.assert_array_equal(actual, cpu_demands[result.scenario_id])
                features.append(actual.reshape(-1)/ppc["baseMVA"])
                labels.append(result.solution.x().numpy())
                versions.append(version)
                all_results += 1
                if len(features) == 16:
                    transfer = time.perf_counter()
                    x = torch.tensor(np.stack(features), dtype=torch.float32, device=device)
                    y = torch.tensor(np.stack(labels), dtype=torch.float32, device=device)
                    torch.xpu.synchronize()
                    transfer_s += time.perf_counter()-transfer
                    assert not y.requires_grad and y.grad_fn is None
                    train_start = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    loss = (predictor(x)-y).square().mean()
                    loss.backward()
                    optimizer.step()
                    torch.xpu.synchronize()
                    training_s += time.perf_counter()-train_start
                    losses.append(float(loss))
                    features.clear()
                    labels.clear()
            with torch.no_grad():
                for param in generator.parameters():
                    param.add_(.0001)
        elapsed = time.perf_counter()-started
        resources = monitor.finish(elapsed)
        scheduling = pool.stats.copy()
    import psutil
    assert all(not psutil.pid_exists(pid) for pid in worker_pids)
    assert all(np.isfinite(losses)) and any(not torch.equal(a, b) for a, b in zip(initial, predictor.parameters()))
    report = dict(passed=True, host=platform.node(), pbs_job=os.environ.get("PBS_JOBID"),
                  source_sha256=source_hashes(), source=source, xpu=torch.xpu.get_device_name(0),
                  labels=all_results, updates=len(losses), generator_versions=sorted(set(versions)),
                  startup_s=startup, wall_s=elapsed, generation_s=generation_s, transfer_s=transfer_s,
                  training_s=training_s, label_wait_s=wait_s, labels_per_s=all_results/elapsed,
                  first_loss=losses[0], last_loss=losses[-1], detached_labels=True,
                  solver_xpu_initialized=False, clean_shutdown=True, max_cache_bytes=max_cache_bytes,
                  cpu_workers=node_workers, scheduling=scheduling, **resources)
    save(args.output, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


def source_hashes():
    files = list((ROOT/"distillation/solver/sparse_solver").glob("*.py"))+[Path(__file__)]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def pipeline_stress():
    """A lazy 10,000-scenario request with known objectives and injected failures."""
    import numpy as np
    from pypower.api import case14
    from distillation.solver import ACOPF, PyPowerSolver
    from distillation.solver.sparse_solver import SolverPool, Topology, Scenario
    base = case14()
    rng = np.random.default_rng(42)
    templates = []
    for i in range(100):
        ppc = deepcopy(base)
        ppc["branch"] = ppc["branch"][rng.permutation(len(ppc["branch"]))]
        templates.append(Topology(i, ppc))
    objectives = []
    for i in range(5):
        ppc = deepcopy(base)
        ppc["bus"][:, 2:4] *= 1+.001*(i-2)
        reference = PyPowerSolver().solve(ACOPF(ppc))
        assert reference.success
        objectives.append(reference.objective)
    bad_ids = {17, 5088, 9998, 9999}
    def inputs():
        for i in range(10000):
            changes = {"bus": base["bus"].copy()}
            changes["bus"][:, 2:4] *= 1+.001*((i % 5)-2)
            if i == 17:
                changes["bus"][0, 2] = np.nan
            if i == 5088:
                changes["bus"] = np.zeros((3, 2))
            if i == 9998:
                changes["gen"] = base["gen"].copy()
                changes["gen"][:, 7] = 0
            if i == 9999:
                changes["gen"] = base["gen"].copy()
                changes["gen"][:, 8:10] = (.01, 0)
            yield Scenario(templates[i//100], i, changes)
    seen, failed, pids, memory = set(), {}, set(), []
    largest_gap = 0.
    with SolverPool(workers=8, cpu_budget=8, task_size=8, max_pending=8,
                    cache_entries=4, cache_mib=16, local_rank=0, local_world_size=1) as pool:
        pool.solve(Scenario(templates[0], f"warm-{i}") for i in range(64))
        monitor = ResourceMonitor()
        start = time.perf_counter()
        for record in pool.stream(inputs(), request_id="stress", generator_version=42):
            i = record.scenario_id
            assert record.ids == ("stress", i//100, i) and i not in seen
            seen.add(i)
            pids.add(record.worker["pid"])
            assert record.generator_version == 42 and not record.worker["xpu_initialized"]
            assert record.worker["cache_bytes"] <= 16*1024**2 and record.worker["cache_entries"] <= 4
            if i in bad_ids:
                assert not record.success and record.y is None
                failed[i] = record.status
            else:
                assert record.success, (record.ids, record.failure_reason)
                gap = abs(record.solution.objective-objectives[i % 5])/abs(objectives[i % 5])
                largest_gap = max(largest_gap, gap)
                assert gap <= 1e-5
                np.testing.assert_array_equal(record.y["bus"][:, 0], record.solution.theta)
                np.testing.assert_array_equal(record.y["generator"][:, 0], record.solution.pg)
            if len(seen) in (1000, 5000, 10000):
                monitor.poll()
                memory.append(dict(completed=len(seen), peak_rss_gib=monitor.peak_rss/1024**3))
        elapsed = time.perf_counter()-start
        resources = monitor.finish(elapsed)
        stats = pool.stats.copy()
        assert len(seen) == 10000 and set(failed) == bad_ids
        assert stats["peak_pending_tasks"] <= 8 and stats["peak_buffered_results"] <= 64
        assert pool.solve([Scenario(templates[0], "after-stress")])[0].success
    import psutil
    assert all(not psutil.pid_exists(pid) for pid in pids)
    return dict(passed=True, scenarios=10000, topologies=100, successes=9996, failures=failed,
                max_relative_cost_gap=largest_gap, wall_s=elapsed, memory=memory,
                clean_shutdown=True, source_sha256=source_hashes(), scheduling=stats, **resources)


def pipeline_report(directory):
    """Rebuild the concise report from measured JSON artifacts."""
    import numpy as np
    directory = Path(directory)
    campaigns = sorted(directory.glob("compute_*/benchmark.json"))
    complete = [p for p in campaigns if json.loads(p.read_text()).get("passed")]
    if not complete:
        raise ValueError("No completed compute-node benchmark")
    folder = complete[-1].parent
    paths = [folder/name for name in ("benchmark.json", "validation.json", "training.json")]
    bench, validation, training = [json.loads(p.read_text()) for p in paths]
    assert validation["passed"] and training["passed"]
    rows = bench["rows"]
    repeated = [r for r in rows if r["count"] == 10000 and r["reused_request"]]
    best = max(repeated, key=lambda r: r["labels_per_s"])
    near_best = [r for r in repeated if r["labels_per_s"] >= .9*best["labels_per_s"]]
    economical = min(near_best, key=lambda r: (r["workers"], -r["labels_per_s"]))
    def table(headers, records):
        return ["| " + " | ".join(headers) + " |", "|" + "---|"*len(headers)] + [
            "| " + " | ".join(map(str, row)) + " |" for row in records]
    lines = ["# Sparse CPU solver pipeline", "", f"PBS job: `{bench['pbs_job']}` (EpiCalib, two debug nodes).",
             f"Benchmark host: `{bench['host']}`. Numerical/NN host: `{validation['host']}`.", "",
             "The implementation uses persistent CPU workers; XPUs remain available for the neural networks.",
             f"All {sum(r['count'] for r in rows):,} benchmark solves succeeded. "
             f"The {len(validation['rows'])}-sample numerical regression and "
             f"{training['labels']}-label XPU training integration also passed.", "",
             "## Numerical agreement", "",
             "Cold starts, float64, one numerical thread. Candidate batches have up to four compatible scenarios. "
             "Candidate time includes native input conversion, sparse preparation, optimization and final checks. "
             "PYPOWER uses serial reference calls from preconstructed ACOPF objects; dataset decoding is excluded.", ""]
    lines += table(["Buses", "Passed", "Sparse batches (s)", "PYPOWER serial (s)", "Ratio"], [
        [r["case"], f"{r['passed']}/{r['samples']}", f"{r['candidate_total_s']:.3f}",
         f"{r['pypower_serial_s']:.3f}", f"{r['pypower_serial_s']/r['candidate_total_s']:.1f}×"]
        for r in validation["summaries"]])
    balance = max(r["checks"]["balance"] for r in validation["rows"])
    cost = max(r["relative_cost_gap"] for r in validation["rows"])
    lines += ["", f"Maximum independent balance residual: {balance:.3g} p.u.; maximum relative objective "
              f"gap: {cost:.3g}. Agreement with PYPOWER establishes neither convexity nor global optimality.", "",
              "## A 10,000-scenario request", "",
              "This scheduling benchmark uses 100 equipment orderings of case14 and 100 scenarios per ordering. "
              "Five demand factors spanning ±0.2% are repeated. These are permutations of one physical topology, "
              "not 100 distinct physical networks. Every scenario is optimized from a cold start; solutions are never cached.", "",
              "The table uses each immediately repeated request in a persistent pool. It includes snapshot copying, "
              "submission, queueing, sparse preparation, solving and result collection. Initial dataset decoding, "
              "template creation and worker startup are excluded. The first requests and startup costs are retained in JSON.", ""]
    lines += table(["Workers", "Scenarios/task", "Request (s)", "Labels/s", "First 32 labels (s)",
                    "Occupied CPU cores", "Peak summed RSS (GiB)"], [
        [r["workers"], r["task_size"], f"{r['wall_s']:.3f}", f"{r['labels_per_s']:.1f}",
         f"{r['first_minibatch_s']:.3f}", f"{r['occupied_cores']:.1f}", f"{r['peak_rss_gib']:.2f}"]
        for r in repeated])
    lines += ["", f"Highest measured throughput: {best['labels_per_s']:.1f} labels/s with {best['workers']} workers "
              f"and {best['task_size']} scenarios/task. The smallest tested worker allocation reaching 90% of that "
              f"rate was {economical['workers']} workers with {economical['task_size']} scenarios/task "
              f"({economical['labels_per_s']:.1f} labels/s). These are measurements of this workload, not promises "
              "for newly generated networks or concurrent large GNN training.", "",
              "CPU occupancy includes the controller and workers. RSS sums process resident memory and double-counts "
              "shared pages; it is not proportional/unique memory. OS thread counts in JSON are distinct from occupied cores. "
              "Worker caches are bounded by entry count and array-storage bytes; process memory also includes Python/Torch "
              "imports, active tasks and allocators.", "", "## Ten case2000 scenarios", "",
              "Five equipment orderings, two scenarios each, one scenario per task. Pool size can exceed the ten active solves.", ""]
    large = [r for r in rows if r.get("workload") == "case2000"]
    lines += table(["Pool workers", "Workers used", "Request (s)", "Labels/s", "Occupied CPU cores", "RSS (GiB)"], [
        [r["workers"], r["workers_used"], f"{r['wall_s']:.3f}", f"{r['labels_per_s']:.2f}",
         f"{r['occupied_cores']:.1f}", f"{r['peak_rss_gib']:.2f}"] for r in large])
    lines += ["", "## Training integration", "",
              f"A generator MLP produced perturbed case118 demands and a predictor MLP trained with MSE on an "
              f"{training['xpu']}. Across {len(training['generator_versions'])} generator versions, "
              f"{training['cpu_workers']} CPU workers supplied {training['labels']} feasible labels for "
              f"{training['updates']} predictor updates. Labels were detached; no solver worker initialized an XPU. "
              "Every graph–label pair was checked by ID and actual demand values. Workers exited cleanly.", "",
              f"After {training['startup_s']:.3f} s of pool startup/warmup, the run took {training['wall_s']:.3f} s "
              f"({training['labels_per_s']:.1f} labels/s): generator computation {training['generation_s']:.3f} s, "
              f"CPU/XPU transfers {training['transfer_s']:.3f} s, predictor updates {training['training_s']:.3f} s, "
              f"and blocking label acquisition {training['label_wait_s']:.3f} s. The latter includes dispatch/collection work. "
              f"Peak summed CPU RSS was {training['peak_rss_gib']:.2f} GiB. This is an integration smoke test with small MLPs, "
              "not a benchmark of the eight production GNN families.", "", "## Failure handling and limits", "",
              "Tests cover changed edges, impedances, taps, shunts, bounds and equipment order; generated buses/branches; "
              "unequal scenario counts; immutable snapshots; cache misses/eviction; malformed inputs; numerical failure; "
              "iteration/time limits; duplicate IDs; early stream closure; and a killed worker. Expected scenario failures "
              "retain IDs with no training label. Unexpected worker errors propagate. Elapsed budgets are cooperative "
              "between iterations, so factorization may overrun a deadline. Nonconvergence does not prove infeasibility."]
    stress = directory/"stress.json"
    if stress.exists():
        s = json.loads(stress.read_text())
        assert s["passed"]
        paths.append(stress)
        lines += ["", f"The separate {s['scenarios']:,}-scenario failure test returned {s['successes']:,} valid labels "
                  f"and exactly {len(s['failures'])} injected failures. Valid objectives agreed with PYPOWER within "
                  f"relative {s['max_relative_cost_gap']:.3g}. Peak pending tasks/results were "
                  f"{s['scheduling']['peak_pending_tasks']}/{s['scheduling']['peak_buffered_results']}; "
                  "pool reuse and clean shutdown passed."]
    lines += ["", "## Sources and reproduction", "",
              "Run `python tests/validate_sparse_solver.py --report-directory distillation/solver/reports/sparse_pipeline` "
              "to regenerate this report. The numerical/input hashes, input archive members, package versions and "
              "benchmark configuration are retained in JSON. Frozen measurement sources and the PBS script are in "
              f"[{folder.name}/source]({folder.name}/source/).", ""]
    for p in paths:
        rel = p.relative_to(directory)
        lines.append(f"- [{rel}]({rel}): SHA-256 `{hashlib.sha256(p.read_bytes()).hexdigest()}`.")
    lines += ["", "Historical CPU/XPU measurements and pre-refactor sources remain in `historical/` and "
              "`historical_source/`. Historical device timings do not describe this CPU pipeline.", ""]
    if (directory/"renaming.md").exists():
        lines += ["The package and reports now use `sparse_solver` and `sparse_pipeline`. Recorded JSON, logs and "
                  "frozen source snapshots retain their original paths and bytes to preserve provenance. "
                  "See [path changes](renaming.md) for the current equivalents.", ""]
    path = directory/"report.md"
    path.write_text("\n".join(lines))
    print(path)
    return 0


def save(path, report):
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix+".tmp")
        clean = json.loads(json.dumps(report), parse_constant=lambda _: None)
        tmp.write_text(json.dumps(clean, indent=2, allow_nan=False)+"\n")
        tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--linear-solver", choices=("auto", "sparse", "dense"), default="auto")
    parser.add_argument("--cases", type=int, nargs="+", default=[14, 30, 57, 118, 500, 2000])
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--large-samples", type=int, default=3)
    parser.add_argument("--pool", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--max-dense-gib", type=float, default=4.)
    parser.add_argument("--cost-rtol", type=float, default=1e-5)
    parser.add_argument("--feasibility-tol", type=float, default=1e-6)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--pipeline-checks", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--training-smoke", action="store_true")
    parser.add_argument("--compute-campaign", action="store_true")
    parser.add_argument("--pipeline-stress", action="store_true")
    parser.add_argument("--report-directory", type=Path)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--task-sizes", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--requests", type=int, nargs="+", default=[10, 100, 10000])
    parser.add_argument("--smoke", action="store_true", help="Two samples/small case, one/large case")
    args = parser.parse_args()
    if args.smoke:
        args.samples, args.large_samples = 2, 1
    if min(args.samples, args.large_samples, args.pool, args.batch_size, args.threads) < 1:
        parser.error("Sample counts, pool, batch size and threads must be positive")
    if min(args.workers+args.task_sizes+args.requests) < 1:
        parser.error("Worker counts, task sizes and request sizes must be positive")
    if max(args.samples, args.large_samples) > args.pool:
        parser.error("Sample counts cannot exceed the selection pool")
    if not set(args.cases) <= {14, 30, 57, 118, 500, 2000}:
        parser.error("Unsupported case")
    budget = max(1, len(os.sched_getaffinity(0))//2)
    if args.threads > budget:
        parser.error(f"Use at most half the available logical CPUs ({budget})")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    import numpy as np
    import torch
    import scipy
    from threadpoolctl import threadpool_limits
    from distillation.solver import ACOPF, PyPowerSolver
    from distillation.solver.sparse_solver import XPUSolver
    from distillation.solver.sparse_solver.solver import synchronize
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    solver = XPUSolver(device=device, linear_solver=args.linear_solver, max_iter=args.max_iter,
                       max_dense_gib=args.max_dense_gib, feasibility_tol=args.feasibility_tol)
    sources = list((ROOT/"distillation/solver/sparse_solver").glob("*.py")) + [Path(__file__)]
    report = dict(started=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                  python=platform.python_version(), torch=torch.__version__, scipy=scipy.__version__,
                  config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  device_name=torch.xpu.get_device_name(device) if device.type == "xpu" else platform.processor(),
                  rows=[], summaries=[])
    with threadpool_limits(limits=args.threads):
        if args.report_directory:
            return pipeline_report(args.report_directory)
        if args.pipeline_stress:
            result = pipeline_stress()
            save(args.output, result)
            print(json.dumps(result, indent=2))
            return 0
        if args.compute_campaign:
            from mpi4py import MPI
            args.output.mkdir(parents=True, exist_ok=True)
            directory = args.output
            if MPI.COMM_WORLD.Get_rank() == 0:
                args.output = directory/"benchmark.json"
                return benchmark(args)
            args.output = directory/"training.json"
            training_smoke(args)
            args.output = directory/"validation.json"
            args.pipeline_checks = True
        if args.benchmark:
            return benchmark(args)
        if args.training_smoke:
            return training_smoke(args)
        numerical_checks(device)
        report["numerical_checks"] = "passed"
        if args.pipeline_checks:
            report["pipeline_checks"] = pipeline_checks()
        save(args.output, report)
        if args.checks_only:
            report["completed"] = datetime.now(timezone.utc).isoformat()
            report["passed"] = True
            save(args.output, report)
            return 0
        for case in args.cases:
            count = args.samples if case < 500 else args.large_samples
            selected = samples(args.raw_root, case, count, args.pool, args.seed)
            case_rows, batch_seconds, reference_seconds = [], [], []
            print(f"Case {case}: {count} cold-start samples, batch size {args.batch_size}", flush=True)
            for offset in range(0, count, args.batch_size):
                subset = selected[offset:offset+args.batch_size]
                models = [ACOPF(data) for data, _ in subset]
                originals = [deepcopy(m.ppc) for m in models]
                synchronize(device)
                start = time.perf_counter()
                results = solver.solve_batch([data for data, _ in subset])
                synchronize(device)
                elapsed = time.perf_counter()-start
                batch_seconds.append(elapsed)
                for model, result, original, (_, source) in zip(models, results, originals, subset):
                    start = time.perf_counter()
                    reference = PyPowerSolver(tol=args.feasibility_tol).solve(model)
                    ref_s = time.perf_counter()-start
                    reference_seconds.append(ref_s)
                    gap = abs(result.objective-reference.objective)/max(1., abs(reference.objective))
                    unchanged = all(np.array_equal(model.ppc[key], original[key]) for key in ("bus", "gen", "branch", "gencost"))
                    passed = result.success and reference.success and gap <= args.cost_rtol and unchanged
                    row = dict(**source, passed=bool(passed), objective=result.objective,
                               reference_objective=reference.objective, relative_cost_gap=gap,
                               checks=result.checks, reference_checks=reference.checks,
                               reference_converged=reference.converged, reference_s=ref_s,
                               batch_wall_s=elapsed, batch_id=f"{case}:{offset}",
                               inputs_unchanged=unchanged, solver=result.info)
                    report["rows"].append(row)
                    case_rows.append(row)
                    state = "PASS" if passed else "FAIL"
                    print(f"  {source['index']:4d} {state} iterations={result.info['iterations']:2d} "
                          f"cost gap={100*gap:.3g}% balance={result.checks['balance']:.2e}", flush=True)
                    save(args.output, report)
            summary = dict(case=case, samples=count, passed=sum(r["passed"] for r in case_rows),
                           candidate_total_s=sum(batch_seconds), pypower_serial_s=sum(reference_seconds),
                           max_cost_gap_pct=100*max(r["relative_cost_gap"] for r in case_rows),
                           max_balance_pu=max(r["checks"]["balance"] for r in case_rows))
            report["summaries"].append(summary)
            save(args.output, report)
    report["completed"] = datetime.now(timezone.utc).isoformat()
    report["passed"] = bool(report["rows"]) and all(r["passed"] for r in report["rows"])
    save(args.output, report)
    print("\n| Case | Passed | Candidate batches (s) | PYPOWER serial (s) | Max cost gap (%) |")
    print("|---|---:|---:|---:|---:|")
    for s in report["summaries"]:
        print(f"| {s['case']} | {s['passed']}/{s['samples']} | {s['candidate_total_s']:.3f} | "
              f"{s['pypower_serial_s']:.3f} | {s['max_cost_gap_pct']:.3g} |")
    print("Candidate timings include native input conversion, sparse preparation, solve and validation; exclude dataset decoding.")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
