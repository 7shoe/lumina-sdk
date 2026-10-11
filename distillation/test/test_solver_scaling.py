"""Check benchmark failure accounting, label isolation and worker cancellation."""
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from distillation.solver.scaling import RAW_ROOT, WorkerPool, load_network, paired_results


@pytest.fixture
def input_file(tmp_path):
    archive = RAW_ROOT / "pglib_opf_case14_ieee_0.tar.gz"
    if not archive.exists():
        pytest.skip("Local OPFData archive unavailable")
    with tarfile.open(archive, "r|gz") as handle:
        member = next(m for m in handle if m.isfile() and m.name.endswith(".json"))
        obj = json.load(handle.extractfile(member))
    obj["solution"] = {"deliberately": "not a usable solution"}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(obj))
    return path


def test_target_labels_never_enter_network(input_file):
    data, objective = load_network(input_file)
    assert objective > 0
    assert data["bus"].x.shape == (14, 4)
    for store in data.stores:
        assert not set(store.keys()) & {"y", "edge_label", "objective"}


def test_warm_speedups_require_success_and_comparable_objectives():
    rows = []
    for index in range(4):
        for mode in ("cold", "previous"):
            rows.append(dict(solver="scipy", case=14, index=index, phase="paired", mode=mode,
                             success=not (index == 1 and mode == "previous"),
                             objective=110 if index == 2 and mode == "previous" else 100,
                             solve_s=2 if mode == "cold" else 1))
    rows.pop()  # Last pair is incomplete.
    pairs = paired_results(rows)
    assert len(pairs) == 1
    assert pairs[0]["index"] == 0
    assert pairs[0]["speedup"] == 2


def test_timeout_stops_batch_and_terminates_worker(tmp_path, input_file):
    config = dict(thread_counts=[1], solvers=["pypower"], timeout=.001,
                  memory_gib=16, poll_seconds=.005)
    pool = WorkerPool(config, tmp_path, [min(os.sched_getaffinity(0))])
    rows = []
    tasks = [dict(id=str(i), input=str(input_file), solver="pypower", threads=1) for i in range(2)]
    try:
        result = pool.run(tasks, 1, rows.append)
        assert [row["status"] for row in rows] == ["timeout", "skipped"]
        assert result["success_count"] == 0
        assert not pool.slots[0]["process"].is_alive()
    finally:
        pool.close()
