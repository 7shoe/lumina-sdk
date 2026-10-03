import pytest
import torch

from lumina.trainer.opf import utils


@pytest.mark.parametrize("cuda,xpu,count,local,kind,index,backend", [
    (True, True, 4, 2, "cuda", 2, "nccl"),
    (True, False, 1, 7, "cuda", 0, "nccl"),
    (False, True, 12, 7, "xpu", 7, "xccl"),
    (False, True, 1, 7, "xpu", 0, "xccl"),
    (False, False, 0, 0, "cpu", None, "gloo"),
])
def test_selection_and_binding_order(
    monkeypatch, cuda, xpu, count, local, kind, index, backend
):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)
    for name in ["cuda", "xpu"]:
        api = getattr(torch, name)
        monkeypatch.setattr(api, "device_count", lambda: count)
        monkeypatch.setattr(api, "set_device",
                            lambda value, name=name: calls.append(("bind", name, value)))
    monkeypatch.setattr(utils.dist, "is_xccl_available", lambda: True)
    monkeypatch.setattr(
        utils.dist, "init_process_group",
        lambda **kwargs: calls.append(("init", kwargs["backend"])),
    )
    for name, value in [("RANK", local), ("LOCAL_RANK", local), ("WORLD_SIZE", 8)]:
        monkeypatch.setenv(name, str(value))
    result = utils.init_distributed_runtime(local, local, 8)
    assert result == (local, local, 8, index)
    expected = [] if kind == "cpu" else [("bind", kind, index)]
    assert calls == expected + [("init", backend)]


def test_xpu_without_xccl_is_a_clear_error(monkeypatch):
    for name, value in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]:
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 1)
    monkeypatch.setattr(torch.xpu, "set_device", lambda index: None)
    monkeypatch.setattr(utils.dist, "is_xccl_available", lambda: False)
    with pytest.raises(RuntimeError, match="native XCCL"):
        utils.init_distributed_runtime(0, 0, 1)


def test_explicit_cpu_does_not_bind_accelerator(monkeypatch):
    def unexpected(index):
        pytest.fail("CPU override must not bind an accelerator")
    monkeypatch.setattr(torch.cuda, "set_device", unexpected)
    monkeypatch.setattr(torch.xpu, "set_device", unexpected)
    assert utils.select_device(device="cpu") == torch.device("cpu")


def test_xpu_rank_outside_visible_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="visible XPU device count"):
        utils.select_device(local_rank=2)



@pytest.mark.parametrize("kind", ["cuda", "xpu"])
def test_explicit_index_overrides_local_rank(monkeypatch, kind):
    calls = []
    monkeypatch.setattr(getattr(torch, kind), "set_device", calls.append)
    assert utils.select_device(local_rank=7, device=f"{kind}:2") == torch.device(kind, 2)
    assert calls == [2]


def test_explicit_backend_override(monkeypatch):
    for name, value in [("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)]:
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    calls = []
    monkeypatch.setattr(utils.dist, "init_process_group", lambda **kwargs: calls.append(kwargs))
    # A mock group verifies the explicit backend is forwarded even if it differs
    # from the automatic CPU default; this is not an actual CPU/NCCL launch.
    assert utils.init_distributed_runtime(0, 0, 1, backend="nccl") == (0, 0, 1, None)
    assert calls == [dict(backend="nccl", init_method="env://", world_size=1, rank=0)]


def test_legacy_imports_are_shared_helpers():
    from lumina.utils import model
    assert utils.select_cuda_device_index is model.select_cuda_device_index
    assert utils.select_device is model.select_device
    for count in [0, -1, None, "invalid", "1"]:
        assert model.select_cuda_device_index(7, count) == 0


def test_lightweight_selector_does_not_import_trainers():
    import subprocess
    import sys
    code = (
        "import sys; sys.modules['lumina.trainer'] = None; "
        "sys.modules['lumina.dataset'] = None; "
        "from lumina.utils.model import select_device; "
        "assert select_device(device='cpu').type == 'cpu'"
    )
    subprocess.run([sys.executable, "-B", "-c", code], check=True, timeout=45)


def load_example(monkeypatch, name):
    import importlib.util
    from pathlib import Path
    examples = Path(__file__).resolve().parents[2] / "example" / "opf"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("xpu_test_" + name, examples / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("local_variable", [
    "MPI_LOCALRANKID", "SLURM_LOCALID", "LOCAL_RANK", "PALS_LOCAL_RANKID",
])
def test_mpi_launch_rank_discovery(monkeypatch, local_variable):
    import sys
    from types import SimpleNamespace
    for name in ["RANK", "WORLD_SIZE", "MPI_LOCALRANKID", "SLURM_LOCALID",
                 "LOCAL_RANK", "PALS_LOCAL_RANKID"]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(local_variable, "3")
    training = load_example(monkeypatch, "train_opf_ddp")
    monkeypatch.setattr(training, "MPI", SimpleNamespace(COMM_WORLD=SimpleNamespace(
        Get_rank=lambda: 7, Get_size=lambda: 8,
    )))
    calls = []
    def runtime(local_rank, global_rank, world_size):
        calls.append((local_rank, global_rank, world_size))
        return local_rank, global_rank, world_size, 3
    monkeypatch.setattr(training, "init_distributed_runtime", runtime)
    monkeypatch.setitem(sys.modules, "train_opf_ddp", training)
    assert training.init_ddp() == (3, 7, 8)
    # LOCAL_RANK alone is a local hint, not a complete torchrun environment.
    evaluation = load_example(monkeypatch, "test_opf_ddp")
    assert evaluation.init_ddp() == (3, 7, 8)
    assert calls == [(3, 7, 8), (3, 7, 8)]


def test_evaluation_torchrun_contract(monkeypatch):
    for name, value in [("RANK", "5"), ("WORLD_SIZE", "8"), ("LOCAL_RANK", "1")]:
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(utils, "init_distributed_runtime",
                        lambda local, rank, size: (local, rank, size, 1))
    evaluation = load_example(monkeypatch, "test_opf_ddp")
    assert evaluation.init_ddp() == (1, 5, 8)
    monkeypatch.delenv("WORLD_SIZE")
    with pytest.raises(KeyError, match="WORLD_SIZE"):
        evaluation.init_ddp()
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("RANK")
    with pytest.raises(KeyError, match="RANK"):
        evaluation.init_ddp()


def test_mpi_local_rank_precedence(monkeypatch):
    from types import SimpleNamespace
    training = load_example(monkeypatch, "train_opf_ddp")
    monkeypatch.setattr(training, "MPI", SimpleNamespace(COMM_WORLD=SimpleNamespace(
        Get_rank=lambda: 7, Get_size=lambda: 8,
    )))
    monkeypatch.setattr(training, "init_distributed_runtime",
                        lambda **kw: (kw["local_rank"], kw["global_rank"], kw["world_size"], 0))
    names = ["MPI_LOCALRANKID", "SLURM_LOCALID", "LOCAL_RANK", "PALS_LOCAL_RANKID"]
    for index, name in enumerate(names):
        monkeypatch.setenv(name, str(index + 1))
    for index, name in enumerate(names):
        assert training.init_ddp() == (index + 1, 7, 8)
        monkeypatch.delenv(name)


def test_evaluation_env_uses_pals_local_rank(monkeypatch):
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("PALS_LOCAL_RANKID", "2")
    monkeypatch.setattr(utils, "init_distributed_runtime",
                        lambda local, rank, size: (local, rank, size, 2))
    assert load_example(monkeypatch, "test_opf_ddp").init_ddp() == (2, 5, 8)
