"""Regressions for accelerator-independent production trainer defects."""
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data import Data, HeteroData

from lumina.model.opf.losses import OPFLossManager
from lumina.trainer.opf.trainer import BaseOPFTrainer, MultiCaseOPFTrainer, OPFTrainer


@pytest.mark.parametrize("trainer_cls", [OPFTrainer, MultiCaseOPFTrainer])
def test_real_loss_manager_construction(trainer_cls):
    trainer = trainer_cls.__new__(trainer_cls)
    trainer.loss_type = "mse"
    trainer.device = torch.device("cpu")
    trainer.log_normalized_violation = True
    trainer.global_rank = 1
    trainer.case_names = ["case14", "case30"]
    trainer._initialize_loss_managers()
    managers = (list(trainer.loss_managers.values())
                if trainer_cls is MultiCaseOPFTrainer else [trainer.loss_manager])
    assert len(managers) == (2 if trainer_cls is MultiCaseOPFTrainer else 1)
    assert all(isinstance(manager, OPFLossManager) for manager in managers)


class _ForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        # Exercise DDP's unused-parameter bookkeeping as well as reductions.
        self.unused = torch.nn.Parameter(torch.ones(1))

    def forward(self, inputs, edge_index_dict=None, minmax_scaling=None):
        if isinstance(inputs, dict):
            return {name: self.linear(value) for name, value in inputs.items()}
        return self.linear(inputs.x)


def _distributed_forward(rank, rendezvous, heterogeneous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(12)
        trainer = BaseOPFTrainer.__new__(BaseOPFTrainer)
        trainer.device = torch.device("cpu")
        trainer.model_type = "RGAT" if heterogeneous else "GCN"
        trainer.minmax_scaling = True
        trainer.model = DistributedDataParallel(_ForwardModel(), find_unused_parameters=True)
        optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)
        initial = trainer.model.module.linear.weight.detach().clone()
        for step in range(3):
            # Different batches ensure bypassing DDP cannot pass by identical gradients.
            x = torch.tensor([[1., 2.], [3., 4.]]) + rank * 2 + step
            if heterogeneous:
                batch = HeteroData()
                batch["bus"].x = x
                batch["bus", "line", "bus"].edge_index = torch.tensor([[0], [1]])
            else:
                batch = Data(x=x, node_type=torch.zeros(2, dtype=torch.long))
            optimizer.zero_grad(set_to_none=True)
            trainer.forward(batch)["bus"].square().mean().backward()
            assert trainer.model.module.unused.grad is None
            grads = torch.cat([p.grad.flatten() for p in trainer.model.parameters()
                               if p.grad is not None])
            assert torch.isfinite(grads).all()
            peers = [torch.empty_like(grads) for _ in range(2)]
            dist.all_gather(peers, grads)
            torch.testing.assert_close(peers[0], peers[1], atol=1e-7, rtol=1e-6)
            optimizer.step()
        params = torch.cat([p.detach().flatten() for p in trainer.model.parameters()])
        peers = [torch.empty_like(params) for _ in range(2)]
        dist.all_gather(peers, params)
        torch.testing.assert_close(peers[0], peers[1], atol=1e-7, rtol=1e-6)
        assert not torch.equal(initial, trainer.model.module.linear.weight)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("heterogeneous", [False, True])
def test_production_forward_synchronizes_real_ddp(tmp_path, heterogeneous):
    mp.spawn(_distributed_forward,
             args=((tmp_path / "rendezvous").as_uri(), heterogeneous),
             nprocs=2, join=True)
