"""Reuse SDK backbones with one explicit output convention for all models."""
from __future__ import annotations

import torch
from torch import nn

from lumina.model.opf.hetero_model import HEAT, HGT, RGAT, OPFHeteroGNN
from lumina.model.opf.homo_model import get_gnnNets
from lumina.utils.graph_utils import convert_opf_to_homo


HOMOGENEOUS_MODELS = {"GCN", "GAT", "GIN", "Transformer"}
SUPPORTED_MODELS = ("GCN", "GAT", "RGAT", "GIN", "HGT", "HEAT", "HeteroGNN", "Transformer")


def apply_bounds(predictions, batch):
    """Match the SDK hetero sigmoid head; leave voltage angles unbounded."""
    bx, gx = batch["bus"].x, batch["generator"].x
    bus, gen = predictions["bus"], predictions["generator"]
    return {
        "bus": torch.stack((bus[:, 0],
            bx[:, 1].float() + bus[:, 1].sigmoid() * (bx[:, 2] - bx[:, 1]).float()), dim=1),
        "generator": torch.stack((
            gx[:, 2].float() + gen[:, 0].sigmoid() * (gx[:, 3] - gx[:, 2]).float(),
            gx[:, 5].float() + gen[:, 1].sigmoid() * (gx[:, 6] - gx[:, 5]).float()), dim=1),
    }


class BenchmarkModel(nn.Module):
    """Heterogeneous batches remain available for physical evaluation.

    Only bus/gen outputs are supervised. Homogeneous conversion uses the SDK's
    64-node-feature/32-edge-feature representation, without supervising dummy
    load/shunt labels. These conventions are recorded in the run protocol.
    """
    def __init__(self, sample, kind, options):
        super().__init__()
        self.kind = kind
        if kind in {"HGT", "RGAT"}:
            cls = {"HGT": HGT, "RGAT": RGAT}[kind]
            self.backbone = cls(
                metadata=sample.metadata(),
                input_channels={k: sample[k].x.shape[1] for k in sample.node_types},
                out_channels=2, **options)
        elif kind in {"HEAT", "HeteroGNN"}:
            # HEAT needs relation feature widths to create its edge projections.
            # HeteroGNN's GAT backend also consumes the real relation features.
            metadata = {
                "nodes": {k: sample[k].x.shape[1] for k in sample.node_types},
                "edges": {k: sample[k].edge_attr.shape[1]
                          if "edge_attr" in sample[k] and sample[k].edge_attr is not None else 0
                          for k in sample.edge_types},
            }
            cls = {"HEAT": HEAT, "HeteroGNN": OPFHeteroGNN}[kind]
            self.backbone = cls(metadata=metadata, input_channels=metadata["nodes"],
                                out_channels=2, **options)
        elif kind in HOMOGENEOUS_MODELS:
            homo = convert_opf_to_homo(sample)
            params = {"model_name": kind, "readout": "mean",
                      "edge_dim": homo.edge_attr.shape[1], **options}
            self.backbone = get_gnnNets(homo.x.shape[1], 2, params)
        else:
            raise ValueError(f"Unsupported experiment model: {kind}")

    def forward(self, batch):
        if self.kind in {"HGT", "RGAT"}:
            raw = self.backbone({k: v.float() for k, v in batch.x_dict.items()},
                                batch.edge_index_dict, minmax_scaling=False)
        elif self.kind in {"HEAT", "HeteroGNN"}:
            raw = self.backbone({k: v.float() for k, v in batch.x_dict.items()},
                                batch.edge_index_dict,
                                edge_attr_dict={k: v.float() for k, v in batch.edge_attr_dict.items()},
                                minmax_scaling=False)
        else:
            homo = convert_opf_to_homo(batch)
            homo.x = homo.x.float()
            homo.edge_attr = homo.edge_attr.float()
            values = self.backbone(homo)
            raw = {"bus": values[homo.node_type == 0],
                   "generator": values[homo.node_type == 1]}
        return apply_bounds(raw, batch)
