"""Differentiable SDK adapters and graph-balanced regression shared by DD/DCGM/HMDC."""
from contextlib import redirect_stdout
import io

import torch
from torch_geometric.data import Batch, HeteroData

from distillation.conversion.matrix_form import (E_LINE,
                                                 E_XFMR,
                                                 LINK_TYPES)

from experiments.lumina_bench.runner import build_model
from experiments.lumina_bench.models import (HOMOGENEOUS_MODELS,
                                             SUPPORTED_MODELS)

from lumina.utils.graph_utils import convert_opf_to_homo


class PrecisionModel(torch.nn.Module):
    """The same initialized SDK backbone/head, with an explicit floating dtype.

    BenchmarkModel deliberately forces FP32. Mixed matching derivatives sometimes
    require FP64; this adapter changes precision, not the architecture or head.
    State dictionary names remain compatible with BenchmarkModel.
    """
    def __init__(self, model, dtype):
        super().__init__()
        self.kind, self.backbone = model.kind, model.backbone.to(dtype=dtype)

    def forward(self, batch):
        dtype = next(self.parameters()).dtype
        if self.kind in ('HGT', 'RGAT'):
            raw = self.backbone({k: v.to(dtype) for k,v in batch.x_dict.items()},
                                batch.edge_index_dict, minmax_scaling=False)
        elif self.kind in ('HEAT', 'HeteroGNN'):
            raw = self.backbone({k: v.to(dtype) for k,v in batch.x_dict.items()},
                                batch.edge_index_dict,
                                edge_attr_dict={k: v.to(dtype) for k,v in batch.edge_attr_dict.items()},
                                minmax_scaling=False)
        elif self.kind in HOMOGENEOUS_MODELS:
            homogeneous = convert_opf_to_homo(batch)
            homogeneous.x = homogeneous.x.to(dtype)
            homogeneous.edge_attr = homogeneous.edge_attr.to(dtype)
            values = self.backbone(homogeneous)
            raw = dict(bus=values[homogeneous.node_type == 0], generator=values[homogeneous.node_type == 1])
        else:
            raise ValueError(f'Unsupported precision adapter: {self.kind}')
        bx, gx = batch['bus'].x, batch['generator'].x
        bus, gen = raw['bus'], raw['generator']
        return dict(bus=torch.stack((bus[:,0], bx[:,1].to(dtype) + bus[:,1].sigmoid() *
                                      (bx[:,2]-bx[:,1]).to(dtype)), dim=1),
                    generator=torch.stack((gx[:,2].to(dtype) + gen[:,0].sigmoid() *
                                            (gx[:,3]-gx[:,2]).to(dtype),
                                            gx[:,5].to(dtype) + gen[:,1].sigmoid() *
                                            (gx[:,6]-gx[:,5]).to(dtype)), dim=1))


def benchmark_graph(data, *, labels=False):
    """Match the SDK processor using torch operations, preserving input gradients.

    Native bus [kV, type, vmin, vmax] becomes [kV, vmin, vmax, one-hot(type,4)].
    Powers/costs are already per unit. Equipment reverse links and store ordering
    match build_heterodata_from_grid exactly. No labels enter inference batches.
    """
    graph = HeteroData()
    graph.baseMVA = float(data.x.detach().reshape(-1)[0])
    for kind, width in (("bus", 4), ("generator", 11), ("load", 2), ("shunt", 2)):
        x = data[kind].x.reshape(-1, width)
        if kind == "bus":
            bus_type = x[:, 1].long()
            if not torch.equal(x[:, 1], bus_type.to(x.dtype)) or not ((bus_type >= 1) & (bus_type <= 4)).all():
                raise ValueError("Invalid native bus type")
            onehot = torch.nn.functional.one_hot(bus_type - 1, 4).to(x.dtype)
            x = torch.cat((x[:, [0, 2, 3]], onehot), dim=1)
        graph[kind].x = x.clone()
        if labels and kind in ("bus", "generator"):
            graph[kind].y = data[kind].y.clone()
    for edge, width in ((E_LINE, 9), (E_XFMR, 11)):
        graph[edge].edge_index = data[edge].edge_index.clone()
        graph[edge].edge_attr = data[edge].edge_attr.reshape(-1, width).clone()
    for edge in LINK_TYPES:
        graph[edge].edge_index = data[edge].edge_index.clone()
        graph[edge[::-1]].edge_index = data[edge].edge_index.flip(0).clone()
    if labels:
        graph.objective = data.objective.clone()
    return graph


def benchmark_batch(graphs):
    return Batch.from_data_list([benchmark_graph(graph) for graph in graphs]).cpu()


def targets(graphs):
    """Keep supervision separate from graph features and from OPF solution metadata."""
    return {kind: torch.cat([graph[kind].y.detach().clone() for graph in graphs])
            for kind in ("bus", "generator")}


def mse(prediction, target, batch=None):
    """Equal weight per graph and target type, including graphs of different sizes."""
    losses = []
    for kind in ("bus", "generator"):
        squared = (prediction[kind] - target[kind].to(prediction[kind])).square().mean(dim=1)
        if batch is None:
            losses.append(squared.mean())
        else:
            ids = batch[kind].batch
            totals = squared.new_zeros(batch.num_graphs).index_add(0, ids, squared)
            counts = torch.bincount(ids, minlength=batch.num_graphs).clamp_min(1)
            losses.append((totals / counts).mean())
    return sum(losses) / 2


def model_options(kind, width=32, layers=2):
    if kind not in SUPPORTED_MODELS:
        raise ValueError(f'Unknown benchmark model: {kind}')
    if kind == 'HeteroGNN':
        return dict(hidden_channels=width, num_layers=layers, backend='gat')
    if kind in ('HGT', 'RGAT'):
        return dict(hidden_channels=width, num_layers=layers, num_heads=4)
    if kind == 'HEAT':
        return dict(hidden_channels=width, num_layers=layers, attention_heads=4)
    return dict(hidden_dim=width, num_layers=layers, dropout=0.0)


def model_for(graph, config, seed, device):
    kind = config['model']
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        torch.manual_seed(seed)
        model = build_model({'model_type': kind, 'models': {kind: config['model_options']}},
                            benchmark_graph(graph), torch.device('cpu'))
    if config.get('model_precision') == 'float64':
        model = PrecisionModel(model, torch.float64)
    return model.to(device).eval()


