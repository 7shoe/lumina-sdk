"""Differentiable SDK adapters and graph-balanced regression shared by DD/DCGM/HMDC."""
from contextlib import redirect_stdout
from copy import deepcopy
from functools import lru_cache
import hashlib
import io
from pathlib import Path

import torch
from distillation.experiments.staging import local_path
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


def validate_initialization(config):
    mode = config.get('initialization', 'random')
    if mode not in ('random', 'fixed', 'pretrained'):
        raise ValueError('initialization must be random, fixed or pretrained')
    if mode == 'pretrained' and not config.get('pretrained_checkpoint'):
        raise ValueError('pretrained initialization requires pretrained_checkpoint')
    if config.get('pretrained_checkpoint') and mode != 'pretrained':
        raise ValueError('Set initialization=pretrained to use pretrained_checkpoint')


def initialization_seed(config, seed):
    """The construction seed; pretrained identity is additionally pinned by hash."""
    return config.get('model_seed', 1_000_000) if config.get('initialization') in ('fixed', 'pretrained') else seed


@lru_cache(maxsize=8)
def _read_pretrained(path, mtime_ns, size):
    """Cache CPU weights, never the pretraining optimizer or a mutable student."""
    content = local_path(path).read_bytes()
    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    spec = payload.get('specification', {})
    if spec.get('format') not in ('lumina-t1-v1', 'lumina-t2-v1'):
        raise ValueError('Expected a T1 or T2 best.pt, final.pt or last.pt checkpoint')
    if spec.get('split_seed') != 42 or not isinstance(spec.get('model_options'), dict):
        raise ValueError('Checkpoint must record model_options and the benchmark split_seed=42')
    weights = payload.get('model', {})
    if not weights or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Checkpoint has missing or nonfinite model weights/buffers')
    metadata = {'sha256': hashlib.sha256(content).hexdigest(), 'specification': spec,
                'samples': payload.get('samples', payload.get('state', {}).get('samples'))}
    return weights, metadata


def clear_pretrained_cache():
    # Some filesystems give consecutive same-size writes identical mtime/ctime.
    # Re-read source bytes at a run boundary before validating pinned identities.
    _read_pretrained.cache_clear()


def pretrained_config(config, *, refresh=False):
    """Resolve saved architecture and checkpoint identity, preserving target data."""
    validate_initialization(config)
    if config.get('initialization') != 'pretrained':
        return config
    if refresh:
        clear_pretrained_cache()
    path = Path(config['pretrained_checkpoint']).expanduser().resolve()
    stat = path.stat()
    _, metadata = _read_pretrained(str(path), stat.st_mtime_ns, stat.st_size)
    spec = metadata['specification']
    if spec['model'] != config['model']:
        raise ValueError(f'Checkpoint model {spec["model"]} does not match requested {config["model"]}')
    if 'pretrained' in config and config['pretrained'] != metadata:
        raise ValueError('Pretrained checkpoint changed; use the original weights or a new run directory')
    return config | {'pretrained_checkpoint': str(path), 'pretrained': deepcopy(metadata),
                     'model_options': deepcopy(spec['model_options']), 'model_seed': spec['seed']}


def model_for(graph, config, seed, device, *, initial_state=None):
    """Build an independent student; embedded states allow source-free replay."""
    if initial_state is None:
        config = pretrained_config(config)
        if config.get('initialization') == 'pretrained':
            path = Path(config['pretrained_checkpoint'])
            stat = path.stat()
            initial_state, metadata = _read_pretrained(str(path), stat.st_mtime_ns, stat.st_size)
            if metadata != config['pretrained']:
                raise ValueError('Pretrained checkpoint changed while loading')
    seed = initialization_seed(config, seed)
    kind = config['model']
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        torch.manual_seed(seed)
        model = build_model({'model_type': kind, 'models': {kind: config['model_options']}},
                            benchmark_graph(graph), torch.device('cpu'))
    if config.get('model_precision') == 'float64':
        model = PrecisionModel(model, torch.float64)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    return model.to(device).eval()


def model_from_export(artifact, device, seed=None, *, model=None):
    """Initialize DD/DCGM replay, or an explicitly named HMDC pair member."""
    config = artifact['config']
    if 'model_checkpoints' in artifact:
        if model not in artifact['model_checkpoints']:
            raise ValueError('Specify a model with an embedded HMDC checkpoint')
        student = artifact['model_checkpoints'][model]
        config = config | {'model': model, 'model_options': student['model_options'],
                           'model_seed': student['seed']}
    else:
        student = artifact.get('model_checkpoint', {})
    if seed is None:
        seed = config.get('model_seed', 1_000_000)
    elif config.get('initialization') == 'fixed':
        config = config | {'initialization': 'random'}  # Explicit unseen-seed replay.
    weights = student['initial_state'] if config.get('initialization') == 'pretrained' else None
    return model_for(artifact['physical_solved_graphs'][0], config, seed, device, initial_state=weights)


def train_student(model, optimizer, synthetic, *, epochs=1, batch_size=16, seed=42, observe=None):
    """Ordinary SGD on detached inputs; no trajectory contributes to data gradients."""
    rng = torch.Generator().manual_seed(seed)
    device = next(model.parameters()).device
    model.eval()  # No stochastic augmentation/dropout in this controlled comparison.
    # Keep calibrated BatchNorm frozen in this adaptation's student training.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    value, updates, presentations = None, 0, 0
    if observe is not None:
        observe(updates, presentations)
    for _ in range(epochs):
        order = torch.randperm(len(synthetic.anchors), generator=rng).tolist()
        for start in range(0, len(order), batch_size):
            batch, labels = synthetic.batch(order[start:start + batch_size], detached=True)
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = mse(model(batch), labels, batch)
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                               for p in model.parameters()):
                raise FloatingPointError('Nonfinite ordinary student SGD')
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite ordinary student SGD parameters after update')
            value = float(loss.detach())
            updates += 1
            presentations += batch.num_graphs
            if observe is not None:
                observe(updates, presentations)
    return value
