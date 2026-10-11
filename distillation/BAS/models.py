"""Existing Lumina models and an explicit load-only, fixed-size reference MLP."""
from contextlib import redirect_stdout
import io

import torch
from torch import nn

from experiments.lumina_bench.models import SUPPORTED_MODELS, HOMOGENEOUS_MODELS
from experiments.lumina_bench.runner import build_model, seed_all


class LoadMLP(nn.Module):
    """Three sigmoid hidden layers by default: linear physical outputs

    Input: flattened P/Q loads
    Targets: bus [angle, magnitude] followed by generator [P,Q]
    
    Reporducibility note:
        This fixes ambiguities in the paper's output encoding
        NOT a claim to recover the authors' unpublished architecture
    """
    def __init__(self,
                 sample,
                 layers=3,
                 dropout=.1,
                 width=None):
        """
        Feedforward AC-OPF surrogate
        """
        super().__init__()
        
        # 
        self.counts = {k: sample[k].num_nodes for k in ('load', 'bus', 'generator')}
        outputs = 2 * (self.counts['bus'] + self.counts['generator'])
        width = outputs if width is None else width
        modules, inputs = [], 2 * self.counts['load']
        for _ in range(layers):
            modules.extend((nn.Linear(inputs, width), nn.Sigmoid(), nn.Dropout(dropout)))
            inputs = width
        
        # model
        self.hidden = nn.Sequential(*modules)
        self.head = nn.Linear(width, outputs)

    def forward(self, batch):
        for kind, count in self.counts.items():
            if not torch.all(batch[kind].ptr.diff() == count):
                raise ValueError('LoadMLP requires fixed bus/generator/load counts')
        x = batch['load'].x.float().reshape(batch.num_graphs, -1)
        outputs = self.head(self.hidden(x))
        boundary = 2 * self.counts['bus']
        return dict(bus=outputs[:, :boundary].reshape(-1, 2),
                    generator=outputs[:, boundary:].reshape(-1, 2))


def model_options(kind, width=64, layers=3, dropout=.1):
    """
    Placeholders for the 8 lumina-bench models
    """
    
    if kind == 'MLP':
        return dict(layers=layers, dropout=dropout)  # Width follows output dimension.
    if kind in HOMOGENEOUS_MODELS:
        return dict(hidden_dim=width, num_layers=layers, dropout=dropout)
    if kind == 'HGT':
        return dict(hidden_channels=width, num_layers=layers, num_heads=4, dropout=dropout)
    if kind == 'RGAT':
        return dict(hidden_channels=width, num_layers=layers, num_heads=4)
    if kind == 'HEAT':
        return dict(hidden_channels=width, num_layers=layers, attention_heads=4)
    if kind == 'HeteroGNN':
        return dict(hidden_channels=width, num_layers=layers, backend='gat')
    raise ValueError(f'Unknown model: {kind}')


def make_model(config, sample, device):
    """
    Seed init and build the confiured model on device
    """
    
    seed_all(config['model_seed'])
    kind = config['model']
    options = config['model_options']
    with redirect_stdout(io.StringIO()):
        if kind == 'MLP':
            model = LoadMLP(sample, **options).to(device)
        else:
            model = build_model(dict(model_type=kind, models={kind: options}), sample, device)
    return model


def output_weights(model):
    """Weights W^t only, excluding biases; concatenate bus/gen head gradients."""
    
    if isinstance(model, LoadMLP):
        return [model.head.weight]
    backbone = model.backbone
    if hasattr(backbone, 'out_dict'):
        return [backbone.out_dict[k].weight for k in ('bus', 'generator')]
    linears = [module for module in backbone.mlps.modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise ValueError('Cannot identify the supervised final-layer weights')
    return [linears[-1].weight]


def has_dropout(model):
    """
    Detect if dropout setting in model architecture
    (functional dropout with a fixed p=0.1)
    """
    
    for module in model.modules():
        if isinstance(module, nn.Dropout) and module.p > 0:
            return True
        rate = getattr(module, 'dropout', 0.)
        if isinstance(rate, (int, float)) and rate > 0:
            return True
    
    
    return getattr(model, 'kind', '') == 'HeteroGNN'
