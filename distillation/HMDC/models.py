"""Model-pair construction and complete local training state for exact resume."""
from dataclasses import dataclass

import torch

from distillation.synthetic.learning import (benchmark_batch,
                                             model_for,
                                             model_options)
from .features import SpatialSemanticAlignment, forward_features
from .hmdc import GradientBalance


def model_config(config, kind):
    return dict(model=kind, model_options=config.get('models', {}).get(kind, model_options(kind)),
                model_precision='float64', strength=config['strength'],
                batch_size=config['batch_size'], eval_batch=config['eval_batch'],
                evaluation_epochs=config['evaluation_epochs'],
                student_rates=config['student_rates'],
                calibration_panels=config.get('calibration_panels', 2),
                student_rate_factor=config.get('student_rate_factor', .1))


def cpu_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_tree(v) for v in value)
    return value


@dataclass
class Pair:
    """Two models with their alignment, gradient balancing, and optimizer state"""
    models: list
    alignment: SpatialSemanticAlignment
    balance: GradientBalance
    optimizers: list
    alignment_optimizer: torch.optim.Optimizer
    query_ids: list

    def state_dict(self):
        """Collect model and training state for checkpointing"""
        return dict(models=[cpu_state(m) for m in self.models],
                    alignment=cpu_state(self.alignment), balance=self.balance.state_dict(),
                    optimizers=[cpu_tree(o.state_dict()) for o in self.optimizers],
                    alignment_optimizer=cpu_tree(self.alignment_optimizer.state_dict()), query_ids=self.query_ids)

    def load_state_dict(self, state):
        """Restore a checkpoint into an already constructed, matching pair"""
        # model comps
        for model, values in zip(self.models, state['models']):
            model.load_state_dict(values)
        
        # alignment
        self.alignment.load_state_dict(state['alignment'])
        self.balance.load_state_dict(state['balance'])
        
        # optimizer (state)
        for optimizer, values in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(values)
        self.alignment_optimizer.load_state_dict(state['alignment_optimizer'])
        self.query_ids = list(state['query_ids'])


def build_pair(graph, config, rates, seed, device):
    """Build a seeded model pair, feature alignment, and separate optimizers"""
    names = config['pair']
    if len(names) != 2 or len(set(names)) != 2:
        raise ValueError('Choose two different Lumina benchmark architectures')
    
    # hand each architecture a distinct reproducible init seed
    models = [model_for(graph, model_config(config, kind), seed + index * 100_000, device)
              for index, kind in enumerate(names)]
    
    # infer feature widths from a sample graph to size the alignment layers
    with torch.no_grad():
        batch = benchmark_batch([graph]).to(device)
        widths = [[f.shape[1] for f in forward_features(m, batch)[1]] for m in models]
        
    # seed alignment initialization while restoring the CPU RNG state on exit
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 200_000)
        alignment = SpatialSemanticAlignment(*widths).double().to(device)
    alignment_optimizer = torch.optim.Adam(alignment.parameters(), lr=config.get('alignment_rate', .01))
    
    # mutual learning adds a third objective to the grad balancer
    return Pair(models, alignment,
                GradientBalance(3 if config.get('mutual', True) else 2, config.get('balance_interval', 10)),
                [torch.optim.SGD(m.parameters(), lr=rates[name]) for m, name in zip(models, names)],
                alignment_optimizer, [])

