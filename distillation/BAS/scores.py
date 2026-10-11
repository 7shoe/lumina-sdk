"""Per-example BAS metrics; only panel labels enter labeled acquisition scores."""
from contextlib import contextmanager

import torch
from torch import nn

from distillation.selection import input_view
from distillation.training.T1_single_topo.state import capture_rng, restore_rng
from .models import has_dropout, output_weights


def per_graph_loss(predictions, batch, loss='l1'):
    """L1 vector norm per graph, averaged over graphs by the training caller.

    MSE is an explicit alternate experiment: mean over all supervised scalars.
    No equal bus/generator-family weighting or label normalization is implicit.
    """
    total = predictions['bus'].new_zeros(batch.num_graphs)
    counts = total.clone()
    for kind in ('bus', 'generator'):
        delta = predictions[kind] - batch[kind].y.to(predictions[kind])
        values = delta.abs() if loss == 'l1' else delta.square()
        total = total.index_add(0, batch[kind].batch, values.sum(1))
        counts = counts.index_add(0, batch[kind].batch, torch.full_like(values[:, 0], values.shape[1]))
    if loss not in ('l1', 'mse'):
        raise ValueError('Supported losses: l1, mse')
    return total if loss == 'l1' else total / counts


@contextmanager
def scoring_mode(model, stochastic, seed):
    """Functional dropout works, batch-normalization stats and training RNG stay fixed."""
    device = next(model.parameters()).device
    states = [(m, m.training) for m in model.modules()]
    rng = capture_rng(device)
    try:
        if stochastic and not has_dropout(model):
            raise ValueError('Monte Carlo acquisition requires nonzero model dropout')
        torch.manual_seed(seed)
        model.train(stochastic)
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        yield
    finally:
        for module, training in states:
            module.training = training
        restore_rng(rng, device)


def acquisition_scores(model, batch, metric='ig', *, loss='l1', draws=25, seed=42):
    """Return one score per graph; no model update and no second derivatives.

    IG: L2 norm over the example's P/Q load gradients. LG: Frobenius norm over
    supervised output weight gradients for each example, not an averaged loss.
    MCV-P: sum of coordinate population variances. MCV-L: population variance
    of the per-example loss. These norm/reduction choices are recorded assumptions.
    """
    if metric not in ('le', 'ig', 'lg', 'mcv_p', 'mcv_l'):
        raise ValueError(f'Unknown acquisition metric: {metric}')
    stochastic = metric.startswith('mcv')
    if stochastic and draws < 2:
        raise ValueError('At least two Monte Carlo draws required')
    inputs = input_view(batch)
    with scoring_mode(model, stochastic, seed):
        if stochastic:
            with torch.no_grad():
                predictions = [model(inputs) for _ in range(draws)]
                if metric == 'mcv_l':
                    scores = torch.stack([per_graph_loss(p, batch, loss) for p in predictions]).var(0, unbiased=False)
                else:
                    scores = predictions[0]['bus'].new_zeros(batch.num_graphs)
                    for kind in ('bus', 'generator'):
                        variances = torch.stack([p[kind] for p in predictions]).var(0, unbiased=False)
                        scores = scores.index_add(0, inputs[kind].batch, variances.sum(1))
        elif metric == 'le':
            with torch.no_grad():
                scores = per_graph_loss(model(inputs), batch, loss)
        else:
            with torch.enable_grad():
                if metric == 'ig':
                    inputs['load'].x = inputs['load'].x.detach().requires_grad_(True)
                losses = per_graph_loss(model(inputs), batch, loss)
                if metric == 'ig':
                    # In eval mode these backbones do not couple different graphs.
                    gradient, = torch.autograd.grad(losses.sum(), inputs['load'].x)
                    scores = gradient.new_zeros(batch.num_graphs).index_add(
                        0, inputs['load'].batch, gradient.square().sum(1)).sqrt()
                else:
                    weights, scores = output_weights(model), []
                    for i in range(batch.num_graphs):
                        gradients = torch.autograd.grad(losses[i], weights, retain_graph=i + 1 < batch.num_graphs)
                        scores.append(sum(g.double().square().sum() for g in gradients).sqrt())
                    scores = torch.stack(scores)
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise FloatingPointError('Nonfinite or negative acquisition score')
    return scores.detach().cpu().double().numpy()
