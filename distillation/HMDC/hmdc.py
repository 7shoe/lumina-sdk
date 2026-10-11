"""HMDC equations 3–13, adapted to regression and feasible graph inputs.

Reference: Moon, Kim & Park, ECCV 2024; official code commit
38c5de3cbf2c7df596ffee75298fd0ee491c8195. 

This is independent implementation:
summed squared parameter-gradient matching, GBM, SSD and mutual real-data SGD.
"""
import torch

from distillation.synthetic.learning import mse
from .features import forward_features, node_graph_ids


def gradient_distance(synthetic, real, parameters):
    """Official implementation's squared L2 SUM, including biases/scalars.

    Missing gradients represent zero for graph relations unused on one side.
    Parameters unused on both sides contribute zero. Real gradients are detached.
    """
    terms = []
    for a, b, p in zip(synthetic, real, parameters):
        if a is None and b is None:
            continue
        a = torch.zeros_like(p) if a is None else a
        b = torch.zeros_like(p) if b is None else b.detach()
        terms.append((a - b).square().sum())
    if not terms:
        raise ValueError('No participating model parameters')
    return sum(terms)


def matching_terms(models, alignment, synthetic, real, *, mutual=True):
    """Two supervised matching terms and matching of MD gradients in BOTH models."""
    if len(models) != 2:
        raise ValueError('HMDC requires exactly two models')
    device = next(models[0].parameters()).device
    real = real.clone().detach().to(device)
    batch, labels = synthetic.batch()
    batch = batch.to(device)
    parameters = [tuple(p for p in m.parameters() if p.requires_grad) for m in models]
    real_features, syn_features, terms, diagnostics = [], [], [], {}
    for i, (model, params) in enumerate(zip(models, parameters)):
        model.eval()
        pred_r, features_r = forward_features(model, real)
        pred_s, features_s = forward_features(model, batch)
        loss_r = mse(pred_r, {k: real[k].y for k in ('bus', 'generator')}, real)
        loss_s = mse(pred_s, labels, batch)
        grad_r = torch.autograd.grad(loss_r, params, retain_graph=True, allow_unused=True)
        grad_s = torch.autograd.grad(loss_s, params, retain_graph=True, create_graph=True, allow_unused=True)
        terms.append(gradient_distance(grad_s, grad_r, params))
        real_features.append(features_r); syn_features.append(features_s)
        diagnostics[f'model_{i}'] = dict(real_mse=float(loss_r.detach()), synthetic_mse=float(loss_s.detach()))
    if mutual:
        real_md = alignment(*real_features, node_graph_ids(real), real.num_graphs)
        syn_md = alignment(*syn_features, node_graph_ids(batch), batch.num_graphs)
        params = parameters[0] + parameters[1]
        grad_r = torch.autograd.grad(real_md, params, retain_graph=True, allow_unused=True)
        grad_s = torch.autograd.grad(syn_md, params, retain_graph=True, create_graph=True, allow_unused=True)
        terms.append(gradient_distance(grad_s, grad_r, params))
        diagnostics['mutual'] = dict(real=float(real_md.detach()), synthetic=float(syn_md.detach()))
    return terms, diagnostics


class GradientBalance:
    """Accumulate observed max-absolute data gradients; detached min(A)/A weights."""
    def __init__(self, terms=3, interval=10, epsilon=1e-6):
        if terms not in (2, 3) or interval < 1 or epsilon <= 0:
            raise ValueError('Invalid GBM configuration')
        self.interval, self.epsilon = interval, epsilon
        self.accumulator = torch.full((terms,), epsilon, dtype=torch.float64)
        self.steps = 0

    def observe(self, gradients):
        maxima = [max((float(g.detach().abs().max()) for g in group if g is not None and g.numel()),
                      default=0.) for group in gradients]
        values = torch.tensor(maxima, dtype=torch.float64)
        if values.shape != self.accumulator.shape or not torch.isfinite(values).all():
            raise FloatingPointError('Invalid GBM observations')
        self.accumulator += values

    def weights(self):
        return self.accumulator.min() / self.accumulator

    def state_dict(self):
        return dict(accumulator=self.accumulator.clone(), steps=self.steps,
                    interval=self.interval, epsilon=self.epsilon)

    def load_state_dict(self, state):
        if state['interval'] != self.interval or state['epsilon'] != self.epsilon:
            raise ValueError('GBM configuration differs on resume')
        self.accumulator.copy_(state['accumulator']); self.steps = state['steps']


def normalize_data_gradients(synthetic, epsilon=1e-6):
    """Reference image/channel max normalization, per graph/physical family here."""
    for edit in synthetic.edits:
        for parameter in edit.parameters():
            if parameter.grad is not None and parameter.numel():
                parameter.grad.div_(parameter.grad.abs().max() + epsilon)


def data_gradients(models, alignment, balance, synthetic, real, *, mutual=True, balanced=True):
    """Only synthetic gradients are populated; model/alignment weights stay fixed."""
    terms, diagnostics = matching_terms(models, alignment, synthetic, real, mutual=mutual)
    parameters = tuple(synthetic.parameters())
    if balanced and balance.steps % balance.interval == 0:
        balance.observe([torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                         for term in terms])
    weights = balance.weights() if balanced else torch.ones(len(terms), dtype=torch.float64)
    loss = sum(term * weight.to(term) for term, weight in zip(terms, weights))
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for p, g in zip(parameters, gradients):
        p.grad = torch.zeros_like(p) if g is None else g.detach()
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('Nonfinite HMDC objective or data gradient')
    if balanced:
        normalize_data_gradients(synthetic, balance.epsilon)
    balance.steps += 1
    return dict(loss=float(loss.detach()), terms=[float(v.detach()) for v in terms],
                weights=weights.tolist(), accumulator=balance.accumulator.tolist(), **diagnostics)


def model_step(models, alignment, optimizers, real, *, mutual=True):
    """Ordinary SGD on REAL examples: each task loss plus mutual feature MSE."""
    device = next(models[0].parameters()).device
    batch = real.clone().detach().to(device)
    features, losses = [], []
    for model in models:
        model.eval()
        predictions, hidden = forward_features(model, batch)
        features.append(hidden)
        losses.append(mse(predictions, {k: batch[k].y for k in ('bus', 'generator')}, batch))
    md = alignment(*features, node_graph_ids(batch), batch.num_graphs) if mutual else losses[0].new_zeros(())
    # The MD term appears once in the joint objective: each model gets its own
    # partial derivative, exactly L_task_i + L_MD (not a doubled MD gradient).
    loss = sum(losses) + md
    params = tuple(p for model in models for p in model.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(loss, params, allow_unused=True)
    if not torch.isfinite(loss) or any(g is not None and not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Nonfinite HMDC real-data model step')
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, gradients):
        p.grad = None if g is None else g.detach()
    for optimizer in optimizers:
        optimizer.step()
    return dict(supervised=[float(v.detach()) for v in losses], mutual=float(md.detach()))


def alignment_step(models, alignment, optimizer, real):
    """Fit SSD on real features with BOTH backbone states frozen."""
    batch = real.clone().detach().to(next(models[0].parameters()).device)
    with torch.no_grad():
        features = [forward_features(model.eval(), batch)[1] for model in models]
    optimizer.zero_grad(set_to_none=True)
    loss = alignment(*features, node_graph_ids(batch), batch.num_graphs)
    gradients = torch.autograd.grad(loss, tuple(alignment.parameters()))
    if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
        raise FloatingPointError('Nonfinite SSD alignment step')
    for p, g in zip(alignment.parameters(), gradients):
        p.grad = g.detach()
    optimizer.step()
    return float(loss.detach())
