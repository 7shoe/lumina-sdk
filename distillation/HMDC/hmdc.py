"""HMDC equations 3–13, adapted to regression and feasible graph inputs.

Reference: Moon, Kim & Park, ECCV 2024; official code commit see below
Git commit hash: 38c5de3cbf2c7df596ffee75298fd0ee491c8195

This is independent implementation:
summed squared parameter-gradient matching, GBM, SSD and mutual real-data SGD.
"""

import os
import torch
from torch_geometric.data import Batch

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
    for parameter in synthetic.parameters():
        if parameter.grad is not None and parameter.numel():
            parameter.grad.div_(parameter.grad.abs().max() + epsilon)


def microbatch_gradients(models, alignment, balance, synthetic, real, size, *, mutual=True, balanced=True):
    """Exact squared-gradient objective in two passes, with bounded activation memory.

    Accumulate the FULL mean parameter gradients first. Then differentiate
    2 <g_syn - g_real, g_syn_chunk> with the residual detached. Squaring each
    chunk's gradient distance instead would define a different objective.
    Graphs have independent input parameters, so their GBM maxima and final
    input derivatives can be assembled on CPU without retaining GPU graphs.
    """
    if type(size) is not int or size < 1:
        raise ValueError('microbatch must be a positive integer')
    device = next(models[0].parameters()).device
    parameters = [tuple(p for p in model.parameters() if p.requires_grad) for model in models]
    term_parameters = parameters + ([parameters[0]+parameters[1]] if mutual else [])
    count = len(term_parameters)

    def losses(batch, labels):
        batch = batch.to(device)
        features, values = [], []
        for model in models:
            predictions, hidden = forward_features(model.eval(), batch)
            features.append(hidden); values.append(mse(predictions, labels, batch))
        if mutual:
            values.append(alignment(*features, node_graph_ids(batch), batch.num_graphs))
        return values

    def chunks(total):
        return [list(range(start, min(start+size, total))) for start in range(0, total, size)]

    real_graphs = real.clone().detach().cpu().to_data_list()
    synth_chunks = chunks(len(synthetic.anchors))
    totals, means = [], []
    for source in ('real', 'synthetic'):
        sums = [[torch.zeros_like(p) for p in params] for params in term_parameters]
        averages = [0.] * count
        groups = chunks(len(real_graphs)) if source == 'real' else synth_chunks
        total = len(real_graphs) if source == 'real' else len(synthetic.anchors)
        for indices in groups:
            if source == 'real':
                batch = Batch.from_data_list([real_graphs[i] for i in indices])
                labels = {k:batch[k].y for k in ('bus', 'generator')}
            else:
                batch, labels = synthetic.batch(indices, detached=True)
            values = losses(batch, labels)
            weight = len(indices)/total
            for i, (value, params) in enumerate(zip(values, term_parameters)):
                gradients = torch.autograd.grad(value, params, retain_graph=i<count-1, allow_unused=True)
                with torch.no_grad():
                    for dst, grad in zip(sums[i], gradients):
                        if grad is not None: dst.add_(grad, alpha=weight)
                averages[i] += float(value.detach())*weight
            del values, gradients, batch, labels
        totals.append(sums); means.append(averages)
    residuals = [[s-r for s,r in zip(syn, ref)] for syn,ref in zip(totals[1], totals[0])]
    terms = [sum(float(v.square().sum()) for v in group) for group in residuals]
    # remove explicitely
    del totals
    
    # collect model inputs
    inputs = tuple(synthetic.parameters())
    positions = {id(p):i for i,p in enumerate(inputs)}
    derivatives = [[torch.zeros_like(p) for p in inputs] for _ in range(count)]
    for indices in synth_chunks:
        batch, labels = synthetic.batch(indices)
        values = losses(batch, labels)
        local = tuple(p for i in indices for p in synthetic.edits[i].parameters())
        if synthetic.target_offsets:
            local += tuple(p for i in indices for p in synthetic.target_offsets[i].parameters())
        weight = len(indices)/len(synthetic.anchors)
        for i, (value, params, residual) in enumerate(zip(values, term_parameters, residuals)):
            gradients = torch.autograd.grad(value, params, retain_graph=True, create_graph=True, allow_unused=True)
            dot = sum((g*r).sum() for g,r in zip(gradients,residual) if g is not None)
            ds = torch.autograd.grad(2*weight*dot, local, retain_graph=i<count-1, allow_unused=True)
            for p,g in zip(local,ds):
                if g is not None: derivatives[i][positions[id(p)]].copy_(g.detach())
        del values, gradients, ds, dot, batch, labels
    
    if balanced and balance.steps % balance.interval == 0:
        balance.observe(derivatives)
    
    # compute loss
    weights = balance.weights() if balanced else torch.ones(count, dtype=torch.float64)
    for index,p in enumerate(inputs):
        p.grad = sum(group[index]*float(w) for group,w in zip(derivatives,weights))
    loss = sum(v*float(w) for v,w in zip(terms,weights))
    
    # sadded stability to HDMC grads
    if not torch.isfinite(torch.tensor(loss, dtype=torch.float64)) or any(not torch.isfinite(p.grad).all() for p in inputs):
        raise FloatingPointError('Nonfinite microbatched HMDC objective or data gradient')
    
    # balanced diagnostics (norm grad)
    if balanced:
        normalize_data_gradients(synthetic, balance.epsilon)
    balance.steps += 1
    diagnostics = {f'model_{i}':dict(real_mse=means[0][i],synthetic_mse=means[1][i]) for i in range(2)}
    
    # mutual diagnostics
    if mutual:
        diagnostics['mutual'] = dict(real=means[0][2],synthetic=means[1][2])
    
    return dict(loss=loss, terms=terms, weights=weights.tolist(), accumulator=balance.accumulator.tolist(),
                microbatch=size, **diagnostics)


def data_gradients(models, alignment, balance, synthetic, real, *, mutual=True, balanced=True, microbatch=None):
    """Only synthetic gradients are populated; model/alignment weights stay fixed."""
    size = int(os.environ.get('STUDY_HMDC_MICROBATCH', '0')) if microbatch is None else microbatch
    if size:
        return microbatch_gradients(models, alignment, balance, synthetic, real, size,
                                   mutual=mutual, balanced=balanced)
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


def real_chunks(real, microbatch):
    size = int(os.environ.get('STUDY_HMDC_MICROBATCH','0')) if microbatch is None else microbatch
    if not size or size >= real.num_graphs:
        yield real, 1.
        return
    if type(size) is not int or size < 1: raise ValueError('microbatch must be a positive integer')
    graphs = real.to_data_list()
    for start in range(0,len(graphs),size):
        part = graphs[start:start+size]
        yield Batch.from_data_list(part), len(part)/len(graphs)


def model_step(models, alignment, optimizers, real, *, mutual=True, microbatch=None):
    """Ordinary SGD on REAL examples: each task loss plus mutual feature MSE."""
    device = next(models[0].parameters()).device
    params = tuple(p for model in models for p in model.parameters() if p.requires_grad)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    supervised, mutual_mean = [0.,0.], 0.
    for part,weight in real_chunks(real,microbatch):
        batch = part.clone().detach().to(device)
        features, losses = [], []
        for model in models:
            predictions, hidden = forward_features(model.eval(), batch)
            features.append(hidden)
            losses.append(mse(predictions, {k: batch[k].y for k in ('bus', 'generator')}, batch))
        md = alignment(*features, node_graph_ids(batch), batch.num_graphs) if mutual else losses[0].new_zeros(())
        # Each model gets L_task_i + L_MD; MD appears once, not twice.
        loss = sum(losses) + md
        gradients = torch.autograd.grad(loss, params, allow_unused=True)
        if not torch.isfinite(loss) or any(g is not None and not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Nonfinite HMDC real-data model step')
        for p,g in zip(params,gradients):
            if g is not None:
                if p.grad is None:p.grad = g.detach()*weight
                else:p.grad.add_(g.detach(),alpha=weight)
        supervised = [a+float(v.detach())*weight for a,v in zip(supervised,losses)]
        mutual_mean += float(md.detach())*weight
    for optimizer in optimizers:
        optimizer.step()
    return dict(supervised=supervised, mutual=mutual_mean)


def alignment_step(models, alignment, optimizer, real, *, microbatch=None):
    """Fit SSD on real features with BOTH backbone states frozen."""
    optimizer.zero_grad(set_to_none=True)
    mean = 0.
    for part,weight in real_chunks(real,microbatch):
        batch = part.clone().detach().to(next(models[0].parameters()).device)
        with torch.no_grad():
            features = [forward_features(model.eval(), batch)[1] for model in models]
        loss = alignment(*features, node_graph_ids(batch), batch.num_graphs)
        gradients = torch.autograd.grad(loss, tuple(alignment.parameters()))
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Nonfinite SSD alignment step')
        for p,g in zip(alignment.parameters(),gradients):
            if p.grad is None:p.grad = g.detach()*weight
            else:p.grad.add_(g.detach(),alpha=weight)
        mean += float(loss.detach())*weight
    optimizer.step()
    return mean
