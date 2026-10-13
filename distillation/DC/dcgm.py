"""Gradient matching with feasible physical inputs; no unrolled student trajectory.

Reference: Zhao, Mopuri & Bilen (ICLR 2021), row-wise cosine distance, synthetic
SGD with momentum, ordinary detached student SGD, repeated random restarts.
Solver-gated inputs and shared solver/frozen/learned targets are explicit adaptations.
"""
import torch
from torch import nn

from distillation.synthetic.inputs import SyntheticInputs, feasible_step, solve_inputs
from distillation.synthetic.learning import mse, train_student



def gradient_distance(synthetic, real, parameters):
    """Reference 'ours' metric: summed per-output-row cosines, ignoring 1D biases.

    None gradients become zero for a consistently indexed parameter list. The
    epsilon is added to the product of norms, exactly as in the reference code.
    """
    terms = []
    for gs, gr, p in zip(synthetic, real, parameters):
        if p.ndim < 2:
            continue
        gs = torch.zeros_like(p) if gs is None else gs
        gr = torch.zeros_like(p) if gr is None else gr.detach()
        # Nearly parallel gradients cause severe FP32 cancellation in mixed
        # derivatives. Evaluate the SAME cosine formula in double precision.
        a, b = gs.reshape(p.shape[0], -1).double(), gr.reshape(p.shape[0], -1).double()
        terms.append((1 - (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)).sum())
    if not terms:
        raise ValueError('Gradient matching needs at least one matrix-valued parameter')
    return sum(terms)


def prepare_student(model, real):
    """Calibrate BatchNorm on real data and freeze its statistics for this round."""
    model.eval()  # Disable architectural dropout; SGD still differentiates weights.
    normalizers = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if normalizers:
        for module in normalizers:
            module.train()
        with torch.no_grad():
            model(real)
        for module in normalizers:
            module.eval()


def matching_loss(model, synthetic, real):
    """Same current weights for real/synthetic gradients; real gradients stop here."""
    device = next(model.parameters()).device
    real = real.clone().to(device)
    prepare_student(model, real)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    real_loss = mse(model(real), {k: real[k].y for k in ('bus', 'generator')}, real)
    gr = torch.autograd.grad(real_loss, parameters, allow_unused=True)
    batch, labels = synthetic.batch()
    batch = batch.to(device)
    synthetic_loss = mse(model(batch), labels, batch)
    gs = torch.autograd.grad(synthetic_loss, parameters, create_graph=True, allow_unused=True)
    loss = gradient_distance(gs, gr, parameters)
    def norm(gradients):
        return float(torch.stack([g.detach().square().sum() for g in gradients if g is not None]).sum().sqrt())
    diagnostics = dict(real_mse=float(real_loss.detach()), synthetic_mse=float(synthetic_loss.detach()),
                       real_gradient_norm=norm(gr), synthetic_gradient_norm=norm(gs),
                       unused_real=sum(g is None for g in gr), unused_synthetic=sum(g is None for g in gs))
    return loss, diagnostics


def data_gradients(model, synthetic, real):
    """Populate only synthetic .grad; model weights and optimizer remain untouched."""
    loss, diagnostics = matching_loss(model, synthetic, real)
    parameters = tuple(synthetic.parameters())
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    for p, g in zip(parameters, gradients):
        p.grad = torch.zeros_like(p) if g is None else g.detach()
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in parameters):
        raise FloatingPointError('Nonfinite matching loss or synthetic gradient')
    return float(loss.detach()), diagnostics
