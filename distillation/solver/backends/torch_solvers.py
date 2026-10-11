"""Experimental routines adapted from acopf/core/solvers/torch_solvers.py.

NewtonPF is a differentiable power-flow layer, not an OPF optimizer. ReducedGradient
uses projected Adam and soft inequality penalties; convergence is not guaranteed.
"""
import torch
from torch.func import jacrev

from ..solution import Solution


class NewtonPF(torch.nn.Module):
    def __init__(self, model, iters=20, tol=1e-10):
        super().__init__()
        if iters < 1 or tol <= 0:
            raise ValueError("Need positive iterations and tolerance")
        self.m, self.iters, self.tol = model, iters, tol
        self.register_buffer("nonref", torch.arange(model.nb, device=model.lo.device)[torch.arange(model.nb, device=model.lo.device) != model.ref])
        slack = (model.gen_bus == model.ref).nonzero().flatten()
        if not len(slack):
            raise ValueError("NewtonPF needs a generator at the reference bus")
        self.slack = int(slack[0])

    def assemble(self, state, controls):
        m, (pg, qg) = self.m, controls
        theta = state.new_zeros(m.nb).index_copy(0, self.nonref, state[:m.nb-1])
        vm = state[m.nb-1:2*m.nb-1]
        pg = torch.cat((pg[:self.slack], state[-1:], pg[self.slack+1:]))
        return m.pack(theta, vm, pg, qg)

    def residual(self, state, controls):
        return self.m.balance(self.assemble(state, controls))

    def init_state(self, controls):
        pg, _ = controls
        vm = self.m.flat_start()[self.m.nb:2*self.m.nb]
        return torch.cat((pg.new_zeros(self.m.nb-1), vm, pg[self.slack:self.slack+1]))

    def forward(self, controls, s0=None):
        if any(t.shape != (self.m.ng,) or t.dtype != self.m.lo.dtype or t.device != self.m.lo.device
               for t in controls):
            raise ValueError("Controls must match model generator count, float64 dtype and device")
        state = self.init_state(controls) if s0 is None else s0
        for _ in range(self.iters):
            residual = self.residual(state, controls)
            if float(residual.detach().abs().max()) < self.tol:
                break
            step = torch.linalg.solve(jacrev(self.residual)(state, controls), residual)
            scale = 1.
            for _ in range(16):
                trial = state - scale * step
                voltage = trial[self.m.nb-1:2*self.m.nb-1]
                error = self.residual(trial, controls).detach().square().sum()
                if bool((voltage > 0).all()) and bool(error < residual.detach().square().sum()):
                    state = trial
                    break
                scale *= 0.5
            else:
                raise RuntimeError("NewtonPF line search stalled")
        if not torch.isfinite(state).all() or float(self.residual(state, controls).detach().abs().max()) > self.tol:
            raise RuntimeError("NewtonPF did not converge")
        return state


class ReducedGradient(torch.nn.Module):
    """Experimental projected Adam; reports feasibility separately from stationarity."""
    name = "reduced_gradient"

    def __init__(self, model, iters=300, lr=1e-3, rho=1e4, pf_iters=20, tol=1e-6, gtol=1e-6):
        super().__init__()
        if iters < 1 or lr <= 0 or rho <= 0:
            raise ValueError("Need positive iterations, learning rate and penalty")
        self.m, self.pf = model, NewtonPF(model, iters=pf_iters)
        self.iters, self.lr, self.rho, self.tol, self.gtol = iters, lr, rho, tol, gtol

    def penalty(self, x):
        m = self.m
        bounds = (m.lo - x).clamp_min(0).square().sum() + (x - m.hi).clamp_min(0).square().sum()
        return m.inequalities(x).clamp_min(0).square().sum() + bounds

    def solve(self, model=None, x0=None):
        if model is not None and model is not self.m:
            raise ValueError("ReducedGradient is bound to its constructor's model")
        m = self.m
        start = m.flat_start() if x0 is None else x0.x(m)
        if start.shape != m.lo.shape or not torch.isfinite(start).all():
            raise ValueError("Invalid initial solution")
        controls = [t.detach().clone().requires_grad_() for t in m.unpack(start)[2:]]
        lower, upper = m.unpack(m.lo)[2:], m.unpack(m.hi)[2:]
        optimizer = torch.optim.Adam(controls, lr=self.lr)
        state, current, converged, message = None, start, False, "Iteration limit reached"
        projected = float("inf")
        for iteration in range(self.iters):
            try:
                state = self.pf(controls, None if state is None else state.detach())
            except (RuntimeError, torch.linalg.LinAlgError) as exc:
                message = str(exc)
                break
            current = self.pf.assemble(state, controls)
            loss = m.cost(current) / m.cost_scale + self.rho * self.penalty(current)
            if not torch.isfinite(loss):
                message = "Non-finite merit function"
                break
            optimizer.zero_grad()
            loss.backward()
            projected = max(float((u - (u - u.grad).clamp(lo, hi)).detach().abs().max())
                            for u, lo, hi in zip(controls, lower, upper))
            if projected <= self.gtol:
                converged, message = True, "Projected gradient tolerance reached"
                break
            if iteration == self.iters - 1:
                break
            optimizer.step()
            with torch.no_grad():
                for u, lo, hi in zip(controls, lower, upper):
                    u.clamp_(lo, hi)
        return Solution.from_x(current, m, converged, self.name, tol=self.tol,
                               iterations=iteration+1, projected_gradient=projected, message=message)
