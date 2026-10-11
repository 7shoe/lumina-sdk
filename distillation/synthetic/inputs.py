"""Shared learnable physical inputs, detached labels and transactional feasibility gate."""
from copy import deepcopy

import torch
from torch import nn

from distillation.conversion.matrix_form import mat2opf, opf2mat
from distillation.solver import ACOPF
from distillation.synthetic import Perturbation, audit_networks, teaching_targets
from distillation.synthetic.learning import benchmark_batch


class SyntheticInputs(nn.Module):
    """Fixed sampled topology, bounded operational parameters, detached AC-OPF labels."""
    def __init__(self, graphs, strength=0.02):
        super().__init__()
        if not graphs or not 0 < strength < 1:
            raise ValueError('Nonempty graphs and 0 < strength < 1 required')
        self.anchors = [opf2mat(g.clone().detach().cpu()) for g in graphs]
        self.strength = strength
        self.edits = nn.ModuleList([Perturbation(m) for m in self.anchors])
        self.labels = teaching_targets(graphs)
        with torch.no_grad():
            for p in self.parameters():
                p.zero_()

    def matrices(self):
        return [edit(anchor, self.strength) for edit, anchor in zip(self.edits, self.anchors)]

    def batch(self, indices=None, *, detached=False):
        indices = list(range(len(self.anchors))) if indices is None else list(indices)
        graphs = [mat2opf(self.edits[i](self.anchors[i], self.strength), include_solution=False)
                  for i in indices]
        batch = benchmark_batch(graphs)
        labels = {k: torch.cat([self.labels[i][k] for i in indices]).detach()
                  for k in ('bus', 'generator')}
        return (batch.detach() if detached else batch), labels



def solve_inputs(synthetic):
    matrices = synthetic.matrices()
    checks, solutions = audit_networks(matrices)
    ok = bool(checks.success.all())
    graphs = ([s.to_hetero(ACOPF(m)) for m, s in zip(matrices, solutions)] if ok else [])
    return ok, graphs, checks.to_dict(orient='records')


def feasible_step(synthetic, optimizer, *, audit=solve_inputs,
                  fractions=(1., .5, .25, .125, .0625)):
    """Apply the data gradient, backtrack infeasibility, refresh detached labels.

    The objective is not line-searched: this is reference SGD plus a physical
    feasibility gate. A rejected update restores all parameters and SGD momentum.
    """
    before, moments = deepcopy(synthetic.state_dict()), deepcopy(optimizer.state_dict())
    optimizer.step()
    proposal = deepcopy(synthetic.state_dict())
    attempts = []
    for fraction in fractions:
        synthetic.load_state_dict({k: before[k] + fraction * (v - before[k]) for k, v in proposal.items()})
        ok, graphs, checks = audit(synthetic)
        attempts.append(dict(fraction=fraction, feasible=ok))
        if ok:
            synthetic.labels = teaching_targets(graphs)
            return True, graphs, checks, attempts
    synthetic.load_state_dict(before)
    optimizer.load_state_dict(moments)
    return False, [], checks, attempts


