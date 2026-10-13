"""Shared physical inputs, teaching-target policies and transactional feasibility gate."""
from copy import deepcopy

import torch
from torch import nn

from distillation.conversion.matrix_form import mat2opf, opf2mat
from distillation.solver import ACOPF
from distillation.synthetic import Perturbation, audit_networks, teaching_targets
from distillation.synthetic.learning import benchmark_batch

TARGET_SEMANTICS = {
    'solver': 'fresh detached AC-OPF labels',
    'frozen': 'initial fixed teaching targets; not certified solutions of edited grids',
    'learned': 'independent learned teaching targets; not certified AC-OPF solutions',
}


def teaching_values(base, offsets, reference, mode, legacy_strength=0.):
    """One target policy for DD/DCGM/HMDC; retain the reference-bus angle gauge."""
    if mode in ('solver', 'frozen'):
        return base
    result = {}
    for kind, value in base.items():
        delta = offsets[kind]
        if mode == 'legacy':
            delta = legacy_strength * value.abs().clamp_min(.1) * delta.tanh()
        if kind == 'bus':
            delta = torch.stack((delta[:, 0] * (~reference), delta[:, 1]), dim=1)
        result[kind] = value + delta
    return result


class SyntheticInputs(nn.Module):
    """Fixed topology and feasible physical inputs, independently of teaching targets."""
    def __init__(self, graphs, strength=0.02, *, target_mode='solver'):
        super().__init__()
        if not graphs or not 0 < strength < 1:
            raise ValueError('Nonempty graphs and 0 < strength < 1 required')
        if target_mode not in TARGET_SEMANTICS:
            raise ValueError('target_mode must be solver, frozen or learned')
        self.target_mode = target_mode
        self.anchors = [opf2mat(g.clone().detach().cpu()) for g in graphs]
        self.strength = strength
        self.edits = nn.ModuleList([Perturbation(m) for m in self.anchors])
        self.labels = teaching_targets(graphs)
        # No additional state keys for legacy solver-only checkpoints.
        self.target_offsets = nn.ModuleList([
            nn.ParameterDict({k: nn.Parameter(torch.zeros_like(v)) for k, v in row.items()})
            for row in self.labels] if target_mode == 'learned' else [])
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
        values = [teaching_values(self.labels[i], self.target_offsets[i] if self.target_offsets else None,
                  self.anchors[i].params.bus_type == 3, self.target_mode) for i in indices]
        labels = {k: torch.cat([row[k] for row in values])
                  for k in ('bus', 'generator')}
        return (batch.detach(), {k: v.detach() for k, v in labels.items()}) if detached else (batch, labels)

    def refresh_labels(self, graphs):
        if self.target_mode == 'solver':
            self.labels = teaching_targets(graphs)

    def snapshot(self):
        return dict(anchor_graphs=[mat2opf(m) for m in self.anchors],
                    synthetic_state=deepcopy(self.state_dict()), labels=deepcopy(self.labels),
                    target_mode=self.target_mode, target_semantics=TARGET_SEMANTICS[self.target_mode])

    @classmethod
    def from_export(cls, artifact):
        """Restore teaching targets exactly, including DD exports without their rates."""
        mode = artifact.get('target_mode', artifact['config'].get('target_mode', 'solver'))
        data = cls(artifact.get('anchor_graphs', artifact['physical_solved_graphs']),
                   artifact['config']['strength'], target_mode=mode)
        if 'synthetic_state' in artifact:
            keys = data.state_dict()
            data.load_state_dict({k: v for k, v in artifact['synthetic_state'].items() if k in keys})
            data.labels = deepcopy(artifact.get('labels', artifact.get('seed_targets', data.labels)))
        elif 'teaching_targets' in artifact:
            counts = {k: [len(row[k]) for row in data.labels] for k in ('bus', 'generator')}
            split = {k: value.detach().cpu().split(counts[k]) for k, value in artifact['teaching_targets'].items()}
            data.labels = [{k: values[i].clone() for k, values in split.items()} for i in range(len(data.anchors))]
        return data


def optimizer_for(synthetic, config):
    groups = [{'params': synthetic.edits.parameters(), 'lr': config['data_rate']}]
    if synthetic.target_offsets:
        groups.append({'params': synthetic.target_offsets.parameters(),
                       'lr': config.get('target_rate', config['data_rate'])})
    return torch.optim.SGD(groups, lr=config['data_rate'], momentum=.5)


@torch.no_grad()
def changes_summary(synthetic):
    """Physical scale movement and actual target drift from the initial OPF labels."""
    families = {}
    for edit in synthetic.edits:
        for key, value in edit.delta.items():
            families.setdefault(key, []).append(synthetic.strength * value.tanh().abs())
    result = {}
    for key, values in families.items():
        values = torch.cat(values)
        result[key] = dict(mean_scale_change_pct=float(values.mean()) * 100 if values.numel() else 0.,
                           max_scale_change_pct=float(values.max()) * 100 if values.numel() else 0.)
    labels = getattr(synthetic, 'seed_targets', None)
    if labels is None:
        labels = synthetic.labels
    drift = []
    for i, anchor in enumerate(synthetic.anchors):
        current = teaching_values(labels[i], synthetic.target_offsets[i] if synthetic.target_offsets else None,
            anchor.params.bus_type == 3, synthetic.target_mode, getattr(synthetic, 'target_strength', 0.))
        initial = mat2opf(anchor)
        drift.extend((v - initial[k].y).flatten() for k, v in current.items())
    result['target_drift_rms'] = float(torch.cat(drift).square().mean().sqrt())
    return result



def solve_inputs(synthetic):
    matrices = synthetic.matrices()
    checks, solutions = audit_networks(matrices)
    ok = bool(checks.success.all())
    graphs = ([s.to_hetero(ACOPF(m)) for m, s in zip(matrices, solutions)] if ok else [])
    return ok, graphs, checks.to_dict(orient='records')


def feasible_step(synthetic, optimizer, *, audit=solve_inputs,
                  fractions=(1., .5, .25, .125, .0625)):
    """Backtrack infeasible/nonfinite proposals; refresh only solver-mode labels.

    The objective is not line-searched: this is reference SGD plus a physical
    feasibility gate. A rejected update restores all parameters and SGD momentum.
    """
    if not fractions or any(not 0 < f <= 1 for f in fractions):
        raise ValueError('Backtracking fractions must be in (0, 1]')
    before, moments, labels = deepcopy(synthetic.state_dict()), deepcopy(optimizer.state_dict()), deepcopy(synthetic.labels)
    def rollback():
        synthetic.load_state_dict(before)
        optimizer.load_state_dict(moments)
        synthetic.labels = labels
    attempts, checks = [], []
    try:
        optimizer.step()
        proposal = deepcopy(synthetic.state_dict())
        for fraction in fractions:
            synthetic.load_state_dict({k: before[k] + fraction * (v - before[k]) for k, v in proposal.items()})
            finite = all(torch.isfinite(p).all() for p in synthetic.parameters())
            if not finite:
                attempts.append(dict(fraction=fraction, feasible=False, finite=False))
                continue
            ok, graphs, checks = audit(synthetic)
            attempts.append(dict(fraction=fraction, feasible=ok))
            if ok:
                synthetic.refresh_labels(graphs)
                return True, graphs, checks, attempts
    except Exception:
        rollback()
        raise
    rollback()
    return False, [], checks, attempts
