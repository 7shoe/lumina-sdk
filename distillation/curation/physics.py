"""Train-label physical margins and transfers, computed without another OPF solve."""
import torch

from distillation.conversion.matrix_form import opf2mat
from distillation.experiments.speculative_followup_two.physical import native


def margins(value, lower, upper):
    """Fixed bounds have no normalized slack; keep their absolute displacement."""
    width = upper-lower
    active = width > 1e-10
    scale = torch.where(active, width, torch.ones_like(width))
    return torch.cat(((value-lower)/scale, (upper-value)/scale))


def descriptor(graph):
    p = opf2mat(native(graph)).params
    va, vm = graph['bus'].y.double().T
    pg, qg = graph['generator'].y.double().T
    voltage = torch.polar(vm, va)
    sf, st = p.branch_flows(voltage)
    rated = p.rate_a > 0
    # Unrated branches have no thermal margin; their actual transfers still enter.
    rate = torch.where(rated, p.rate_a, torch.ones_like(p.rate_a))
    loading = torch.cat((torch.where(rated, sf.abs()/rate, 0.),
                         torch.where(rated, st.abs()/rate, 0.)))
    injection = torch.zeros(p.N, dtype=torch.complex128)
    injection.index_add_(0, p.gen_bus, torch.complex(pg, qg))
    injection.index_add_(0, p.load_bus, -torch.complex(p.pd, p.qd))
    injection.index_add_(0, p.shunt_bus, -torch.complex(p.gs, -p.bs)*vm[p.shunt_bus].square())
    angle = va[p.fr]-va[p.to]-p.shift
    values = torch.cat((loading, margins(vm, p.vmin, p.vmax),
        margins(pg, p.pmin, p.pmax), margins(qg, p.qmin, p.qmax),
        injection.real, injection.imag, sf.real, sf.imag, st.real, st.imag,
        (sf+st).real, (sf+st).imag, angle.sin(), angle.cos()))
    if not torch.isfinite(values).all():
        raise ValueError('Nonfinite physics descriptor')
    return values.numpy()

