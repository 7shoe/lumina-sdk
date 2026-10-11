"""Lossless OPFData ↔ matrix conversion, network edits, and residual checks.

Preserves unedited records; parameter edits do not refresh solution labels.
See README.md for the matrix algebraic derivation/details & usage, conventions, limitations
"""


from __future__ import annotations

from dataclasses import dataclass, field, replace
from copy import deepcopy
from functools import wraps
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.data import HeteroData, Batch

# ----------------------------------------------------- #
# OPFData columns
# ---------------------------------------------------#
BUS_X_COLS = ("base_kv", "bus_type", "vmin", "vmax")
BUS_Y_COLS = ("va", "vm")
GEN_X_COLS = ("mbase", "pg_set", "pmin", "pmax", "qg_set",
              "qmin", "qmax", "vg", "c2", "c1", "c0")
GEN_Y_COLS = ("pg", "qg")
LOAD_X_COLS = ("pd", "qd")
SHUNT_X_COLS = ("bs", "gs")
LINE_ATTR_COLS = ("angmin", "angmax", "b_fr", "b_to", "br_r", "br_x",
                  "rate_a", "rate_b", "rate_c")
XFMR_ATTR_COLS = ("angmin", "angmax", "br_r", "br_x", "rate_a", "rate_b",
                  "rate_c", "tap", "shift", "b_fr", "b_to")

# Solution branch features.  OPFData paper, Appendix A (arXiv:2406.07234) lists
# them as  pt, qt, pf, qf  -- NOT pf, qf, pt, qt.  Run ``infer_label_order(data)``
# once on a real sample to confirm; a wrong order shows up as a flow residual of
# order 2|p_f| in ``residuals`` (since p_f ~ -p_t up to losses).
BRANCH_LABEL_COLS = ("pt", "qt", "pf", "qf")

# unified per-branch parameter names (lines get tap=1, shift=0)
BRANCH_COLS = ("angmin", "angmax", "br_r", "br_x", "b_fr", "b_to",
               "rate_a", "rate_b", "rate_c", "tap", "shift")

E_LINE = ("bus", "ac_line", "bus")
E_XFMR = ("bus", "transformer", "bus")
E_GEN = ("generator", "generator_link", "bus")
E_LOAD = ("load", "load_link", "bus")
E_SHUNT = ("shunt", "shunt_link", "bus")
LINK_TYPES = {E_GEN: "gen_bus", E_LOAD: "load_bus", E_SHUNT: "shunt_bus"}

def _cols(mat: Tensor, names: Tuple[str, ...]) -> Dict[str, Tensor]:
    if not isinstance(mat, Tensor) or mat.dtype not in (torch.float32, torch.float64):
        raise TypeError("Core feature/label tables must be float32 or float64 tensors")
    if mat.ndim == 1 and mat.numel() == 0:
        mat = mat.reshape(0, len(names))  # PyG torch.tensor([]) convention
    if mat.ndim != 2 or mat.shape[1] != len(names):
        raise ValueError(f"Expected [n,{len(names)}], got {tuple(mat.shape)}")
    return {n: mat[:, i].clone() for i, n in enumerate(names)}


def _stack(d: Dict[str, Tensor], names: Tuple[str, ...]) -> Tensor:
    return torch.stack([d[n] for n in names], dim=1)


def _onehot(idx: Tensor, n_rows: int, n_cols: int, dtype=torch.float64) -> Tensor:
    """Dense {0,1} matrix with M[r, idx[r]] = 1."""
    M = torch.zeros(n_rows, n_cols, dtype=dtype, device=idx.device)
    M[torch.arange(n_rows, device=idx.device), idx] = 1.0
    return M


# ---------------------------------------------------- #
# Fixed problem data  (the "d" in  x = x(u | d))
# ----------------------------------------------------- #
@dataclass
class OPFParams:
    """
    Everything the AC-OPF treats as given.  Stored raw and exact; the
    matrices A, C_*, D_y, Y, Y_f, Y_t are derived properties (float64/complex128).
    """
    base_mva: Tensor          # [1]   graph-level context
    # ---- topology (index form)
    fr: Tensor                # [E]   from-bus of each branch
    to: Tensor                # [E]   to-bus
    n_line: int               # branches [:n_line] are ac lines, [n_line:] transformers
    gen_bus: Tensor           # [G]
    load_bus: Tensor          # [L]
    shunt_bus: Tensor         # [S]
    n_bus: int
    
    # ---- bus data
    base_kv: Tensor; bus_type: Tensor; vmin: Tensor; vmax: Tensor
    
    # ---- branch parameters
    br_r: Tensor; br_x: Tensor; b_fr: Tensor; b_to: Tensor
    tap: Tensor; shift: Tensor
    rate_a: Tensor; rate_b: Tensor; rate_c: Tensor
    angmin: Tensor; angmax: Tensor
    
    # ---- generator bounds / costs (+ legacy setpoints kept for exactness)
    pmin: Tensor; pmax: Tensor; qmin: Tensor; qmax: Tensor
    c2: Tensor; c1: Tensor; c0: Tensor
    mbase: Tensor; pg_set: Tensor; qg_set: Tensor; vg: Tensor
    
    # ---- loads and shunts
    pd: Tensor; qd: Tensor
    bs: Tensor; gs: Tensor

    # ---- sizes
    @property
    def E(self) -> int: return int(self.fr.numel())
    @property
    def N(self) -> int: return self.n_bus
    @property
    def G(self) -> int: return int(self.gen_bus.numel())
    @property
    def L(self) -> int: return int(self.load_bus.numel())
    @property
    def S(self) -> int: return int(self.shunt_bus.numel())

    # ---- incidence matrices
    @property
    def Cf(self) -> Tensor: return _onehot(self.fr, self.E, self.N)       # [E,N]
    @property
    def Ct(self) -> Tensor: return _onehot(self.to, self.E, self.N)       # [E,N]
    @property
    def A(self) -> Tensor: return self.Cf - self.Ct                        # [E,N]
    @property
    def Cg(self) -> Tensor: return _onehot(self.gen_bus, self.G, self.N).T    # [N,G]
    @property
    def Cd(self) -> Tensor: return _onehot(self.load_bus, self.L, self.N).T   # [N,L]
    @property
    def Cs(self) -> Tensor: return _onehot(self.shunt_bus, self.S, self.N).T  # [N,S]

    # ---- per-branch / per-bus complex parameters
    @property
    def y(self) -> Tensor:
        """Series admittance y_e = 1/(r_e + j x_e)."""
        return 1.0 / torch.complex(self.br_r.double(), self.br_x.double())
    @property
    def D_y(self) -> Tensor: return torch.diag(self.y)
    @property
    def t(self) -> Tensor:
        """Complex tap t_e = tap_e exp(j shift_e)  (= 1 for ac lines)."""
        return torch.polar(self.tap.double(), self.shift.double())
    @property
    def y_sh(self) -> Tensor:
        """Bus shunt admittance  y_sh = C_s (g_s + j b_s)  in C^N."""
        ysh = torch.zeros(self.N, dtype=torch.complex128, device=self.fr.device)
        return ysh.index_add(0, self.shunt_bus, torch.complex(self.gs.double(), self.bs.double()))
    @property
    def S_d(self) -> Tensor:
        """Bus load S_d = C_d (p_d + j q_d)  in C^N."""
        sd = torch.zeros(self.N, dtype=torch.complex128, device=self.fr.device)
        return sd.index_add(0, self.load_bus, torch.complex(self.pd.double(), self.qd.double()))

    # admittance matrices
    def admittance(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Return dense (Y, Y_f, Y_t), the same objects as makeYbus.

        Y_ff = (y + j b_fr)/|t|^2,  Y_ft = -y/conj(t),  Y_tf = -y/t,  Y_tt = y + j b_to
        Y_f  = diag(Y_ff) C_f + diag(Y_ft) C_t
        Y_t  = diag(Y_tf) C_f + diag(Y_tt) C_t
        Y    = C_f^T Y_f + C_t^T Y_t + diag(y_sh)
        """
        Yff, Yft, Ytf, Ytt = self.branch_admittances()
        e = torch.arange(self.E, device=self.fr.device)
        Yf = torch.zeros(self.E, self.N, dtype=torch.complex128, device=self.fr.device)
        Yf.index_put_((e, self.fr), Yff, accumulate=True)
        Yf.index_put_((e, self.to), Yft, accumulate=True)
        Yt = torch.zeros(self.E, self.N, dtype=torch.complex128, device=self.fr.device)
        Yt.index_put_((e, self.fr), Ytf, accumulate=True)
        Yt.index_put_((e, self.to), Ytt, accumulate=True)
        Y = torch.zeros(self.N, self.N, dtype=torch.complex128, device=self.fr.device)
        Y.index_add_(0, self.fr, Yf)
        Y.index_add_(0, self.to, Yt)
        Y += torch.diag(self.y_sh)
        return Y, Yf, Yt

    @property
    def Y(self) -> Tensor: return self.admittance()[0]
    @property
    def Yf(self) -> Tensor: return self.admittance()[1]
    @property
    def Yt(self) -> Tensor: return self.admittance()[2]

    def branch_admittances(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        validate_params(self, physics=True)
        y, t = self.y, self.t
        return ((y + 1j * self.b_fr.double()) / self.tap.double().square(),
                -y / t.conj(), -y / t, y + 1j * self.b_to.double())

    def branch_flows(self, V: Tensor) -> Tuple[Tensor, Tensor]:
        """O(E) evaluation, equal to diag(Cf V) conj(Yf V) and its to-side analog."""
        if V.shape != (self.N,) or V.device != self.fr.device:
            raise ValueError("V must have shape [N] on the parameter device")
        ff, ft, tf, tt = self.branch_admittances()
        vf, vt = V[self.fr], V[self.to]
        return vf * (ff * vf + ft * vt).conj(), vt * (tf * vf + tt * vt).conj()

    def with_series_admittance(self, y: Tensor) -> "OPFParams":
        """Numerically set r+jx=1/y. This is an edit, not an exact round trip."""
        if y.shape != self.br_r.shape or y.device != self.br_r.device:
            raise ValueError("y must have shape [E] on the parameter device")
        if not torch.isfinite(y).all() or (y == 0).any():
            raise ValueError("Series admittances must be finite and nonzero")
        z = 1.0 / y.to(torch.complex128)
        p = replace(self, br_r=z.real.to(self.br_r.dtype), br_x=z.imag.to(self.br_x.dtype))
        validate_params(p, physics=True)
        return p

    def cost(self, pg: Tensor) -> Tensor:
        pg = pg.double()
        return (self.c2.double() * pg ** 2 + self.c1.double() * pg + self.c0.double()).sum()


# --------------------------------------------- #
# Decision variables / solver output  (y = (x, u))
# --------------------------------------------- #
@dataclass
class OPFSolution:
    va: Tensor; vm: Tensor        # state  x = (theta, |V|)       [N]
    pg: Tensor; qg: Tensor        # control u = (P_g, Q_g)        [G]
    pf: Tensor; qf: Tensor; pt: Tensor; qt: Tensor   # derived branch flows [E]
    objective: Tensor             # scalar  C(P_g*)

    @property
    def V(self) -> Tensor:
        return torch.polar(self.vm.double(),
                           self.va.double())
    @property
    def S_g(self) -> Tensor:
        return torch.complex(self.pg.double(),
                             self.qg.double())
    @property
    def S_f(self) -> Tensor:
        return torch.complex(self.pf.double(),
                             self.qf.double())
    @property
    def S_t(self) -> Tensor: 
        return torch.complex(self.pt.double(),
                             self.qt.double())


@dataclass
class OPFMatrices:
    params: OPFParams
    sol: Optional[OPFSolution]
    meta: Dict[str, Any] = field(default_factory=dict)


# Serialization helpers keep raw records independent of derived matrices.
def _clone(value):
    if isinstance(value, Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _clone(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone(v) for v in value]
    if isinstance(value, tuple):
        return type(value)(*[_clone(v) for v in value]) if hasattr(value, '_fields') else tuple(_clone(v) for v in value)
    return deepcopy(value)


def _clone_data(data):
    out = HeteroData()
    for k, v in data.items():
        out[k] = _clone(v)
    for key in list(data.node_types) + list(data.edge_types):
        store = out[key]  # preserve even an empty store
        for k, v in data[key].items():
            store[k] = _clone(v)
    return out


def _index(idx, shape, upper, name):
    if idx.dtype != torch.int64 or tuple(idx.shape) != tuple(shape):
        raise ValueError(f'{name}: expected int64 tensor of shape {shape}')
    if idx.numel() and ((idx < 0).any() or (idx >= upper).any()):
        raise ValueError(f'{name}: index out of bounds [0,{upper})')


def validate_params(p: OPFParams, *, physics=False):
    """Check representability; physics=True also checks the supported circuit model."""
    if p.N <= 0 or not 0 <= p.n_line <= p.E:
        raise ValueError('Invalid bus count or branch partition')
    device = p.fr.device
    for name, value in vars(p).items():
        if isinstance(value, Tensor) and value.device != device:
            raise ValueError(f'Mixed core devices: {name}')
    for name, count, cols in [('bus', p.N, BUS_X_COLS), ('branch', p.E, BRANCH_COLS),
                              ('generator', p.G, GEN_X_COLS), ('load', p.L, LOAD_X_COLS),
                              ('shunt', p.S, SHUNT_X_COLS)]:
        for col in cols:
            v = getattr(p, col)
            if v.shape != (count,) or v.dtype not in (torch.float32, torch.float64):
                raise ValueError(f'{name}.{col}: invalid shape or dtype')
    for name, count in [('fr', p.E), ('to', p.E), ('gen_bus', p.G),
                        ('load_bus', p.L), ('shunt_bus', p.S)]:
        _index(getattr(p, name), (count,), p.N, name)
    if p.base_mva.numel() != 1 or not torch.isfinite(p.base_mva).all() or not (p.base_mva > 0).all():
        raise ValueError('Expected one positive finite baseMVA, not a batch')
    if not (p.tap[:p.n_line] == 1).all() or not (p.shift[:p.n_line] == 0).all():
        raise ValueError('ac_line rows cannot encode non-unit taps or phase shifts')
    if not physics:
        return
    for name in ('br_r', 'br_x', 'b_fr', 'b_to', 'tap', 'shift', 'gs', 'bs',
                 'pd', 'qd', 'c2', 'c1', 'c0', 'vmin', 'vmax', 'bus_type'):
        if not torch.isfinite(getattr(p, name)).all():
            raise ValueError(f'Nonfinite physical parameter: {name}')
    if ((p.br_r == 0) & (p.br_x == 0)).any() or (p.tap <= 0).any():
        raise ValueError('Zero branch impedance or nonpositive transformer tap')
    if not ((p.bus_type == 1) | (p.bus_type == 2) | (p.bus_type == 3)).all():
        raise ValueError('Physics requires active buses with types 1, 2 or 3')
    for low, high in [('vmin', 'vmax'), ('pmin', 'pmax'), ('qmin', 'qmax'), ('angmin', 'angmax')]:
        a, b = getattr(p, low), getattr(p, high)
        if torch.isnan(a).any() or torch.isnan(b).any() or (a > b).any():
            raise ValueError(f'Invalid bounds: {low}, {high}')
    if (p.vmin <= 0).any():
        raise ValueError('This polar model requires strictly positive voltage bounds')


def opf2mat(data: HeteroData, *, branch_label_cols=BRANCH_LABEL_COLS) -> OPFMatrices:
    """Snapshot a single OPFData record and expose raw parameters + derived matrices.

    The template preserves all metadata and partial labels. It is not a compressed
    representation. For alternative label conventions pass an explicit permutation.
    """
    if isinstance(data, Batch) or not isinstance(data, HeteroData):
        raise TypeError('Expected one HeteroData, not Batch')
    order = tuple(branch_label_cols)
    if len(order) != 4 or set(order) != {'pf', 'qf', 'pt', 'qt'}:
        raise ValueError('branch_label_cols must be a permutation of pf,qf,pt,qt')
    for key in ('bus', 'generator', 'load', 'shunt'):
        if key not in data.node_types or 'x' not in data[key]:
            raise ValueError(f'Missing required node features: {key}')
    for key in (E_LINE, E_XFMR, *LINK_TYPES):
        if key not in data.edge_types or 'edge_index' not in data[key]:
            raise ValueError(f'Missing required edge_index: {key}')
    bus, gen = _cols(data['bus'].x, BUS_X_COLS), _cols(data['generator'].x, GEN_X_COLS)
    load, shunt = _cols(data['load'].x, LOAD_X_COLS), _cols(data['shunt'].x, SHUNT_X_COLS)
    n = len(bus['vmin'])
    ln, xf = data[E_LINE], data[E_XFMR]
    la, xa = _cols(ln.edge_attr, LINE_ATTR_COLS), _cols(xf.edge_attr, XFMR_ATTR_COLS)
    nl, nx = len(la['br_r']), len(xa['br_r'])
    for key, count in ((E_LINE, nl), (E_XFMR, nx)):
        _index(data[key].edge_index, (2, count), n, str(key))
    la['tap'], la['shift'] = torch.ones_like(la['br_r']), torch.zeros_like(la['br_r'])
    # Concatenating f32 and f64 is exact; original per-store dtypes live in template.
    br = {k: torch.cat([la[k], xa[k]]) for k in BRANCH_COLS}
    ei = torch.cat([ln.edge_index, xf.edge_index], dim=1)
    links = {}
    for et, name in LINK_TYPES.items():
        count = {'generator': len(gen['pmin']), 'load': len(load['pd']), 'shunt': len(shunt['gs'])}[et[0]]
        edge = data[et].edge_index
        if edge.dtype != torch.int64 or edge.shape != (2, count):
            raise ValueError(f'{et}: expected int64 [2,{count}]')
        src, dst = edge
        if not torch.equal(src.sort().values, torch.arange(count, device=src.device)):
            raise ValueError(f'{et}: expected exactly one bus per equipment row')
        _index(dst, (count,), n, name)
        out = torch.empty_like(dst)
        out[src] = dst
        links[name] = out
    params = OPFParams(base_mva=data.x.clone(), fr=ei[0].clone(), to=ei[1].clone(),
                       n_line=nl, n_bus=n, **links, **bus, **br, **gen, **load, **shunt)
    validate_params(params)
    required_labels = [('bus', 'y'), ('generator', 'y'), (E_LINE, 'edge_label'), (E_XFMR, 'edge_label')]
    sol = None
    if all(attr in data[key] for key, attr in required_labels) and 'objective' in dict(data.items()):
        by, gy = _cols(data['bus'].y, BUS_Y_COLS), _cols(data['generator'].y, GEN_Y_COLS)
        ll, xl = _cols(ln.edge_label, order), _cols(xf.edge_label, order)
        expected = [(by, n), (gy, params.G), (ll, nl), (xl, nx)]
        for cols, count in expected:
            if any(v.shape != (count,) or v.device != ei.device for v in cols.values()):
                raise ValueError('Solution shape/device does not match parameters')
        if not isinstance(data.objective, Tensor) or data.objective.numel() != 1 or data.objective.device != ei.device:
            raise ValueError('Objective must be a single tensor value on the core device')
        sol = OPFSolution(**by, **gy, **{k: torch.cat([ll[k], xl[k]]) for k in order}, objective=data.objective.clone())
    schema = dict(bus=BUS_X_COLS, gen=GEN_X_COLS, load=LOAD_X_COLS, shunt=SHUNT_X_COLS,
                  line=LINE_ATTR_COLS, transformer=XFMR_ATTR_COLS, bus_y=BUS_Y_COLS,
                  gen_y=GEN_Y_COLS, branch_y=order)
    return OPFMatrices(params, sol, {'template': _clone_data(data), 'schema': schema,
                                    'sizes': (params.N, params.G, params.L, params.S, nl, nx)})


def _write(store, key, value):
    """Retain source dtype/shape/device; refuse a silently lossy export."""
    if key not in store:
        store[key] = value.clone()
        return
    old = store[key]
    if not isinstance(old, Tensor) or value.numel() != old.numel() or value.device != old.device:
        raise ValueError(f'{key}: incompatible edit shape/device')
    target = value.reshape(old.shape).to(old.dtype)
    restored = target.to(value.dtype).reshape(value.shape)
    equal = (restored == value)
    if value.is_floating_point() or value.is_complex():
        equal = equal | (torch.isnan(restored) & torch.isnan(value))
    if not equal.all():
        raise ValueError(f'{key}: edit not exactly representable in original dtype {old.dtype}')
    # Casting f32 -> f64 -> f32 may quiet a signaling NaN. Keep original
    # bytes for unchanged numeric values/NaNs; allow explicit signed-zero edits.
    # For trainable values, selecting source bytes would cut gradients wherever
    # values equal the archive (including the initial optimization step).
    if target.is_floating_point() and not value.requires_grad:
        same = (target == old) | (torch.isnan(target) & torch.isnan(old))
        same = same & ((target != 0) | (torch.signbit(target) == torch.signbit(old)))
        target = torch.where(same, old, target)
    store[key] = target.clone()


def mat2opf(mats: OPFMatrices, *, include_solution=True) -> HeteroData:
    """Rebuild a record using its source template; no inverse from Y alone exists.

    Partial labels are passed through. Complete sol, if present, overrides them.
    params edits do not automatically recompute labels or objective.
    Set include_solution=False to export only problem inputs and omit even
    partial labels held in the source template. Default True preserves the
    original exact round-trip API; it does not certify label validity.
    """
    p, s, meta = mats.params, mats.sol, mats.meta
    if not include_solution:
        s = None
    if 'template' not in meta or 'schema' not in meta:
        raise ValueError('Lossless export requires meta from opf2mat')
    validate_params(p)
    if (p.N, p.G, p.L, p.S, p.n_line, p.E-p.n_line) != meta['sizes']:
        raise ValueError('Changing counts or line/transformer partition is unsupported')
    if s is not None:
        for name, count in [('va', p.N), ('vm', p.N), ('pg', p.G), ('qg', p.G),
                            ('pf', p.E), ('qf', p.E), ('pt', p.E), ('qt', p.E)]:
            v = getattr(s, name)
            if v.shape != (count,) or v.device != p.fr.device or v.dtype not in (torch.float32, torch.float64):
                raise ValueError(f'Invalid solution tensor shape/device/dtype: {name}')
    data, schema = _clone_data(meta['template']), meta['schema']
    if not include_solution:
        _strip_solution(data)
    _write(data, 'x', p.base_mva)
    for key, name in [('bus', 'bus'), ('generator', 'gen'), ('load', 'load'), ('shunt', 'shunt')]:
        _write(data[key], 'x', _stack(vars(p), schema[name]))
    for et, sl, name in [(E_LINE, slice(None, p.n_line), 'line'),
                         (E_XFMR, slice(p.n_line, None), 'transformer')]:
        _write(data[et], 'edge_index', torch.stack([p.fr[sl], p.to[sl]]))
        _write(data[et], 'edge_attr', _stack({k: getattr(p, k)[sl] for k in schema[name]}, schema[name]))
        if s is not None:
            _write(data[et], 'edge_label', _stack({k: getattr(s, k)[sl] for k in schema['branch_y']}, schema['branch_y']))
    if s is not None:
        _write(data['bus'], 'y', _stack(vars(s), schema['bus_y']))
        _write(data['generator'], 'y', _stack(vars(s), schema['gen_y']))
        _write(data, 'objective', s.objective)
    for et, name in LINK_TYPES.items():
        bus_of = getattr(p, name)
        original = meta['template'][et].edge_index
        new = torch.stack([original[0], bus_of[original[0]]])
        _write(data[et], 'edge_index', new)
        rev = et[::-1]
        if not torch.equal(original, new) and rev in data.edge_types:
            # Preserve independent reverse-edge ordering and edge attributes.
            re = data[rev].edge_index
            count = bus_of.numel()
            if re.shape != (2, count) or re.dtype != torch.int64 or not torch.equal(re[1].sort().values, torch.arange(count, device=re.device)):
                raise ValueError('Cannot update non-bijective reverse equipment links')
            old_bus = torch.empty_like(bus_of)
            old_bus[original[0]] = original[1]
            if not torch.equal(re[0], old_bus[re[1]]):
                raise ValueError('Reverse links contradict the original forward mapping')
            _write(data[rev], 'edge_index', torch.stack([bus_of[re[1]], re[1]]))
    return data


def _max(value):
    return value.max().item() if value.numel() else 0.0


def _viol(lower, x, upper):
    return max(_max(torch.relu(lower.double() - x)), _max(torch.relu(x - upper.double())))


def residuals(mats: OPFMatrices, *, zero_rating='error') -> Dict[str, float]:
    """Feasibility/label diagnostics; no optimality claim. Empty maxima equal zero.

    zero_rating: 'error' refuses ambiguous zero ratings (default); 'literal'
    enforces |S|<=0; 'unmonitored' omits zero-rated branches. +inf is unbounded.
    Cost-relative residual is NaN when objective=0; cost_abs is always reported.
    Voltage/thermal/power residuals are p.u.; angle/ref_angle are radians.
    """
    p, s = mats.params, mats.sol
    if s is None:
        raise ValueError('residuals requires all solution labels and objective')
    validate_params(p, physics=True)
    expected = {'va': p.N, 'vm': p.N, 'pg': p.G, 'qg': p.G,
                'pf': p.E, 'qf': p.E, 'pt': p.E, 'qt': p.E}
    for name, count in expected.items():
        v = getattr(s, name)
        if v.shape != (count,) or v.device != p.fr.device or not torch.isfinite(v).all():
            raise ValueError(f'Invalid solution tensor: {name}')
    if s.objective.numel() != 1 or not torch.isfinite(s.objective).all():
        raise ValueError('Invalid objective')
    if (s.vm < 0).any():
        raise ValueError('Voltage magnitudes cannot be negative')
    if zero_rating not in ('error', 'literal', 'unmonitored'):
        raise ValueError('Unknown zero_rating policy')
    rate = p.rate_a.double()
    if torch.isnan(rate).any() or (rate < 0).any():
        raise ValueError('Thermal ratings must be nonnegative or +inf')
    if zero_rating == 'error' and (rate == 0).any():
        raise ValueError('Zero rate_a: choose literal or unmonitored explicitly')
    monitor = torch.isfinite(rate)
    if zero_rating == 'unmonitored':
        monitor = monitor & (rate > 0)
    V = s.V
    Sf, St = p.branch_flows(V)
    inj = torch.zeros(p.N, dtype=torch.complex128, device=V.device).index_add(0, p.gen_bus, s.S_g)
    out = torch.zeros_like(inj).index_add(0, p.fr, Sf).index_add(0, p.to, St)
    bal = inj - p.S_d - out - V.abs().square() * p.y_sh.conj()
    dth = s.va.double()[p.fr] - s.va.double()[p.to]
    cost_abs = abs(p.cost(s.pg).item() - s.objective.item())
    ref = p.bus_type == 3
    return {'balance': _max(bal.abs()), 'flow_f': _max((Sf-s.S_f).abs()),
            'flow_t': _max((St-s.S_t).abs()), 'cost_abs': cost_abs,
            'cost_rel': cost_abs / abs(s.objective.item()) if s.objective.item() != 0 else float('nan'),
            'thermal': max(_max(torch.relu(Sf.abs()[monitor]-rate[monitor])),
                           _max(torch.relu(St.abs()[monitor]-rate[monitor]))),
            'angle': _viol(p.angmin, dth, p.angmax),
            'ref_angle': _max(s.va.double()[ref].abs()) if ref.any() else float('nan'),
            'reference_bus_count': int(ref.sum().item()),
            'v_bounds': _viol(p.vmin, s.vm.double(), p.vmax),
            'pg_bounds': _viol(p.pmin, s.pg.double(), p.pmax),
            'qg_bounds': _viol(p.qmin, s.qg.double(), p.qmax)}


def infer_label_order(data: HeteroData) -> Dict[str, float]:
    """Compare two candidate flow schemas without global mutation.

    This is evidence, not a universal identification theorem: zero/symmetric
    flows can tie, and bad units/parameters can make both candidates poor.
    No universal 1e-6 residual threshold is assumed.
    """
    m = opf2mat(data)
    if 'y' not in data['bus']:
        raise ValueError('Bus voltage labels are required')
    by = _cols(data['bus'].y, BUS_Y_COLS)
    Sf, St = m.params.branch_flows(torch.polar(by['vm'].double(), by['va'].double()))
    out = {}
    for cand in (('pt', 'qt', 'pf', 'qf'), ('pf', 'qf', 'pt', 'qt')):
        labels = [_cols(data[et].edge_label, cand) for et in (E_LINE, E_XFMR)]
        cols = {k: torch.cat([q[k] for q in labels]).double() for k in cand}
        out[','.join(cand)] = max(_max((Sf-torch.complex(cols['pf'], cols['qf'])).abs()),
                                _max((St-torch.complex(cols['pt'], cols['qt'])).abs()))
    return out


def _equal(a, b):
    if isinstance(a, Tensor):
        if not isinstance(b, Tensor) or a.shape != b.shape or a.dtype != b.dtype or a.device != b.device or a.layout != b.layout:
            return False
        if a.layout != torch.strided:
            raise TypeError('Exact checker supports strided tensor attributes only')
        # Byte comparison distinguishes signed zero and treats identical NaNs as equal.
        aa = a.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
        bb = b.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
        return torch.equal(aa, bb)
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return list(a) == list(b) and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        import struct
        return struct.pack('d', a) == struct.pack('d', b)
    try:
        result = a == b
        return bool(result.all()) if hasattr(result, 'all') else bool(result)
    except (TypeError, ValueError):
        return False


def equal_records(a: HeteroData, b: HeteroData) -> bool:
    """Stored content equality, including dtype/device/shape and tensor bit patterns."""
    return (a.node_types == b.node_types and a.edge_types == b.edge_types
            and _equal(dict(a.items()), dict(b.items()))
            and all(_equal(dict(a[key].items()), dict(b[key].items()))
                    for key in list(a.node_types)+list(a.edge_types)))


def check_roundtrip(data: HeteroData, **kwargs) -> bool:
    return equal_records(data, mat2opf(opf2mat(data, **kwargs)))


# ---------------------------------------------------------------------------
# Discrete structural edits: rebuild the record rather than resizing its archive.
# ---------------------------------------------------------------------------
_NODE_COLS = {'bus': BUS_X_COLS, 'generator': GEN_X_COLS,
              'load': LOAD_X_COLS, 'shunt': SHUNT_X_COLS}
_BRANCH_TYPES = {'ac_line': E_LINE, 'transformer': E_XFMR}
_BRANCH_SCHEMA = {'ac_line': LINE_ATTR_COLS, 'transformer': XFMR_ATTR_COLS}
_EQUIPMENT_LINK = {'generator': E_GEN, 'load': E_LOAD, 'shunt': E_SHUNT}


def _strip_solution(data):
    for key, attr in [('bus', 'y'), ('generator', 'y'),
                      (E_LINE, 'edge_label'), (E_XFMR, 'edge_label')]:
        if attr in data[key]:
            del data[key][attr]
    if 'objective' in dict(data.items()):
        del data['objective']


def validate_network(p: OPFParams, *, require_connected=True,
                     require_reference=True):
    """Check parameters, no self-loops, and connected components/reference counts.

    Default: one connected grid and exactly one reference bus. With
    require_connected=False, each island must still have exactly one reference
    unless require_reference=False. Returns islands as lists of bus row indices.
    This discrete CPU graph check neither solves nor certifies OPF feasibility.
    """
    validate_params(p, physics=True)
    if (p.fr == p.to).any():
        raise ValueError('Self-loop branches are unsupported; remove or merge explicitly')
    if not torch.isfinite(p.base_kv).all() or (p.base_kv <= 0).any():
        raise ValueError('base_kv must be finite and positive')
    if not torch.isfinite(p.mbase).all() or (p.mbase <= 0).any():
        raise ValueError('Generator mbase must be finite and positive')
    for name in ('rate_a', 'rate_b', 'rate_c'):
        v = getattr(p, name)
        if torch.isnan(v).any() or (v < 0).any():
            raise ValueError(f'{name} must be nonnegative or +inf')
    parent = list(range(p.N))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in zip(p.fr.detach().cpu().tolist(), p.to.detach().cpu().tolist()):
        parent[find(i)] = find(j)
    groups = {}
    for i in range(p.N):
        groups.setdefault(find(i), []).append(i)
    islands = list(groups.values())
    if require_connected and len(islands) != 1:
        raise ValueError(f'Network has {len(islands)} islands; reconnect it or allow islands explicitly')
    refs = (p.bus_type == 3).detach().cpu().tolist()
    if require_reference:
        for buses in islands:
            if sum(refs[i] for i in buses) != 1:
                raise ValueError(f'Island {buses[:8]} requires exactly one reference bus (bus_type=3)')
    return islands


def _atomic_edit(method):
    """A failed edit leaves the editor unchanged; successful edits stay isolated."""
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        previous = self._data
        self._data = _clone_data(previous)
        try:
            result = method(self, *args, **kwargs)
            # Allow disconnected intermediate networks while assembling a batch.
            candidate = opf2mat(self._data, branch_label_cols=self._label_order)
            validate_network(candidate.params, require_connected=False, require_reference=False)
            return result
        except Exception:
            self._data = previous
            raise
    return wrapped


class NetworkEditor:
    """Small, discrete topology editor, initialized from OPFMatrices.

    Input is never mutated. The editor detaches tensors and drops all old solution
    labels (oly the OPFData core schema: unknown metadata/stores cause
    an error unless drop_extra=True explicitly permits discarding them

    Edit methods mutate this isolated editor. build() returns a fresh OPFMatrices
    with a resized archival template; ordinary mat2opf() then works unchanged.
    Every operation is atomic on failure; a build failure leaves edits available
    for repair. Intermediate disconnected graphs are permitted, but build()
    defaults to a connected network with exactly one reference bus.

    All indices are CURRENT, zero-based row indices. Branch methods use indices
    local to 'ac_line' or 'transformer', not the concatenated p.fr array. Additions
    return the new local index. Removal returns an old->new map (-1 for removed).
    merge_buses returns a bus map mapping BOTH old buses to the retained bus.
    Other component/branch indices can shift after cascading deletion/merge.

    This is a proposal mechanism for an outer discrete search, not a topology
    optimizer. After build(), create fresh trainable continuous tensors and reset
    optimizer state affected by changes in shape/order. Structural editing does
    not backpropagate through additions, removals or merges.
    """

    def __init__(self, mats: OPFMatrices, *, drop_extra=False):
        src = mat2opf(mats, include_solution=False)
        allowed = {key: {'x', 'num_nodes'} for key in _NODE_COLS}
        allowed.update({E_LINE: {'edge_index', 'edge_attr'},
                        E_XFMR: {'edge_index', 'edge_attr'}})
        allowed.update({et: {'edge_index'} for et in (*LINK_TYPES, *(et[::-1] for et in LINK_TYPES))})
        extra = [f'global.{k}' for k in dict(src.items()) if k != 'x']
        for key in list(src.node_types) + list(src.edge_types):
            if key not in allowed:
                extra.append(str(key))
            else:
                extra += [f'{key}.{attr}' for attr in src[key] if attr not in allowed[key]]
        if extra and not drop_extra:
            raise ValueError('Structural edits cannot remap unknown metadata: '
                             + ', '.join(extra) + '. Use drop_extra=True to discard it explicitly.')
        self.dropped_extra = tuple(extra)
        self._label_order = tuple(mats.meta['schema']['branch_y'])
        self._data = HeteroData()
        self._data.x = src.x.detach().clone()
        for key, cols in _NODE_COLS.items():
            if 'num_nodes' in src[key] and int(src[key].num_nodes) != src[key].x.reshape(-1, len(cols)).shape[0]:
                raise ValueError(f'{key}: explicit num_nodes contradicts feature rows')
            self._data[key].x = src[key].x.detach().clone().reshape(-1, len(cols))
        for kind, et in _BRANCH_TYPES.items():
            self._data[et].edge_attr = src[et].edge_attr.detach().clone().reshape(-1, len(_BRANCH_SCHEMA[kind]))
            self._data[et].edge_index = src[et].edge_index.detach().clone()
        for kind, et in _EQUIPMENT_LINK.items():
            bus_of = getattr(mats.params, LINK_TYPES[et]).detach().clone()
            if et[::-1] in src.edge_types:
                rev = src[et[::-1]].edge_index
                n = len(bus_of)
                if (rev.dtype != torch.int64 or rev.shape != (2, n)
                    or not torch.equal(rev[1].sort().values, torch.arange(n, device=rev.device))
                    or not torch.equal(rev[0], bus_of[rev[1]])):
                    raise ValueError(f'Inconsistent reverse equipment link: {et[::-1]}')
            self._set_links(kind, bus_of)
        validate_network(mats.params, require_connected=False, require_reference=False)

    def _set_links(self, kind, buses):
        et = _EQUIPMENT_LINK[kind]
        rows = torch.arange(len(buses), dtype=torch.int64, device=buses.device)
        edges = torch.stack([rows, buses])
        self._data[et].edge_index = edges
        self._data[et[::-1]].edge_index = edges.flip(0)

    def _bus(self, bus):
        return self._id(bus, self._data['bus'].x.shape[0], 'bus')

    @staticmethod
    def _id(index, size, kind):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < size:
            raise ValueError(f'{kind}: expected integer row in [0,{size}), got {index}')
        return index

    @staticmethod
    def _row(table, columns, values, defaults=None):
        values = {**(defaults or {}), **values}
        if set(values) != set(columns):
            raise ValueError(f'Missing fields {set(columns)-set(values)}; unknown fields {set(values)-set(columns)}')
        scalars = []
        for key in columns:
            value = torch.as_tensor(values[key], dtype=table.dtype, device=table.device).detach()
            if value.numel() != 1:
                raise ValueError(f'{key} must be scalar')
            scalars.append(value.reshape(()))
        return torch.stack(scalars).reshape(1, -1)

    def _append_equipment(self, kind, bus, values):
        self._bus(bus)
        table = self._data[kind].x
        defaults = {'load': {'qd': 0.0}, 'shunt': {'gs': 0.0, 'bs': 0.0},
                    'generator': {'mbase': self._data.x.item(), 'pg_set': 0.0,
                                  'qg_set': 0.0, 'vg': 1.0}}
        row = self._row(table, _NODE_COLS[kind], values, defaults[kind])
        index = table.shape[0]
        self._data[kind].x = torch.cat([table, row])
        et = _EQUIPMENT_LINK[kind]
        buses = self._data[et].edge_index[1]
        self._set_links(kind, torch.cat([buses, buses.new_tensor([bus])]))
        return index

    @_atomic_edit
    def add_bus(self, *, base_kv, vmin, vmax, bus_type=None,
                load=None, generator=None, shunt=None):
        """Append a bus, optionally with equipment dictionaries; returns bus index.

        generator requires pmin,pmax,qmin,qmax,c2,c1,c0; load requires pd (qd=0).
        Shunt defaults gs=bs=0. New bus defaults to type 2 with a generator, else 1.
        Add a connecting branch before build(). No synthetic solution is invented.
        """
        table = self._data['bus'].x
        kind = (2 if generator is not None else 1) if bus_type is None else bus_type
        row = self._row(table, BUS_X_COLS, dict(base_kv=base_kv, bus_type=kind, vmin=vmin, vmax=vmax))
        index = table.shape[0]
        self._data['bus'].x = torch.cat([table, row])
        for equipment, values in [('load', load), ('generator', generator), ('shunt', shunt)]:
            if values is not None:
                self._append_equipment(equipment, index, values)
        return index

    @_atomic_edit
    def add_load(self, bus, *, pd, qd=0.0):
        return self._append_equipment('load', bus, dict(pd=pd, qd=qd))

    @_atomic_edit
    def add_shunt(self, bus, *, gs=0.0, bs=0.0):
        return self._append_equipment('shunt', bus, dict(gs=gs, bs=bs))

    @_atomic_edit
    def add_generator(self, bus, *, pmin, pmax, qmin, qmax, c2, c1, c0,
                      mbase=None, pg_set=0.0, qg_set=0.0, vg=1.0):
        """Append a distinct generator; bus_type is unchanged. Costs use p.u. Pg."""
        return self._append_equipment('generator', bus, dict(
            pmin=pmin, pmax=pmax, qmin=qmin, qmax=qmax, c2=c2, c1=c1, c0=c0,
            mbase=self._data.x.item() if mbase is None else mbase,
            pg_set=pg_set, qg_set=qg_set, vg=vg))

    @_atomic_edit
    def add_branch(self, fr, to, *, br_r, br_x, rate_a, angmin, angmax,
                   kind='ac_line', b_fr=0.0, b_to=0.0, rate_b=None, rate_c=None,
                   tap=1.0, shift=0.0):
        """Append a branch; returns its index WITHIN the selected branch kind.

        Ratings and impedance are p.u., angles radians. Transformer tap is on
        the from side. rate_b/rate_c default to rate_a and are not used by OPF.
        """
        self._bus(fr); self._bus(to)
        if kind not in _BRANCH_TYPES:
            raise ValueError('kind must be ac_line or transformer')
        if kind == 'ac_line' and (tap != 1 or shift != 0):
            raise ValueError('Use kind=transformer for non-unit tap/phase shift')
        et, columns = _BRANCH_TYPES[kind], _BRANCH_SCHEMA[kind]
        values = dict(br_r=br_r, br_x=br_x, b_fr=b_fr, b_to=b_to, rate_a=rate_a,
                      rate_b=rate_a if rate_b is None else rate_b,
                      rate_c=rate_a if rate_c is None else rate_c, angmin=angmin, angmax=angmax)
        if kind == 'transformer':
            values.update(tap=tap, shift=shift)
        table, edges = self._data[et].edge_attr, self._data[et].edge_index
        index = table.shape[0]
        self._data[et].edge_attr = torch.cat([table, self._row(table, columns, values)])
        self._data[et].edge_index = torch.cat([edges, edges.new_tensor([[fr], [to]])], dim=1)
        return index

    @_atomic_edit
    def rewire_branch(self, index, fr, to, *, kind='ac_line'):
        """Keep oriented branch parameters unchanged, change its endpoint buses."""
        self._bus(fr); self._bus(to)
        et = _BRANCH_TYPES[kind]
        self._id(index, self._data[et].edge_attr.shape[0], kind)
        self._data[et].edge_index[:, index] = self._data[et].edge_index.new_tensor([fr, to])

    @_atomic_edit
    def move_equipment(self, kind, index, bus):
        """kind is generator/load/shunt; preserves individual equipment records."""
        self._bus(bus)
        et = _EQUIPMENT_LINK[kind]
        self._id(index, self._data[kind].x.shape[0], kind)
        buses = self._data[et].edge_index[1].clone()
        buses[index] = bus
        self._set_links(kind, buses)

    @_atomic_edit
    def set_values(self, kind, index, **values):
        """Update named features on one bus/generator/load/shunt/branch row.

        Example: set_values('generator', 0, pmax=2.0, qmin=-0.5, qmax=0.5).
        Branch index is local to kind='ac_line' or 'transformer'. No base changes
        or line-to-transformer reclassification are performed implicitly.
        """
        if kind in _NODE_COLS:
            table, columns = self._data[kind].x, _NODE_COLS[kind]
        elif kind in _BRANCH_TYPES:
            table, columns = self._data[_BRANCH_TYPES[kind]].edge_attr, _BRANCH_SCHEMA[kind]
        else:
            raise ValueError('Unknown component kind')
        self._id(index, table.shape[0], kind)
        row = {key: table[index, col] for col, key in enumerate(columns)}
        row.update(values)
        table[index] = self._row(table, columns, row)[0]

    @staticmethod
    def _mapping(mask):
        mapping = torch.full((len(mask),), -1, dtype=torch.int64, device=mask.device)
        mapping[mask] = torch.arange(int(mask.sum().item()), device=mask.device)
        return mapping

    def _keep_branches(self, kind, mask):
        et = _BRANCH_TYPES[kind]
        self._data[et].edge_attr = self._data[et].edge_attr[mask]
        self._data[et].edge_index = self._data[et].edge_index[:, mask]

    def _keep_equipment(self, kind, mask):
        et = _EQUIPMENT_LINK[kind]
        self._data[kind].x = self._data[kind].x[mask]
        self._set_links(kind, self._data[et].edge_index[1, mask])

    @_atomic_edit
    def remove_branch(self, index, *, kind='ac_line'):
        """True branch outage: removes its series and both charging contributions."""
        et = _BRANCH_TYPES[kind]
        count = self._data[et].edge_attr.shape[0]
        self._id(index, count, kind)
        mask = torch.ones(count, dtype=torch.bool, device=self._data.x.device)
        mask[index] = False
        mapping = self._mapping(mask)
        self._keep_branches(kind, mask)
        return mapping

    @_atomic_edit
    def remove_equipment(self, kind, index):
        """Delete one generator, load or shunt and its two attachment links."""
        if kind not in _EQUIPMENT_LINK:
            raise ValueError('kind must be generator, load or shunt')
        count = self._data[kind].x.shape[0]
        self._id(index, count, kind)
        mask = torch.ones(count, dtype=torch.bool, device=self._data.x.device)
        mask[index] = False
        mapping = self._mapping(mask)
        self._keep_equipment(kind, mask)
        return mapping

    def _delete_bus_row(self, bus):
        count = self._data['bus'].x.shape[0]
        mask = torch.ones(count, dtype=torch.bool, device=self._data.x.device)
        mask[bus] = False
        mapping = self._mapping(mask)
        self._data['bus'].x = self._data['bus'].x[mask]
        for et in _BRANCH_TYPES.values():
            self._data[et].edge_index = mapping[self._data[et].edge_index]
        for kind, et in _EQUIPMENT_LINK.items():
            self._set_links(kind, mapping[self._data[et].edge_index[1]])
        return mapping

    @_atomic_edit
    def remove_bus(self, bus, *, cascade=False):
        """Remove a bus. Default refuses attached equipment/branches.

        cascade=True explicitly deletes all incident branches and all equipment
        attached to this bus (including its demand and generation capacities).
        Surviving bus indices are renumbered; returns old->new bus mapping.
        """
        self._bus(bus)
        if self._data['bus'].x.shape[0] == 1:
            raise ValueError('Cannot remove the last bus')
        for kind, et in _BRANCH_TYPES.items():
            attached = (self._data[et].edge_index == bus).any(dim=0)
            if attached.any() and not cascade:
                raise ValueError('Bus has incident branches; use cascade=True or reconnect them')
            self._keep_branches(kind, ~attached)
        for kind, et in _EQUIPMENT_LINK.items():
            attached = self._data[et].edge_index[1] == bus
            if attached.any() and not cascade:
                raise ValueError('Bus has attached equipment; use cascade=True or move it')
            self._keep_equipment(kind, ~attached)
        return self._delete_bus_row(bus)

    @_atomic_edit
    def merge_buses(self, keep, remove):
        """Contract two adjacent equal-base-kV buses into one (NOT Kron reduction).

        Intersect voltage bounds, retain a reference designation if either bus
        has one, and relocate all equipment WITHOUT combining generators/costs.
        Keep all external branches, including parallel ones. Delete internal
        AC lines and move their total charging susceptance to an explicit shunt.
        Internal transformer branches, disjoint voltage bounds, or internal
        angle bounds excluding zero cause rejection. Bus metadata/setpoints
        otherwise follow the retained bus/equipment records.

        This is a changed OPF model: internal branch thermal constraints disappear.
        No claim of equivalent feasible sets, optimal costs or voltage solutions.
        """
        self._bus(keep); self._bus(remove)
        if keep == remove:
            raise ValueError('Choose two distinct buses')
        buses = self._data['bus'].x
        if not torch.isclose(buses[keep, 0], buses[remove, 0], rtol=1e-6, atol=0):
            raise ValueError('Merge requires equal base_kv; cross-voltage contraction is unsupported')
        low = torch.maximum(buses[keep, 2], buses[remove, 2])
        high = torch.minimum(buses[keep, 3], buses[remove, 3])
        if low > high:
            raise ValueError('Merged voltage bounds have empty intersection')
        internal = {}
        for kind, et in _BRANCH_TYPES.items():
            f, t = self._data[et].edge_index
            internal[kind] = ((f == keep) & (t == remove)) | ((f == remove) & (t == keep))
        if not any(mask.any().item() for mask in internal.values()):
            raise ValueError('Only adjacent buses can be merged')
        if internal['transformer'].any():
            raise ValueError('Cannot contract across a transformer; explicit remodelling is required')
        attrs = self._data[E_LINE].edge_attr[internal['ac_line']]
        if (attrs[:, 0] > 0).any() or (attrs[:, 1] < 0).any():
            raise ValueError('Internal line angle bounds exclude equal bus angles')
        charging = (attrs[:, 2] + attrs[:, 3]).sum()
        if charging != 0:
            self._append_equipment('shunt', keep, dict(gs=0.0, bs=charging))
        buses[keep, 2], buses[keep, 3] = low, high
        types = (buses[keep, 1].item(), buses[remove, 1].item())
        buses[keep, 1] = 3 if 3 in types else (2 if 2 in types else 1)
        for kind, et in _BRANCH_TYPES.items():
            self._keep_branches(kind, ~internal[kind])
            edges = self._data[et].edge_index
            self._data[et].edge_index = torch.where(edges == remove, keep, edges)
        for kind, et in _EQUIPMENT_LINK.items():
            attached = self._data[et].edge_index[1]
            self._set_links(kind, torch.where(attached == remove, keep, attached))
        mapping = self._delete_bus_row(remove)
        mapping[remove] = mapping[keep]
        return mapping

    def build(self, *, require_connected=True, require_reference=True) -> OPFMatrices:
        """Validate and snapshot edits into a fresh, unlabeled OPFMatrices.

        build does not solve OPF or ensure power adequacy/thermal feasibility.
        Caller must supply a valid reference designation; none is chosen silently.
        Returned tensors are independent of subsequent editor operations.
        """
        mats = opf2mat(self._data, branch_label_cols=self._label_order)
        validate_network(mats.params, require_connected=require_connected,
                         require_reference=require_reference)
        return mats


if __name__ == '__main__':
    import argparse
    from torch_geometric.datasets import OPFDataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--case', default='pglib_opf_case118_ieee')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--zero-rating', choices=['error', 'literal', 'unmonitored'], default='error')
    args = parser.parse_args()
    ds = OPFDataset(args.root, case_name=args.case, split='train', num_groups=1)
    data = ds[args.index]
    print('round trip exact:', check_roundtrip(data))
    print('candidate label errors:', infer_label_order(data))
    print('physical diagnostics:', residuals(opf2mat(data), zero_rating=args.zero_rating))
