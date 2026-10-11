"""Sparse float64 AC-OPF, analytic derivatives and PIPS interior-point steps."""
from copy import copy
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np
import torch
from pypower.makeYbus import makeYbus
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from ..problem import ACOPF, as_ppc
from ..solution import Solution


def digest(*arrays):
    value = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        value.update(str((a.shape, a.dtype.str)).encode())
        value.update(a.tobytes())
    return value.hexdigest()


def structural_signature(ppc):
    """Equipment order, connectivity, reference and constraint masks, not values."""
    b, g, e = (np.asarray(ppc[k]) for k in ("bus", "gen", "branch"))
    bounds = np.r_[b[:, 12], g[:, 9], g[:, 4]]
    upper = np.r_[b[:, 11], g[:, 8], g[:, 3]]
    al = (e[:, 11] != 0) & (e[:, 11] > -360)
    ah = (e[:, 12] != 0) & (e[:, 12] < 360)
    return digest(b[:, :2], g[:, 0], e[:, :2], e[:, 5] > 0,
                  bounds == upper, np.isfinite(bounds), np.isfinite(upper),
                  al, ah, al & ah & (e[:, 11] == e[:, 12]))


class SparseProblem:
    """CPU snapshot in original row order; preparation and checks stay sparse."""
    def __init__(self, network, cache=None):
        if isinstance(network, (ACOPF, SparseProblem)):
            self.ppc, self.template, self.label_columns = as_ppc(network.ppc)
            self.template = network.template
            self.label_columns = network.label_columns
        else:
            self.ppc, self.template, self.label_columns = as_ppc(network)
        self.base = float(self.ppc.get("baseMVA", 0))
        arrays = []
        for key, width in (("bus", 13), ("gen", 10), ("branch", 13), ("gencost", 5)):
            array = np.array(self.ppc.get(key, []), dtype=float, copy=True)
            if array.ndim != 2 or array.shape[1] < width or not np.isfinite(array).all():
                raise ValueError(f"{key} must be a finite matrix with at least {width} columns")
            self.ppc[key] = array.copy()
            arrays.append(array)
        bus, gen, br, gc = arrays
        self.nb, self.ng, self.nl = len(bus), len(gen), len(br)
        if not np.isfinite(self.base) or self.base <= 0 or not self.nb or not self.ng:
            raise ValueError("Need positive baseMVA, buses and generators")
        if set(self.ppc) & {"A", "N", "H", "Cw", "userfcn"}:
            raise ValueError("Extended PYPOWER objectives/constraints are not supported")
        if np.any(~np.isin(bus[:, 1], [1, 2, 3])) or np.any(gen[:, 7] <= 0) or np.any(br[:, 10] <= 0):
            raise ValueError("Remove offline equipment/buses before solving")
        if gen.shape[1] > 10 and np.any(gen[:, 10:16]):
            raise ValueError("Generator capability curves are not supported")
        refs = np.flatnonzero(bus[:, 1] == 3)
        if len(refs) != 1:
            raise ValueError("Expected exactly one reference bus")
        self.ref = int(refs[0])
        ids = bus[:, 0]
        if len(set(ids)) != self.nb or np.any(ids != np.floor(ids)) or np.any(ids < 0):
            raise ValueError("Bus IDs must be unique nonnegative integers")
        lookup = {value: index for index, value in enumerate(ids)}
        try:
            gen[:, 0] = [lookup[value] for value in gen[:, 0]]
            br[:, 0] = [lookup[value] for value in br[:, 0]]
            br[:, 1] = [lookup[value] for value in br[:, 1]]
        except KeyError as exc:
            raise ValueError(f"Equipment refers to unknown bus {exc.args[0]}") from exc
        bus[:, 0] = np.arange(self.nb)
        if np.any(br[:, 0] == br[:, 1]):
            raise ValueError("Self-loop branches are not supported")
        if np.any((br[:, 2] == 0) & (br[:, 3] == 0)) or np.any(br[:, 8] < 0) or np.any(br[:, 5] < 0):
            raise ValueError("Invalid impedance, tap or thermal rating")
        graph = coo_matrix((np.ones(self.nl), (br[:, 0].astype(int), br[:, 1].astype(int))),
                           shape=(self.nb, self.nb))
        if connected_components(graph, directed=False, return_labels=False) != 1:
            raise ValueError("Solve connected networks separately")
        if len(gc) != self.ng or np.any(gc[:, 0] != 2) or np.any(~np.isin(gc[:, 3], [1, 2, 3])):
            raise ValueError("Only polynomial active-generation costs of degree <=2 are supported")
        if np.any((gen[:, 8] == 0) & (gen[:, 9] < 0)):
            raise ValueError("PYPOWER dispatchable loads are not supported")
        coefficients = np.zeros((self.ng, 3))
        for index, row in enumerate(gc):
            count = int(row[3])
            if len(row) < 4 + count:
                raise ValueError("Missing polynomial cost coefficients")
            coefficients[index, -count:] = row[4:4 + count]
        self.ppc["bus"][self.ref, 8] = 0
        def tensor(a, dtype=torch.float64):
            return torch.as_tensor(np.array(a, copy=True), dtype=dtype)
        self.gen_bus = tensor(gen[:, 0], torch.long)
        self.f, self.t = tensor(br[:, 0], torch.long), tensor(br[:, 1], torch.long)
        self.rate, self.monitored = tensor(br[:, 5]/self.base), tensor(br[:, 5] > 0, torch.bool)
        self.c = tensor(coefficients)
        self.Sd = tensor((bus[:, 2] + 1j*bus[:, 3])/self.base, torch.complex128)
        amin = np.where((br[:, 11] != 0) & (br[:, 11] > -360), np.radians(br[:, 11]), -np.inf)
        amax = np.where((br[:, 12] != 0) & (br[:, 12] < 360), np.radians(br[:, 12]), np.inf)
        self.angle_lo, self.angle_hi = tensor(amin), tensor(amax)
        lo = np.r_[np.full(self.nb, -np.inf), bus[:, 12], gen[:, 9]/self.base, gen[:, 4]/self.base]
        hi = np.r_[np.full(self.nb, np.inf), bus[:, 11], gen[:, 8]/self.base, gen[:, 3]/self.base]
        lo[self.ref] = hi[self.ref] = 0
        if np.any(lo > hi) or np.any(amin > amax) or np.any(bus[:, 12] <= 0):
            raise ValueError("Invalid voltage, dispatch or angle bounds")
        self.lo, self.hi = tensor(lo), tensor(hi)
        self.signature = structural_signature(self.ppc)
        key = ("admittance", digest(self.base, bus[:, [0, 4, 5]], br[:, [0, 1, 2, 3, 4, 8, 9, 10]]))
        admittance = cache.get(key) if cache is not None else None
        self.admittance_cache_hit = admittance is not None
        if admittance is None:
            admittance = makeYbus(self.base, bus, br)
            if cache is not None:
                cache.put(key, admittance)
        self.Ybus, self.Yf, self.Yt = admittance

    def unpack(self, x):
        n, g = self.nb, self.ng
        return x[..., :n], x[..., n:2*n], x[..., 2*n:2*n+g], x[..., 2*n+g:]

    def cost(self, x):
        power = self.unpack(x)[2]*self.base
        return (self.c[:, 0]*power.square()+self.c[:, 1]*power+self.c[:, 2]).sum(-1)

    def _voltage(self, x):
        theta, vm, _, _ = self.unpack(x.detach().cpu().numpy())
        return vm*np.exp(1j*theta)

    def branch_flows(self, x):
        v = self._voltage(x)
        return (torch.from_numpy(v[self.f.numpy()]*(self.Yf@v).conj()),
                torch.from_numpy(v[self.t.numpy()]*(self.Yt@v).conj()))

    def balance(self, x):
        v = self._voltage(x)
        _, _, pg, qg = self.unpack(x.detach().cpu().numpy())
        s = -self.Sd.numpy()-v*(self.Ybus@v).conj()
        np.add.at(s, self.gen_bus.numpy(), pg+1j*qg)
        return torch.from_numpy(np.r_[s.real, s.imag])

    def angles(self, x):
        return x[self.f]-x[self.t]


@dataclass
class COO:
    row: torch.Tensor
    col: torch.Tensor
    shape: tuple

    def mv(self, values, x, transpose=False):
        row, col = (self.col, self.row) if transpose else (self.row, self.col)
        size = self.shape[int(transpose)]
        out = values.new_zeros((len(values), size))
        return out.index_add_(1, row, values * x[:, col])

    def dense(self, values):
        out = values.new_zeros((len(values), np.prod(self.shape)))
        out.index_add_(1, self.row * self.shape[1] + self.col, values)
        return out.reshape(len(values), *self.shape)


def local_power(x, indices, cross, diagonal=0):
    """Complex power and its first/second derivatives in [angle_i,angle_j,v_i,v_j]."""
    a, b, vi, vj = x[:, indices].unbind(-1)
    c = cross * torch.exp(torch.complex(torch.zeros_like(a), a - b))
    s = c * vi * vj
    di, dj = c * vj, c * vi
    zero = torch.zeros_like(s)
    jac = torch.stack((1j*s, -1j*s, di + 2*diagonal*vi, dj), -1)
    hess = torch.stack((-s, s, 1j*di, 1j*dj,
                        s, -s, -1j*di, -1j*dj,
                        1j*di, -1j*di, zero + 2*diagonal, c,
                        1j*dj, -1j*dj, c, zero), -1).reshape(*s.shape, 4, 4)
    return s + diagonal*vi.square(), jac, hess


class NetworkBatch:
    """One shared sparsity layout; loads, impedances, limits and costs may differ."""
    def tensor(self, array, dtype=torch.float64):
        return torch.as_tensor(np.asarray(array), dtype=dtype, device=self.device)

    def long(self, array):
        return self.tensor(array, torch.long)

    def __init__(self, models, device):
        self.device = torch.device(device)
        self.batch = len(models)
        first = models[0]
        self.nb, self.ng = n, ng = first.nb, first.ng
        self.nx = nx = 2*n + 2*ng
        self.ref = first.ref
        self.cost_mult = 1e-4

        def cpu(t):
            return t.detach().cpu().numpy()

        def tensor(a, dtype=torch.float64):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

        f, t, gb = (cpu(getattr(first, k)) for k in ("f", "t", "gen_bus"))
        if np.any(f == t):
            raise ValueError("Self-loop branches are not supported")
        mon = cpu(first.monitored)
        lo, hi = cpu(first.lo), cpu(first.hi)
        al, ah = cpu(first.angle_lo), cpu(first.angle_hi)
        fixed = lo == hi
        afixed = np.isfinite(al) & (al == ah)
        masks = (fixed, np.isfinite(lo), np.isfinite(hi), afixed,
                 np.isfinite(al), np.isfinite(ah))
        for m in models:
            if (m.nb, m.ng, m.nl, m.ref) != (n, ng, first.nl, first.ref):
                raise ValueError("Batch models must share dimensions and reference bus")
            for key in ("f", "t", "gen_bus", "monitored"):
                if not np.array_equal(cpu(getattr(m, key)), cpu(getattr(first, key))):
                    raise ValueError(f"Batch models differ in {key}; group by topology")
            ml, mh, mal, mah = (cpu(getattr(m, k)) for k in ("lo", "hi", "angle_lo", "angle_hi"))
            other = (ml == mh, np.isfinite(ml), np.isfinite(mh),
                     np.isfinite(mal) & (mal == mah), np.isfinite(mal), np.isfinite(mah))
            if any(not np.array_equal(a, b) for a, b in zip(masks, other)):
                raise ValueError("Batch bound masks differ; group these models separately")
            if m.ppc["gen"].shape[1] > 10 and np.any(m.ppc["gen"][:, 10:16]):
                raise ValueError("Generator capability curves are not supported")

        self.fixed = self.long(np.flatnonzero(fixed))
        self.pg = self.long(np.arange(2*n, 2*n+ng))

        # Include structural zeros so numeric cancellations cannot change the layout.
        keys = np.unique(np.r_[f*n+t, t*n+f, np.arange(n)*(n+1)])
        yr, yc = keys // n, keys % n
        self.yrow = self.long(yr)
        yi = np.stack((yr, yc, n+yr, n+yc), -1)
        self.yindices = self.long(yi)
        fi, ti = np.r_[f[mon], t[mon]], np.r_[t[mon], f[mon]]
        bi = np.stack((fi, ti, n+fi, n+ti), -1)
        self.bindices = self.long(bi)
        self.nflow = len(fi)
        self.yr, self.yc, self.f, self.t, self.mon = yr, yc, f, t, mon

        # Linear constraints: fixed variables/angles, then one-sided bounds.
        eq, iq, eb, ib = [], [], [], []
        def add(rows, rhs, columns, coefficients, values):
            rows.append((columns, coefficients))
            rhs.append(values)
        for j in np.flatnonzero(fixed):
            add(eq, eb, [j], [1.], ("lo", j, 1.))
        for j in np.flatnonzero(afixed):
            add(eq, eb, [f[j], t[j]], [1., -1.], ("angle_hi", j, 1.))
        for bounds, sign in (("hi", 1.), ("lo", -1.)):
            mask = np.isfinite(cpu(getattr(first, bounds))) & ~fixed
            for j in np.flatnonzero(mask):
                add(iq, ib, [j], [sign], (bounds, j, sign))
        for bounds, sign in (("angle_hi", 1.), ("angle_lo", -1.)):
            mask = np.isfinite(cpu(getattr(first, bounds))) & ~afixed
            for j in np.flatnonzero(mask):
                add(iq, ib, [f[j], t[j]], [sign, -sign],
                    (bounds, j, sign))
        def linear(rows, rhs):
            rr, cc, vv, gr, gc, gv, gi = [], [], [], [], [], [], []
            for i, (cols, vals) in enumerate(rows):
                rr.extend([i]*len(cols))
                cc.extend(cols)
                vv.extend(vals)
                for j, v in zip(cols, vals):
                    for k, w in zip(cols, vals):
                        gr.append(j)
                        gc.append(k)
                        gv.append(v*w)
                        gi.append(i)
            coo = COO(self.long(rr), self.long(cc), (len(rows), nx))
            values = tensor(vv).expand(self.batch, -1)
            target = rhs
            return coo, values, target, (gr, gc, gv, gi)
        self.ae, self.aev, self.eq_spec, _ = linear(eq, eb)
        self.ai, self.aiv, self.iq_spec, gram = linear(iq, ib)
        self.neq, self.niq = 2*n+len(eq), self.nflow+len(iq)
        self.gram_values, self.gram_rows = tensor(gram[2]), self.long(gram[3])

        jr = np.r_[np.repeat(yr, 4), np.repeat(yr+n, 4), gb, gb+n, 2*n+cpu(self.ae.row)]
        jc = np.r_[yi.ravel(), yi.ravel(), np.arange(2*n, nx), cpu(self.ae.col)]
        self.jg = COO(self.long(jr), self.long(jc), (self.neq, nx))
        hr = np.r_[np.repeat(np.arange(self.nflow), 4), self.nflow+cpu(self.ai.row)]
        hc = np.r_[bi.ravel(), cpu(self.ai.col)]
        self.jh = COO(self.long(hr), self.long(hc), (self.niq, nx))
        def pairs(indices):
            return np.repeat(indices, 4, axis=1).ravel(), np.tile(indices, (1, 4)).ravel()
        hyr, hyc = pairs(yi)
        hbr, hbc = pairs(bi)
        self.hessian = COO(self.long(np.r_[hyr, hbr, np.arange(2*n, 2*n+ng)]),
                           self.long(np.r_[hyc, hbc, np.arange(2*n, 2*n+ng)]), (nx, nx))
        self.gram = COO(self.long(np.r_[hbr, gram[0]]), self.long(np.r_[hbc, gram[1]]), (nx, nx))
        self.bind(models)

    def bind(self, models):
        """Refresh every scenario value while retaining only the sparse layout."""
        self.batch = len(models)
        def stack(name):
            return np.stack([getattr(m, name).detach().cpu().numpy() for m in models])
        self.lo, self.hi, self.cost = (self.tensor(stack(k)) for k in ("lo", "hi", "c"))
        self.base = self.tensor([m.base for m in models])[:, None]
        demand = stack("Sd")
        self.demand = self.tensor(np.concatenate((demand.real, demand.imag), axis=1))
        self.rates = self.tensor(np.tile(stack("rate")[:, self.mon], (1, 2)))
        self.aev = self.aev[:1].expand(self.batch, -1)
        self.aiv = self.aiv[:1].expand(self.batch, -1)
        bound_values = {name: stack(name) for name in ("lo", "hi", "angle_lo", "angle_hi")}
        def bounds(spec):
            columns = [sign*bound_values[name][:, j] for name, j, sign in spec]
            return self.tensor(np.stack(columns, axis=1) if columns else np.empty((self.batch, 0)))
        self.be, self.bi = bounds(self.eq_spec), bounds(self.iq_spec)
        rows, f, t = np.flatnonzero(self.mon), self.f[self.mon], self.t[self.mon]
        def entries(matrix, r, c):
            return np.asarray(matrix[r, c]).ravel() if len(r) else np.empty(0, complex)
        y, cross, diagonal = [], [], []
        for m in models:
            if isinstance(m, SparseProblem):
                ybus, yf, yt = m.Ybus, m.Yf, m.Yt
            else:
                bus, branch = m.ppc["bus"].copy(), m.ppc["branch"].copy()
                bus[:, 0], branch[:, 0], branch[:, 1] = np.arange(self.nb), self.f, self.t
                ybus, yf, yt = makeYbus(m.base, bus, branch)
            y.append(entries(ybus, self.yr, self.yc).conj())
            cross.append(np.r_[entries(yf, rows, t), entries(yt, rows, f)].conj())
            diagonal.append(np.r_[entries(yf, rows, f), entries(yt, rows, t)].conj())
        self.y, self.bcross, self.bdiag = (self.tensor(np.stack(a), torch.complex128)
                                         for a in (y, cross, diagonal))

    def initial(self, x0=None):
        lower = self.lo.masked_fill(~self.lo.isfinite(), -1e10)
        upper = self.hi.masked_fill(~self.hi.isfinite(), 1e10)
        x = (lower + upper)/2
        x[:, :self.nb] = 0.
        if x0 is not None:
            x = torch.as_tensor(x0, dtype=torch.float64, device=self.device).clone()
            if x.shape != self.lo.shape or not torch.isfinite(x).all():
                raise ValueError("x0 must be finite and have shape (batch, variables)")
            x = torch.maximum(self.lo, torch.minimum(self.hi, x))
        x[:, self.fixed] = self.lo[:, self.fixed]
        return x

    def evaluate(self, x):
        s, dy, hy = local_power(x, self.yindices, self.y)
        sf, ds, hs = local_power(x, self.bindices, self.bcross, self.bdiag)
        flow_jac = 2*(sf.conj()[..., None]*ds).real
        flow_hess = 2*(ds.conj()[..., :, None]*ds[..., None, :] + sf.conj()[..., None, None]*hs).real
        g = self.demand.clone()
        g[:, :self.nb].index_add_(1, self.yrow, s.real)
        g[:, self.nb:].index_add_(1, self.yrow, s.imag)
        # Generator entries in Jg have derivative -1.
        offset = 8*self.yindices.shape[0]
        grows = self.jg.row[offset:offset+2*self.ng]
        g.index_add_(1, grows, -x[:, 2*self.nb:])
        g = torch.cat((g, self.ae.mv(self.aev, x)-self.be), -1)
        h = torch.cat((sf.abs().square()-self.rates.square(), self.ai.mv(self.aiv, x)-self.bi), -1)
        jg = torch.cat((dy.real.flatten(1), dy.imag.flatten(1),
                        x.new_full((self.batch, 2*self.ng), -1), self.aev), -1)
        jh = torch.cat((flow_jac.flatten(1), self.aiv), -1)
        power = x[:, self.pg]*self.base
        c = self.cost
        objective = (c[..., 0]*power.square()+c[..., 1]*power+c[..., 2]).sum(-1)
        grad = torch.zeros_like(x)
        grad[:, self.pg] = (2*c[..., 0]*power+c[..., 1])*self.base*self.cost_mult
        return dict(f=objective*self.cost_mult, grad=grad, g=g, h=h, jg=jg, jh=jh,
                    hy=hy, flow_hess=flow_hess, flow_jac=flow_jac)

    def curvature(self, data, lam, mu, slack):
        weights = torch.complex(lam[:, self.yrow], -lam[:, self.yrow+self.nb])
        hy = (weights[..., None, None]*data["hy"]).real.flatten(1)
        hb = (mu[:, :self.nflow, None, None]*data["flow_hess"]).flatten(1)
        hc = 2*self.cost[..., 0]*self.base.square()*self.cost_mult
        hessian = torch.cat((hy, hb, hc), -1)
        d = mu/slack
        jac = data["flow_jac"]
        gram = (d[:, :self.nflow, None, None]*jac[..., :, None]*jac[..., None, :]).flatten(1)
        linear = d[:, self.nflow+self.gram_rows]*self.gram_values
        return hessian, torch.cat((gram, linear), -1)


class KKT:
    def __init__(self, network, backend, memory_gib, residual_tol):
        self.n = network.nx + network.neq
        self.backend, self.tol = backend, residual_tol
        h, gram, j = network.hessian, network.gram, network.jg
        rows = torch.cat((h.row, gram.row, network.nx+j.row, j.col))
        cols = torch.cat((h.col, gram.col, j.col, network.nx+j.row))
        keys, inverse = np.unique((rows*self.n+cols).cpu().numpy(), return_inverse=True)
        self.inverse = network.long(inverse)
        self.pattern = COO(network.long(keys//self.n), network.long(keys % self.n), (self.n, self.n))
        self.nnz = len(keys)
        self.indices = keys % self.n
        self.indptr = np.r_[0, np.cumsum(np.bincount(keys//self.n, minlength=self.n))]
        self.dense_bytes_per_sample = 3*8*self.n*self.n
        self.chunk = int(memory_gib*1024**3 // self.dense_bytes_per_sample)
        if backend == "dense" and self.chunk < 1:
            needed = self.dense_bytes_per_sample/1024**3
            raise ValueError(f"Dense KKT needs an estimated {needed:.2f} GiB/sample; increase max_dense_gib")

    def assemble(self, hessian, gram, jac):
        values = torch.cat((hessian, gram, jac, jac), -1)
        return values.new_zeros((len(values), self.nnz)).index_add_(1, self.inverse, values)

    def solve(self, values, rhs, active):
        """Check backward residuals; a failed member does not abort other members."""
        answer = torch.zeros_like(rhs)
        error = rhs.new_full((len(rhs),), math.inf)
        chosen = active.nonzero().flatten()
        if self.backend == "sparse":
            for i in chosen.tolist():
                matrix = csr_matrix((values[i].numpy(), self.indices, self.indptr), shape=(self.n, self.n))
                b = rhs[i].numpy()
                try:
                    factor = splu(matrix.tocsc())
                    x = factor.solve(b)
                    for _ in range(2):
                        residual = b-matrix@x
                        relative = np.linalg.norm(residual, np.inf)/(1+np.linalg.norm(b, np.inf))
                        if relative <= self.tol:
                            break
                        x += factor.solve(residual)
                    answer[i] = torch.from_numpy(x)
                    error[i] = np.linalg.norm(b-matrix@x, np.inf)/(1+np.linalg.norm(b, np.inf))
                except RuntimeError:
                    pass
        else:
            for chunk in chosen.split(self.chunk):
                matrix = self.pattern.dense(values[chunk])
                b = rhs[chunk, :, None]
                factor, pivots, info = torch.linalg.lu_factor_ex(matrix, check_errors=False)
                x = torch.linalg.lu_solve(factor, pivots, b)
                for _ in range(2):
                    x = x + torch.linalg.lu_solve(factor, pivots, b-matrix@x)
                residual = (b-matrix@x).abs().amax(dim=(1, 2))/(1+b.abs().amax(dim=(1, 2)))
                error[chunk] = torch.where(info == 0, residual, math.inf)
                answer[chunk] = x.squeeze(-1)
        valid = torch.isfinite(answer).all(-1) & (error <= self.tol)
        return answer, valid, error


def synchronize(device):
    if device.type == "xpu":
        torch.xpu.synchronize(device)


def maximum(x):
    return x.abs().amax(-1) if x.shape[-1] else x.new_zeros(len(x))


class XPUSolver:
    """solve(model, x0=None) or solve_batch(models, x0=None) -> Solution(s).

Batch members need the same equipment order and finite/fixed constraint masks.
Initial points are optional primal vectors or Solutions; defaults use no labels.
"""
    name = "xpu_pips"

    def __init__(self, *, device="cpu", linear_solver="auto", max_iter=150,
                 feasibility_tol=1e-6, optimality_tol=1e-8, cost_tol=1e-9,
                 linear_tol=1e-7, max_dense_gib=4., history=False, verbose=False,
                 time_limit=None, cache=None):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "xpu"):
            raise ValueError("device must be cpu or xpu[:index]")
        if self.device.type == "xpu" and not torch.xpu.is_available():
            raise RuntimeError("XPU requested but unavailable; run on an Aurora compute node")
        if linear_solver == "auto":
            linear_solver = "sparse" if self.device.type == "cpu" else "dense"
        if linear_solver not in ("sparse", "dense"):
            raise ValueError("linear_solver must be auto, sparse or dense")
        if linear_solver == "sparse" and self.device.type != "cpu":
            raise ValueError("Sparse LU is CPU-only; XPU uses dense device LU without CPU fallback")
        if max_iter < 0 or int(max_iter) != max_iter:
            raise ValueError("max_iter must be a nonnegative integer")
        if any(not math.isfinite(v) or v <= 0 for v in
               (feasibility_tol, optimality_tol, cost_tol, linear_tol, max_dense_gib)):
            raise ValueError("Tolerances and memory limit must be positive and finite")
        self.linear_solver, self.max_iter = linear_solver, max_iter
        self.feasibility_tol, self.optimality_tol = feasibility_tol, optimality_tol
        self.cost_tol, self.linear_tol = cost_tol, linear_tol
        self.max_dense_gib, self.history, self.verbose = max_dense_gib, history, verbose
        if time_limit is not None and (not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be positive seconds or None")
        self.time_limit, self.cache = time_limit, cache

    def solve(self, network, x0=None):
        return self.solve_batch([network], None if x0 is None else [x0])[0]

    @torch.no_grad()
    def solve_batch(self, networks, x0=None):
        if not networks:
            return []
        start = time.perf_counter()
        models = [m if isinstance(m, SparseProblem) else SparseProblem(m, self.cache) for m in networks]
        if any(m.signature != models[0].signature for m in models):
            raise ValueError("Batch structures differ; use SolverPool for mixed topologies")
        key = ("layout", models[0].signature, str(self.device), self.linear_solver,
               self.max_dense_gib, self.linear_tol)
        cached = self.cache.get(key) if self.cache is not None else None
        layout_hit = cached is not None
        if cached is None:
            template = NetworkBatch(models[:1], self.device)
            kkt = KKT(template, self.linear_solver, self.max_dense_gib, self.linear_tol)
            cached = (template, kkt)
            if self.cache is not None:
                self.cache.put(key, cached)
        net, kkt = copy(cached[0]), cached[1]
        net.bind(models)
        x = net.initial()
        starts = [None]*len(models) if x0 is None else list(x0)
        if len(starts) != len(models):
            raise ValueError("One initial point is required per batch member")
        for i, value in enumerate(starts):
            if value is None:
                continue
            value = value.x() if isinstance(value, Solution) else torch.as_tensor(value)
            value = value.to(dtype=torch.float64, device=self.device).detach()
            if value.shape != x[i].shape or not torch.isfinite(value).all():
                raise ValueError("x0 must be a finite vector matching the problem size")
            x[i] = value.clamp(min=net.lo[i], max=net.hi[i])
        data = net.evaluate(x)
        slack = torch.maximum(-data["h"], torch.ones_like(data["h"]))
        mu, lam = torch.ones_like(slack), torch.zeros_like(data["g"])
        gamma = x.new_ones(net.batch)
        active = torch.ones(net.batch, dtype=torch.bool, device=self.device)
        converged = torch.zeros_like(active)
        iterations = torch.zeros(net.batch, dtype=torch.int64, device=self.device)
        messages = ["iteration limit"]*net.batch
        previous_cost = data["f"].clone()
        linear_error = x.new_zeros(net.batch)
        trace = []
        synchronize(self.device)
        setup_s = time.perf_counter()-start
        solve_start = time.perf_counter()
        finished = [None]*net.batch
        for iteration in range(self.max_iter+1):
            lag_grad = data["grad"] + net.jg.mv(data["jg"], lam, True) + net.jh.mv(data["jh"], mu, True)
            primal = torch.maximum(maximum(data["g"]), maximum(data["h"].clamp_min(0)))
            grad = maximum(lag_grad)/(1+torch.maximum(maximum(lam), maximum(mu)))
            comp = (slack*mu).sum(-1)/(1+maximum(x))
            cost = (data["f"]-previous_cost).abs()/(1+previous_cost.abs())
            finite = torch.isfinite(x).all(-1) & torch.isfinite(primal+grad+comp+cost)
            done = active & finite & (primal <= self.feasibility_tol) & (grad <= self.optimality_tol)
            done &= (comp <= self.optimality_tol) & (cost <= self.cost_tol)
            converged |= done
            iterations[active] = iteration
            for i in done.nonzero().flatten().tolist():
                messages[i] = "converged"
            failed = active & ~finite
            for i in failed.nonzero().flatten().tolist():
                messages[i] = "nonfinite iterate"
            active &= ~(done | failed)
            if self.time_limit is not None and time.perf_counter()-start >= self.time_limit:
                for i in active.nonzero().flatten().tolist():
                    messages[i] = "elapsed time limit"
                active[:] = False
            for i in (~active).nonzero().flatten().tolist():
                if finished[i] is None:
                    finished[i] = time.perf_counter()-solve_start
            if self.history:
                trace.append(dict(iteration=iteration, objective=(data["f"]/net.cost_mult).cpu().tolist(),
                                  primal=primal.cpu().tolist(), gradient=grad.cpu().tolist(),
                                  complementarity=comp.cpu().tolist()))
            if self.verbose:
                print(f"{iteration:3d} active={int(active.sum())} balance/limits={float(primal.max()):.3e} "
                      f"stationarity={float(grad.max()):.3e} complementarity={float(comp.max()):.3e}", flush=True)
            if not active.any() or iteration == self.max_iter:
                break
            previous_cost = data["f"].clone()
            hessian, gram = net.curvature(data, lam, mu, slack)
            values = kkt.assemble(hessian, gram, data["jg"])
            correction = (mu*data["h"]+gamma[:, None])/slack
            rhs = -torch.cat((lag_grad+net.jh.mv(data["jh"], correction, True), data["g"]), -1)
            step, valid, error = kkt.solve(values, rhs, active)
            linear_error[active] = error[active]
            failed = active & ~valid
            for i in failed.nonzero().flatten().tolist():
                messages[i] = "KKT factorization/residual failed"
            active &= valid
            dx, dl = step[:, :net.nx], step[:, net.nx:]
            dz = -data["h"]-slack-net.jh.mv(data["jh"], dx)
            dm = -mu+(gamma[:, None]-mu*dz)/slack
            def fraction(value, delta):
                if not value.shape[-1]:
                    return value.new_ones(len(value))
                ratios = torch.where(delta < 0, -value/delta, math.inf)
                return (0.99995*ratios.amin(-1)).clamp(max=1.)
            ap, ad = fraction(slack, dz), fraction(mu, dm)
            good_step = torch.isfinite(ap+ad) & (ap >= 1e-8) & (ad >= 1e-8)
            for i in (active & ~good_step).nonzero().flatten().tolist():
                messages[i] = "step too small"
            active &= good_step
            x = torch.where(active[:, None], x+ap[:, None]*dx, x)
            slack = torch.where(active[:, None], slack+ap[:, None]*dz, slack)
            lam = torch.where(active[:, None], lam+ad[:, None]*dl, lam)
            mu = torch.where(active[:, None], mu+ad[:, None]*dm, mu)
            gamma = 0.1*(slack*mu).sum(-1)/max(1, net.niq)
            data = net.evaluate(x)
        synchronize(self.device)
        solve_s = time.perf_counter()-solve_start
        results = []
        for i, m in enumerate(models):
            info = dict(iterations=int(iterations[i]), message=messages[i], device=str(self.device),
                        linear_solver=self.linear_solver, setup_batch_s=setup_s, solve_batch_s=solve_s,
                        batch_size=net.batch, kkt_size=kkt.n, kkt_nnz=kkt.nnz,
                        primal_residual=float(primal[i]), stationarity=float(grad[i]),
                        complementarity=float(comp[i]), linear_residual=float(linear_error[i]),
                        warm_start=starts[i] is not None, layout_cache_hit=layout_hit,
                        admittance_cache_hit=m.admittance_cache_hit,
                        solve_s=finished[i] if finished[i] is not None else solve_s)
            if self.history:
                info["history"] = [{k: (v[i] if isinstance(v, list) else v) for k, v in row.items()} for row in trace]
            validation_start = time.perf_counter()
            result = Solution.from_x(x[i].cpu(), m, bool(converged[i]), self.name,
                                     tol=self.feasibility_tol, **info)
            result.info["validation_s"] = time.perf_counter()-validation_start
            results.append(result)
        return results


class SparseSolver(XPUSolver):
    """CPU-only production solver; XPUSolver retains the historical device checks."""
    name = "sparse_pips"

    def __init__(self, **options):
        if options.pop("device", "cpu") != "cpu" or options.pop("linear_solver", "sparse") != "sparse":
            raise ValueError("SparseSolver uses CPU sparse factorization")
        super().__init__(device="cpu", linear_solver="sparse", **options)
