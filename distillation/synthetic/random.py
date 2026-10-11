"""Fresh connected topologies and empirical equipment draws in OPFData units.

This is a proposal distribution, not a model of real-grid geography or an OPF
feasibility certificate. No source edges or solution labels are copied.
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from distillation.conversion.matrix_form import (BUS_X_COLS,
                                                 GEN_X_COLS,
                                                 LOAD_X_COLS,
                                                 SHUNT_X_COLS,
                                                 LINE_ATTR_COLS,
                                                 XFMR_ATTR_COLS,
                                                 E_LINE,
                                                 E_XFMR,
                                                 LINK_TYPES,
                                                 opf2mat,
                                                 validate_network
                                                 )
from .topology import bus_limits


def sample_sizes(count, original_buses, *, steps=1, seed=42, bus_range=None):
    """Log-spaced sizes covering both hard limits in EACH equal-size batch."""
    low, high = bus_limits(original_buses)
    if bus_range is not None:
        if (len(bus_range) != 2 or any(int(n) != n for n in bus_range)
                or not low <= bus_range[0] <= original_buses <= bus_range[1] <= high):
            raise ValueError("Invalid bus_range")
        low, high = map(int, bus_range)
    
    if count < 1 or int(count) != count or steps < 1 or int(steps) != steps or count % steps:
        raise ValueError("Need a positive count forming equal-size batches")
    
    count, steps = int(count), int(steps)
    
    width = count // steps
    sizes = [original_buses] if width == 1 else np.rint(np.geomspace(low, high, width)).astype(int)
    rng = np.random.default_rng(seed)
    
    return np.concatenate([rng.permutation(sizes) for _ in range(steps)]).tolist()


def random_network(reference, n_buses, *, original_buses=None, seed=42,
                   spread=0.5, density=(1.0, 1.6), placement="electrical"):
    """Propose an unlabeled grid of exactly n_buses; the caller must solve it.

    A random tree within each voltage level plus a random tree between levels
    guarantees connectedness. Random chords vary cycle count. Branch/equipment
    tuples are bootstrapped, retaining R/X, P/Q and bound ordering, then jittered
    by factors in [1/(1+spread), 1+spread]. Reference voltage levels and transformer
    orientations are retained; equipment placement and branch endpoints are new.
    Electrical placement favors stronger buses for larger equipment; uniform
    placement is available as an ablation. Neither guarantees feasibility.
    """
    
    # conversion
    p = opf2mat(reference.clone().detach().cpu()).params
    anchor = p.N if original_buses is None else original_buses
    low, high = bus_limits(anchor)
    
    # validate network attrs
    if int(n_buses) != n_buses or not max(2, low) <= n_buses <= high:
        raise ValueError(f"Random bus count must be an integer in [{max(2, low)}, {high}]")
    if not np.isfinite(spread) or not 0 <= spread < 1:
        raise ValueError("Use 0 <= spread < 1")
    if len(density) != 2 or not np.isfinite(density).all() or not 0 < density[0] <= density[1]:
        raise ValueError("density must be a positive ordered pair")
    if placement not in ("electrical", "uniform"):
        raise ValueError("placement must be electrical or uniform")
    validate_network(p)
    
    # check
    if not p.G or not p.L or not p.n_line or p.pd.sum() <= 0 or p.pmax.sum() <= 0:
        raise ValueError("Reference needs lines, positive demand and generation capacity")
    n = int(n_buses)
    rng = np.random.default_rng(seed)
    values = {name: value.numpy() for name, value in vars(p).items() if isinstance(value, torch.Tensor)}

    def factor(size=None):
        return np.exp(rng.uniform(-np.log1p(spread), np.log1p(spread), size))

    def table(columns, indices):
        return np.column_stack([values[name][indices] for name in columns]).copy()

    kv, frequency = np.unique(values["base_kv"], return_counts=True)
    if n < len(kv):
        raise ValueError("Requested size cannot retain all reference voltage levels")
    
    # Each observed level survives; the remaining buses follow empirical proportions.
    allocation = 1 + rng.multinomial(n - len(kv), frequency / p.N)
    bus_rows = np.concatenate([rng.choice(np.flatnonzero(values["base_kv"] == level), size)
                               for level, size in zip(kv, allocation)])
    rng.shuffle(bus_rows)
    bus = table(BUS_X_COLS, bus_rows)
    groups = {level: np.flatnonzero(bus[:, 0] == level) for level in kv}
    center, half = (bus[:, 2] + bus[:, 3]) / 2, (bus[:, 3] - bus[:, 2]) / 2 * factor(n)
    bus[:, 2], bus[:, 3] = center - half, center + half
    bus[:, 1] = 1
    fr_kv, to_kv = values["base_kv"][values["fr"]], values["base_kv"][values["to"]]
    if np.any(fr_kv[:p.n_line] != to_kv[:p.n_line]):
        raise ValueError("Random initialization requires same-voltage AC lines in the reference")
    branches, occupied = [], set()
    borrowed_line_rows = 0

    def connect(fr, to, row):
        pair = tuple(sorted((int(fr), int(to))))
        if fr == to or pair in occupied:
            return False
        occupied.add(pair)
        branches.append((int(fr), int(to), int(row)))
        return True

    for level, buses in groups.items():
        order = rng.permutation(buses)
        rows = np.flatnonzero(fr_kv[:p.n_line] == level)
        if not len(rows):
            rows = np.arange(p.n_line)
            borrowed_line_rows += len(buses) - 1
        for i in range(1, len(order)):
            connect(order[i], rng.choice(order[:i]), rng.choice(rows))

    # Random spanning tree of voltage levels, using only observed transformer pairs.
    parent = {level: level for level in kv}

    def root(level):
        while parent[level] != level:
            level = parent[level]
        return level

    for row in rng.permutation(np.arange(p.n_line, p.E)):
        a, b = fr_kv[row], to_kv[row]
        if root(a) != root(b):
            connect(rng.choice(groups[a]), rng.choice(groups[b]), row)
            parent[root(a)] = root(b)
    if len({root(level) for level in kv}) != 1:
        raise ValueError("Reference transformer voltage levels are disconnected")

    density_scale = rng.uniform(*density)
    target_edges = min(n * (n - 1) // 2, max(n - 1, round(p.E / p.N * n * density_scale)))
    target_transformers = max(sum(row >= p.n_line for _, _, row in branches),
                              round((p.E - p.n_line) / p.N * n * density_scale))
    for draw in range(30 * target_edges):
        if len(branches) == target_edges:
            break
        have_transformers = sum(row >= p.n_line for _, _, row in branches)
        rows = np.arange(p.n_line, p.E) if have_transformers < target_transformers else np.arange(p.n_line)
        if draw >= 10 * target_edges:
            rows = np.arange(p.E)  # A saturated branch kind may borrow the other's remaining pairs.
        row = int(rng.choice(rows))
        connect(rng.choice(groups[fr_kv[row]]), rng.choice(groups[to_kv[row]]), row)
    if len(branches) != target_edges:
        raise ValueError("Cannot fill requested branch density with compatible distinct endpoints")

    data = HeteroData()
    data.x = p.base_mva.detach().clone().double()
    support = np.zeros(n)
    for a, b, row in branches:
        support[[a, b]] += 1 / np.hypot(values["br_r"][row], values["br_x"][row])
    sizes, rescaling = {}, {}
    for kind, columns, index_name, source_count in (
        ("generator", GEN_X_COLS, "gen_bus", p.G),
        ("load", LOAD_X_COLS, "load_bus", p.L),
        ("shunt", SHUNT_X_COLS, "shunt_bus", p.S),
    ):
        count = max(int(kind != "shunt"), round(source_count * n / p.N * rng.uniform(0.85, 1.15)))
        rows = rng.choice(source_count, count) if source_count else np.array([], dtype=int)
        if kind == "generator" and not np.any(values["pmax"][rows] > 0):
            rows[0] = rng.choice(np.flatnonzero(values["pmax"] > 0))
        x = table(columns, rows)
        # Avoid artificial concentration: use distinct buses within each voltage
        # level until they are exhausted, then start another shuffled placement.
        levels = values["base_kv"][values[index_name][rows]]
        positions = np.empty(count, dtype=int)
        for level in np.unique(levels):
            slots = np.flatnonzero(levels == level)
            buses = groups[level]
            order = []
            for _ in range((len(slots) - 1) // len(buses) + 1):
                ranking = (rng.permutation(buses) if placement == "uniform" else
                           buses[np.argsort(-(np.log(support[buses]) + 0.5 * rng.gumbel(size=len(buses))))])
                order.extend(ranking)
            if placement == "electrical":
                demand = np.abs(x[:, columns.index("pmax")]) if kind == "generator" else np.linalg.norm(x, axis=1)
                slots = slots[np.argsort(-demand[slots], kind="stable")]
            positions[slots] = order[:len(slots)]
        scale = factor(count)
        names = ("pmin", "pmax", "qmin", "qmax") if kind == "generator" else columns
        for name in names:
            x[:, columns.index(name)] *= scale
        if kind in ("generator", "load"):
            key = "pmax" if kind == "generator" else "pd"
            total = x[:, columns.index(key)].sum()
            if total <= 0:
                raise ValueError(f"Resampled {kind} has no positive {key}")
            normalization = values[key].sum() * n / p.N * factor() / total
            if not 0.25 <= normalization <= 4:
                raise ValueError(f"Resampled {kind} needs excessive capacity/demand rescaling")
            for name in names:
                x[:, columns.index(name)] *= normalization
            rescaling[kind] = float(normalization)
        if kind == "generator":
            x[:, columns.index("pg_set")] = 0
            x[:, columns.index("qg_set")] = 0
            for name in ("c2", "c1", "c0"):
                x[:, columns.index(name)] *= factor(count)
            bus[positions, 1] = 2
            bus[positions[x[:, columns.index("pmax")].argmax()], 1] = 3
        data[kind].x = torch.tensor(x, dtype=torch.float64)
        et = next(et for et in LINK_TYPES if et[0] == kind)
        edges = torch.tensor(np.stack((np.arange(count), positions)), dtype=torch.long)
        data[et].edge_index, data[et[::-1]].edge_index = edges, edges.flip(0)
        sizes[f"{kind}s"] = count
    data["bus"].x = torch.tensor(bus, dtype=torch.float64)
    for et, columns, is_line in ((E_LINE, LINE_ATTR_COLS, True), (E_XFMR, XFMR_ATTR_COLS, False)):
        selected = [(a, b, row) for a, b, row in branches if (row < p.n_line) == is_line]
        rows = np.array([row for _, _, row in selected], dtype=int)
        x = table(columns, rows)
        length, rating = factor(len(rows)), factor(len(rows))
        for name in ("br_r", "br_x", "b_fr", "b_to"):
            x[:, columns.index(name)] *= length
        for name in ("rate_a", "rate_b", "rate_c"):
            x[:, columns.index(name)] *= rating
        data[et].edge_index = torch.tensor([(a, b) for a, b, _ in selected], dtype=torch.long).reshape(-1, 2).T
        data[et].edge_attr = torch.tensor(x, dtype=torch.float64)
    matrices = opf2mat(data)
    validate_network(matrices.params)
    return matrices, dict(seed=int(seed), requested_buses=n, buses=n, branches=len(branches),
                          density_scale=float(density_scale), borrowed_line_rows=borrowed_line_rows,
                          placement=placement, normalization=rescaling,
                          reference_buses=p.N, reference_demand_pu=float(p.pd.sum()),
                          reference_capacity_pu=float(p.pmax.sum()),
                          sampled_demand_pu=float(matrices.params.pd.sum()), **sizes)
