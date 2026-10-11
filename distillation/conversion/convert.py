"""Conversion between torch_geometrics's HeteroData annd AC-OPF solver's PYPOWER format

DETAILS:
(OPFData-native format via torch_geometric) -> PYPOWER/MATPOWER case dict.
Pure data re-arrangement & unit conversion from OPFData to PYPOWER solver input.

CAVEAT:
Column orders below were verified on pglib_opf_case118_ieee by checking that the
label state satisfies Kirchhoff to ~1e-5 (see `validate.kirchhoff_mismatch`).

Note the transformer order differs from the line order.
"""
from __future__ import annotations
import numpy as np

# HeteroData column indices constants
BUS  = dict(base_kv=0, bus_type=1, vmin=2, vmax=3)
GEN  = dict(mbase=0, pg=1, pmin=2, pmax=3, qg=4, qmin=5, qmax=6, vg=7, c2=8, c1=9, c0=10)

# new
LINE = dict(angmin=0, angmax=1, b_fr=2, b_to=3, br_r=4, br_x=5, rate_a=6, rate_b=7, rate_c=8)
XFMR = dict(angmin=0, angmax=1, br_r=2, br_x=3, rate_a=4, rate_b=5, rate_c=6,
            tap=7, shift=8, b_fr=9, b_to=10)
EDGE_LABEL = dict(pt=0, qt=1, pf=2, qf=3)   # solved branch flows; verify with the diagnostic


def _bus_of(data, t):
    """Bus index (0-based) of each node of type t, via its *_link edge."""
    return data[t, f"{t}_link", "bus"].edge_index[1].numpy()


def _branches(data, etype, c, base):
    st = data["bus", etype, "bus"]
    ei = st.edge_index.numpy() + 1
    a = st.edge_attr.numpy().astype(float)
    n = len(a)
    
    # lines have unity tap and zero phase shift; transformers supply both.
    tap = a[:, c["tap"]] if "tap" in c else np.ones(n)
    shift = np.degrees(a[:, c["shift"]]) if "shift" in c else np.zeros(n)
    
    # combine end charging susceptances into PYPOWER's total BR_B
    # F_BUS T_BUS BR_R BR_X BR_B RATE_A RATE_B RATE_C TAP SHIFT STATUS ANGMIN ANGMAX
    return np.column_stack([
        ei[0], ei[1], a[:, c["br_r"]], a[:, c["br_x"]], a[:, c["b_fr"]] + a[:, c["b_to"]],
        a[:, c["rate_a"]] * base, a[:, c["rate_b"]] * base, a[:, c["rate_c"]] * base,
        tap, shift, np.ones(n), np.degrees(a[:, c["angmin"]]), np.degrees(a[:, c["angmax"]]),
    ])


def hetero_to_ppc(data):
    """Convert one OPFData `HeteroData` sample to a PYPOWER case dict.

    Loads and shunts are aggregated onto their bus rows (MW/MVAr at 1 p.u.);
    powers go p.u. -> MW, angles rad -> deg, cost coefficients p.u. -> MW basis;
    bus indices 0-based -> 1-based. Solution fields (.y, edge_label, objective)
    are ignored. Initial VM/VA are a flat start (1.0 p.u., 0 deg).
    """
    base = float(data.x.item())
    nb = data["bus"].num_nodes
    bx = data["bus"].x.numpy().astype(float)
    gx = data["generator"].x.numpy().astype(float)
    lx = data["load"].x.numpy().astype(float)
    sx = data["shunt"].x.numpy().astype(float)

    # BUS_I TYPE PD QD GS BS AREA VM VA BASE_KV ZONE VMAX VMIN
    bus = np.zeros((nb, 13))
    bus[:, 0] = np.arange(1, nb + 1)
    bus[:, 1] = bx[:, BUS["bus_type"]]
    
    # accumulate contributions when multiple loads/shunts share a bus
    np.add.at(bus[:, 2], _bus_of(data, "load"), lx[:, 0] * base)
    np.add.at(bus[:, 3], _bus_of(data, "load"), lx[:, 1] * base)
    np.add.at(bus[:, 4], _bus_of(data, "shunt"), sx[:, 1] * base)   # GS from gs
    np.add.at(bus[:, 5], _bus_of(data, "shunt"), sx[:, 0] * base)   # BS from bs
    
    # use a flat voltage start and default area/zone identifiers
    bus[:, 6] = 1; bus[:, 7] = 1.0; bus[:, 8] = 0.0
    bus[:, 9] = bx[:, BUS["base_kv"]]; bus[:, 10] = 1
    bus[:, 11] = bx[:, BUS["vmax"]]; bus[:, 12] = bx[:, BUS["vmin"]]

    # GEN_BUS PG QG QMAX QMIN VG MBASE STATUS PMAX PMIN (+11 OPF output columns)
    # ... remaining columns hold optional generator parameters
    ng = len(gx)
    gen = np.zeros((ng, 21))
    gen[:, 0] = _bus_of(data, "generator") + 1
    gen[:, 1] = gx[:, GEN["pg"]] * base;   gen[:, 2] = gx[:, GEN["qg"]] * base
    gen[:, 3] = gx[:, GEN["qmax"]] * base; gen[:, 4] = gx[:, GEN["qmin"]] * base
    gen[:, 5] = gx[:, GEN["vg"]];          gen[:, 6] = gx[:, GEN["mbase"]]
    gen[:, 7] = 1
    gen[:, 8] = gx[:, GEN["pmax"]] * base; gen[:, 9] = gx[:, GEN["pmin"]] * base

    # rescale polynomial coefficients so cost is unchanged when power uses MW
    # MODEL=2 STARTUP SHUTDOWN NCOST=3 c2 c1 c0  ($/MW^2h, $/MWh, $/h)
    gencost = np.column_stack([
        2 * np.ones(ng), np.zeros(ng), np.zeros(ng), 3 * np.ones(ng),
        gx[:, GEN["c2"]] / base**2, gx[:, GEN["c1"]] / base, gx[:, GEN["c0"]],
    ])

    # keep a deterministic branch order
    branch = np.vstack([_branches(data, "ac_line", LINE, base),
                        _branches(data, "transformer", XFMR, base)])

    return {"version": "2",
            "baseMVA": base,
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "gencost": gencost}