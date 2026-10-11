"""pytest distillation/test/test_matrix_conversion.py -v"""
import sys
from pathlib import Path

import pytest
import torch

root = next(p for p in Path(__file__).resolve().parents
            if (p / "distillation/conversion/matrix_form.py").is_file())
sys.path.insert(0, str(root))

from distillation.acopf.opf_data import load_dataset
from distillation.conversion.matrix_form import (
    E_LINE, E_XFMR, NetworkEditor, equal_records,
    mat2opf, opf2mat, validate_network,
)

# random indices of graphs/test cases
INDICES = [1, 50, 250, 500, 1000, 1500, 2000, 3000, 5000]

@pytest.fixture(scope="session")
def ds():
    return load_dataset()

@pytest.fixture(params=INDICES)
def data(ds, request):
    return ds[request.param].clone()

def checked(mats):
    """
    val network
    """
    
    validate_network(mats.params)
    out = mat2opf(mats)
    assert mats.sol is None and "objective" not in dict(out.items())
    assert "y" not in out["bus"] and "y" not in out["generator"]
    assert all("edge_label" not in out[et] for et in (E_LINE, E_XFMR))
    assert equal_records(out, mat2opf(opf2mat(out)))
    return out


def extended(data):
    p = opf2mat(data).params
    edit = NetworkEditor(opf2mat(data))
    bus = edit.add_bus(
        base_kv=p.base_kv[0], vmin=p.vmin[0], vmax=p.vmax[0],
        load={"pd": 0.2, "qd": 0.05}, shunt={"bs": 0.01},
        generator=dict(pmin=0, pmax=1, qmin=-0.5, qmax=0.5,
                       c2=1, c1=10, c0=0),
    )
    branch = edit.add_branch(0, bus, br_r=0.01, br_x=0.1,
                            rate_a=2, angmin=-0.5, angmax=0.5)
    return edit, bus, branch


def test_round_trip(data):
    assert equal_records(data, mat2opf(opf2mat(data)))


def test_parameter_edits(data):
    """
    check if networks indeed editable
    """
    
    original = data.clone()
    mats = opf2mat(data)
    target_y = 2 * mats.params.y
    mats.params = mats.params.with_series_admittance(target_y)
    
    # minor additive/muliplicative pertubation
    p = mats.params
    p.pd *= 1.01
    p.qd *= 1.01
    p.bs *= 1.01
    p.gs *= 1.01
    p.pmax += 0.01
    p.qmax += 0.01
    p.vmax += 0.01
    p.angmax += 0.01
    modified = opf2mat(mat2opf(mats, include_solution=False))
    checked(modified)
    torch.testing.assert_close(modified.params.y, target_y, rtol=1e-6, atol=0)
    for name in ("pd", "qd", "bs", "gs", "pmax", "qmax", "vmax", "angmax"):
        assert torch.equal(getattr(modified.params, name), getattr(p, name))
    assert equal_records(data, original)


def test_add_remove(data):
    original = data.clone()
    p = opf2mat(data).params
    edit, bus, _ = extended(data)
    out = checked(edit.build())
    for kind in ("bus", "generator", "load", "shunt"):
        assert out[kind].num_nodes == data[kind].num_nodes + 1
    q = opf2mat(out).params
    assert q.E == p.E + 1
    assert q.gen_bus[-1] == q.load_bus[-1] == q.shunt_bus[-1] == bus
    mapping = edit.remove_bus(bus, cascade=True)
    assert mapping[-1] == -1
    restored = checked(edit.build())
    baseline = checked(NetworkEditor(opf2mat(data)).build())
    assert equal_records(restored, baseline)
    assert equal_records(data, original)


def test_rewire_and_outage(data):
    edit, bus, branch = extended(data)
    edit.rewire_branch(branch, 1, bus)
    out = checked(edit.build())
    assert out[E_LINE].edge_index[:, branch].tolist() == [1, bus]
    edit.remove_branch(branch)
    with pytest.raises(ValueError, match="islands"):
        edit.build()
    edit.add_branch(0, bus, br_r=0.01, br_x=0.1,
                    rate_a=2, angmin=-0.5, angmax=0.5)
    checked(edit.build())


def test_merge(data):
    p = opf2mat(data).params
    edit, bus, _ = extended(data)
    mapping = edit.merge_buses(keep=0, remove=bus)
    assert mapping[0] == mapping[bus] == 0
    q = opf2mat(checked(edit.build())).params
    assert (q.N, q.E, q.G, q.L, q.S) == (p.N, p.E, p.G+1, p.L+1, p.S+1)
    assert q.gen_bus[-1] == q.load_bus[-1] == q.shunt_bus[-1] == 0
    torch.testing.assert_close(q.pd.sum(), p.pd.sum() + 0.2)
    assert torch.equal(q.c2[:-1], p.c2)


def test_export_gradient(data):
    mats = opf2mat(data)
    mats.params.pd = torch.nn.Parameter(mats.params.pd.detach().clone())
    out = mat2opf(mats, include_solution=False)
    out["load"].x[:, 0].sum().backward()
    torch.testing.assert_close(mats.params.pd.grad, torch.ones_like(mats.params.pd))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *sys.argv[1:]]))
