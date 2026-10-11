# Shared utility `distillation.conversion`

It is essential to transflate between the models' canonical input of networks (`torch_geometric.HeteroData`) and the representation (from MATLAB) that is tailored to `PYPOWER` solver.
Since network topology, connectivity, and operational constraints become learnable for the synthetic pipeline, we need to (1.) modify pre-existing networks and (2.) recompute the AC-OPF solution.
This requieres a (somewhat) bijective mapping between the linear matrix form and the input to the solver. 

## Usage

```python
from distillation.conversion.matrix_form import (
    NetworkEditor, mat2opf, opf2mat, residuals,
)

mats = opf2mat(data)
restored = mat2opf(mats)  # Exact unedited record round trip.
checks = residuals(mats)  # Requires complete solution labels.

# Export fixed-size parameter edits without stale solution labels.
changed_data = mat2opf(mats, include_solution=False)

# Add a bus with a load and connect it to the existing network.
edit = NetworkEditor(mats)
bus = edit.add_bus(
    base_kv=230, vmin=0.95, vmax=1.05,
    load={"pd": 0.2, "qd": 0.05},
)
edit.add_branch(
    0, bus, br_r=0.01, br_x=0.1, rate_a=2.0,
    angmin=-0.5, angmax=0.5,
)
changed = edit.build()
changed_data = mat2opf(changed)  # Topology edits discard old labels.
```

Powers and admittances use per unit; angles use radians. Voltage bases use kV, power bases use MVA, and costs use coefficients defined for per-unit generation. Unified branch arrays contain AC lines first, then transformers.

## Components

**`opf2mat` / `mat2opf`**

- Convert between one OPFData record and `OPFMatrices`.
- Preserve attributes, values (including NaN payloads and signed zero), dtype, shape, device, and store order for unedited round trips. Export rejects lossy dtype casts.
- Do not preserve aliasing, strides, object identity, subclass state, or original autograd history.
- `include_solution=True` preserves stored labels by default; it does not validate them after parameter edits. Use `False` to export edited inputs without old labels.

**`OPFMatrices`**

- `params` stores topology, raw electrical parameters, loads, bounds, and costs.
- `sol` stores voltage, dispatch, branch flows, and objective only when all five solution blocks exist.
- `meta` stores the cloned source template needed for reconstruction. Keep it: derived matrices alone cannot recover every original attribute. This deliberately duplicates data.

**`OPFParams`**

- Derives incidence matrices, series admittances, complex taps, bus loads, and admittance matrices `Y`, `Yf`, and `Yt` from raw parameters.
- Supports continuous tensor edits with autograd. Parameter changes do not refresh solution labels or objective values.
- `with_series_admittance(y)` sets raw impedance using `r + j*x = 1/y`; reciprocal rounding makes this an edit, not an exact round trip.
- Dense matrix views are intended for small cases. Branch-flow evaluation uses branchwise operations.

**`NetworkEditor`**

- Adds, removes, rewires, and updates buses, equipment, lines, and transformers; also merges adjacent buses.
- Owns changes to component counts and the line/transformer partition. `build()` validates a connected network with one reference bus and rebuilds the source template.
- Discards obsolete labels and autograd history. Recreate trainable tensors and optimizer state afterward.
- Rejects unknown extra metadata by default; `drop_extra=True` permits discarding it.
- Bus merging creates a new physical model: removed branches' thermal limits disappear. Connectedness does not guarantee OPF feasibility.

**`residuals` / `infer_label_order`**

- Check stored labels against power balance, branch flows, bounds, angle differences, reference angle, and objective consistency. Residuals do not certify optimality or KKT conditions.
- Zero thermal ratings raise by default. Set `zero_rating='unmonitored'` to ignore those limits or `'literal'` to enforce zero flow.
- Branch-label order is a per-conversion option retained with the result. The default is `pt, qt, pf, qf`; `infer_label_order(data)` checks a labeled record.

## Conventions and limits

Input uses the OPFData Appendix A schema: four node types, two branch stores, and three equipment links. Core floating-point tensors are `float32` or `float64` on one device; indices are `int64`.

Physics assumes nonzero series impedance, positive taps, and zero branch-end conductance. Endpoint susceptances `b_fr` and `b_to` are already separate; do not halve them again. The Laplacian simplification applies only when every complex tap equals one, including transformers. Phase shifts can make `Y` nonsymmetric; it is generally not Hermitian. Angle checks retain unwrapped `va`.

Unknown attributes are preserved during ordinary conversion but are not physically modeled. No arbitrary-`Y` inversion, unit rebasing, equivalent network reduction, or gradients through topology edits or external solvers are provided.

## Validation

The source documentation reports 35 CPU topology-edit tests plus 20 regression checks covering reindexing, rollback, metadata handling, label invalidation, round trips, continuous gradients, and three public IEEE-118 records. These tests were not rerun when preparing this README.

## References

- [OPFData, Appendix A](https://arxiv.org/html/2406.07234v1)
- [PyG OPFDataset (2.6.1)](https://pytorch-geometric.readthedocs.io/en/2.6.1/_modules/torch_geometric/datasets/opf.html)
- [MATPOWER makeYbus](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html)
