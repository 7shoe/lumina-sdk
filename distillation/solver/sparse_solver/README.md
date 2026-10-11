# Sparse CPU AC-OPF

`SparseSolver` produces float64 AC-OPF labels using analytic derivatives, sparse LU and PIPS-style interior-point steps. Keeping grid matrices sparse avoids the storage and computation of dense systems. PYPOWER also uses sparse PIPS; this package adds scenario batching, primal warm starts and persistent CPU workers (`SolverPool`) for mixed topologies. Our SciPy backend uses dense `trust-constr`. Production solves stay on CPUs: the tested dense XPU implementation scaled poorly on large grids, and CPU solving leaves XPUs available for neural networks.

## Usage

```python
from distillation.solver.sparse_solver import SparseSolver

result = SparseSolver().solve(network)  # OPFData graph, OPFMatrices or PYPOWER case
if result.success:
    print(result.objective)
```

`success` requires convergence and feasibility checks. Labels are detached; gradients do not pass through the solver.

## Recorded runtime

Seconds per problem, measured 2026-10-09 with cold starts, float64 and one CPU thread. Batched times are amortized over their problems.

| Buses | [Sparse CPU](../reports/sparse_pipeline/compute_8917594/validation.json) | PYPOWER CPU | [Dense XPU (archived)](../reports/sparse_pipeline/historical/validation_xpu.json) | [SciPy CPU](../reports/scaling.md) |
| --- | ---: | ---: | ---: | ---: |
| 14 | 0.012 | 0.242 | 0.028 | 0.679 |
| 30 | 0.014 | 0.228 | 0.100 | 0.878 |
| 57 | 0.020 | 0.273 | 0.133 | 3.380 |
| 118 | 0.046 | 0.493 | 0.135 | 50.793 |
| 500 | 0.217 | 2.475 | 2.555 | — |
| 2000 | 1.313 | 31.032 | 129.472 | — |

Sparse/PYPOWER share a compute-node run: 32 problems per case, or 3 above 118 buses. Sparse and XPU use batches up to four; PYPOWER and SciPy run serially. XPU used another compute node; SciPy averaged ten problems on a login node. Cross-run comparisons are indicative. **—** means untested.
