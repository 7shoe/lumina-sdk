# AC-OPF solvers

We implemented (or at least adapted) solvers in order to replicate OPFData labels but 
also to squeeze most out of ALCF's Aurora when online optimization is required, e.g.
for dataset condensation/distillation routines. 

The adapted solvers are `ScipySolver` and `PyPowerSolver` (see `./backend/reference.py`).
A sparse solver `SparseSolver` sits in `./sparse_solver/solver.py`

### Details
`solve(network)` runs PYPOWER/PIPS on native OPFData, `OPFMatrices`, or a PYPOWER
case dictionary. A usable result requires `result.success`: both convergence and
independent feasibility checks. SciPy supports primal warm starts; PYPOWER's PIPS
wrapper constructs its own initial point and rejects `x0`.

```python
from distillation.solver import ACOPF, PyPowerSolver, ScipySolver

model = ACOPF(network)
reference = PyPowerSolver().solve(model)
refined = ScipySolver().solve(model, x0=reference)
```

`ReducedGradient` is experimental. `NewtonPF` solves power flow for supplied
generator controls; it is not an OPF optimizer.

### Scaling benchmark
A small scaling experiment on the key distillation/condensation case sizes (identical to lumina-bench) subsets, 
shows the scaling on Aurora.

```bash
source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh
python distillation/solver/scaling.py
```

The default run reads eleven native JSON records per case: one pilot/source and
ten targets. It measures cases 14, 30, 57, 118, 500 and 2000 with PYPOWER, and
cases through 118 with the dense SciPy backend. It tests 1/2/5/10 concurrent
PYPOWER workers, 1/2 SciPy workers, and 1/2/4 threads per solve. Worker affinity
and BLAS/PyTorch thread limits keep the CPU allocation within half the available
logical CPUs. The thread sweep uses three targets; concurrency uses all ten.

Each task has a 600-second deadline and a 16-GiB RSS guard. Reaching either stops
further work for that solver/case. The SciPy size guard is separate from these
measured limits and can be changed with `--scipy-max-buses`.

Warm-start comparisons use the computed solution of the preceding distinct
problem, with randomized cold/warm execution order. Stored target labels are
never used as starts. Only converged, feasible pairs with comparable final costs
enter the speedup statistics. Source-solve costs are saved separately.

Reports contain latency, throughput, CPU occupancy, RSS, thread sensitivity,
warm-start results and descriptive size regressions. PNG/SVG plots, raw timings,
input hashes, exact JSON records, source snapshots and library versions accompany
`report.md` under `reports/scaling_<UTC>/`.

```bash
# One reference solve per size.
python distillation/solver/scaling.py --solvers pypower --pilot-only

# Small end-to-end check.
python distillation/solver/scaling.py --cases 14 --batch 2 --workers 1 2 --thread-counts 1

# Continue an interrupted run; completed observations are reused.
python distillation/solver/scaling.py --resume /absolute/path/to/run

# Recompute the report without solving.
python distillation/solver/scaling.py --report-only /absolute/path/to/run
```

An unfinished run requires unchanged inputs and solver/benchmark source files.
Resuming a completed run only refreshes its report. A batch
partly measured before an interruption retains its individual solve results but
does not contribute a misleading combined throughput measurement.
Measurement code and the current report generator are saved separately, so
plotting updates preserve the provenance of existing timings.

To add a backend, implement `solve(model, x0=None) -> Solution` and register its
module, factory, warm-start support and optional size guard in `SOLVERS` in
`scaling.py`. Runtime settings and paths are available through `--help`.

```bash
python -m pytest distillation/test/test_acopf_solver.py distillation/test/test_solver_scaling.py -q
```
