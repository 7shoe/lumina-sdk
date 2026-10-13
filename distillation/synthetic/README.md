# Shared synthetic AC-OPF networks

`distillation.synthetic` provides feasible synthetic grids for DD, DCGM and HMDC.
Use `conservative` to edit training networks or `random` to draw fresh topologies
and equipment. Accepted graphs carry freshly checked AC-OPF labels.

```python
from distillation.synthetic import generate

result = generate(
    training_graphs, mode="random", original_buses=30,
    bus_range=(23, 38), count=12, steps=3, seed=42,
)
graphs = result.physical_graphs
```

Supply native OPFData graphs from the training split. Campaign size ranges are
30 → 23–38, 57 → 43–71 and 118 → 89–148 buses; pass `bus_range` explicitly when
calling `generate`. Feasibility checks do not establish global optimality.

The Python needed for benchmark generation and training stays here. External PBS
launchers call `python -m distillation.synthetic.campaign --manifest PATH`.
Normal DD/DCGM/HMDC runs do not import `synthetic.aux`.

`learning.py` also shares model initialization: `random` (default), `fixed`
(`model_seed`), or `pretrained` (T1/T2 weights and buffers). DD and DCGM use:

```python
config.update(initialization="pretrained", pretrained_checkpoint="/path/to/final.pt")
```

HMDC supplies a checkpoint for each architecture:

```python
config.update(initialization="pretrained", pair=["GIN", "GAT"],
              evaluation_models=["GIN", "GAT"], pretrained_checkpoints={
                  "GIN": "/path/to/GIN/final.pt", "GAT": "/path/to/GAT/final.pt"})
```

Include a checkpoint for every `evaluation_models` entry. Saved architecture
options take precedence; the target `case` and its data remain independently
configured. Models stay trainable, with fresh method-specific optimizers.
DD resets each trajectory; DCGM/HMDC reset at their existing restart boundaries.
HMDC's alignment and gradient balancing still restart with the pair.

Hashes pin source checkpoints for resume; keep those files unchanged and
available. Exports embed initial weights (HMDC: pair members), allowing
`learning.model_from_export(artifact, device, model="GIN")` without the source
file; omit `model` for DD/DCGM. Evaluation reports `pretrained_baseline`, matched
real-data adaptation, and the number of distinct initializations. These runs
measure adaptation from specified weights. Initialization does not change each
method's teaching-label policy or learning objective.

New `default_config` functions in DD, DCGM and HMDC share **100 grids, batches
of 10, 100 downstream updates, width 128 / three layers, FP64, ±10% operational
edits and ±25% bus limits**. Explicit configurations in old runs retain their
settings. `solver`, `frozen` and `learned` target policies are shared: every input
must remain solvable, while only solver-mode teaching targets are certified
solutions. Learned targets are additive regression variables; reference-bus
angles stay fixed. Target learning is an extension of DCGM/HMDC's fixed-label
reference algorithms. HMDC applies its existing gradient balancing/normalization
to the additional target variables, with a separate `target_rate`.

`inputs.py` handles policies, rollback and exact target export. `evaluation.py`
shares ordinary SGD, training-only rate calibration, learning curves and paired
controls. DD additionally retains its primary learned-rate curriculum result;
`common_evaluation.json` measures its dataset using the same ordinary-SGD
procedure as DCGM/HMDC. Fixed-mode runs also evaluate unseen random weights.
Final evaluation resumes by arm. Physical/target drift is recorded during
condensation; physical solutions and teaching targets are always saved separately.

Replay DCGM/HMDC exports with their saved teaching targets and starting weights:

```python
from distillation.synthetic.evaluation import fit_export
model = fit_export(artifact, device="xpu:0")  # HMDC: also model="GIN"
```

Use [the comparison manifest builder](../experiments/comparison.py) for matched
references, initial grids, precision, model settings and evaluation budgets.
DCGM retains cosine gradient matching and synthetic-data student SGD; HMDC
retains paired gradient matching, alignment and real-data mutual training.

New studies can use [one YAML and PBS interface](../experiments/configs/README.md).
`artifacts.load_dataset(run_directory)` checks dataset identity and loads SDK graphs
with their saved teaching targets. `artifacts.train_from_artifact` supports recorded,
scratch, or explicit-checkpoint initialization and returns downstream provenance.
Physical OPF solutions remain separately available. Each study publishes a Lustre
catalog plus per-run `artifact.json` and `recipe.json`; `latest.pt` remains the
resumable synthesis checkpoint.

[Detailed documentation](aux/README.md), standalone generation, reporting,
plotting, the example PBS script and historical results live in `aux/`.
