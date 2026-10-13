# DD / DCGM / HMDC / RAMBO configuration

The small discovery study is [discovery_pilot.yaml](discovery_pilot.yaml), profile
`pilot`: GIN/GAT/HEAT, cases30/57, 30 condensation runs plus four RAMBO/uniform
libraries reused by 18 student evaluations. It requests 32 nodes for one hour,
800 active synthesis seconds per condition, 100 grids and 100 student updates.
`verify_run` checks interruption/resume and exported-student replay without
shortening synthesis to the two-update smoke profile. Adaptation is T1 case30
to case57; final ordinary-SGD reports include source-case retention. HMDC uses
GIN+HEAT with GAT evaluated only after synthesis. See the
[experiment plan](../README.md#discovery-pilot-2026-10-13).

```bash
python -m distillation.experiments.study prepare --config distillation/experiments/configs/discovery_pilot.yaml --profile pilot
python -m distillation.experiments.study submit --root /lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/discovery_20261013/pilot --nodes 32
```

Edit [condensation.yaml](condensation.yaml). `reference` expands the full study;
`smoke` uses the same 100 grids, model sizes and 100 downstream updates, with two
synthesis updates and smaller real-data panels. Extra smoke selections exercise
target policies, case118 with variable sizes, and T1 case57 pretrained weights.
Smoke results establish execution correctness, not data usefulness.
`integration_smoke` is a smaller nine-condition check: four grids and four student
updates, covering all four routines, both RAMBO samplers, two RAMBO student models
and T1 case57 pretrained GIN adaptation. It exercises actual Julia generation.

RAMBO integration validated on 2026-10-12: PBS `8923806`, one node / 12 XPUs,
22m06s; all nine `integration_smoke` conditions passed (DD 1, DCGM 1, HMDC 1,
RAMBO 6). CPU checks passed 140 tests and six real Julia/student conditions.
Generation/evaluation resume, physical targets, split consistency, checksums and
exported-student replay were verified. See the
[validation report](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/data_methods_20261012_v2/integration_smoke/validation.md)
and [catalog](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/data_methods_20261012_v2/integration_smoke/catalog.csv).
These four-grid checks do not establish data usefulness.

Earlier condensation qualification on 2026-10-12: PBS `8923581`, three nodes / 36 XPUs, 34m55s;
25/25 conditions passed (DD 5, DCGM 12, HMDC 8), alongside 151 distinct CPU tests.
DCGM covered all eight models; HMDC covered four pairs spanning all eight; DD used GIN.
See the Lustre [validation report](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/comparison_100_20261012_v2/smoke/validation.md)
and [dataset catalog](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/comparison_100_20261012_v2/smoke/catalog.csv).
Case118 `random_size` rejected both DD/DCGM updates at the AC-OPF gate, so those
smoke artifacts retain their initial data. HMDC accepted both updates.

```bash
python -m distillation.experiments.study preview --profile smoke
python -m distillation.experiments.study prepare --profile smoke
```

`prepare` freezes code and resolved configurations, copies training-only references,
and imports only matching initial-grid cache entries. Missing entries are generated
on compute nodes. Changing scientific settings, data, or code requires a new
`campaign` name. The default ±25% bus limits are 23–38, 43–71 and 89–148 for
cases30/57/118. `strength` controls subsequent operational edits; generation uses
0.05 for conservative and 0.5 for random protocols unless explicitly overridden.
Topology remains fixed after initialization in the reference study.

Run the methods together, or submit one job per method:

```bash
python -m distillation.experiments.study submit --root <prepared-root> --layout combined --nodes 3
python -m distillation.experiments.study submit --root <prepared-root> --layout separate --nodes 1
# A single method, including resubmission:
python -m distillation.experiments.study submit --root <prepared-root> --methods DCGM --nodes 1
# Retry with optional Copper package caching disabled; keep frozen code/data and local data staging:
python -m distillation.experiments.study submit --root <prepared-root> --nodes 32 --no-copper
```

`--nodes` is **per job**. Every condition uses one node / 12 XPUs; additional nodes
run independent conditions. A one-hour allocation is not a promise to finish the
entire matrix. Resubmitting the same study resumes unfinished synthesis/evaluation
and skips completed work. The queue can serialize separate jobs. Frozen source,
12 replicas and the same configuration are required for continuation.
`--no-copper` writes a separate submission manifest recording that execution
override; it preserves the original manifest, dataset IDs and frozen runtime.

`execution.staging: true` copies each node's assigned processed cases, references,
pretrained weights and RAMBO template archives to `/tmp` once, verifies checksums,
and shares those copies across its 12 workers. Datasets and verified RAMBO templates
are reused within each worker. Evaluations of the same RAMBO library stay on one
node; existing shared initialization and generation checkpoints remain on Lustre.
Canonical paths, sample IDs, splits and experiment identities do not change.
`execution.copper: true` uses Aurora's Copper module for custom Python packages;
the system framework installation stays in place. Both switches can be disabled.
The launcher supports 1–64 nodes, counts startup against `allocation_seconds`, and
records per-node copy sizes, timings and dataset-cache hits in `jobs/startup_*.json`.
All outputs stay on Lustre; `/tmp` is only a disposable input cache.

Staging validated on 2026-10-13: PBS `8923910`, two nodes / 24 XPUs, 15m49s;
all nine integration conditions passed, including resume and exported-data replay.
Each node copied 1.98 GB in 48–53s; each worker loaded its case once and reused it.
Cold startup still took about 8m22s to reach the first experiment. This verifies
reuse, not a measured speedup or 64-node throughput; scheduling through 64 nodes
was checked on CPU. See the Lustre
[validation report](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/startup_staging_20261013/integration_smoke/validation.md).

The full default matrix has 234 conditions: 216 condensation conditions
(3 cases × 3 topology protocols × 3 target modes × 8 models/pairs), plus 18 RAMBO
evaluations (3 cases × 2 samplers × 3 models). Change matrix lists
to select experiments; adding initialization regimes or synthesis seeds multiplies
this count. The split stays at seed42; synthesis seeds change grid generation,
training model seed streams and real sampling. Evaluation seed pools stay paired.
`real_batch` is per XPU. HMDC also consumes real examples for model/alignment
training and candidate selection; matching batches alone understate its exposure.

All eight benchmark models are accepted. To adapt pretrained models, select
`T1_case57` in `matrix.initializations`, edit its checkpoint pattern, or supply a
`checkpoints` mapping by model. Checkpoint architectures override `model_defaults`.
Source cases remain distinct from the matrix's target cases; checkpoint hashes
and specifications are recorded. Keep source checkpoints unchanged for resume.

RAMBO's `models` are evaluators: no model participates in generation. Its own
`domains` and `samplers` control load bounds and RAMBO versus uniform generation;
the matrix's synthetic topology/target modes do not multiply RAMBO conditions.
The topology stays at the nominal case size and targets are the verified final
Julia/Ipopt solutions. Adding `paper: [0.5, 2.0]` defines a separate load-domain
experiment. Source freezing includes the Julia environment lock and nominal cases.

Each `generation_id` identifies one CPU/Julia library, reused across model and
initialization evaluations. One team leader generates it sequentially; the 12 XPUs
then evaluate independent students. Existing libraries require no new solves.
Generation checkpoints every solver stage, can pause or exhaust its attempt budget,
and must reach the requested count before evaluation. GPU allocation time includes
this CPU phase; solver calls and generation time are recorded separately. Pretrained
RAMBO settings affect only student training, with an unchanged-checkpoint control.

Outputs are in `<paths.output>/<campaign>/<profile>/`. Start with `catalog.csv`
or `catalog.json`; each row has dataset identity, completion status, artifact path,
checksum, ordinary-SGD results and a suggested downstream model directory.
Refresh it with `python -m distillation.experiments.study catalog --root <root>`.
RAMBO rows identify dataset/evaluation recipes; group by `generation_id` to count
distinct generated libraries. Their source files live under
`<paths.generated>/<campaign>/<profile>/<generation_id>/` on Lustre, alongside
`dataset_selection/RAMBO` artifacts. The common study catalog points to both roots.

Within a run, `best.pt` is the validation-selected dataset, `distilled.pt` (DD) or
`condensed.pt` (DCGM/HMDC) is the final dataset, and `latest.pt` resumes synthesis.
`artifact.json` explains targets, origins and split use; `recipe.json` gives exact
training settings. COMPLETE artifacts are checksum-verified by the loader.
`frozen`/`learned` teaching targets may differ from physical OPF solutions.
RAMBO instead exports the entire completed library as `dataset.pt`; there is no
validation-selected dataset. Its `evaluation.partial.json` resumes by evaluation
arm, while the shared generation directory's `state.json` resumes the solver.
Both routes expose `artifact.json`, `recipe.json`, and the same loader below.

```python
from distillation.synthetic.artifacts import load_dataset, train_from_artifact

data = load_dataset("/path/to/run")
graphs = data["graphs"]  # SDK inputs with the actual teaching targets in .y
model, provenance = train_from_artifact("/path/to/run", device="xpu:0",
    model="GIN", initialization="scratch", seed=123)
# initialization="recorded": original recipe, including embedded pretrained weights.
# initialization="checkpoint", checkpoint=".../best.pt": adapt another T1/T2 model.
# recipe="ordinary": shuffled SGD; DD's default native recipe uses learned rates.
```

Save downstream weights and returned provenance under the catalog's
`downstream_output`, with model/initialization/seed subdirectories. These are new
training experiments and must not overwrite the dataset. Other architectures can
consume `load_dataset` graphs with the benchmark's graph-balanced loss. No OPF
re-solve or original pretrained checkpoint is needed for recorded export replay.
Reproduction still requires the saved feature/model implementation and environment;
environment versions are recorded in each job's provenance.

Compare DD `common_evaluation.json` with DCGM/HMDC `evaluation.json` for ordinary
SGD. DD's primary `evaluation.json` additionally reports learned-curriculum results.
HMDC reports each evaluated architecture separately. Selection and reporting panels
are disjoint subsets of validation; test data remain untouched.
RAMBO's `evaluation.json` reports `generated`, equal-size `random_real`, and
`untrained` or `pretrained_baseline` arms. It uses the same train-only learning-rate
calibration, graph-balanced MSE, student seed pools, batch/epoch schedule and final
validation IDs. The selection panel is reserved but unused for RAMBO dataset choice.
Equal retained counts do not imply equal generation cost or real-label exposure.
