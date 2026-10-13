# Dataset condensation with gradient matching for AC-OPF

New experiments use `campaign.default_config(case, model)`: **100 grids, batches
of 10, 100 downstream SGD updates, width 128 / three layers, FP64 and ±10%
operational ranges**. All eight benchmark models and `random`/`fixed`/`pretrained`
initialization are supported. Eleven matching rounds with one synthetic epoch
between rounds span 100 student updates before restarting. This restart schedule
is separate from the total condensation-update budget.

Target modes are `solver` (refresh OPF labels), `frozen` (retain initial labels),
and `learned` (optimize independent targets, `target_rate`). Every mode keeps
solver-verified inputs; frozen/learned targets need not solve those inputs.
Learned targets are an explicit extension of the paper. Cosine matching, detached
real gradients, ordinary student SGD and feasibility-only acceptance remain.
Exports preserve actual teaching targets, physical solutions and resume state;
use `distillation.synthetic.evaluation.fit_export` for exact replay. Learning
curves, physical/target drift and unseen-seed fixed-mode controls are recorded.
Shared settings and launch preparation are in [synthetic](../synthetic/README.md)
and [comparison.py](../experiments/comparison.py). The 48-grid study below records
the earlier configurations and results.

Status: DCGM core, notebook, resumable MPI runner and numerical tests implemented;
the DD/DCGM protocol campaign is complete (61 conditions).
The original plan and user decisions remain below. DD, DCGM and HMDC share physical
grid and model-initialization utilities. Their objectives stay outside that utility. See
[`../synthetic/README.md`](../synthetic/README.md).

Pretrained adaptation uses `initialization: pretrained` and
`pretrained_checkpoint: /path/to/T1-or-T2/final.pt`; `case` specifies the target
case independently. Each student restart loads those weights, then ordinary
synthetic-data training continues between matching rounds. Calibration and
real-data controls use the same starting checkpoint. Its hash is checked on
resume, initial weights are exported, and evaluation includes
`pretrained_baseline`. This option changes initialization, preserving the
gradient-matching objective, student schedule and solver-label policy.
`random` remains the default; `fixed` resets to `model_seed`.

## Method and scope

Adapt [Zhao, Mopuri and Bilen, Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929)
and [`../mnist_mvp/dcgm.py`](../mnist_mvp/dcgm.py) to cases 30, 57 and 118, with
separate GAT, GIN and HeteroGNN experiments. Compare protocols using matched
splits, initialization pools and synthetic sample budgets. DD and DCGM retain
their own training schedules and corresponding evaluation controls.
The original campaign used fresh random initializations and the seed-42 dataset split.
Keep validation and test examples out of synthesis.

At the same current model weights, minimize

\[
J(S;\theta)=D\left(\nabla_\theta L(S;\theta),
\operatorname{stopgrad}\nabla_\theta L(R;\theta)\right).
\]

Detach real gradients; retain the synthetic gradient graph. Alternate data
updates with ordinary model training on detached synthetic data, and restart
the model periodically. This needs mixed second derivatives, without DD's
differentiation through a sequence of model updates. Start with the reference
row-wise cosine distance and log gradient norms. Explicitly handle unused GNN
relations and one-dimensional weights (the reference skips the latter).
A magnitude-sensitive distance is a later, separately reported ablation.

Use graph-balanced regression loss: equal weight for each graph and for bus and
generator targets, regardless of bus count. Match gradients per case; do not
invent classification labels or copy MNIST's images-per-class schedule.
Parameter gradients have compatible shapes across different graph sizes.
Operating-regime stratification can be tested later.

## Adaptation plan

1. **Shared physical inputs first.** Move topology proposals, bounded operational
   edits, feasibility checks and initialization into `distillation.synthetic`.
   Support conservative edits with backtracking and fresh random topologies with
   training-reference equipment sampling. New runs keep buses within ±25%,
   rounding half buses upward: 30 → 23–38, 57 → 43–71, 118 → 89–148,
   anchored to the original case. Measure batch diversity and size coverage.

    Answer: Yes, appears sufficient now.

2. **Small DC core.** Add `DC/dcgm.py` with matching loss and alternating updates,
   plus a short notebook. Reuse conversion, adapters, splits and solvers. Aim for
   roughly 150–250 readable core lines; add launchers/checkpoint handling
   separately when needed.

    Answer: OK. 

3. **Feasible updates.** Audit proposed physical inputs with fresh AC-OPF solves.
   Backtrack or reject failures, restoring optimizer state on rejected updates.
   Discrete topology selection uses the method's objective outside the utility;
   recreate shape-dependent optimizer state after structural edits.

    Answer: Yes, but keep what you import from `distillation.synthetic`.

4. **Separate supervision.** Store inputs, verified solver solutions and detached
   teaching targets separately. Use fresh solver labels in the shared pipeline
   now. Independent target optimization and pretrained-model predictions are
   deferred, with only an explicit placeholder.

    Answer: OK

5. **Training and evaluation.** Calibrate a fixed student learning rate and
   ordinary detached training schedule. Do not inherit DD's learned ordered
   rates by default. Evaluate fresh models on condensed, random real and initial
   synthetic sets under equal budgets. Report prediction quality, physical
   residuals and seed variability, rather than matching loss alone.

   Answer: OK

6. **Validate before scaling.** Check gradients numerically, data/model update
   separation, solver rejection/rollback, mixed-size batches and CPU/XPU second
   derivatives. Add resumable distributed runs after the local implementation
   and synthetic generator are understood.

   Answer: OK

## Relation to MNIST and the original method

The local prototype retains the central algorithm: detached real gradients,
differentiable synthetic gradients, synthetic SGD with momentum, ordinary
student SGD between matching rounds and random student restarts. The
[authors' implementation](https://github.com/VICO-UoE/DatasetCondensation)
is the comparison point.

Local experimental differences include a 50k/10k train/validation partition,
pixel scaling without the reference MNIST mean/std normalization, different
ConvNet/LeNet configurations, explicit RNG streams, one synthesized set per
invocation with repeated evaluation seeds, and consistent final-update/evaluation
endpoints. These are defensible but prevent claiming an exact reproduction.
Repeat independently condensed sets to measure synthesis variance.

AC-OPF adds constrained continuous inputs, varying discrete graph structure,
regression targets and external feasibility gates. These are scientifically
reasonable extensions when documented and ablated. They restrict the search
space and introduce selection bias; feasibility alone does not establish grid
realism or improved learning. Random electrical inputs require a physical
parameterization and solver screening; they cannot be unrestricted independent
noise in the sense of image pixels.

## Questions and decisions

- **Topology — answered.** Build a shared utility for DD, DC and HMDC. Support
  (1) conservative edits and (2) independently sampled topologies and operational
  parameters now. Permit broader size coverage within the original hard limits.
  Keep per-method topology scoring separate.

  Answer: Yes, shared utility `distillation.synthetic`. It is ok to debug it for this or the previous data distillation run that should be on debug-scaling right now.
  When you conduct experiments for this DCGM, run different synthetitization modes (request more compute nodes, up to 16 for debug-scaling to get a feeling what works.)

- **Teaching labels — deferred by request.** Reserve (3) independent detached
  labels, including pretrained neural predictions, as a placeholder. Solver
  solutions certify physical inputs; future teacher predictions must never be
  presented as AC-OPF feasibility certificates.

  Answer: Ok. 

- **DC delivery/scheduling — subsequently authorized.** Start with the compact module and
  notebook. Decide restart frequency, matching rounds, student steps, topology
  update frequency and distributed launchers after pipeline measurements. No
  training campaign was part of recording that initial plan. The later request
  explicitly authorized implementation and debug-scaling experiments.

  OK.

Compare conservative/random initializations at matched graph and bus budgets.
Track requested/accepted sizes, branch density, equipment counts, demand/capacity,
solver failures, backtracking and structural diversity. Fixed-ID signatures can
count bus renumberings as different; supplement them with structural statistics.
Save generation settings, seeds and training-reference provenance with each set.

## Implemented algorithm and explicit deviations

[`dcgm.py`](dcgm.py) contains the algorithm; [`campaign.py`](campaign.py) contains
experiment plumbing. [`dcgm_mvp.ipynb`](dcgm_mvp.ipynb) runs a small alternating
example. Input adapters and graph-balanced MSE live in `distillation.synthetic.learning`.

The notebook stores synthetic data, solver labels and student weights on Lustre at
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/DCGM/mvp/<model>/case<case>/<protocol>/mvp.pt`.
Distributed runs store `latest.pt`, `best.pt`, `condensed.pt` and `student.pt` in
each manifest experiment's `output` directory. Existing campaign manifests already
use Lustre under `dataset_distillation/protocols_20261010`; use a Lustre output
directory for new campaigns as well.

- Real parameter gradients are detached; synthetic gradients retain their graph.
  The distance is the authors' summed row-wise cosine, with `1e-6` added to the
  product of norms. The same formula and SDK backbones run in FP64 after FP32
  smoke checks exposed cancellation in small mixed derivatives. The precision
  adapter preserves architecture, initialized weights and output bounds.
  Vector/scalar parameters are omitted, and unused parameter
  gradients become zeros. No gradient-norm balancing or cross-architecture
  matching is introduced.
- Synthetic inputs use SGD (learning rate 0.1, momentum 0.5). Each replica uses
  ordinary student SGD without momentum, with a fixed rate calibrated on
  training-only data. Five matching rounds share a student initialization;
  one shuffled synthetic epoch follows each round except the last. Then the
  student restarts. Synthetic optimizer momentum persists across restarts.
- Twelve XPUs average gradients of the **same synthetic dataset**, from twelve
  independent students of the **same architecture** and separate real batches.
  This estimates the expectation over initialization/training trajectories.
  It is neither heterogeneous-model condensation nor DD trajectory differentiation.
- The reference's class-wise image loss becomes per-case graph-balanced
  regression. All 48 synthetic graphs enter each matching loss. Student training
  uses three batches of 16. Dropout is disabled for reproducibility; BatchNorm,
  if present, is calibrated on real data and frozen for the matching round.
- Physical parameters remain within bounded ±2% edits of the generated inputs.
  Failed AC-OPF proposals backtrack through fractions 1, 1/2, 1/4, 1/8, 1/16.
  Complete rejection restores inputs **and SGD momentum**. Accepted solves
  refresh labels, detached from autograd. This is a changing, stop-gradient
  supervision surrogate; it does not differentiate through the OPF optimizer
  and does not claim to be the exact derivative of solver-consistent targets.
- These protocol comparisons hold sampled topology fixed during optimization.
  Conservative and fresh-random generation are separate initial distributions;
  random-size batches span the permitted interval at exactly the base case's
  total bus budget. Conservative batches can differ slightly in total buses.
  Generation is solver-screened and therefore biased toward feasible proposals.
- Validation compares fresh students trained on condensed, initial synthetic,
  random real, and no training. All trained controls receive the same ordinary
  SGD budget. Ten evaluation epochs and twelve fresh student seeds are a debug
  budget, not the paper's full evaluation or repeated independent synthesis.
  Evaluation uses fixed-rate SGD without momentum, decay or augmentation; the
  authors' image evaluation uses longer training with momentum, weight decay
  and a learning-rate drop. BatchNorm, if added to a backbone, also stays frozen
  during our student training, unlike the reference training helper. The three
  current backbones have no BatchNorm layers.
  DD's learned three-step curriculum and DCGM's ordinary training are different
  evaluation protocols; their raw scores do not isolate a method advantage.
  DD's real-subset control also reuses rates learned for the synthetic data;
  it is not a separately optimized real-data curriculum. Its initial-synthetic
  control with the same learned rates isolates physical-input and associated
  solver-label changes from learning the rates themselves.

Every matching update samples 64 real training graphs per XPU (768 total),
cycling through the full training pool after excluding 48 reference IDs.
Validation uses 256 selection examples and a disjoint 1,024-example final
panel. Test data are never accessed. The final real-example count is recorded
per experiment, rather than assuming the entire accessible pool was visited.
The 48 initialization references and the real-subset control are inherited from
the earlier campaign's seeded pool (their canonical IDs lie in archive group 0).
They are not a fresh uniform draw from all 240,000 training examples. Matching
uses the full remaining training population; `original.pt` and `split.json`
record the distinction. Broader reference-pool coverage remains an untested
initialization choice.

Checkpoints store physical anchors, input offsets, labels, optimizer momentum,
all twelve current students, and the sampler cursor. The launcher can resume
from `latest.pt` with the same manifest. Each allocation is limited to one
hour, with a reserve for final evaluation and atomic saves. `best.pt` is selected
on validation; `condensed.pt` records the last update, including unsuccessful
experiments where the initial data remain best.

## Campaign artifacts

The protocol campaign is stored at
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/protocols_20261010`.
Each method/model/case/protocol directory contains `config.json`, `original.pt`,
`initial.pt`, cached generation reports, `split.json`, `device_checks.json`,
`history.jsonl`, `latest.pt`, dataset exports, student checkpoints and
`evaluation.json`. Frozen source copies and SHA256 manifests are stored at the
campaign root. Failures and their diagnostics are retained.

The bus-range change arrived after eighteen DD conditions initialized. Those
conditions retain their original ranges; the remaining nine use ±25%. All DCGM
conditions use ±25%. Separate new DD comparisons cover the six affected
random-size conditions; no started run's topology range is silently rewritten.
One additional DD GAT/case118 conservative control uses the corrected shared
initialization. An early cached initialization chose different operational
backtracking after roundoff in the former parameterization. Both attempts are
retained; this control removes that initialization difference from comparisons.

## Qualification and stability findings

The local suite has 151 passing tests across DCGM, shared generation, DD and
matrix conversion. Notebook code cells execute successfully on CPU in FP64.
Three 12-XPU smoke configurations pass all 36 matching-gradient comparisons and
complete feasible updates, evaluation and export. A distributed four-update
resume comparison covers 864 tensors (synthetic parameters/momentum/labels and
all twelve students); the largest difference from uninterrupted training is
4.09e-14.

Full-run random restarts exposed GAT/case30, GAT/case118 and HeteroGNN/case118
initializations that diverged at the twelve-seed calibration winners (1e-4
for GAT/case30 and 1e-5 for the two case118 configurations). All three failures
were reproduced on CPU. Those attempts are archived rather than
omitted. The affected model/case retries calibrate on
48 independent training-only model seeds, then use a fixed rate ten times
smaller than the best stable candidate. The expanded panel independently
rejects 1e-4 for GAT/case30. The same revised rule applies to all three generation
protocols within each affected model/case; no student seed is skipped and no model gradient is clipped.
Other conditions retain their recorded calibration protocol.

Completion records and the authoritative resume manifest/source pair are
recorded in the campaign-root README. Initial failed smoke logs, full-run
attempts, configurations, and source revisions remain available for audit.

## Completed protocol study: 2026-10-10

All 27 DCGM conditions and 34 DD conditions completed. Seven DD conditions are
additional size/input controls, so the preferred comparison has 54 cells:
two methods × three architectures × three cases × three generation protocols.
Each dataset contains 48 synthetic graphs. All 732 per-condition XPU checks
passed, all final evaluation controls were finite, and selection/final validation
IDs are disjoint. Test data were not read. The final job exited successfully.

The preferred DD conditions visited 49,152–239,952 distinct real training
examples; DCGM visited 46,080–239,952. Fifteen of 27 DD conditions and thirteen
of 27 DCGM conditions visited the full available pool. These counts describe
the completed search, not necessarily the earlier validation-selected iterate.
`analysis/results.csv` records both counts; six DCGM conditions selected their
initial synthetic dataset (`best_iteration=0`).

Learned operational changes usually had small effects. Across preferred DD
conditions, the largest mean improvement over initial synthetic inputs under
the same learned rates was about 0.33%. DCGM GIN/case57 improved mean MSE over
its initial synthetic inputs by 1.47% for random-fixed and 0.95% for random-size
generation (12/12 and 11/12 paired student wins). Both remained worse than the
random-real control. Larger GAT/case57 mean changes depended on a few sensitive
student initializations: the random-size mean improved about 29%, while its
median paired ratio was 0.9999993. This is not evidence of a broad 29% gain.

The useful conclusion is that the shared pipeline produces feasible and more
structurally varied grids, and both algorithms execute and resume correctly.
These short runs do not establish a general advantage for optimized synthetic
data. Topologies were fixed after generation; the observed topology diversity
comes from initialization. One synthesis seed and twelve student seeds do not
measure variance across independently distilled datasets.

Accepted physical edits also refresh the solver labels. Consequently, changes
relative to initialization combine input edits with changes in the AC-OPF
solution targets. The case57 diagnostic in `analysis/case57_label_drift.json`
records these changes; they should not be attributed solely to input features.

Start with the campaign's `README.md`, `analysis/datasets.json` (preferred
exports/checkpoint roles), `analysis/README.md` (all results), and
`analysis/protocol_comparison.pdf`. The authoritative combined resume pair is
`campaign_resume_v8.json` with `dcgm_source_v5`; original DD conditions use
`dd_resume.json` with `dd_source_v2`. Completed conditions skip training.

## Load the synthetic data for later training

```python
import torch
from distillation.synthetic.learning import benchmark_graph

artifact = torch.load('path/to/experiment/best.pt', map_location='cpu', weights_only=False)
native_grids = artifact['physical_solved_graphs']  # includes verified bus/generator labels
training_graphs = [benchmark_graph(g, labels=True) for g in native_grids]
```

`best.pt` is chosen using the selection panel; `condensed.pt` is the final DCGM
iterate. Native grids retain the physical OPFData schema and per-unit quantities.
`inputs` is the detached SDK batch and `teaching_targets` is separate supervision.
Use the recorded `config`, `student_rate`, and evaluation schedule to reproduce
this campaign. Training for a different duration or with a different optimizer
is a new downstream utility experiment. DD exports instead include their
learned ordered `schedule`; replay that schedule for its reported DD score.

Both formats embed a matching student checkpoint in `best.pt['model_checkpoint']`.
The separate DCGM `student.pt` is trained on the best dataset with a final-panel
seed. DD's separate `student.pt` belongs to the last exported dataset and can
differ from its validation-selected best student. The campaign's
`analysis/datasets.json` records these roles and the preferred artifact paths.
