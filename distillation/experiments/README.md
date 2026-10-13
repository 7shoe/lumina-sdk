# Experiments

For new DD/DCGM/HMDC/RAMBO studies, use the shared
[configuration and launch guide](configs/README.md) and [one YAML](configs/condensation.yaml).
It prepares frozen per-run configurations, combined or separate PBS submissions,
and readable/checksummed dataset catalogs on Lustre. The historical pilots below
retain their original settings and results.
RAMBO contributes fixed-topology operating-point generation and its matched
uniform control, followed by the common student evaluation. BAS remains a separate
selection campaign. RAMBO libraries are reused across models and initialization
regimes; the catalog records their generation identities and evaluation recipes.

This directory contains experiment notes and PBS launchers for:

- data selection (BAS);
- operating-point generation (RAMBO);
- synthetic-data routines (DD, DC, HMDC);
- training Lumina benchmark models on single or multiple topologies (T1, T2).

## Discovery pilot (2026-10-13)

[Executable configuration](configs/discovery_pilot.yaml): **GIN, GAT, HEAT**,
cases **30/57**, one synthesis seed (42), **100 grids**, 10 batches of 10 and
100 downstream SGD updates. Condensation uses learned targets with AC-OPF-verified
inputs. Connectivity stays fixed after generation; operational attributes and
teaching targets are optimized. No synthesis hyperparameter search.

| Condition | Scope | DD | DCGM | HMDC |
| --- | --- | ---: | ---: | ---: |
| Conservative, fresh initialization | Three models, both cases | 6 | 6 | 2 |
| Conservative, fixed initialization | Three models, case30 | 3 | 3 | 1 |
| Conservative, pretrained adaptation | Three T1 case30 checkpoints → case57 | 3 | 3 | 1 |
| Random fixed-size generation, fresh initialization | GAT, case30 | 1 | 1 | 0 |
| **Condensation runs** | | **13** | **13** | **4** |

HMDC synthesizes with **GIN+HEAT**; GAT is an additional evaluation architecture
and does not select data. RAMBO and uniform each generate one 100-example library
per case: **four libraries**, reused across three models from scratch and for
case30→57 adaptation (**18 evaluation recipes**). These retain solver targets
and fixed nominal topology. Altogether: **34 datasets, 48 run directories**.

Request **32 nodes / 384 XPUs for one hour**, with one 12-XPU team per active
condition. Only two teams need a second dataset assignment. Each condensation
condition gets 800 active synthesis seconds or 1,000 outer/matching updates,
whichever comes first. Initialization and evaluation cost additional time;
unfinished work resumes from the same frozen manifest. Qualification, a deliberate
checkpoint/resume, and export replay accompany the useful training budget.

First submission **8924199** failed before Python/training startup (exit 127,
17m15s): the launcher reported 12 unreachable nodes after Copper initialization.
The root cause is not established. Retry **8924284** requests the same 32 nodes
and one hour with `--no-copper`; verified local data staging remains enabled,
and the frozen source, experiment configurations and dataset IDs are unchanged.
All 48 recipes were still pending at resubmission. Preflight:
99 CPU regression tests passed; 25 scheduling/configuration checks passed after
the final placement change (overlapping tests). XPU results are pending. The
frozen source contains 142 files; all 300 initial grids reused matching caches.
Three targeted submission tests also passed for the recorded caching override.

Primary comparisons use ordinary SGD, matched train-only learning-rate calibration,
100 real-grid and initial-synthetic controls, paired student seeds, and the same
1,024 reporting validation examples per case. Keep DD's learned-curriculum results
separate. Report both validation-selected and final datasets under ordinary SGD;
the three methods retain their different selection objectives. Adaptation also
reports the unchanged checkpoint, matched real-data fine-tuning, and 256 source-case
validation examples for retention. Test splits remain untouched.

Use a **20% validation-MSE reduction against the matched real control** as a
practical screening threshold, checking paired student outcomes and physical
residuals. Compare with the initial grids to distinguish generation gains from
condensation gains. Twelve student evaluations are conditional on one synthetic
dataset; they are not twelve independent synthesis experiments. Only promising
datasets advance to a separate 500-update ordinary-SGD evaluation with controls.

Durable outputs: `/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/discovery_20261013/pilot/`.
Start with `catalog.csv`; `campaign.yaml`, `resolved.json` and `manifest.json`
pin the settings and inputs, `source/` freezes the runtime, and each run exports
`artifact.json`, `recipe.json`, checkpoints and the actual teaching targets.
RAMBO source libraries are in `dataset_selection/RAMBO/discovery_20261013/pilot/`;
the common catalog links them. The broader matrices below remain future plans.

## Agreed full DD experiment (2026-10-13; planned)

This records the agreed experiment matrix; the executable YAML still describes
the earlier study. This full experiment has not been prepared or submitted.

Use all eight models (**GCN, GAT, GIN, Transformer, HGT, RGAT, HEAT, HeteroGNN**)
and all five cases (**14, 30, 57, 118, 500**); exclude case2000. Use one synthesis
seed, **42**, with the benchmark train/validation/test split also fixed at 42.
The synthesis seed is distinct from the student model initialization regime.

For every model/target-case pair, run exactly these four settings:

| Purpose | Generation protocol | Target mode | Model initialization | Runs |
| --- | --- | --- | --- | ---: |
| Main DD | `conservative` | `learned` | `random`: fresh weights for each outer update/replica | 40 |
| Physical-label control | `conservative` | `solver` | `random` | 40 |
| Fixed-initialization ablation | `conservative` | `learned` | `fixed`: reset to the same `model_seed` | 40 |
| Cross-case adaptation | `conservative` | `learned` | `pretrained`: reset to the same T1 checkpoint | 40 |
| **Total** | | | | **160** |

No `random_fixed`, `random_size`, `frozen` targets or additional synthesis seeds
are included. Connectivity stays fixed after conservative initialization;
operational parameters, teaching targets where selected, and student rates are
learned. All modes retain solver-verified physical inputs. Learned teaching
targets need not equal the physical solutions.

Retain **100 grids, 10 batches of 10, 10 passes = 100 student SGD updates**, with
100 learned per-update rates. Use the current width-128 / three-layer models,
FP64, operational strength 0.10 and +/-25% bus limits; pretrained runs retain
the checkpoint's architecture. Fixed random initialization uses `model_seed`
1,000,000. Evaluation includes unseen initializations for that ablation.

Planned adaptation sources use the same model's **T1 `best.pt`, seed42**:

| Target case | T1 source case |
| ---: | ---: |
| 14 | 30 |
| 30 | 14 |
| 57 | 30 |
| 118 | 57 |
| 500 | 118 |

Reuse existing T1 weights; T2 is a broader multi-topology reference, not a
comparison with identical training-case exposure. Keep the equal-size real-data
and initial-grid controls, and the unchanged-checkpoint control for adaptation.
DD's learned-curriculum results and its ordinary-SGD evaluation must remain
separate when comparing against DCGM/HMDC.

Retain the current **2,400-second active synthesis budget**, with the existing
1,000-outer-update cap, periodic checkpoints and resumable evaluation. Plan
approximately **160 one-hour node allocations**, one 12-XPU node per condition:
three waves of **64 + 64 + 32 nodes**. This is a resource estimate, not a promise
of completion: startup, generation and evaluation add time, and case500 DD has
not been benchmarked. References for cases14/500 and the expanded YAML still need
preparation before launch. Multiple student evaluations do not constitute
independent synthesis repetitions.

## Agreed DCGM and HMDC experiments (2026-10-13; planned)

Mirror the DD design above with **one synthesis seed (42)** and cases
**14, 30, 57, 118, 500**. Retain each method's default hyperparameters, including
training-only student-rate calibration; no additional HPO sweep. These decisions
are recorded here and still need to be encoded in the executable YAML.

DCGM uses all eight individual models listed above. HMDC uses exactly these
ordered pairs, with both pair members as the primary downstream evaluators:

- `GAT + RGAT`
- `Transformer + HGT`
- `GCN + GIN`
- `HeteroGNN + HGT`
- `GIN + HEAT`

| Generation | Targets | Initialization at each model/pair restart | DCGM runs | HMDC runs |
| --- | --- | --- | ---: | ---: |
| `conservative` | `learned` | Fresh random weights | 40 | 25 |
| `conservative` | `solver` | Fresh random weights | 40 | 25 |
| `conservative` | `learned` | Same fixed weights | 40 | 25 |
| `conservative` | `learned` | Same T1 pretrained weights | 40 | 25 |
| **Total** | | | **160** | **100** |

HMDC adaptation loads each pair member's own T1 checkpoint, using the same
source/target mapping and checkpoint seed as DD. All eight architectures are
covered; report by pair and model because GIN and HGT each occur twice.
Evaluation outside the generating pair is deferred to a separate transfer study.

Share the 100 initial grids/reference IDs per case, batches of 10, operational
strength 0.10, +/-25% bus limits, feasibility requirements, architectures and FP64
precision. Preserve the same train split, selection/reporting validation panels,
evaluation seed pools, equal-size real-data and initial-grid controls, and
unchanged-checkpoint adaptation control. Fixed-mode evaluation includes unseen
initializations. Match **100 ordinary-SGD downstream updates**; compare DCGM/HMDC
with DD's `common_evaluation.json`, reporting DD's native learned-rate curriculum
separately. Learned targets remain explicit extensions of the condensation methods.

DCGM defaults: `data_rate=0.1`, `target_rate=0.003`, `restart_rounds=11`,
`student_epochs=1`. Its student rate is calibrated then constant; ten intervening
synthetic epochs give 100 student SGD updates per restart cycle.

HMDC defaults: `data_rate=0.005`, `target_rate=0.003`, `restart_rounds=100`,
`real_steps=2`, `alignment_rate=0.01`, `mutual_rate_factor=0.1`,
`query_interval=10`, `candidate_pool=256`, `real_batch=64`, `mutual=true`,
`balanced=true`, `balance_interval=10`. Its pair trains on real data with mutual
feature learning; the restart count measures condensation rounds. Keep that
schedule distinct from DCGM's synthetic student training and DD's unrolled updates.

Retain 2,400 active synthesis seconds, the existing 1,000-update cap and resumable
checkpoints/evaluation. Record actual updates and real-example exposure; equal
compute budgets do not imply equal real-data exposure, especially for HMDC.
The agreed DD/DCGM/HMDC plan therefore contains **420 synthesis runs** in total.

## Updated DD pilot (2026-10-12)

**Completed: 96/96 conditions**, including saved synthetic curricula and final
student evaluations. PBS job `8922460` used 32 nodes / 384 XPUs and finished in
46 minutes. The main comparison has 81 conditions; another 15 examine fixed
model initialization and operational ranges. Historical 48-grid results are
retained separately below.

The reference is **100 grids, 10 batches of 10, 10 passes = 100 student SGD
updates**. All three target modes jointly learn **100 independent per-update
learning rates**; the starting rate is only an initialization. The number of
student updates is fixed during distillation. Existing grids can be used for
longer training, but optimizing explicitly for a longer trajectory requires
continuing distillation with that trajectory and its rate schedule; the grids
need not be reinitialized. Outer distillation ran for an 800-second budget per
condition, producing 14–133 outer iterations rather than exactly 100.

Use [`dd_pilot.py`](dd_pilot.py) to prepare/qualify the experiment and
[`dd_pilot.pbs`](dd_pilot.pbs) as the external PBS launcher. Set `DD_SOURCE` to
an immutable source snapshot, `DD_OUTPUT` to the experiment root on Lustre, and
`DD_STAGE` to `qualify`, `calibrate`, or `pilot`. Qualification/calibration each
use one debug node. `python -m distillation.experiments.dd_pilot manifest --root
<DD_OUTPUT>` writes the pilot manifest after qualification and calibration pass.
The pilot uses up to 32 nodes, 12 XPUs per node; resubmitting the same manifest
and source resumes incomplete experiments and skips completed ones.

The independent choices are generation protocol (`conservative`, `random_fixed`,
`random_size`), target mode (`solver`, `frozen`, `learned`), model initialization
(`random`, `fixed`), operational range, and curriculum. All eight benchmark
backbones are supported; this pilot compares GAT/GIN/HeteroGNN, cases 30/57/118,
and all nine generation/target combinations. Additional case30 runs compare
fixed initialization and strengths 0.02/0.10/0.20. All synthetic inputs require
successful AC-OPF audits; independent teaching targets are stored separately
from physical solutions. Topology stays fixed after initialization in this pilot.

| Target mode | Teaching targets during distillation |
| --- | --- |
| `solver` | Re-solve changed grids and refresh targets; gradients treat the current targets as constants and do not differentiate through OPF. |
| `frozen` | Keep the initial OPF targets while learning grid attributes and rates; targets may cease to solve the modified grid. |
| `learned` | Optimize targets alongside grid attributes and rates; teaching targets need not be physical OPF solutions. |

Default students have width 128 and three layers. Dropout is disabled in the
DD trajectory and matched controls. Starting rates are selected on training
examples, shared across protocols for each model/case. The real-data control
gets a separate train-only rate sweep. Seeds, full configurations, source hashes,
learning curves, physical-parameter changes and resumable checkpoints accompany
the exported curricula. Test data remain untouched. A pilot condition has one
synthesis seed; multiple evaluation seeds do not substitute for independent
synthesis replications.

Pilot output root:
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/dd_100x100_20261012`.
Use `best.pt` for the validation-selected curriculum, including its ordered
inputs, teaching targets and learned rates. `distilled.pt` is the final curriculum;
`latest.pt` is the resumable optimization state. `evaluation.json` compares fresh
student training on synthetic and real examples. Fixed-initialization results
and transfer to unseen model initializations are distinguished.

### Validation after training exclusively on synthetic grids

**This evaluation ran.** For each main condition, the evaluator builds 12 fresh,
randomly initialized students (seeds 20,000,000–20,000,011), trains each for all
100 ordered updates using only the selected synthetic curriculum, then measures
error on **512 real validation examples**. There is no real-data fine-tuning or
pretrained student checkpoint in these evaluations. Real training examples were
used earlier by the outer distillation objective to learn the curriculum.

The curriculum is selected using a different panel of 128 validation examples.
The 512-example reporting panel is disjoint from that selection panel; it is a
subset of validation, not the full validation split. The test split is untouched.
The evaluation path is `final_evaluation → panel → evaluation → inner` in
[`DD/campaign.py`](../DD/campaign.py).

Each cell below is **mean validation MSE ± population standard deviation across
the 12 student initializations**, read from `evaluation.json: best_distilled`.
Lower is better. The [shared loss](../synthetic/learning.py) gives equal weight
to each graph and to bus versus generator targets, averaging their two output
channels within each type. It uses the encoded voltage and generation targets.
These are endpoint scores after 100 updates; seed variability is not a confidence
interval or replication of the distillation experiment.

| Model | Case | Generation | `solver` | `frozen` | `learned` |
| --- | ---: | --- | ---: | ---: | ---: |
| GAT | 30 | conservative | 0.1553 ± 0.1865 | 0.1530 ± 0.1829 | 0.1546 ± 0.1855 |
| GAT | 30 | random_fixed | 0.1440 ± 0.1599 | 0.1438 ± 0.1610 | 0.1205 ± 0.1163 |
| GAT | 30 | random_size | 0.1308 ± 0.1324 | 0.1493 ± 0.1643 | 0.1476 ± 0.1615 |
| GAT | 57 | conservative | 0.6707 ± 0.3850 | 0.6652 ± 0.3992 | 0.5994 ± 0.2874 |
| GAT | 57 | random_fixed | 0.9731 ± 0.9436 | 0.7505 ± 0.8687 | 1.0211 ± 1.2122 |
| GAT | 57 | random_size | 0.6923 ± 0.5609 | 1.0691 ± 1.1213 | 1.0061 ± 1.0382 |
| GAT | 118 | conservative | 0.7537 ± 0.2979 | 0.8039 ± 0.3376 | 0.8039 ± 0.3376 |
| GAT | 118 | random_fixed | 0.7945 ± 0.2959 | 0.7906 ± 0.2963 | 0.7905 ± 0.2963 |
| GAT | 118 | random_size | 0.7982 ± 0.2946 | 0.8027 ± 0.3102 | 0.8075 ± 0.3097 |
| GIN | 30 | conservative | 0.0513 ± 0.0041 | 0.0513 ± 0.0041 | 0.0407 ± 0.0017 |
| GIN | 30 | random_fixed | 0.0536 ± 0.0043 | 0.0536 ± 0.0043 | 0.0533 ± 0.0042 |
| GIN | 30 | random_size | 0.0534 ± 0.0040 | 0.0534 ± 0.0040 | 0.0524 ± 0.0039 |
| GIN | 57 | conservative | 0.3248 ± 0.0107 | 0.3248 ± 0.0108 | 0.3111 ± 0.0084 |
| GIN | 57 | random_fixed | 0.5177 ± 0.0311 | 0.5131 ± 0.0299 | 0.3667 ± 0.0175 |
| GIN | 57 | random_size | 0.5803 ± 0.0771 | 0.5802 ± 0.0771 | 0.5771 ± 0.0764 |
| GIN | 118 | conservative | 0.5000 ± 0.0034 | 0.4993 ± 0.0035 | 0.4949 ± 0.0034 |
| GIN | 118 | random_fixed | 0.5574 ± 0.0054 | 0.5572 ± 0.0054 | 0.5570 ± 0.0054 |
| GIN | 118 | random_size | 0.5571 ± 0.0062 | 0.5571 ± 0.0062 | 0.5567 ± 0.0062 |
| HeteroGNN | 30 | conservative | 2.9888 ± 1.6410 | 2.9888 ± 1.6411 | 2.9871 ± 1.6408 |
| HeteroGNN | 30 | random_fixed | 3.2818 ± 1.8006 | 3.0139 ± 1.5490 | 2.9804 ± 1.5231 |
| HeteroGNN | 30 | random_size | 2.9228 ± 1.5335 | 3.1125 ± 1.7071 | 2.9999 ± 1.6072 |
| HeteroGNN | 57 | conservative | 2.4108 ± 1.0068 | 2.4104 ± 1.0067 | 2.4108 ± 1.0068 |
| HeteroGNN | 57 | random_fixed | 2.4166 ± 1.0027 | 2.4171 ± 1.0024 | 2.4171 ± 1.0024 |
| HeteroGNN | 57 | random_size | 2.4242 ± 1.0072 | 2.4234 ± 1.0071 | 2.4227 ± 1.0071 |
| HeteroGNN | 118 | conservative | 6.7692 ± 2.7880 | 5.5850 ± 2.1240 | 5.8470 ± 2.1970 |
| HeteroGNN | 118 | random_fixed | 9.5682 ± 4.2821 | 9.3846 ± 4.2840 | 9.5730 ± 4.3192 |
| HeteroGNN | 118 | random_size | 11.3635 ± 6.5120 | 11.1759 ± 5.8515 | 11.3153 ± 6.5171 |

### How far off are we from the full training 

- **Full data:** 240,000 distinct grids, trained for about 4.17 epochs / 15,625 updates.
- **Synthetic only:** 100 grids, trained for 10 passes / 100 updates, with learned per-update learning rates.

The synthetic column consistently uses **conservative generation + learned targets**. Both columns use the same MSE definition; lower is better. The MSE ratio is synthetic-only MSE divided by full-data MSE.

| Model | Case | Full-data val MSE | Synthetic-only val MSE | MSE ratio |
|---|---:|---:|---:|---:|
| GAT | 30 | 0.009815 | 0.154560 | 15.7× |
| GAT | 57 | 0.011775 | 0.599406 | 50.9× |
| GAT | 118 | 0.014556 | 0.803853 | 55.2× |
| GIN | 30 | 0.009868 | 0.040671 | 4.1× |
| GIN | 57 | 0.006455 | 0.311108 | 48.2× |
| GIN | 118 | 0.015857 | 0.494909 | 31.2× |
| HeteroGNN | 30 | 0.000588 | 2.987060 | 5076.5× |
| HeteroGNN | 57 | 0.213532 | 2.410829 | 11.3× |
| HeteroGNN | 118 | 0.177691 | 5.846972 | 32.9× |

**Comparison caveat:** These results are indicative rather than fully matched. Full-data results average three seeds on all 30,000 validation examples; synthetic results average twelve seeds on a 512-example subset. Optimizers and training dropout also differ.

### Comparison with 100 real training examples

The following compares each condition with its saved `real_subset_tuned_rates`
control: a fresh student trained on 100 real train examples for 100 updates,
using a constant rate selected by a separate train-only sweep. Synthetic student
rates are learned per update. Relative differences are computed within each
condition before taking medians; raw MSEs from different cases are not pooled.

| Target mode | Lower mean MSE than real control | Median MSE change vs real | Conservative only: median change vs real |
| --- | ---: | ---: | ---: |
| `solver` | 9/27 | +7.90% | −0.75% |
| `frozen` | 10/27 | +3.61% | −2.39% |
| `learned` | 11/27 | +2.81% | −4.63% |

Negative changes mean lower MSE. Learned targets give the lowest mean MSE among
the three modes in 15/27 matched model/case/generation settings. Conservative
generation beats the real control in 21/27 comparisons; random_fixed does so in
5/27 and random_size in 4/27. These counts describe point estimates, not
statistically established wins; GAT and HeteroGNN have substantial seed variance.

To separate changes to the synthetic examples from the learned rate schedule,
`initial_topology_learned_rates` trains on the initial generated grids and their
initial OPF targets using the final learned rates. Relative to this control,
median MSE reductions are only **0.026% / 0.029% / 0.140%** for
solver/frozen/learned. Thus learning targets helps in some settings, but typical
improvements from changing the examples remain small. Connectivity was held
fixed during this pilot, so these results do not measure online topology search.

The 15 additional conditions are indexed separately: nine fixed-initialization
conditions and six range ablations. Fixed-initialization primary scores repeat
the same model seed and must not be interpreted as 12 independent students.
Unseen-seed transfer has explicit nonfinite outcomes in some ablations; inspect
`nonfinite_initializations` rather than treating a completed run as universal
numerical success. All 81 main synthetic evaluations in the table are finite.

What has not run in this DD pilot is longer synthetic-only student training or
training to convergence, evaluation on the full validation/test splits, or
independently replicated distillation runs. Learning curves are available every ten updates on a
32-example validation probe; those curve values differ from the 512-example
endpoint metric above.

### Comparison with the full 240,000-example training split

The question here is how much full-data accuracy the 100-grid curriculum retains.
The preceding 100-real-example control does not measure that. Completed
[T1 runs](../training/T1_single_topo/README.md) provide a full-data reference:
each case has 300,000 examples split into **240,000 train / 30,000 validation /
30,000 test**, with the same canonical split and seed 42 used by DD.

T1 trains for **1,000,000 sample presentations** (about 4.17 passes over the
entire train split), using 15,625 AdamW updates with batch size 64. The full-data
column below uses the **final 1M-budget checkpoint**, averaged over model seeds
42, 43 and 44, so every included model has seen every training example. It does
not use the separately selected `best.pt`, which can precede a complete epoch.
The synthetic column consistently uses `conservative` generation and `learned`
targets, averaged over the 12 fresh students described above: 100 SGD updates,
100 unique grids and 1,000 sample presentations.

Both columns use the DD MSE convention. T1 reports the sum of bus and generator
MSEs (`sdk_mse`), whereas DD averages those two terms; therefore **T1 values are
divided by two** below. Its differently weighted `mse` field is not substituted.

| Model | Case | Full real-data MSE | 100 synthetic grids: MSE | Synthetic / full MSE |
| --- | ---: | ---: | ---: | ---: |
| GAT | 30 | 0.009815 | 0.154560 | 15.7× |
| GAT | 57 | 0.011775 | 0.599406 | 50.9× |
| GAT | 118 | 0.014556 | 0.803853 | 55.2× |
| GIN | 30 | 0.009868 | 0.040671 | 4.1× |
| GIN | 57 | 0.006455 | 0.311108 | 48.2× |
| GIN | 118 | 0.015857 | 0.494909 | 31.2× |
| HeteroGNN | 30 | 0.000588 | 2.987060 | 5076.5× |
| HeteroGNN | 57 | 0.213532 | 2.410829 | 11.3× |
| HeteroGNN | 118 | 0.177691 | 5.846972 | 32.9× |

**The current 100-update synthetic curricula do not recover full-data accuracy.**
All 81 main synthetic-condition mean MSEs exceed their corresponding full-data
reference, including the generation/target combinations outside this table.
This does not establish the best accuracy achievable by training longer on the
100 grids, or by distilling them for a longer student trajectory.

This is an indicative comparison, not a matched re-evaluation: T1 uses all
30,000 validation examples, while DD uses their verified 512-example subset;
the model seeds, optimizers, dropout during training and update counts also
differ. Both use width 128 and `num_layers=3`. A controlled follow-up should
evaluate both training routes on the same validation examples and model seeds,
and report synthetic-only learning curves beyond 100 updates. The existing
full-data checkpoints can be reused for that evaluation.

Source values for T1 are `validation.sdk_mse / 2` in
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1/checkpoints/<model>/pglib_opf_case<case>_ieee/seed<seed>/history/001000000.json`;
DD values are `best_distilled.mean` in the corresponding `evaluation.json`.
The means use three full-data training runs versus twelve synthetic-trained
student initializations. These are existing saved results; this comparison did
not launch new training or infer convergence from the 1M-sample budget.

### Locate and reproduce a result

The [dataset index](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/dd_100x100_20261012/datasets.json)
links all 96 curricula. Main results are under
`<root>/experiments/<model>/case<case>/<generation>/<target>/evaluation.json`.
`best_distilled.mean`, `.std`, `.per_initialization` and `.model_seeds` contain
the table's source values; `.learning_curves` holds the probe trajectories.
For example, [GAT/case57/conservative/learned](/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/dd_100x100_20261012/experiments/GAT/case57/conservative/learned/evaluation.json)
has validation MSE **0.5994 ± 0.2874** after synthetic-only training.

Use that condition's `best.pt` and the [fresh-student replay example](../DD/README.md)
to reproduce its ordered inputs, teaching targets and all 100 learned rates.
The pilot's exact code is preserved in `<root>/source_v4`, alongside its source
hash inventory. This section reports saved evaluations; updating this analysis
did not launch additional training.


## Historical implementation details & early results (48 grids)

The tables in this section are the earlier recorded results, before the
100-update pilot above. They use a different curriculum and are retained as
historical notes rather than as the current result table.

## 1. Dataset Distillation (`DD`)

Three modes on how to synthetically generate electr. networks
- conservative: starts from training-reference grids (local topology edits + bounded operational perturbations + feasibility backtracking)
- random_fixed: generates fresh connected topologies, all with the base case’s bus count
- random_size: generates fresh connected topologies with varying bus counts, while preserving the aggregate base-case bus budget within each synthetic batch

Seeds: One fresh model per XPU; each new outer loop gets a freshly seeded batch of model.

#### Training Regimes

| Model | Case | conservative | random_fixed | random_size |
|---|---:|---:|---:|---:|
| GAT | 30 | 0.64 | 0.66 | 0.75 |
| GAT | 57 | 1.31 | 1.35 | 1.33 |
| GAT | 118 | 2.44 | 2.69 | 6.44 |
| GIN | 30 | 0.07 | 0.07 | 0.07 |
| GIN | 57 | 0.49 | 0.57 | 0.57 |
| GIN | 118 | 0.53 | 0.54 | 0.54 |
| HeteroGNN | 30 | 542.76 | 8.39 | 10.59 |
| HeteroGNN | 57 | 3.91 | 339.67 | 4.08 |
| HeteroGNN | 118 | 187.67 | 938.94 | 1680.43 |

*Random seed training. Conservative modifications of topology much more effective.*

$M=48$ synthetic grids in 3 batches (16 grids each). Most likely this can be extended significantly... batch size is $16$, not memory-limited.

#### Results: Mixed

| Model | Case | Conservative | Random fixed | Random size |
|---|---:|---:|---:|---:|
| GAT | 30 | +0.2% | +3.7% | +16.8% |
| GAT | 57 | +0.3% | +4.6% | −2.4% |
| GAT | 118 | ≈0.0% | −2.3% | +21.7% |
| GIN | 30 | +0.0% | +1.2% | +1.1% |
| GIN | 57 | −0.0% | +6.9% | +4.8% |
| GIN | 118 | −0.0% | +0.2% | +0.1% |
| HeteroGNN | 30 | +0.1% | −7.1% | −58.5% |
| HeteroGNN | 57 | +1.6% | −10.9% | −45.1% |
| HeteroGNN | 118 | +0.2% | −13.7% | −1.9% |

*Advantage of training on 48 synthetic vs. real grids. Training on synthetic data achieves lower mean MSE in 10 of 27 configurations. Training consists of just 3 updates, rather than many epochs.*

This earlier run used only three student updates and did not examine fixed
model initialization. The new pilot above uses 100 updates and includes that
initialization regime.

## Earlier improvement notes and current status

### DD
1. **Implemented:** student learning curves every ten updates on 32 validation
   examples, plus physical-parameter drift and acceptance diagnostics. Full
   per-batch evaluation on the reporting panel is still not recorded.

2. **Partly completed:** all eight models are supported and passed XPU
   qualification. The full pilot covers three; comparable distillation results
   for the other five models remain to be collected.

3. **Implemented range comparisons:** the original ±2% operational range became
   ±10% in the main pilot, with case30 strength 0.02/0.20 ablations. Online
   connectivity/node edits were not enabled in this pilot.

Increasing the existing AC-OPF edit parameters is the simpler experiment—and worth doing. There are distinct controls:

   Control                                          Effect
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   Increase DD strength, e.g. 0.02 → 0.10 → 0.20    Opens a wider range around the initial physical parameters
  ───────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────
   Increase outer_rate                              Makes optimizer steps more aggressive, including learned learning-rate updates
  ───────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────────────────
   Enable periodic topology search                  Allows connectivity/node edits during DD; changing strength alone does not

  I would increase the allowed range first while keeping the optimizer step size fixed.

4. **Implemented:** `target_mode` is independent of topology initialization
   (conservative, random_fixed, random_size). All three variants below ran in
   the main pilot:

   Target mode    What happens during distillation?                                            Interpretation
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   solver         Recompute detached AC-OPF targets after accepted grid changes.               Current physically consistent approach.
  ─────────────  ───────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────
   frozen         Initialize targets from AC-OPF once, then keep them fixed while learning     Fixed teaching targets; they may cease to solve the modified grid.
                  grid parameters and learning rates.
  ─────────────  ───────────────────────────────────────────────────────────────────────────  ────────────────────────────────────────────────────────────────────
   learned        Initialize targets from AC-OPF, then optimize them independently             Learned teaching targets; they need not satisfy AC-OPF.
                  alongside grid parameters and learning rates.

5. **Implemented:** 100 synthetic grids and 100 student updates, with a separate
   learned rate for every update. Longer optimized trajectories and independent
   synthesis replications remain follow-up experiments.



## 2. Dataset Condensation (DC) w/ Gradient Matching

This is an earlier run.

Validation MSEs after training exclusively on **48 synthetic grids**, averaged over **12 fresh students** on the same **1,024 validation examples**. Each cell is **DD / DCGM**; lower is better. DD results include repeat runs with bus ranges corrected to ±25%.

| Model | Case | Conservative | Random fixed | Random size |
|---|---:|---:|---:|---:|
| GAT | 30 | 0.640 / 0.611 | 0.657 / 0.646 | 0.754 / 0.626 |
| GAT | 57 | 1.315 / 1.333 | 1.351 / 1.338 | 1.327 / 1.181 |
| GAT | 118 | 2.442 / 2.316 | 2.686 / 2.295 | 6.440 / 2.307 |
| GIN | 30 | 0.0651 / 0.0510 | 0.0655 / 0.0542 | 0.0655 / 0.0538 |
| GIN | 57 | 0.487 / 0.361 | 0.572 / 0.714 | 0.573 / 0.705 |
| GIN | 118 | 0.532 / 0.502 | 0.544 / 0.555 | 0.545 / 0.535 |
| HeteroGNN | 30 | 542.76 / 4.06 | 8.39 / 4.09 | 10.59 / 4.88 |
| HeteroGNN | 57 | 3.91 / 0.617 | 339.67 / 0.631 | 4.08 / 0.618 |
| HeteroGNN | 118 | 187.67 / 224.83 | 938.94 / 154.95 | 1680.4 / 199.27 |

**Comparison caveat:** DCGM often achieved lower absolute MSE, but its students received ten times as many updates: DD used 3 learned-rate steps; DCGM used 30 fixed-rate steps. DCGM also used FP64. This comparison evaluates the complete experimental setups and does not isolate an algorithmic advantage.


## Matched DD / DCGM / HMDC configuration (implemented)

New default configurations share 100 grids, 10 batches of 10, width 128 / three
layers, FP64, ±10% operational ranges, ±25% bus limits and a 100-update downstream
SGD budget. Pretrained runs instead use the checkpoint's saved architecture.
All three methods support solver/frozen/learned targets and random/fixed/pretrained
initialization. Learned targets are an explicit extension of the condensation
papers; physical inputs remain solver-verified in every mode.

DCGM uses 11 matching rounds with one synthetic epoch between rounds (100 student
updates before restart). HMDC retains its own paired-model, real-data mutual
training and restart schedule. Synthesis updates and downstream training updates
are reported separately. Neither condensation method adopts DD's unrolled loss
or learned per-update rates.

DD's primary evaluation still replays its learned curriculum. Its additional
`common_evaluation.json` uses the same ordinary shuffled SGD, train-only calibrated
fixed rate, validation IDs and paired student seeds as DCGM/HMDC. Compare this
report's `results.best` with their `evaluation.json: results.best` (HMDC:
`results.<model>.best`). Same-budget real and initial-synthetic controls accompany
all methods. Fixed-mode reports distinguish unseen random initializations.
One hour of synthesis is an equal resource budget, not an equal number of real
examples or gradient computations, especially for HMDC's paired models.

Prepare a manifest without submitting jobs:

```bash
python -m distillation.experiments.comparison \
  --root /lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/comparison_100 \
  --references /lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation/dd_100x100_20261012/references \
  --models GAT GIN HeteroGNN --pairs GAT,GIN GIN,HeteroGNN
```

An external PBS script uses 12 MPI ranks per node and calls
`python -m distillation.synthetic.campaign --manifest <root>/manifest.json`.
The builder shares references and cached initial grids across methods and target
modes. It supports all eight models, other pairs, and a model-to-checkpoint JSON
for pretrained adaptation. Outputs and resumable datasets belong on Lustre.
These are new capabilities; the historical result tables above are unchanged.

Implementation validation: 206 distinct tests passed across the regression and
follow-up suites, covering all target policies, finite-difference derivatives,
rollback, pretrained identity checks, exact resume/replay and final-evaluation
resume. Thirteen additional CPU scale checks used 100 case-30 grids, width 128,
three layers and FP64: eight DCGM models (including 100 downstream updates), four
HMDC pairs covering all eight models, and one full 100-step DD meta-gradient.
New comparison manifests enable CPU/XPU qualification before training; this
implementation validation did not submit accelerator campaigns.

## Very speculative

1. Why stick with AC-OPT... Why not learn other, similar optimization problems on the grid ... use that to pretrain
```bash
AC-OPF → OPT: Learn graph optimization problems with AC-OPF-like structure but modified objectives or constraints. Use their solutions as synthetic training
  labels, optimizing for performance on real AC-OPF while discouraging trivial, uncoupled problems.
```
