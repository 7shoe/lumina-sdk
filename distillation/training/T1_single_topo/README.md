# T1: one model per case

Train GCN, GAT, GIN, Transformer, HGT, RGAT, HEAT, and HeteroGNN (GAT backend)
separately on cases 14, 30, 57, 118, and 500, using MSE loss and seeds 42, 43, and 44.
This gives **120 runs**: 8 models × 5 cases × 3 seeds.
Settings and paths live in [config.yaml](config.yaml).

## Protocol

- **1,000,000 training samples seen per run**, about 4.17 epochs of 240,000 samples.
  The final batch is shortened to meet the budget exactly. Holdout samples
  do not count toward this budget.
- Use `distillation.data.split.split_indices()` with the full canonical case name
  and split seed 42: 240,000 train, 30,000 validation, 30,000 test samples.
  The runner rejects any other split seed.
- Model seeds 42, 43, and 44 vary initialization and model randomness, including
  dropout. The separate `shuffle_seed: 42` gives identical training sample order
  across model seeds, including the final partial epoch.
- Shuffle training samples each epoch; never train on validation or test samples.
- MSE = mean squared bus error (Va, Vm) + mean squared generator error (Pg, Qg).
  AdamW, learning rate 0.001, weight decay 0.0001, gradient clipping at 1.0,
  FP32, batch size 64, no learning-rate schedule.
- Reuse Lumina benchmark backbones, bounded output heads, and physical metrics.
  Select `best.pt` by validation `sdk_mse + violation`, as in the benchmark.
  `final.pt` contains the weights at the full 1M budget.
- Evaluate the selected checkpoint on the entire validation and test splits,
  storing separate sample-weighted results.

This is the 1M-budget T1-style experiment, not an exact reproduction of a 2M run.
The three model seeds share the same data split and sample sequence for comparison.

## Prepare once

The original SDK caches do not record canonical sample IDs. Preparation reads
the local tarballs, verifies IDs, and orders graphs by numeric `example_ID`.
It writes memory-mapped groups under
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/processed/T1`.
There is no download or shared extraction directory.

From the repository root; both PBS scripts bill the `EpiCalib` account:

```bash
qsub distillation/training/T1_single_topo/prepare_T1.pbs
```

Preparation uses two nodes and four CPU workers per node. All seeds share its
prepared data, so preparation does not need more nodes. Re-submit if it runs out
of time: completed groups are checked and skipped. Training requires all five
case manifests. Incomplete preparation exits nonzero, so a PBS `afterok`
dependency will not start training prematurely.

## Train and resume

```bash
qsub distillation/training/T1_single_topo/run_T1.pbs
```

The default requests 12 nodes for one hour, running six independent model/seed
runs per node on XPU tiles 0, 2, 4, 6, 8, and 10: 72 concurrent runs. Each run uses one tile;
it does not use distributed training. Workers claim unfinished runs using file locks.
At job startup, one process per node reads the prepared groups sequentially into
the operating system's file cache. This avoids slow random reads on cold nodes.
It does not alter the cache files, splits, or training order.
The scripts disable MPI's progress timeout because finished workers may wait
more than ten minutes for larger runs at the final collective.

**Re-submit the same command to continue.** Workers skip completed runs and load
`last.pt` for interrupted runs. The output path is stable across PBS job IDs.
Concurrent submissions cannot train the same run simultaneously.

For one run per node, including when checking case118/case500 memory use:

```bash
qsub -v T1_PPN=1 distillation/training/T1_single_topo/run_T1.pbs
```

`T1_PPN` accepts 1–6; six passed the 120-combination XPU smoke test. A 1M budget roughly
halves training work relative to 2M, but does not guarantee case500 finishes
in one hour. Workers stop cooperatively after 55 minutes and resume in later jobs.
The 120-run campaign can span several submissions.

Other PBS overrides: `T1_CONFIG` (absolute YAML path), `T1_REPO`, `T1_ACTIVATE`,
`T1_SECONDS` (default 3300), and `T1_PREP_PPN` (preparation workers per node).
Keep `T1_SECONDS` below the requested walltime with time left to save checkpoints.
Use a config copy to select cases/models or add seeds. Changing an existing run's
training settings, model, source code, or dataset requires a new output root.

With the Python environment activated, inspect pending work without training:

```bash
python -m distillation.training.T1_single_topo.campaign --dry-run
```

Run one pair directly with the same resume behavior:

```bash
python -m distillation.training.T1_single_topo.train \
  --model HGT --case case500 --seed 42 --device xpu:0
```

## Checkpoints and results

Under `/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1`:

```text
checkpoints/<model>/<canonical-case>/seed<seed>/
  last.pt             # Model, optimizer, RNG, epoch/cursor, sample count
  best.pt             # Selected validation checkpoint
  final.pt            # Full-budget checkpoint
  config.json         # Frozen settings and code/data fingerprints
  split.json          # Exact sample IDs in each split
  status.json         # Progress and phase
  history/*.json      # Validation at each half epoch and the final budget
  run.log             # Appended across campaign submissions
  complete.json       # Written only after both holdout results exist
results/<model>/<canonical-case>/seed<seed>/
  val.json
  test.json
```

`last.pt` is replaced atomically every half epoch (120,000 training samples),
every five minutes at a training batch boundary, and on cooperative shutdown.
It also records the validation phase and checkpoint selection state. An abrupt
kill replays work since the last saved checkpoint. Exact RNG restoration is
tested on CPU; XPU operations may still have hardware-level nondeterminism.

Evaluation progress is saved separately every 32 batches and at shutdown, so an
interrupted holdout evaluation resumes without more optimizer steps. Completion
requires checksums for `best.pt`, `final.pt`, `val.json`, and `test.json`.

Results include:

- `opf_solution_error` / `sdk_mse`, plus scalar and per-output MSEs.
- `constraint_violation` / `violation`: (power-balance L2 + thermal-excess L2)
  divided by the square root of bus count. Bound residuals are also reported.
- `cost_difference`: predicted minus reference generation cost, with percentage
  difference when reference costs are nonzero.
- Training budget, selected-checkpoint sample count, split identity, and sample count.

## Local checks

```bash
python -m pytest -q distillation/training/T1_single_topo/tests
bash -n distillation/training/T1_single_topo/{prepare_T1,run_T1}.pbs
```

CPU tests cover all eight models, exact interrupted/uninterrupted training,
recovery after a failed checkpoint write, interrupted evaluation, canonical IDs,
fixed splits/sample order across model seeds, duplicate locks, and skipping completed runs.
