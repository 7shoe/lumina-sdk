# T2: one model per topology collection

Train GCN, GAT, GIN, Transformer, HGT, RGAT, HEAT, and HeteroGNN (GAT backend)
on each collection: **{14,30,57}, {30,57,118}, {57,118,500}**.
MSE loss and initialization seeds 42, 43, and 44 give **72 runs**:
8 models × 3 collections × 3 seeds. Settings are in [config.yaml](config.yaml).

## Data and training

Reuse T1's prepared caches and the original `split_indices()` membership with
split seed **42**. Each topology contributes 240,000 training, 30,000 validation,
and 30,000 test samples. Holdouts are never used for optimizer steps.

Each run sees **1,000,000 training samples in total**. The sampler interleaves
three independently shuffled training streams, giving the first listed topology
333,334 samples and the other two 333,333 each. One combined epoch has 720,000
samples, so the budget is about 1.39 combined epochs. Shuffle seed 42 keeps the
training sequence identical across model seeds; model initialization and dropout
use the run's seed.

Each graph contributes its mean bus MSE (Va, Vm) plus mean generator MSE (Pg, Qg).
Average these graph losses, so larger topologies do not receive extra weight
because they contain more nodes. Use AdamW, learning rate 0.001, weight decay
0.0001, gradient clipping 1.0, FP32, batch size 64, and no learning-rate schedule.

Save checkpoints every half combined epoch (360,000 samples), every five minutes,
and at cooperative shutdown. Validate at half-epoch boundaries and at the final
budget. Select `best.pt` by mean validation `sdk_mse + violation`, matching T1;
`final.pt` records the full-budget weights and optimizer state.

## Submit and resume

Run T2 after T1 has completed. No new data preparation is needed.

```bash
source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh
qsub distillation/training/T2_multi_topo/smoke_T2.pbs
qsub distillation/training/T2_multi_topo/run_T2.pbs
```

Wait for the smoke test to pass before submitting training. The smoke test uses
two nodes in `debug`, checks all 72 combinations on real mixed-topology batches,
and exercises training interruption, optimizer/RNG restoration, and holdouts.
Its small sample subsets and outputs are separate from the full campaign.

Training requests **12 nodes in `debug-scaling`**, billed to **EpiCalib**.
Six workers per node use XPU tiles 0, 2, 4, 6, 8, and 10, one per physical GPU.
All 72 runs can train concurrently. Each run uses one tile; it is independent
of other workers. `T2_PPN` accepts 1–6; `T2_CONFIG` selects an absolute config path.
Before training, one process per node reads all prepared groups sequentially to
warm the operating system's file cache and avoid slow initial random reads.
MPI's progress timeout is disabled so early finishers can wait for larger runs.

Re-submit the same PBS script to continue unfinished work. Workers skip completed
runs and restore model, optimizer, RNG, epoch, and cursor from `last.pt`.
They stop after 55 minutes to save progress before the one-hour allocation ends.
`T2_SECONDS` can change this deadline while leaving time to save checkpoints.
File locks prevent duplicate workers from training the same run.

```bash
python -m distillation.training.T2_multi_topo.campaign --dry-run
python -m distillation.training.T2_multi_topo.train \
  --model HGT --collection case57_case118_case500 --seed 42 --device xpu:0
```

## Outputs

Under `/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T2`:

```text
checkpoints/<model>/<collection>/seed<seed>/
  last.pt, best.pt, final.pt
  config.json, split.json, status.json, complete.json
  history/*.json
  run.log
results/<model>/<collection>/seed<seed>/
  val.json, test.json
  val/<canonical-case>.json
  test/<canonical-case>.json
```

The selected checkpoint is evaluated on all **90,000 validation and 90,000 test
samples** per collection. Each topology also gets its own 30,000-sample report.
Combined results use metric sums and sample counts from the per-topology reports.
An interrupted evaluation resumes at its saved cursor without retraining.

Reports contain OPF solution error, constraint violation, signed cost difference
(prediction minus reference), component MSEs, and training/checkpoint sample counts.
Metric definitions and bounded output heads match [T1](../T1_single_topo/README.md).
Completion requires both checkpoints and all eight result files with matching
checksums. Data, source code, and settings are fingerprinted to prevent an
incompatible resume.

## Checks

```bash
python -m pytest -q distillation/training/T2_multi_topo/tests/test_t2.py
bash -n distillation/training/T2_multi_topo/{run_T2,smoke_T2}.pbs
```

CPU tests cover all eight models with different graph sizes, exact resume across
an epoch boundary, fixed data across initialization seeds, balanced sample counts,
per-graph MSE, interrupted evaluation, and repair of missing per-topology results.

## Campaign monitoring

The active campaign uses a monitor that waits for T1, re-submits unfinished runs,
checks every checkpoint and holdout, and only then starts T2:

```bash
source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh
setsid --fork python -u -m distillation.training.monitor_campaigns \
  </dev/null >>/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/campaign-monitor.log 2>&1
```

This launches the monitor independently of the terminal or chat session.
It reads the recorded jobs in each campaign's `jobs/submissions.json`, keeps
`jobs/training-status.json` current, and reduces later node requests as runs finish.
A monitor lock prevents duplicate submissions. A failed job stops the monitor for
inspection; a clean time-limited job resumes automatically. After verification,
`results/summary.csv` and `results/summary.json` collect the individual reports,
and `jobs/verified.json` records the audit.
