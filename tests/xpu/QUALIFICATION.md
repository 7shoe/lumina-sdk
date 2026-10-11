# Aurora XPU qualification record

Date: 2026-10-03. Starting commit: `3b6d52f`; starting tree clean, with no
applicable `AGENTS.md`. Job `8901893`, host `x4112c6s6b0n0`, one allocated node.

## Environment

The requested activation script selects
`/lus/flare/projects/FRAME-IDP/siebenschuh/envs/reason_venv_2026.1.0/bin/python`;
invoking `python` after activation selects this same executable. Site PyTorch
`2.13.0a0+gitcf30153`, PyG `2.8.0.post1`, NumPy `2.3.5`, driver `1.6.33578+77`.
Twelve Intel Data Center GPU Max 1550 tiles are visible under FLAT hierarchy.
Native XCCL and an actual XPU tensor computation passed preflight. Optional PyG
extensions were absent and their capability flags false. Site PyTorch and other
dependencies were preserved.

The session temporarily switched to a sandbox without `/dev/dri`. XPU probes
returned zero devices and MPI failed before Python with MUNGE permission errors.
Unrestricted access was restored at 06:36 UTC, after which the remaining hardware
jobs ran successfully. Earlier skipped or blocked attempts are not hardware passes.

## Demonstrated hardware execution

Final combined pytest result: **246 PASS, 1 FAIL, 0 SKIP**, in 50.47 seconds.
This includes all 160 synthetic CPU/XPU numerical cases, actual-data CPU controls,
the new identical-gradient XPU test, evaluation/import/device-policy regressions,
trainer prerequisites and real two-process Gloo forward checks. The retained
actual-data XPU failure is described below. `git diff --check` also passed.

| Check | Result |
| --- | --- |
| Preflight, real XPU computation, native XCCL | **PASS** |
| Single-XPU HGT/RGAT, single/multicase, scaling on/off | **PASS**, eight jobs |
| Two-XPU HGT/RGAT, single/multicase, scaling on/off | **PASS**, eight jobs |
| Twelve-XPU HGT/RGAT, single/multicase, scaling on | **PASS**, four jobs |
| Two-XPU HGT/RGAT multicase with distinct per-rank masks | **PASS**, two jobs |
| AdamW CPU/XPU identical-gradient optimizer regression | **PASS** |
| Actual case14 simple single-XPU training/validation | **PASS** |
| Actual case14 HGT/RGAT MPI training on one and two XPUs | **PASS**, four jobs |
| Actual HGT/RGAT checkpoint bound evaluation | **PASS**, two jobs |
| Actual HGT/RGAT checkpoint out-of-sample entry point | **PASS**, two jobs |
| Actual HGT/RGAT checkpoint MPI evaluation on two XPUs | **PASS**, two jobs |
| Actual-data RGAT CPU/XPU parameter/update comparison | **FAIL**, see below |

All **22 synthetic trainer matrix jobs** completed with exit zero. Assertions
checked disjoint sampler indices, synchronized gradients at each optimizer step,
synchronized final parameters, object/tensor collectives, timed validation,
checkpoint schema/loading, public CPU/XPU prediction and CPU aggregate evaluation,
restored AdamW continuation, and normal XCCL teardown. No teardown workaround was
used. Each twelve-XPU log records twelve device assignments and completion messages.
Gathered `(host, physical tile UUID)` pairs must be distinct; masked ranks both
use logical index zero while retaining different physical UUIDs.

Actual-data checks processed all 15,000 case14 group-0 scenarios from an existing
cached archive into an isolated cache. Stored bus features remain float64 and
targets float32; existing model/trainer feature casting was preserved. The simple
example trained for one epoch on 30 scenarios and validated on 15. MPI jobs used
32 global training samples and a 16-sample validation subset, with hidden width
16 and two layers. These are smoke tests, not convergence or performance results.

Both two-XPU training checkpoints contain serialized `xpu:0` tensors. Bound and
out-of-sample evaluation processed two batches of four scenarios, with finite,
nonempty bound statistics and aggregation weight eight. MPI evaluation reported
nonempty finite test metrics. Out-of-sample evaluation exercised the entry point
on case14 itself; this is not a cross-topology generalization claim. Its optional
additional feasibility dictionary was empty; only the reported bound/prediction
metrics were qualified.

## Numerical diagnosis and narrowly scoped test changes

The original proposed thresholds produced **144 PASS, 20 FAIL, 2 SKIP** across
numerical/evaluation tests. All failures involved CPU/XPU AdamW comparisons for
GAT/RGAT without edge features. One reproduction showed nearly cancelling
attention gradients around `1e-10`; AdamW's `eps=1e-8` amplifies their relative
rounding differences into larger parameter updates.

The user explicitly authorized more tolerant cross-device comparisons. A first
3%-of-learning-rate update allowance still left one failure: **163 PASS, 1 FAIL,
2 SKIP**. A diagnostic collector then measured all three steps of the twenty
originally failing cases; it did not count ignored comparison failures as passes.
Maximum synthetic update difference was `6.23018e-5`, cumulative parameter
difference `7.13915e-5`, and prediction/loss difference `2.38419e-7`.

The final test allows an absolute error of 10% of learning rate (`1e-4`) for
**both individual updates and total parameter drift**, only for CPU/XPU AdamW
GAT/RGAT without edge features. This allowance does not accumulate across steps.
CPU controls, Adam, SAGE, edge-feature cases, predictions, losses, gradients,
finiteness, and same-backend DDP synchronization retain their original thresholds.
The focused attention regression passed all sixty selected CPU/XPU cases.

A new independent AdamW test supplies identical CPU/XPU gradients spanning zero,
`1e-12` through `1`, with a sign change over three updates. Parameters, updates,
and first/second optimizer moments pass tight comparisons. This supports the
interpretation that the observed drift comes from gradient rounding sensitivity;
it does not establish real-data trajectory parity.

**Remaining failure:** `test_processed_rgat_steps[xpu]` exceeds the retained
`1e-4` parameter/update allowance on real case14 data. The first mismatch was
`2.76625e-4`. Diagnostic execution through all three steps found maximum cumulative
parameter difference `6.34372e-4` in
`convs.2.convs.<bus___ac_line___bus>.att_dst`; prediction/loss differences were at
most `6.19888e-6`, and gradient comparisons still met their original tolerances.
All values remained finite. No further tolerance increase or production-model
change was made. This numerical qualification remains **FAIL**, even though
actual-data training and evaluation completed successfully.

## Reproduction and evidence

Full original hardware evidence is under:
`/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-xpu-qualification/8901893/`.
Resumed execution, real-data configs/cache/checkpoints and diagnostics are in its
`resumed/` directory. `progress.md`, `matrix.jsonl`, and `actual-matrix.jsonl`
record commands, exit codes, durations, acceptance checks and deviations.
Logs are also copied under the ignored `training_logs/xpu-8901893/` directory in
this checkout. Private environment snapshots are not copied into the repository.

```bash
source /home/siebenschuh/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export LUMINA_XPU_DATA_ROOT=/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-xpu-qualification/8901893/resumed/actual-data
python -c 'import torch; assert torch.xpu.is_available(); print(torch.ones(4, device="xpu").sum().item())'
MPI4PY_RC_INITIALIZE=0 python -m pytest -q tests/xpu tests/trainer/opf \
  tests/evaluator/test_checkpoint_classification.py --tb=short
```

The MPI initialization override applies only to mocked MPI rank-discovery unit
tests in this pytest invocation; real Gloo subprocess tests still execute. Unset
it before actual MPI launches. The combined suite intentionally retains the
real-data numerical failure rather than skipping it.

Representative trainer command (change `-n`/`--ppn` to 1, 2, or 12, select
HGT/RGAT, add `--multi` or `--no-minmax-scaling`, and use a fresh output directory):

```bash
unset MPI4PY_RC_INITIALIZE RANK WORLD_SIZE LOCAL_RANK MPI_LOCALRANKID SLURM_LOCALID PALS_LOCAL_RANKID
export MASTER_ADDR="$(head -n 1 "$PBS_NODEFILE")" MASTER_PORT=29800
timeout -k 15 180 mpiexec --pmi=pmix --envall -n 12 --ppn 12 \
  --cpu-bind depth --depth 4 python tests/xpu/trainer_smoke.py \
  --expect-device xpu --model-type RGAT --multi \
  --output training_logs/xpu-resume/twelve-RGAT-multi-scaled
```

The out-of-sample CLI requires `--test-cases pglib_opf_case14_ieee` for this cache;
the shorthand `case14` caused an unsuccessful download attempt. The run was
repeated successfully with the full name. A temporary driver also initially
assumed W&B-style checkpoint subdirectories without enabling W&B; correcting its
output configuration resolved checkpoint discovery without production changes.

## Pending qualification

**NOT RUN:** multiple nodes (only one allocated), CUDA/HIP hardware regression,
on-disk/sharded datasets, production-sized model configurations, HEAT/homogeneous
models, other precisions and full convergence/performance studies. Multi-node
qualification needs at least two allocated nodes and a shared data/output path.

Current changes are test coverage, diagnosed test tolerances and documentation.
Production mathematics, checkpoint format and dependencies are unchanged. The
changes have not been committed or pushed.
