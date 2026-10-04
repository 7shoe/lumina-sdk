# Small LUMINA-Bench reproduction experiments

Additive experiments for the [LUMINA paper](https://arxiv.org/abs/2605.02133v1).
This directory reuses SDK datasets and all eight backbones: GCN, GAT, GIN,
Transformer, HGT, RGAT, HEAT and HeteroGNN (the HGNN entry).
AL uses a narrow SDK loss-manager extension; existing supervised behavior is preserved.
It adds the missing experiment protocol around them. No `acopf` installation,
W&B account, new package installation, or distributed process group is required.

**Implemented:** T1 single-topology MSE training, saved splits, common bounded
outputs, validation-score checkpoint selection, test/held-out evaluation, and
CPU/XPU checkpoint portability. Single process, one device per experiment.
Upstream Frontier AL is also supported for fixed-topology FP32 runs, including
MSE-to-AL epoch scheduling, checkpointed loss state and gradient diagnostics.
See [AL_UPSTREAM.md](../../AL_UPSTREAM.md) for the reference and remaining differences.
Multi-topology training, DDP, full resume and adaptation are not implemented here.

## Protocol and comparison limits

These are reconstructed experiments, not the authors' exact winning recipes.
The author split IDs, winning hyperparameters, and precise MSE reduction are
unavailable. Configurations deliberately avoid the SDK's large HPC defaults.

- Split each concatenated case/group dataset once with `split_seed`; persist
  all indices and processed-data SHA256 hashes. Training seed does not change
  the split. Group order is recorded. The upstream JSON processor does not sort
  files, so preserve the processed cache and manifest together across runs.
- All eight backbones receive the same sigmoid transformation for Vm/Pg/Qg.
  Angles remain raw. GCN/GAT/GIN/Transformer use the SDK 64/32-dimensional homogeneous
  conversion; only bus/generator targets are supervised. Existing SDK model
  implementations themselves are unchanged. HGT/RGAT use the existing path
  without edge-feature attention; RGAT retains its native behavior.
  HEAT and HeteroGNN receive metadata with relation feature widths and actual
  edge attributes. The extension recipe uses HeteroGNN's `gat` backend, matching
  the shipped SDK choice. Native layer/head/dropout conventions are retained:
  HeteroGNN's `num_layers: 3` means two message-passing blocks with dropout 0.1;
  homogeneous GAT uses one head, and HEAT uses four heads without added dropout.
  On the checked case30 sample, the SDK's default `OPFHomoWrapper` dropped mixed-
  width edge attributes (`edge_attr=None`). Our explicitly selected 64/32 converter
  preserves them. This is a documented recipe difference, not an XPU effect.
- Train FP32 with AdamW, constant learning rate and gradient clipping. No HPO,
  scheduler or early stopping. Stop at the exact samples-seen budget; a positive
  `max_steps` is an additional smoke-test cap. Epochs repeat until the cap.
- Report both `sdk_mse` (bus MSE + generator MSE) and `mse` (mean over all
  bus/generator output entries per sample), plus Va/Vm/Pg/Qg MSE separately.
  The explicit `selection_mse` setting chooses which enters the validation score.
  Default: `sdk_mse + violation`. Test data is used only after checkpoint selection.
- Compute physics on CPU in FP64 from each sample's actual branch, transformer,
  shunt, generation and demand data. Powers and ratings in the processed OPFData
  release-1 cache are **already p.u.**; costs are polynomials on p.u. generation.
  Do not multiply these coefficients or ratings by baseMVA again.
- `balance_l2` is the per-sample L2 norm of concatenated P/Q balance residuals.
  `thermal_l2` uses positive `|S|² - rate_a²` at both ends of each branch; a zero
  rating is unconstrained. `violation` averages `(balance_l2 + thermal_l2)/sqrt(Nbus)`
  over samples. Bounds are separate diagnostics, not included in that score.
  Batch means are never averaged without sample weights. Cost differences are
  signed, relative to costs computed from solver-label generation.
- The squared-flow/both-end convention and MSE reduction are explicitly recorded;
  these still need comparison with the authors' original evaluation code.

Transformer/case30 at 1M samples is a useful Table 11 reference (solution error
0.001146, violation 0.0747), not an acceptance threshold for our untuned recipe.

## Entry points

Run from the SDK checkout, with the previously validated Aurora environment active:

```bash
cd "$HOME/Projects/lumina-sdk"
source "$HOME/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# Choose external paths; these need not match the smoke session's paths.
export LUMINA_DATA_ROOT=/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-paper/data
export LUMINA_RUN_ROOT=/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-paper/runs
export LUMINA_RAW_ROOT=/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/raw/OPFData/dataset_release_1

python -m experiments.lumina_bench prepare \
  --config experiments/lumina_bench/configs/smoke.yaml \
  --data-root "$LUMINA_DATA_ROOT" --raw-root "$LUMINA_RAW_ROOT" \
  --split-file "$LUMINA_DATA_ROOT/case30-group0-split.json"

python -m experiments.lumina_bench.smoke \
  --data-root "$LUMINA_DATA_ROOT" --output "$LUMINA_RUN_ROOT/smoke-001"

python -m experiments.lumina_bench train \
  --config experiments/lumina_bench/configs/pilot.yaml \
  --data-root "$LUMINA_DATA_ROOT" --output "$LUMINA_RUN_ROOT/hgt-pilot-001" \
  --model-type HGT --seed 42 --device xpu:0 \
  --split-file "$LUMINA_DATA_ROOT/case30-group0-split.json"

python -m experiments.lumina_bench evaluate \
  --checkpoint "$LUMINA_RUN_ROOT/hgt-pilot-001/best.pt" \
  --data-root "$LUMINA_DATA_ROOT" --device cpu --batch-size 128 \
  --output "$LUMINA_RUN_ROOT/hgt-pilot-001/cpu-test.json"
```

`--raw-root` symlinks existing archives into an isolated SDK cache; it never
modifies source archives. Without it, the SDK can download missing groups.
**Prepare serially before concurrent jobs:** the upstream processor shares an
extraction directory. `prepare` loads every configured group, not just the first.

Use a fresh output directory for every run. Existing nonempty directories and
evaluation result files are rejected. Explicit XPU requests fail if unavailable.
Select any of the eight names above with `--model-type`, using a configuration
that contains its model options. Use `HeteroGNN` for HGNN.
For held-out evaluation add `--case case57 --groups 0` and prepare that cache
first. Its fixed test split is used, without target-topology training or validation.
Model metadata must be compatible; strict loading/forward errors are not suppressed.

| Config | Data | Training budget | Purpose |
| --- | --- | --- | --- |
| `configs/smoke.yaml` | group 0; 32/16/16 train/val/test subset | 32 samples, 4 steps | Pipeline check; width 16, 2 layers |
| `configs/pilot.yaml` | group 0; full 80/10/10 split | 100k samples | Short learning run; width 128, 3 layers |
| `configs/case30_mse.yaml` | groups 0–19; full split | 1M samples | Measured reconstruction; width 128, 3 layers |
| `configs/case30_mse_remaining.yaml` | groups 0–19; full split | 1M samples | GCN/GAT/GIN/HeteroGNN/HEAT extension of MSE job 8902050 |

The group-0 and full-dataset configurations require separate manifests. Prepare
the full configuration before launching full-data jobs. Configs contain no machine
paths. They use the small experiment schema, not the SDK trainer's YAML schema.
The MPI campaign entry point accepts `--split-file` to reuse the original MSE
campaign's persisted split and validate all data fingerprints before training.

## PBS launch

The launcher runs one independent experiment on one XPU tile. The script body
was tested inside the allocation; `qsub` submission has not been performed.
Its default is the group-0 pilot. Adjust queue/walltime using site policy and
measured pilot duration before the full 1M run.

```bash
export LUMINA_SDK_ROOT="$HOME/Projects/lumina-sdk"
export LUMINA_BENCH_ACTIVATE="$HOME/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh"
export LUMINA_BENCH_CONFIG=experiments/lumina_bench/configs/pilot.yaml
export LUMINA_BENCH_MODEL=HGT
export LUMINA_BENCH_SEED=42

qsub -A FRAME-IDP \
  -v LUMINA_SDK_ROOT,LUMINA_DATA_ROOT,LUMINA_RUN_ROOT,LUMINA_BENCH_ACTIVATE,LUMINA_BENCH_CONFIG,LUMINA_BENCH_MODEL,LUMINA_BENCH_SEED \
  experiments/lumina_bench/aurora.pbs.sh
```

Submit separate model/seed jobs by changing those two variables. There is no
legacy oneCCL setup or dependency installation in the launcher. The environment
must already contain the site-compatible PyTorch and SDK dependencies.

## Outputs and future acopf integration

Each run writes `config.json`, `protocol.json`, `environment.json`, `split.json`,
`history.jsonl`, `initial.pt`, `best.pt`, `last.pt`, and `results.json`. The results
include the best validation metrics, test metrics, ground-truth metrics, effective
split sizes, actual steps/samples, and timing. Provenance includes SDK commit,
experiment source hashes, PyTorch/PyG versions, hostname and device. No credentials
or complete environment dump is collected. Files stay outside the repository.

Checkpoints use `format=lumina-paper-experiment-v1`; they contain CPU weights,
the configuration and a training example needed to reconstruct model metadata.
They are not advertised as ordinary SDK trainer checkpoints. They contain no
optimizer/resume state. The evaluator checks the saved split checksum and data
fingerprints, and loads model weights strictly.

From `acopf`, make this checkout importable and call:

```python
from experiments.lumina_bench.runner import train, evaluate_checkpoint

result = train(config, data_root=cache_path, output=fresh_run_path, device="xpu:0")
metrics = evaluate_checkpoint(checkpoint_path, data_root=cache_path, device="cpu")
```

The namespace intentionally differs from `acopf`'s existing top-level
`lumina_bench`. Models are explicitly FP32 even if a caller changed PyTorch's
default dtype; construction temporarily selects FP32 and restores the caller's
default. No acopf files
are edited or imported. External orchestration owns machine paths and scheduling.

## Checks

```bash
LUMINA_BENCH_TEST_DATA="$LUMINA_DATA_ROOT" \
  python -m pytest -q experiments/lumina_bench/tests
bash -n experiments/lumina_bench/aurora.pbs.sh
```

PYPOWER is needed only for the independent branch-physics tests and is already
present in the validated Aurora environment. The training/evaluation runner does
not depend on it. See `SMOKE.md` for the recorded hardware check.
