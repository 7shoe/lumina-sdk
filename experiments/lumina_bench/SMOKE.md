# Aurora smoke record — 2026-10-03

Job `8901971`, node `x4710c6s4b0n0`, SDK commit `b3c53ea` (`xpu-support`).
Changes are additions under `experiments/lumina_bench/` only. The pre-existing
untracked `tests/xpu/QUALIFICATION.md` was preserved. The sibling `acopf` checkout
was read for integration context and remains unchanged.

Environment: existing `reason_venv_2026.1.0`, site PyTorch
`2.13.0a0+gitcf30153`, PyG `2.8.0.post1`, Intel Max 1550, driver `1.6.33578+77`.
Twelve XPU tiles and native XCCL were visible; a real `torch.ones(..., device='xpu')`
sum returned 4. No dependencies were changed. Experiments used one tile.

## Completed checks

| Check | Result |
| --- | --- |
| Protocol tests, including real case30 ground truth | **12 passed**, no skips; 23.42 seconds |
| PYPOWER reference comparison, line and phase-shifting transformer | **PASS** |
| Sample-weighted aggregation with unequal final batches | **PASS** |
| Split partition, fingerprint and checkpoint split-checksum checks | **PASS** |
| Shared bounded outputs, zero-width generator interval | **PASS** |
| Exact sample budget with a partial final training batch | **PASS** |
| FP32 construction with a caller's FP64 default, identical initialization | **PASS** |
| Non-finite rejection and existing-output protection | **PASS** |
| Checkpoint selection by minimum validation score | **PASS** |
| Standalone evaluation CLI, CPU, batch size 3, 16 test samples | **PASS** |
| PBS script body through MPI inside this allocation | **PASS**, Transformer smoke |
| Python parse/compile and shell syntax | **PASS** |

Cached case30 group 0 was processed into 15,000 graphs. The smoke configuration
uses 32 training, 16 validation and 16 test samples from disjoint persisted splits.
All three models completed four optimizer steps and changed parameters. Checkpoints
were reloaded on CPU and XPU. Evaluation with batch sizes 3 and 5 agreed within
the documented inference tolerance (`atol=2e-5`, `rtol=2e-4`).

| Model | Small model XPU smoke | Max CPU/XPU prediction difference | Width 128 / 3-layer XPU smoke |
| --- | --- | ---: | --- |
| HGT | PASS, 4 steps / 32 samples | 4.76837e-7 | PASS, 4 steps / 256 samples |
| Transformer | PASS, 4 steps / 32 samples | 9.53674e-7 | PASS, 4 steps / 256 samples |
| RGAT | PASS, 4 steps / 32 samples | 7.62939e-6 | PASS, 4 steps / 256 samples |

Solver-label evaluation on the 16-sample test subset gave normalized violation
`1.32555e-6`, balance L2 `6.34605e-6`, thermal L2 `9.14276e-7`, and zero prediction
error. Costs recomputed from labels differed from stored objectives by about
`2.58e-4` on average, consistent with stored float32 values. No ground-truth residual
was artificially set to zero.

These checks establish execution and metric consistency, not convergence, long-run
CPU/XPU training parity, or reproduction of the paper's reported scores. The
100k/1M-sample runs, full 20-group dataset, held-out topologies, AL, and scheduler
submission have not been run. `aurora.pbs.sh` is ready for the subsequent submission
step with paths/account supplied as described in `README.md`.

## Evidence and reproduction

Persistent evidence root:

`/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-paper/8901971/`

- `integration-final/smoke-results.json`: three-model portability summary.
- `integration-final/{HGT,Transformer,RGAT}/`: configs, manifests, histories,
  checkpoints, CPU/XPU evaluations, physics metrics, source hashes and environments.
- `sized/{HGT,Transformer,RGAT}/`: four-step checks using the proposed larger models.
- `launcher/8901971/Transformer-seed42/`: output of the actual MPI launcher body.
- `cli-evaluation.json`: final standalone CPU evaluator output.
- `verification-logs/`: retained test, integration, launcher and evaluation logs.
- `data/`: isolated processed cache; original archived data was symlinked, not changed.

```bash
source "$HOME/Projects/reasonABLE/env/Aurora/activate_reason_venv.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LUMINA_BENCH_TEST_DATA=/lus/flare/projects/FRAME-IDP/siebenschuh/lumina-paper/8901971/data
python -m pytest -q experiments/lumina_bench/tests
python -m experiments.lumina_bench.smoke \
  --data-root "$LUMINA_BENCH_TEST_DATA" --output /path/to/fresh/smoke-output
```

During development, an initial dataset-constructor call used a keyword supported
by a different OPFData interface; that experiment-local call was corrected before
the passing runs above. Existing SDK source was not patched to accommodate it.
