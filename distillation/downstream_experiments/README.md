# Downstream training

Train fresh or pretrained benchmark models on published DD, DCGM, HMDC, BAS,
RAMBO, US and DSS datasets. `dataset_path.py` is the dataset feeder; `RND777`
is a separate uniform subset of the complete OPFData **train** split. Source
exports and historical T1/T2 and synthesis campaigns remain unchanged.

```bash
python -m distillation.downstream_experiments.dataset_path --records --method HMDC --model GAT+RGAT --case 30
python -m distillation.downstream_experiments.scripts.campaign prepare --config distillation/downstream_experiments/config.yaml
qsub -l select=2 -v DOWNSTREAM_MANIFEST=/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/downstream_experiments/downstream_v1/manifest.json distillation/downstream_experiments/run_downstream.pbs
```

Preparation freezes exact artifacts and writes `coverage.json`, including missing
conditions. It does not train models. The YAML describes the full matrix; reduce
cases, students, families and seeds for a pilot. An optional `dataset_ids` list
restricts published inputs to explicit feeder IDs. Pin ambiguous entries using
`canonical_ids`. Changing scientific settings/source requires a new campaign name.

| Family | Comparison |
|---|---|
| Individual M100 | Each creation model/pair -> every fresh student |
| Fixed | Recorded untrained weights versus fresh weights with identical model options; currently GAT/cases 30 and 57 for DD/DCGM |
| Pretrained | Random-created and pretrained-created datasets adapt compatible T1 checkpoints |
| Mixed M100/M400 | Uniform sampling within balanced creator quotas; HMDC pairs are creators |
| Joint/replay M400 | 200 per case; fresh joint training and source-checkpoint adaptation |

`families.adaptation_sizes` enables individual M100 and mixed M400 adaptation.
Target-only adaptation and replay both evaluate source and target cases. Replay
halves target exposure under the same total budget: this measures an allocation
tradeoff, not an isolated causal replay effect. Source replay uses random-created
data; pretrained target data must match the source->target direction.

Controls are deduplicated across families. Missing/partial datasets are never padded.
M400 uses 50 examples per creator with eight creators, 80 per pair with five HMDC
pairs, or 100 per pair with four. Other counts use deterministic balanced remainders.
M100 mixtures distinguish creator diversity from an increase in sample count.

The trainer imports T1 model definitions/AdamW defaults, T2 graph-balanced loss
and T1 physical evaluation. It preserves teaching targets. Fixed-init arms retain
recorded architecture/precision in matched and fresh conditions; the ordinary
benchmark profile uses FP32. This measures reusable data under a common trainer,
not native DD schedule replay (available separately through
`distillation.synthetic.artifacts.train_from_artifact`).

Full **validation** sets select `best.pt` (mean per-case `sdk_mse + violation`).
Full unchanged **test** sets are evaluated after training/checkpoint selection.
Adaptation also evaluates the initial source model on both tests, without using
these results for selection. Some validation examples already selected synthesis
exports; test is the final holdout. Student seeds measure training variability,
not synthesis variability. T1/T2 references have different training budgets.

Under `/lus/flare/projects/FRAME-IDP/siebenschuh/`:

| Location | Contents |
|---|---|
| `acopf_datasets/downstream_experiments/<campaign>/<corpus_id>/` | Portable `dataset.pt`, exact source IDs/indices/hashes, creator counts and unique-input/pair counts |
| `acopf_models/downstream_experiments/<campaign>/` | Frozen manifest, coverage, `results.csv` and `results.json` |
| `.../<campaign>/runs/<student>/<run_id>/` | `initial.pt`, `last.pt`, `best.pt`, `final.pt`, configuration, corpus, progress and full per-case val/test metrics |

PBS uses independent node launches and one model per configured XPU. The default
matches T1: six physical GPUs via FLAT indices `0,2,4,6,8,10`. All twelve visible
tiles can be selected explicitly. Inputs stage once per node. Failed students do
not cancel other nodes; resubmission restores optimizer, RNG, data position and
partial holdout evaluation. There is no gradient averaging across students.

`runtime.py` handles PBS execution separately from the frozen trainer. Each node
reads only its assigned run catalog, with identical run IDs and scientific settings.
Heartbeats cover staging and training; workers stop accepting runs near the deadline
and allow time to save checkpoints. Runtime hashes and launcher settings are saved
in `jobs/<PBS_JOBID>/runtime.json`. Existing campaigns can resume without rewriting
their manifests or checkpoints when only this execution layer changes.
PBS defers aggregate CSV/JSON collection to the separate `collect` command below;
the exact command is saved in `jobs/<PBS_JOBID>/collection.json`. Run it outside
the allocation so scanning all results cannot consume training time or overrun
the wall-time limit. Per-run checkpoints and metrics are saved during training.

`python -m distillation.downstream_experiments.watch --manifest /path/to/manifest.json
--job JOB_ID --max-followups 2 --max-nodes 96` monitors an existing allocation from
outside PBS. It verifies saved artifacts, submits smaller continuations when needed,
then collects reports. `continuation.json` records progress. The monitor must remain
running; the follow-up limit prevents endless retries, and a lock prevents duplicate
monitors. Frozen scientific settings are checked before every continuation.

```bash
python -m distillation.downstream_experiments.scripts.campaign run --manifest /path/to/manifest.json --device cpu --seconds 300
python -m distillation.downstream_experiments.scripts.campaign collect --manifest /path/to/manifest.json
# One condition:
python -m distillation.downstream_experiments.scripts.train --manifest /path/to/manifest.json --run-id RUN-... --device xpu:0
# Protocol tests:
python -m pytest -q distillation/downstream_experiments/tests/test_downstream.py
```

Verified on 2026-10-16: 23 CPU protocol tests; all seven published dataset formats
loaded; all eight models completed a finite CPU update on learned DD graphs.
Two-node XPU job `8934153` passed all 16 DD/RND777 case14 runs, including full
30,000-example validation and test evaluation (3m43s, exit 0). These two-update
runs qualify the pipeline; they are not estimates of final dataset quality.
Results: `acopf_models/downstream_experiments/qualification_xpu_9dd63e96b28c/`.
Reproduce preparation/verification with `python -m
distillation.downstream_experiments.tests.smoke_xpu [--verify /path/to/manifest.json]`.

Runtime qualification: 39 CPU tests passed. Two-node XPU job `8934451` deliberately
failed one checkpointed student at step 207; its sibling and the other node kept
training. The failed run resumed on the other node with an identical checkpoint
checksum. All four students completed 4,000 updates and full validation/test
evaluation. The test induces a process failure, not a hardware outage; see
`tests/isolation_xpu.pbs` and the Flare `qualification_isolation_20261016/` results.
