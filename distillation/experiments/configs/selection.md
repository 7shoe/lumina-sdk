# Selection sidecar

[selection.yaml](selection.yaml) runs BAS-IG, uniform selection and demand-stratified
selection on cases 14/30/57/118/500. It has its own frozen source, manifest and catalog.
Wave 1 configurations, source snapshots, datasets and catalogs are read-only inputs
to later comparisons; this sidecar never prepares or republishes them.

All methods use the existing split (seed 42). Demand buckets use all 240,000 training
inputs. A common 10,240-example panel is excluded from selection, leaving 229,760
eligible examples. Only BAS reads panel labels or trains a selector. Its validation
monitor uses the first 256 validation IDs; acquisition uses the training panel only.

The BAS selector is an MLP with IG scores and the existing patience schedule.
Its pool ceiling is 4,096 and the round ceiling is 500. The actual pool may be smaller:
round completion does not guarantee the acquisition budget was reached. A paused run
resumes; an exhausted, undersized pool is reported as `UNDERFILLED`. Zero-acquisition
runs are explicitly `initial_random_only`, not evidence of a BAS improvement.

Uniform and demand-stratified controls draw their pools directly, without selector
training. Stratification uses fixed training-demand buckets and equal bucket quotas.
All methods retain original OPFData targets; none runs a new OPF solve.

## Run

```bash
python -m distillation.experiments.selection prepare
```

This creates, but does not submit, the 15-condition campaign at:

```text
/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_selection/bas_controls_20261014/
```

Run from its frozen `source/` directory. An external PBS launcher can assign
each manifest index to a separate process/device; no cross-experiment MPI reduction
is required. For example, with `CAMPAIGN` set to the directory above:

```bash
cd "$CAMPAIGN/source"
python -m distillation.experiments.selection run --root "$CAMPAIGN" --index 0 --device xpu:0 --seconds 3300
python -m distillation.experiments.selection catalog --root "$CAMPAIGN"
```

Repeat the same run command to resume. Omit `--index` for sequential execution, or
use `--device cpu`. Preparation rejects changed configurations under an existing
campaign name. Choose a new name for changed settings.

## Datasets

```text
runs/<canonical-case>/<bas|uniform|demand_stratified>/seed42/
  selector/                      # BAS's resumable state and acquisition record
  exports/pool.json               # Immutable pool, provenance, export ordering
  exports/M100/{dataset.pt,artifact.json}
  exports/M800/{dataset.pt,artifact.json}
```

BAS and uniform exports use a seeded permutation of the completed selected pool.
Stratified exports interleave shuffled demand buckets so each size remains balanced.
Both yield nested M100/M800 subsets. Metadata records original IDs, split/source
hashes, selector settings, label exposure, initial-example counts and export seeds.
Incomplete sizes are never padded. Portable exports need no original OPFData files.

```python
from distillation.experiments.datasets import load_dataset
data = load_dataset("/path/to/exports/M100")
graphs, provenance = data["graphs"], data["metadata"]
```

The same adapter reads existing DD/DCGM/HMDC/RAMBO artifacts without rewriting them.
It preserves learned teaching targets and returns no model checkpoints or curriculum.
Fresh-student training is a separate experiment with one common training protocol.

## M800 from existing M100 datasets

`distillation.experiments.datasets.write_union(output_json, member_paths,
expected_count=800)` creates a reference-only manifest for explicitly chosen exports.
Use eight 100-example DD or DCGM datasets from the same case, target mode and
initialization regime. HMDC members are model-pair datasets: five pairs give 500
examples, even if their models cover all eight architectures. Do not repeat a
dataset to manufacture M800.

The union is a pool of separately optimized datasets, not joint condensation of
800 examples. It retains all pairs and reports distinct input graphs and distinct
teaching pairs; shared initialization can cause duplicates. Member checksums and
model identities are recorded. The same `load_dataset` reads the union manifest.
Store it in a new follow-up directory, outside Wave 1.

CPU checks: `python -m pytest distillation/test/test_selection_sidecar.py distillation/test/test_bas.py -q`.

Validated on 2026-10-14: 40 CPU tests passed; real case14 passed on CPU, and
case14/case500 passed on XPUs (job 8929279), including a small training step for
all eight architectures. Scoring 10,240 IG-panel examples took 6.4/7.3 seconds
after data loading. Cold case500 reads dominated startup; these are scoring
times, not end-to-end BAS runtimes. The BAS smoke checks resume and underfill,
not selection quality.

Reports and verified case14 DD/DCGM M800 unions are stored in the sibling
`dataset_selection/selection_support_20261014/` directory. Each union has 800
distinct inputs and teaching pairs. The 15-condition campaign is prepared;
the full selection runs have not been launched.
