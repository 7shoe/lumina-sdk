### Shared Selection utility `distillation.selection`

Selects real OPFData examples. This contrasts the synthetic sample generation. Mostly [BAS](https://arxiv.org/abs/2208.07497v3) and [RAMBO](https://arxiv.org/abs/2304.10912).
Code provides train-only pools, stable sample IDs, label-free input views, fixed scalar partitions, capacity-aware sampling, and subset reload. 
It contains no losses, models, or acquisition policies. BAS is the first consumer; RAMBO can use the pool and artifact contract without buckets. Physical graphs and labels are never optimized.

## Usage

```python
from distillation.selection import TrainingPool, selected_dataset
from distillation.training.T1_single_topo.data import CaseDataset

dataset = CaseDataset(data_root, 'case30')
pool = TrainingPool(dataset, dataset.splits(42), reference_ids)
candidate_inputs = pool.inputs(candidate_ids)
pool.acquire(chosen_ids, reason='my_strategy', details={'round': 0})
training_graphs = pool.labeled(chosen_ids)
selected = selected_dataset(dataset, '/path/to/run/selection.json')
```

Use the authoritative `CaseDataset.splits(42)`. Another random split with the same seed is not equivalent. Existing processed data and processor hashes are reused, and the cache and its preparation code are left unchanged.

## Components

**`TrainingPool(dataset, splits, reference_ids)`**
- Validates a complete, disjoint train/val/test partition.
- Inputs are readable only for training IDs. Labels are readable only for reserved reference IDs or acquired IDs.
- Reference IDs never enter training data.
- This is a programming contract only. OPFData files already contain labels, so simulated acquisition saves no solver work.

**`selection.json` / `selected_dataset`**
- The versioned record holds the source data identity, split hashes, ordered selected/reference IDs, acquisition history, and method metadata.
- `selected_dataset` verifies data identity and split membership, then returns a lazy PyTorch subset. No graph files are duplicated.

**`load_descriptors`**
- Computes total active demand (p.u.) from training inputs only, using the existing grouped tensors. No val/test values enter its cache.
- Reports demand in its actual units. It is not the original paper's regional factor and is not divided by an invented nominal network.

**`Buckets.fit`**
- Fits fixed boundaries to the descriptor vector (used by BAS).
- Merges duplicate boundaries and empty intervals, and reports the actual bucket count.

**`allocate`**
- Uses capped proportional quotas with deterministic largest remainders.
- Exhausted buckets get no samples. Zero-score remainders are split equally among available buckets.

All random decisions use a caller-owned NumPy RNG, independent of model/dropout RNG.

## Portable experiments

The separate [selection sidecar](../experiments/configs/selection.md) exports BAS,
uniform and demand-stratified pools as nested M100/M800 datasets. `artifacts.publish`
accepts a method-independent selection record and preserves original inputs/targets,
IDs and provenance. `artifacts.load_dataset` reloads these without the source cache.
Future selection policies can use this same exporter.

`distillation.experiments.datasets.load_dataset` also reads existing synthetic
exports and explicit dataset unions, without changing the Wave 1 implementation
or reusing its model weights.
