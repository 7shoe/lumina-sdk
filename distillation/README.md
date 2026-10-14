# Distillation Experiments

We use the 8 reference models of [lumina bench](https://arxiv.org/abs/2605.02133v1) `{GCN, GAT, GIN, Transformer, HGT, RGAT, HEAT, HeteroGNN (HGNN)}`.

## 1. Lumina Bench's T1 & T2 Campaigns
These runs emulate the T1 and T2 campaigns of the lumina benchmark. Slightly modified. 1M samples seen per model (equating to ~4.17 epochs per model).
These models serve as (a.) the trained baselines for subsequent fine-tuning experiments, (b.) as AC-OPF surrogate solvers for data condensation/distillation campaigns, 
and (c.) they serve as the baseline of models trained from scratch on randomly sampled data ()

#### T1: Single-Topology Training
8 models across `{14,30,57,118,500,2000}` with 4.17 epochs for 1M sample views.

The model checkpoints are available for subsequent runs
```bash
/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1/checkpoints # /results
```

#### T2: Multi-Topology
The model checkpoints are available for subsequent runs
```bash
/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T2/checkpoints # /results
```

## 2. Data Distillation & Condensation
This second wave of experiments explores new ground. It explored Wang's [Dataset Distillation](https://arxiv.org/abs/1811.10959) (`DD`), Zhao's [Dataset Condensation with Gradient Matching](https://arxiv.org/abs/2006.05929) (`DC(GM)`) and Moon's [Heterogenous Model Data Condensation](https://arxiv.org/abs/2409.14538) (`HMDC`).

### 2.1 Data Distillation (DD)

### 2.2 Data Condensation with Gradient Matching (DCGM)

### 2.3 Heterogenous Model Data Condensation (HMDC)

[HMDC implementation and experiments](HMDC/README.md) adapt the ECCV 2024
heterogeneous-model method with gradient balancing, spatial/semantic feature
alignment, and mutual distillation between two Lumina architectures. The
[runnable notebook](HMDC/aux/hmdc_mvp.ipynb) demonstrates the update sequence.
DD, DCGM and HMDC share the [feasible synthetic-grid pipeline](synthetic/README.md).

## 3. Baselines 
While dataset condensation is underexplored for AC-OPF, there are sampling-based strategies.

### 3.1 Bucketized Active Sampling (BAS)
See [Bucketized Active Sampling for Learning ACOPF](https://arxiv.org/abs/2208.07497).

### 3.2 Routine for Aggregating Maximally Broadcasted Opfs (RAMBO)
See [Scalable Bilevel Optimization](https://arxiv.org/abs/2304.10912)

## Appendix

### I. Appendix: T1 and T2 campaigns
Details on the T1 and T2 campaigns. Based on fixed datasplit of 80/10/10, resulting in 240K training samples available per case.
We fix the samples seen (1M) across T1 and T2 and *not* the effective number of epochs (as this would blend "data diversity" effect and "training exposure").
Three seeds for model instantiation (42,43,44) but always the same seed for data split (`42`) to maintain integrity of the standard data split.
Hence, we go after the core question:

```bash
Does broader coverage of topologies (when controling for samples viewed) improve performance?
```

For T1, each model was trained separately on one case: ~4.17 epochs.

| Case | Training presentations | Unique samples | Epochs |
|---|---:|---:|---:|
| 14  | 1M | 240K | 4.17 |
| 30  | 1M | 240K | 4.17 |
| 57  | 1M | 240K | 4.17 |
| 118 | 1M | 240K | 4.17 |
| 500 | 1M | 240K | 4.17 |

For T2, each model trained jointly on one collection (1M total presentations): ~ 1.39 epochs

| Collection | Case 14 | Case 30 | Case 57 | Case 118 | Case 500 |
|---|---:|---:|---:|---:|---:|
| {14,30,57}   | 333.3K | 333.3K | 333.3K | — | — |
| {30,57,118}  | — | 333.3K | 333.3K | 333.3K | — |
| {57,118,500} | — | — | 333.3K | 333.3K | 333.3K |
