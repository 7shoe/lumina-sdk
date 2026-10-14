# AC-OPF heterogeneous model dataset condensation

HMDC condenses feasible grids using two different Lumina models, gradient
matching, gradient balancing and mutual feature distillation. Generation and
feasibility checks use [`distillation.synthetic`](../synthetic/README.md).

`campaign.default_config(case, pair)` shares DD/DCGM's **100 grids, batches of
10, 100 downstream SGD updates, width 128 / three layers, FP64 and ±10%
operational ranges**. It supports `solver`, `frozen` and `learned` targets;
all inputs remain solver-verified, but only solver targets are physical solutions.
Learned targets extend the reference method and participate in gradient balancing
and per-family normalization, with a separate `target_rate`. Pair training still
uses real data and mutual feature learning. Downstream evaluation uses the same
ordinary SGD and train-only calibration as DCGM, with learning curves and
unseen-weight controls for fixed initialization. Actual teaching targets survive
checkpointing and export; `synthetic.evaluation.fit_export` replays them.

Large cases can use `STUDY_HMDC_MICROBATCH=8` for bounded activation memory.
It accumulates full mean parameter gradients before matching, then recomputes
chunks for input derivatives; the objective and GBM remain the same, apart from
floating-point summation order. Student-training batches are unchanged.

| File | Purpose |
| --- | --- |
| `campaign.py` | Distributed training, evaluation, checkpoints and resume |
| `hmdc.py` | Gradient matching, balancing and update rules |
| `features.py` | Spatial/semantic feature alignment |
| `models.py` | Model pairs and saved training state |
| `selection.py` | Real training-example selection |

External PBS launchers use:

```bash
python -m distillation.synthetic.campaign --manifest /path/to/manifest.json
```

Set manifest `method` to `HMDC` and use 12 MPI ranks per node. Each experiment's
`output` belongs on Lustre under
`/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_datasets/dataset_distillation`.
Resubmit the same manifest and source to resume `latest.pt`. Synthetic datasets
are in `best.pt` and `condensed.pt`; model checkpoints are in `student_<model>.pt`.

For pretrained adaptation, set `initialization: pretrained` and
`pretrained_checkpoints: {GIN: /path/to/GIN/final.pt, GAT: /path/to/GAT/final.pt}`
for a GIN/GAT pair. Supply a checkpoint for every `evaluation_models` entry;
set that list to the pair if only those two are evaluated. The target `case`
is independent of checkpoint training cases. The [shared loader](../synthetic/README.md)
restores architecture options and pins checkpoint hashes. At pair restarts,
backbones reset to their checkpoints and alignment/balancing restart as before;
resume restores the in-progress pair. Real-data training and mutual learning
retain their existing schedule. Evaluation includes `pretrained_baseline` and
matched real-data adaptation; exports embed pair-member starting weights.
`random` remains the default; `fixed` repeats `model_seed`.

Optional tools, notebooks, results and detailed method notes are in
[aux/](aux/README.md). Runtime execution does not import it.

### Reference

Adapted from [Moon et al. (ECCV 2024): “Towards Model-Agnostic Dataset Condensation
by Heterogeneous Models”](https://arxiv.org/abs/2409.14538).
The paper studies image classification. This implementation uses graph
regression, fixed sampled topologies and solver-verified inputs with configurable
teaching targets.

```bibtex
@inproceedings{moon2024towards,
  title={Towards model-agnostic dataset condensation by heterogeneous models},
  author={Moon, Jun-Yeong and Kim, Jung Uk and Park, Gyeong-Moon},
  booktitle={European Conference on Computer Vision},
  pages={234--250},
  year={2024},
  organization={Springer}
}
```
