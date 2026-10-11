# AC-OPF heterogeneous model dataset condensation

HMDC condenses feasible grids using two different Lumina models, gradient
matching, gradient balancing and mutual feature distillation. Generation and
feasibility checks use [`distillation.synthetic`](../synthetic/README.md).

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

Optional tools, notebooks, results and detailed method notes are in
[aux/](aux/README.md). Runtime execution does not import it.

### Reference

Adapted from [Moon et al. (ECCV 2024): “Towards Model-Agnostic Dataset Condensation
by Heterogeneous Models”](https://arxiv.org/abs/2409.14538).
The paper studies image classification. This implementation uses graph
regression, fixed sampled topologies and detached AC-OPF labels with feasibility checks.

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
