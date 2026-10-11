# AC-OPF dataset distillation

Wang-style dataset distillation: differentiate through ordered student SGD steps
to learn synthetic grid parameters, teaching targets and learning rates. Physical
generation and feasibility checks use `distillation.synthetic`; AC-OPF solves
and discrete topology changes remain outside autograd.

| File | Purpose |
| --- | --- |
| `campaign.py` | Distributed training, topology updates, checkpoints and evaluation |
| `mvp.py` | Learnable data, differentiable inner loop and derivative checks |
| `topology.py` | DD scoring and selection of feasible topology proposals |
| `data.py` | Native OPFData loading with the existing seeded split |

External PBS launchers use:

```bash
python -m distillation.DD.campaign --manifest /path/to/manifest.json
```

The manifest contains `experiments`, each with `config` and `output`. Run with
12 MPI ranks per node and one experiment per node. Resubmit the same manifest
to resume `latest.pt`. Training and qualification (`--mode smoke`) do not import
`DD.aux` or `synthetic.aux`.

Notebooks, optional calibration/preparation/replay/reporting tools, example PBS
scripts and historical runs are in [aux/](aux/README.md). Their commands use
`python -m distillation.DD.aux.NAME`.


### Reference

The dataset distillation implementation is that of [Wang et al. (2018): "Dataset Distillation"](https://arxiv.org/abs/1811.10959).
Note, that this work introduced dataset distillation (loosely inspired by model distillation) for convolutional neural networks on MNIST data.

The experiments are encouraging, fixed-seed training allows ~100 synthetic images to alsmost achieve accuracy of training on full 50_000 MNIST images.

```bash
@article{wang2018dataset,
  title={Dataset distillation},
  author={Wang, Tongzhou and Zhu, Jun-Yan and Torralba, Antonio and Efros, Alexei A},
  journal={arXiv preprint arXiv:1811.10959},
  year={2018}
}
```

Caveats: 

(1.) randomly initialized models harder to train

(2.) full MNIST still neded to compute loss -> synthetic images "mediate" the training on real data not replace the need for real data.

(3.) M=100 images split into 10 batches of 10 images (oner per class), routine also learns learning rate, etc. -> very elaborate way to feed data in

(4.) Approach exclusively demonstrated for CNNs on computer vision data (MNIST, CIFAR): question how this extends to GNNs etc.
