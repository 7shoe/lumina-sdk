# AC-OPF dataset distillation

Wang-style dataset distillation: differentiate through ordered student SGD steps
to learn synthetic grid parameters, teaching targets and learning rates. Physical
generation and feasibility checks use `distillation.synthetic`; AC-OPF solves
and discrete topology changes remain outside autograd.

The reference curriculum now uses **100 grids: 10 batches of 10, repeated for
10 passes (100 student SGD updates)**. Each update has its own learned rate.
All eight benchmark backbones are supported; new defaults use width 128, three
layers and FP64, with dropout disabled during the differentiable trajectory.

| Setting | Choices |
| --- | --- |
| `generation_protocol` | `conservative`, `random_fixed`, `random_size` |
| `target_mode` | `solver` refreshes detached OPF labels; `frozen` keeps initial labels; `learned` optimizes independent targets |
| `initialization` | `random` draws fresh weights; `fixed` resets to `model_seed`; `pretrained` resets to a T1/T2 checkpoint |
| `strength` | Operational range around initialization; default 0.10 |
| `batches`, `batch_size`, `passes` | Curriculum partition and repeated passes |

All modes require solver-verified feasible inputs. Frozen/learned targets are
teaching signals, **not AC-OPF solution certificates**. Solver-mode acceptance
scores proposals after refreshing labels; gradients still exclude the solver.
Learned reference-bus angles retain the original angle convention. Online
topology changes currently require solver targets; all initialization protocols
work with all target modes. New defaults keep buses within ±25% of the base case.

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

For multiple experiments per node, use `distillation.synthetic.campaign` with a
manifest `method: DD`. The external launcher and qualification tool are in
[`../experiments/dd_pilot.pbs`](../experiments/dd_pilot.pbs) and
[`../experiments/dd_pilot.py`](../experiments/dd_pilot.py).

Store outputs on Lustre. `best.pt` and `distilled.pt` contain ordered inputs,
teaching targets, rates, physical solutions, model configuration and target
semantics. `latest.pt` resumes optimizer/data state; final evaluation resumes
from `evaluation.partial.json`. Learning curves sample every ten student updates;
logs summarize physical-parameter movement. Final controls include an independently
tuned real-data learning rate, selected on training examples. Fixed-initialization
results and transfer to unseen initializations are reported separately.
New runs also write `common_evaluation.json`: 100 ordinary shuffled SGD updates
with a training-only calibrated fixed rate, matching DCGM/HMDC's downstream
procedure. This additional comparison uses the learned grids and teaching
targets; the primary DD result still replays the learned ordered rates.

To distill an adaptation curriculum from existing weights, add these settings to
an experiment's configuration:

```python
from distillation.DD.campaign import default_config

config = default_config(118, "GIN")  # Target case, independent of pretraining.
config.update(initialization="pretrained", pretrained_checkpoint=
    "/lus/flare/projects/FRAME-IDP/siebenschuh/acopf_models/T1/checkpoints/"
    "GIN/pglib_opf_case57_ieee/seed42/final.pt")
```

T1 and T2 `best.pt`, `final.pt` and `last.pt` are accepted for all eight models.
Loading and replay use the same [shared utility](../synthetic/README.md) as DCGM and HMDC.
Saved model options override DD's architecture defaults; weights and buffers load
strictly. Every outer update and real-data control starts at the same checkpoint,
using fresh inner SGD and learned rates. All generation/target modes remain
available. This optimizes adaptation to **one checkpoint**; repeated XPU workers
are not independent model initializations. Evaluation includes
`pretrained_baseline` before adaptation. The checkpoint hash and pretraining
specification are saved; keep the source file unchanged and available to resume.

Replay an exported curriculum (including pretrained starting weights, without
needing the original checkpoint file):

```python
import torch
from distillation.DD.campaign import model_from_export, fit_export

artifact = torch.load("/path/to/best.pt", map_location="cpu", weights_only=False)
model = model_from_export(artifact, device="xpu:0")
weights, buffers = fit_export(model, artifact)
model.load_state_dict(weights | buffers)
```

The default seed reproduces the fixed regime; pass `seed=123` to assess unseen
random initializations. Pretrained replay always loads the embedded starting
weights. Preserve the exported batch order, rates, targets and all 100 steps
when reproducing the curriculum.

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
