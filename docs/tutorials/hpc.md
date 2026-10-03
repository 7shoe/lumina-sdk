# HPC Training

This guide covers HPC launchers and the Aurora XPU qualification workflow.

!!! note "Substitute the UPPERCASE placeholders for your environment"
    The job scripts below contain `<UPPERCASE>` placeholders that you must replace before submitting:

    - `<HPC_ACCOUNT>` — your allocation ID
    - `<CONDA_ENV_PATH>` — path to your conda env
    - `<LUMINA_REPO_PATH>` — your local clone of `lumina-sdk`

## Polaris (ALCF)

### Single-node script

Use the pre-built Polaris config:

```bash
#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A <HPC_ACCOUNT>

module load conda
conda activate <CONDA_ENV_PATH>
cd <LUMINA_REPO_PATH>

export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
export MASTER_PORT=29500

mpiexec -n 1 -ppn 4 \
 python example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case30 case118 \
  --group_ids 0 1 2 3 4
```

### Multi-node DDP script

Use the pre-built Polaris config:

```bash
#!/bin/bash
#PBS -l select=2:system=polaris
#PBS -l walltime=02:00:00
#PBS -q prod
#PBS -A <HPC_ACCOUNT>

module load conda
conda activate <CONDA_ENV_PATH>
cd <LUMINA_REPO_PATH>

NNODES=$(cat $PBS_NODEFILE | sort | uniq | wc -l)
NGPUS_PER_NODE=4
NTOTGPUS=$((NNODES * NGPUS_PER_NODE))

export MASTER_ADDR=$(hostname).hsn.cm.polaris.alcf.anl.gov
export MASTER_PORT=29500

mpiexec -n ${NTOTGPUS} -ppn ${NGPUS_PER_NODE} \
  python example/opf/train_opf_ddp.py \
  --config configs/config.polaris.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4 5 6 7 8 9
```

## Perlmutter (NERSC)

### Single-node script

```bash
#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH -A <HPC_ACCOUNT>

module load pytorch
cd <LUMINA_REPO_PATH>
pip install -e .

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500} 
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32

srun --ntasks-per-node 4 --gpus-per-task 1 \ 
  python example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 \
  --group_ids 0 1
```

### Multi-node DDP script

```bash
#!/bin/bash
#SBATCH -N 2
#SBATCH -C gpu
#SBATCH -G 8
#SBATCH -t 02:00:00
#SBATCH -q regular
#SBATCH -A <HPC_ACCOUNT>

module load pytorch
cd <LUMINA_REPO_PATH>/
pip install -e .

export SLURM_CPU_BIND="cores"
export MASTER_PORT=${MASTER_PORT:-29500}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)}
export OMP_NUM_THREADS=32

srun --ntasks-per-node 4 --gpus-per-task 1 \ # optional to specify -N if using a subset of nodes
  python example/opf/train_opf_ddp.py \
  --config configs/config.perlmutter.ddp.yaml \
  --cases case14 case118 case2000 \
  --group_ids 0 1 2 3 4
```

## Multi-Node Tips

- **Data staging**: Use `data.staging.root` in config to stage datasets to node-local storage (e.g., `$TMPDIR`)
- **Gradient accumulation**: Set `training.accumulate_grad_batches` to simulate larger batch sizes
- **Sharded datasets**: For large cases, pre-build shards with `scripts/opf_build_shards.py`
- **W&B logging**: Only rank 0 logs to W&B; use `--wandb` flag

## Device Visibility Notes

- **Perlmutter / Polaris**: each rank typically sees all node GPUs, so the trainer uses `LOCAL_RANK` to select the local device.
- **Frontier**: when using `job_submission_scripts/launch_rank_frontier.sh`, each rank gets one visible GPU via `ROCR_VISIBLE_DEVICES=$SLURM_LOCALID`, so the selected device index is `0` inside that process.

## Frontier (OLCF)

Frontier helper scripts are provided as templates under `job_submission_scripts/`.

1. Create or activate a ROCm 7.1.1 environment:

```bash
bash install/frontier/setup_env_frontier_rocm711.sh
```

2. (Optional) preprocess heterogeneous OPF data:

```bash
sbatch job_submission_scripts/job-frontier_data_preprocess.sh
```

3. Launch multi-node DDP training:

```bash
sbatch job_submission_scripts/job-frontier_16n_rocm711.sh
```

Set `FRONTIER_VENV_BIN` (and path overrides below) to match your allocation and filesystem layout before submission.

## Frontier Path Overrides

Use placeholders and override site-specific paths at runtime instead of committing cluster-specific absolute paths:

```bash
python example/opf/train_opf_ddp_frontier.py \
  --config configs/config.frontier.rocm711.yaml \
  --root <DATA_ROOT> \
  --logging_dir <LOG_DIR> \
  --checkpoint_dir <CKPT_DIR>
```

You can also use env vars:

```bash
export LUMINA_ROOT=<DATA_ROOT>
export LUMINA_LOGGING_DIR=<LOG_DIR>
export LUMINA_CHECKPOINT_DIR=<CKPT_DIR>
```

## Aurora (ALCF): demonstrated single-node XPU execution

OPF training and evaluation automatically select CUDA, then XPU, then CPU.
Distributed execution selects NCCL, native XCCL, or Gloo respectively, after
binding the local device. A build without native XCCL raises for XPU DDP;
single-device XPU selection does not require XCCL. There is no legacy CCL fallback.

**Demonstrated coverage (2026-10-03):** CPU/XPU synthetic numerical checks for
SAGE/GAT/RGAT, including GAT edge features, both scaling settings, five losses and
Adam/AdamW; CPU/XPU bound evaluation; all eight one-XPU and eight two-XPU HGT/RGAT
trainer configurations (single/multicase, scaling on/off); four twelve-tile trainer
jobs with scaling enabled; and two jobs with distinct per-rank masks. Native XCCL
checks include distinct physical tile UUIDs, synchronized gradients/parameters,
timed validation, checkpoint loading, public prediction/aggregate evaluation,
optimizer continuation and normal teardown. Actual case14 smoke training passed
on one/two XPUs, together with all three evaluation entry points for HGT/RGAT
checkpoints. Site PyTorch `2.13.0a0+gitcf30153`, PyG `2.8.0.post1`, and NumPy `2.3.5`
were preserved. Earlier CPU validation also covered all sixteen one-/two-rank
Gloo trainer configurations.

**Qualification is partial:** the final combined regression suite had **246
passes and one failure**, with no skips. The failure is actual-data RGAT AdamW
CPU/XPU parameter/update drift; output and gradient comparisons remain close.
Synthetic AdamW attention comparisons use a diagnosed absolute parameter/update
budget of `1e-4` at learning rate `1e-3`. See `tests/xpu/QUALIFICATION.md` in the
repository for thresholds, commands, failures and exact demonstrated scope.

**Not yet qualified:** multi-node execution, on-disk/sharded data, CUDA/HIP
hardware regression, HEAT/homogeneous models, other precisions and production-sized
configurations. Actual-data checks used a small model and bounded training samples;
they do not establish convergence or performance. RGAT's existing implementation
does not configure edge-feature attention.

Use the site `frameworks/2026.1.0` module and a virtual environment inheriting its
PyTorch installation. Review a constrained pip dry run before installing
`.[test,acopf]` and the existing evaluation dependencies `huggingface_hub` and
`safetensors`. Preserve site PyTorch; omit IPEX, legacy CCL bindings, and optional
PyG extensions for the initial baseline. Inspect inherited extensions too:

```bash
python -B - <<'PY'
import importlib.util
import inspect
import socket
import sys
import torch
import torch_geometric
import numpy
from torch_geometric import typing as flags
from torch_geometric.nn.dense.linear import HeteroLinear, HeteroDictLinear

print(socket.gethostname(), sys.executable)
for module in [torch, torch_geometric, numpy]:
    print(module.__name__, module.__version__, module.__file__)
print("XPU count", torch.xpu.device_count(), "XCCL", torch.distributed.is_xccl_available())
for name in ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]:
    spec = importlib.util.find_spec(name)
    print(name, None if spec is None else spec.origin)
    assert spec is None, "Use a clean site-compatible environment for the initial baseline"
for name in ["WITH_PYG_LIB", "WITH_SEGMM", "WITH_GMM", "WITH_TORCH_SCATTER", "WITH_TORCH_SPARSE"]:
    print(name, getattr(flags, name, False))
    assert not getattr(flags, name, False)
for cls in [HeteroLinear, HeteroDictLinear]:
    print(inspect.getsource(cls.forward))
assert torch.xpu.is_available() and torch.xpu.device_count() > 0
assert torch.distributed.is_xccl_available()
PY
```

Run that preflight inside the allocation; a login-node count of zero is insufficient.
Do not patch PyG capability flags to bypass unsupported kernels.

The main `example/opf/train_opf_ddp.py` launcher remains MPI-only. Local-rank
precedence is `MPI_LOCALRANKID`, `SLURM_LOCALID`, `LOCAL_RANK`, then
`PALS_LOCAL_RANKID`; clear stale values before launch. Distributed checkpoint
evaluation (`test_opf_ddp.py`) uses MPI if both `RANK` and `WORLD_SIZE` are absent,
and requires both for an explicit environment launch such as torchrun.
`evaluate_out_of_sample.py` also accepts `--device xpu` or `--device cpu`.
Modeler aggregate constraint postprocessing intentionally remains on CPU.

!!! note "Substitute allocation and shared output paths"
    Set `<COMPUTE_HOST>` to an allocated compute node, `<FREE_PORT>` to a shared
    free port, and use distinct shared output directories for every smoke run.

```bash
export MASTER_ADDR=<COMPUTE_HOST>
export MASTER_PORT=<FREE_PORT>
unset RANK WORLD_SIZE LOCAL_RANK
python -m pytest -q -s tests/xpu/test_numerics.py tests/xpu/test_evaluation.py
timeout 300 mpiexec --pmi=pmix --envall -n 1 --ppn 1 \
  python tests/xpu/trainer_smoke.py --expect-device xpu --output <SINGLE_XPU_OUTPUT>
timeout 300 mpiexec --pmi=pmix --envall -n 2 --ppn 2 \
  python tests/xpu/trainer_smoke.py --expect-device xpu --output <TWO_XPU_OUTPUT>
```

Repeat numerical execution; repeat trainer runs for `--model-type HGT` and `RGAT`,
with `--multi`, and with/without `--no-minmax-scaling`. Progress from two devices
to all process-visible tiles (the planned FLAT layout has twelve), then at least
two nodes. Test full visibility and distinct rank-specific masks. Under a single
device mask every local index is zero: record physical device identities to prove
different ranks use different devices. Record rank/mask/hierarchy environment,
module/driver versions, numerical errors, and complete job exit status.

Set `LUMINA_XPU_DATA_ROOT` to a prestaged raw **and** processed case14 cache to run
`test_processed_rgat_steps`; optionally set `LUMINA_XPU_MODEL_CONFIG` to the intended
model YAML. A missing cache with the root set fails; an unset root skips and leaves
qualification pending. Follow with actual-data simple/MPI training and all three
evaluation entry points, requiring nonempty finite metrics and expected batch
counts. Pass `--minmax_scaling` explicitly to the MPI training CLI. Qualify intended
width/depth/heads, batch size and accumulation, then on-disk and balanced sharded data.

Keep eager FP32 for initial qualification. A single-rank hang or teardown crash
is an unsuccessful job; retain the last stage and external timeout result for site
diagnosis. The smoke script's explicit `LUMINA_SKIP_XCCL_DESTROY=1` option is only
for a reproduced XCCL teardown issue and must be recorded if used. Production
teardown is unchanged; the workaround does not address an initialization/training hang.
See the [ALCF PyTorch guide](https://docs.alcf.anl.gov/aurora/data-science/frameworks/pytorch/)
and [known issues](https://docs.alcf.anl.gov/aurora/known-issues/) for site guidance.

## Existing HPC Documentation

Additional system-specific docs:

- [W&B sweeps on Perlmutter](../wandb_sweep_perlmutter.md)
<!-- - [HuggingFace integration](../huggingface.md) -->
