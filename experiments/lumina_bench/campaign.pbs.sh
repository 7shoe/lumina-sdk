#!/usr/bin/env bash
#PBS -N lumina-case30-9runs
#PBS -q debug-scaling
#PBS -l select=3
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
set -euo pipefail
: "${LUMINA_SDK_ROOT:?}"
: "${LUMINA_DATA_ROOT:?}"
: "${LUMINA_RAW_ROOT:?}"
: "${LUMINA_RUN_ROOT:?}"
: "${LUMINA_BENCH_ACTIVATE:?Activation script is required}"
source "$LUMINA_BENCH_ACTIVATE"
cd "$LUMINA_SDK_ROOT"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset MPI4PY_RC_INITIALIZE RANK WORLD_SIZE LOCAL_RANK MPI_LOCALRANKID SLURM_LOCALID PALS_LOCAL_RANKID
read -r -a campaign_models <<< "${LUMINA_MODELS:-HGT Transformer RGAT}"
read -r -a campaign_seeds <<< "${LUMINA_SEEDS:-42 43 44}"
campaign_tasks=$((${#campaign_models[@]} * ${#campaign_seeds[@]}))
mpiexec --pmi=pmix --envall -n "$campaign_tasks" --ppn "${LUMINA_PPN:-3}" --cpu-bind depth --depth 4 \
  python -u -m experiments.lumina_bench.campaign \
  --config "${LUMINA_CONFIG:-experiments/lumina_bench/configs/case30_mse.yaml}" \
  --data-root "$LUMINA_DATA_ROOT" --raw-root "$LUMINA_RAW_ROOT" \
  --output "$LUMINA_RUN_ROOT/${PBS_JOBID%%.*}" \
  --models "${campaign_models[@]}" --seeds "${campaign_seeds[@]}"
