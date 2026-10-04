#!/usr/bin/env bash
#PBS -N lumina-paper
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
# Supply account with qsub -A; paths via exported variables and qsub -v.
# One process / one XPU per experiment. No legacy CCL or package installation.
set -euo pipefail

: "${LUMINA_DATA_ROOT:?Set the prepared OPFDataset cache root}"
: "${LUMINA_RUN_ROOT:?Set an external run/artifact root}"
: "${LUMINA_SDK_ROOT:?Set the lumina-sdk checkout path}"
cd "$LUMINA_SDK_ROOT"
if [[ -n ${LUMINA_BENCH_ACTIVATE:-} ]]; then
    source "$LUMINA_BENCH_ACTIVATE"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
unset MPI4PY_RC_INITIALIZE RANK WORLD_SIZE LOCAL_RANK MPI_LOCALRANKID SLURM_LOCALID PALS_LOCAL_RANKID

LUMINA_CONFIG="${LUMINA_BENCH_CONFIG:-experiments/lumina_bench/configs/pilot.yaml}"
LUMINA_MODEL="${LUMINA_BENCH_MODEL:-HGT}"
LUMINA_SEED="${LUMINA_BENCH_SEED:-42}"
LUMINA_OUTPUT="$LUMINA_RUN_ROOT/${PBS_JOBID%%.*}/$LUMINA_MODEL-seed$LUMINA_SEED"
mkdir -p "$(dirname -- "$LUMINA_OUTPUT")"
mpiexec --pmi=pmix --envall -n 1 --ppn 1 --cpu-bind depth --depth 4 \
    "${LUMINA_BENCH_PYTHON:-python}" -u -m experiments.lumina_bench train \
    --config "$LUMINA_CONFIG" --data-root "$LUMINA_DATA_ROOT" \
    --output "$LUMINA_OUTPUT" --device xpu:0 \
    --model-type "$LUMINA_MODEL" --seed "$LUMINA_SEED" \
    2>&1 | tee "$LUMINA_OUTPUT.log"
