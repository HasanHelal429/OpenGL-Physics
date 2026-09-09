#!/bin/bash
#SBATCH --job-name=nbody-gpu-remeasure
#SBATCH --account=m5032_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#
# Re-measure the GPU ladder at a high, fixed repetition count.
#
# The first GPU sweep produced a non-monotonic gpu_direct point (N=8,000 read
# slower than N=16,000), which is a measurement artefact rather than a
# property of the solver -- most plausibly a one-off inside the timed window.
# The GPU solvers are cheap enough that 25 reps everywhere costs a couple of
# minutes, so this re-runs the whole ladder rather than patching one point,
# giving a second independent sample of every N.
#
#   sbatch scripts/perlmutter_nbody_bench_gpu_remeasure.sbatch

set -u

BUILD_DIR="${BUILD_DIR:-$SLURM_SUBMIT_DIR/build/linux-release}"
OUT_DIR="${OUT_DIR:-$SCRATCH/nbody-bench/${SLURM_JOB_ID:-manual}}"
BIN="$BUILD_DIR/projects/02_nbody_gravity/02_nbody_gravity"
SWEEP="$SLURM_SUBMIT_DIR/scripts/nbody_bench_sweep.sh"

mkdir -p "$OUT_DIR"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
export OMP_NUM_THREADS=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')

echo "=== node: $(hostname) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1
echo "out dir: $OUT_DIR"
echo

[ -x "$BIN" ] || { echo "ERROR: not built: $BIN" >&2; exit 1; }

bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/new_gpu.csv" \
    --solvers gpu_direct,gpu_bh \
    --ns 1000,2000,4000,8000,16000,32000,64000,128000,256000,512000 \
    --reps 25 --label gpu-node-gpu-r25

echo
echo "=== done. results in $OUT_DIR ==="
ls -l "$OUT_DIR"
