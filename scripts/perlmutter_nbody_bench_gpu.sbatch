#!/bin/bash
#SBATCH --job-name=nbody-bench-gpu
#SBATCH --account=m5032_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive
#SBATCH --time=20:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#
# GPU solver sweep (Phase 10/11 GpuDirect + GpuBarnesHut) on an A100, plus the
# CPU solvers on the SAME node so the GPU-vs-CPU comparison is not confounded
# by running the two halves on different hardware.
#
# A Perlmutter GPU node has one EPYC 7763 (64 cores), not the CPU node's two,
# so these CPU numbers are 64-thread and are NOT the ones to compare against
# the CPU-node sweep -- that is what perlmutter_nbody_bench_cpu.sbatch is for.
#
# The GPU solvers are OpenGL compute shaders and need a live GL context. There
# is no display server here, so this relies on GLContext's EGL backend.
#
#   sbatch scripts/perlmutter_nbody_bench_gpu.sbatch

set -u

BUILD_DIR="${BUILD_DIR:-$SLURM_SUBMIT_DIR/build/linux-release}"
OUT_DIR="${OUT_DIR:-$SCRATCH/nbody-bench/${SLURM_JOB_ID:-manual}}"
BIN="$BUILD_DIR/projects/02_nbody_gravity/02_nbody_gravity"
SWEEP="$SLURM_SUBMIT_DIR/scripts/nbody_bench_sweep.sh"

mkdir -p "$OUT_DIR"

export OMP_PROC_BIND=spread
export OMP_PLACES=cores

PHYS_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')

echo "=== node: $(hostname) ==="
lscpu | grep -E 'Model name|^Socket|^Core|^Thread'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "physical cores: $PHYS_CORES"
echo "out dir: $OUT_DIR"
echo

[ -x "$BIN" ] || { echo "ERROR: not built: $BIN" >&2; exit 1; }

# Fail loudly if the EGL path is unavailable rather than silently producing a
# CPU-only result set.
echo "### GL context check"
"$BIN" --bench gpu_direct --bench-n 1000 --bench-reps 2 || {
    echo "ERROR: GPU bench failed -- no headless GL context?" >&2
    exit 1
}
echo

NS_GPU=1000,2000,4000,8000,16000,32000,64000,128000,256000,512000

export OMP_NUM_THREADS=$PHYS_CORES

echo "### GPU solvers (A100, fp32)"
bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/new_gpu.csv" \
    --solvers gpu_direct,gpu_bh --ns "$NS_GPU" --label gpu-node-gpu
echo

echo "### CPU solvers on the same node, $PHYS_CORES threads (fp64)"
bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/new_gpu_cpuref.csv" \
    --solvers direct,barnes_hut,fmm --ns "$NS_GPU" --label gpu-node-cpu
echo

echo "=== done. results in $OUT_DIR ==="
ls -l "$OUT_DIR"
