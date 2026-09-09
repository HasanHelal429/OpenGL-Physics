#!/bin/bash
#SBATCH --job-name=nbody-bench-cpu
#SBATCH --account=m5032
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#
# Unified-solver (ngrav) scaling sweep on a whole CPU node, driving
# 02_nbody_gravity's own --bench. Deliberately the same node type, thread
# counts and N ladder as the pre-unification sweep, so the two are
# comparable.
#
#   sbatch scripts/perlmutter_nbody_bench_cpu.sbatch

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
lscpu | grep -E 'Model name|^Socket|^Core|^Thread|NUMA node\(s\)'
echo "physical cores: $PHYS_CORES"
echo "out dir: $OUT_DIR"
echo

[ -x "$BIN" ] || { echo "ERROR: not built: $BIN" >&2; exit 1; }

CPU_SOLVERS=direct,barnes_hut,fmm,spherical_fmm
NS_FULL=1000,2000,4000,8000,16000,32000,64000,128000,256000,512000
NS_SERIAL=1000,2000,4000,8000,16000,32000,64000,128000

echo "### full node: $PHYS_CORES threads"
export OMP_NUM_THREADS=$PHYS_CORES
bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/new_cpu_full.csv" \
    --solvers "$CPU_SOLVERS" --ns "$NS_FULL" --label cpu-node-full
echo

echo "### serial: 1 thread"
export OMP_NUM_THREADS=1
bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/new_cpu_serial.csv" \
    --solvers "$CPU_SOLVERS" --ns "$NS_SERIAL" --label cpu-node-serial
echo

# Thread scaling at fixed N, one CSV per thread count, concatenated. Same
# N as the pre-unification run so the speedup numbers line up.
THREAD_N=32000
echo "### thread scaling at N=$THREAD_N"
COMBINED="$OUT_DIR/new_cpu_threads.csv"
: > "$COMBINED"
first=1
for t in $(printf '%s\n' 1 2 4 8 16 32 64 "$PHYS_CORES" | sort -n -u); do
    [ "$t" -gt "$PHYS_CORES" ] && continue
    export OMP_NUM_THREADS=$t
    part="$OUT_DIR/.thr_$t.csv"
    echo "--- $t thread(s) ---"
    bash "$SWEEP" --bin "$BIN" --out "$part" \
        --solvers "$CPU_SOLVERS" --ns "$THREAD_N" --label cpu-node-threads --budget 300
    if [ "$first" = 1 ]; then cat "$part" >> "$COMBINED"; first=0
    else tail -n +2 "$part" >> "$COMBINED"; fi
    rm -f "$part"
done
echo

echo "=== done. results in $OUT_DIR ==="
ls -l "$OUT_DIR"
