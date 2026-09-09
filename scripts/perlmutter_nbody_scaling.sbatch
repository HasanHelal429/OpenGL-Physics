#!/bin/bash
#SBATCH --job-name=nbody-scaling
#SBATCH --account=m5032
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#
# Scaling sweep for the N-body gravity solvers, on a whole CPU node.
#
#   sbatch scripts/perlmutter_nbody_scaling.sbatch
#
# A dedicated node is the point: a Perlmutter login node routinely sits at
# load average 50+ across 30 users, which is far more scatter than the
# effects being measured. --exclusive means nothing else lands on the socket.
#
# Override BUILD_DIR / OUT_DIR in the environment if yours differ:
#   BUILD_DIR=... OUT_DIR=... sbatch scripts/perlmutter_nbody_scaling.sbatch

set -u

BUILD_DIR="${BUILD_DIR:-$SLURM_SUBMIT_DIR/build/linux-release}"
OUT_DIR="${OUT_DIR:-$SCRATCH/nbody-scaling/${SLURM_JOB_ID:-manual}}"

BENCH3D="$BUILD_DIR/projects/02_nbody_gravity/02_nbody_scaling"
BENCH2D="$BUILD_DIR/projects/03_nbody_gravity_2d/03_nbody_scaling_2d"

mkdir -p "$OUT_DIR"

# Pin threads to distinct physical cores. Without this the OpenMP runtime is
# free to migrate threads across the two sockets / 8 NUMA domains, which
# turns the thread-scaling curve into noise.
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

FULL_THREADS=$(nproc --all)
PHYS_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')
: "${PHYS_CORES:=$FULL_THREADS}"

echo "=== node: $(hostname) ==="
lscpu | grep -E 'Model name|^Socket|^Core|^Thread|NUMA node\(s\)'
echo "physical cores: $PHYS_CORES   logical: $FULL_THREADS"
echo "out dir: $OUT_DIR"
echo

for b in "$BENCH3D" "$BENCH2D"; do
    if [ ! -x "$b" ]; then
        echo "ERROR: benchmark not built: $b" >&2
        exit 1
    fi
done

NS_BIG=1000,2000,4000,8000,16000,32000,64000,128000,256000,512000
NS_SERIAL=1000,2000,4000,8000,16000,32000,64000,128000

# ---------------------------------------------------------------------------
# 1. Complexity sweep, full node. This is the headline curve: wall time vs N
#    for every solver, with Direct retired automatically once a single call
#    blows the budget.
# ---------------------------------------------------------------------------
echo "### 3D complexity sweep, $PHYS_CORES threads"
export OMP_NUM_THREADS=$PHYS_CORES
"$BENCH3D" --ns "$NS_BIG" --solvers direct,bh,fmm,sfmm \
    --target-ms 3000 --min-reps 3 --max-reps 15 --budget-ms 30000 \
    --accuracy-max-n 16000 --csv "$OUT_DIR/scaling3d_full.csv"
echo

echo "### 2D complexity sweep, $PHYS_CORES threads"
"$BENCH2D" --ns "$NS_BIG" --solvers direct,bh,fmm \
    --target-ms 3000 --min-reps 3 --max-reps 15 --budget-ms 30000 \
    --accuracy-max-n 16000 --csv "$OUT_DIR/scaling2d_full.csv"
echo

# ---------------------------------------------------------------------------
# 2. Serial complexity sweep. One thread isolates the algorithmic exponent
#    from every parallel effect (spin-up, imbalance, memory bandwidth), so
#    the fitted slopes here are the ones to quote as "the complexity".
# ---------------------------------------------------------------------------
echo "### 3D complexity sweep, 1 thread"
export OMP_NUM_THREADS=1
"$BENCH3D" --ns "$NS_SERIAL" --solvers direct,bh,fmm,sfmm \
    --target-ms 2000 --min-reps 3 --max-reps 10 --budget-ms 25000 \
    --accuracy-max-n 8000 --csv "$OUT_DIR/scaling3d_serial.csv"
echo

echo "### 2D complexity sweep, 1 thread"
"$BENCH2D" --ns "$NS_SERIAL" --solvers direct,bh,fmm \
    --target-ms 2000 --min-reps 3 --max-reps 10 --budget-ms 25000 \
    --accuracy-max-n 8000 --csv "$OUT_DIR/scaling2d_serial.csv"
echo

# ---------------------------------------------------------------------------
# 3. Thread scaling at fixed N -- the actual "what does the big machine buy
#    us" number, per solver. Written as one CSV per thread count and
#    concatenated, since the harness reports its own thread count per row.
# ---------------------------------------------------------------------------
THREAD_N=32000
echo "### 3D thread scaling at N=$THREAD_N"
STRONG="$OUT_DIR/threads3d.csv"
: > "$STRONG"
THREAD_LIST=$(printf '%s\n' 1 2 4 8 16 32 64 "$PHYS_CORES" | sort -n -u)
first=1
for t in $THREAD_LIST; do
    [ "$t" -gt "$PHYS_CORES" ] && continue
    export OMP_NUM_THREADS=$t
    part="$OUT_DIR/.threads3d_$t.csv"
    echo "--- $t thread(s) ---"
    "$BENCH3D" --ns "$THREAD_N" --solvers direct,bh,fmm,sfmm \
        --target-ms 2000 --min-reps 3 --max-reps 10 --budget-ms 60000 \
        --accuracy-max-n 0 --csv "$part"
    if [ "$first" = 1 ]; then cat "$part" >> "$STRONG"; first=0
    else tail -n +2 "$part" >> "$STRONG"; fi
    rm -f "$part"
done
echo

echo "=== done. results in $OUT_DIR ==="
ls -l "$OUT_DIR"
