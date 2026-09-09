#!/bin/bash
#SBATCH --job-name=nbody-bench-ref
#SBATCH --account=m5032
#SBATCH --constraint=cpu
#SBATCH --qos=debug
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=30:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out
#
# Open-source reference gravity solvers (FMM3D / fmm3dpy, pytreegrav) on the
# SAME problem and the SAME node type as perlmutter_nbody_bench_cpu.sbatch,
# so their cost/accuracy overlays directly on new_cpu_full.csv /
# new_cpu_serial.csv.
#
# The problem is fed in as the byte-identical Plummer sphere the ngrav bench
# uses: nbody_bench_reference.py calls `02_nbody_gravity --dump-ic` itself.
#
#   sbatch scripts/perlmutter_reference_bench.sbatch
#
# Env knobs: BUILD_DIR (default $SLURM_SUBMIT_DIR/build/linux-release),
# OUT_DIR (default $SCRATCH/nbody-bench/<jobid>), REF_VENV (default
# $SCRATCH/nbody-ref-venv).

set -u

BUILD_DIR="${BUILD_DIR:-$SLURM_SUBMIT_DIR/build/linux-release}"
OUT_DIR="${OUT_DIR:-$SCRATCH/nbody-bench/${SLURM_JOB_ID:-manual}}"
REF_VENV="${REF_VENV:-$SCRATCH/nbody-ref-venv}"
BIN="$BUILD_DIR/projects/02_nbody_gravity/02_nbody_gravity"
DRIVER="$SLURM_SUBMIT_DIR/scripts/nbody_bench_reference.py"
SWEEP="$SLURM_SUBMIT_DIR/scripts/nbody_bench_sweep.sh"

mkdir -p "$OUT_DIR"

export OMP_PROC_BIND=spread
export OMP_PLACES=cores
PHYS_CORES=$(lscpu | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{print c*s}')

echo "=== node: $(hostname) ==="
lscpu | grep -E 'Model name|^Socket|^Core|^Thread'
echo "physical cores: $PHYS_CORES"
echo "out dir: $OUT_DIR"
echo

[ -x "$BIN" ] || { echo "ERROR: not built: $BIN" >&2; exit 1; }

# --- python env with fmm3d + pytreegrav -----------------------------------
module load python 2>/dev/null || true
if [ ! -x "$REF_VENV/bin/python" ]; then
    echo "### creating reference venv at $REF_VENV"
    python -m venv "$REF_VENV" || { echo "ERROR: venv creation failed" >&2; exit 1; }
    "$REF_VENV/bin/pip" -q install --upgrade pip
    # fmm3d ships manylinux wheels (OpenMP-enabled); pytreegrav is pure-python+numba
    "$REF_VENV/bin/pip" -q install numpy scipy numba fmm3d pytreegrav || {
        echo "ERROR: pip install failed -- try building fmm3d from source (needs gfortran)" >&2
        exit 1
    }
fi
PY="$REF_VENV/bin/python"
echo "### python: $($PY -V)"
$PY -c "import fmm3dpy, pytreegrav, numpy; print('  fmm3dpy', fmm3dpy.__version__ if hasattr(fmm3dpy,'__version__') else 'ok', ' pytreegrav', pytreegrav.__version__ if hasattr(pytreegrav,'__version__') else 'ok')"
echo

NS=1000,2000,4000,8000,16000,32000,64000
NS_SERIAL=1000,2000,4000,8000,16000,32000

# The comparison is run UNSOFTENED (1/r) -- FMM3D's kernel is 1/r and the
# Plummer softening barely moves our FMM's error anyway (it's set by the
# linear-shift local expansion, not the softening). Both sides sweep their
# own accuracy knob so plot_nbody_vs_reference.py can match at a target error.
OUR_THETAS=0.15,0.2,0.3,0.4,0.5           # ngrav tree/FMM opening angles
REF_EPS=1e-1,3e-2,1e-2,3e-3,1e-3,1e-4     # fmm3d requested precisions
REF_THETAS=0.3,0.4,0.5,0.6,0.7            # pytreegrav opening angles

run_pair () {   # $1 = threads, $2 = suffix, $3 = ns
    export OMP_NUM_THREADS=$1
    echo "### ngrav solvers across theta, $1 thread(s)"
    bash "$SWEEP" --bin "$BIN" --out "$OUT_DIR/ngrav_pareto_$2.csv" \
        --solvers barnes_hut,fmm,spherical_fmm --ns "$3" --thetas "$OUR_THETAS" \
        --eps 0 --label "ngrav-$2" --budget 120
    echo "### reference solvers across their accuracy knob, $1 thread(s)"
    $PY "$DRIVER" --ic-bin "$BIN" --out "$OUT_DIR/ref_pareto_$2.csv" \
        --ns "$3" --solvers fmm3d,pytreegrav \
        --eps "$REF_EPS" --theta "$REF_THETAS" --soft-eps 0 \
        --label "ref-$2"
    echo
}

run_pair "$PHYS_CORES" full   "$NS"
run_pair 1             serial "$NS_SERIAL"

echo "=== done. results in $OUT_DIR ==="
ls -l "$OUT_DIR"
echo
echo "plot:  python scripts/plot_nbody_vs_reference.py \\"
echo "         --ngrav-csv $OUT_DIR/ngrav_pareto_full.csv \\"
echo "         --ref-csv   $OUT_DIR/ref_pareto_full.csv"
