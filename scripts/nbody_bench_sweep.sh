#!/usr/bin/env bash
# Drive 02_nbody_gravity's own `--bench` across a ladder of N and a set of
# solvers, and collect the one-line CSVs into one file.
#
# `--bench` measures a single (solver, N) point per process, which is the right
# granularity for it but means a sweep needs a driver. This is that driver: it
# adds the thread count (which --bench does not know about), sizes the
# repetition count down as N grows, and retires a solver from larger N once a
# point exceeds the wall-clock budget.
#
#   nbody_bench_sweep.sh --bin <02_nbody_gravity> --out sweep.csv \
#       --solvers direct,barnes_hut,fmm --ns 1000,4000,16000 [--theta 0.5]
#
# Set OMP_NUM_THREADS before calling; it is recorded in the `threads` column.

set -u

BIN=""
OUT=""
SOLVERS="direct,barnes_hut,fmm,spherical_fmm"
NS="1000,2000,4000,8000,16000,32000,64000,128000,256000,512000"
THETAS=0.5       # comma-separated: each (solver,N) is measured at every theta
EPS=0.02         # Plummer softening length (--bench-eps); 0 -> ~unsoftened
BUDGET=90        # seconds; a point slower than this retires its solver
LABEL=""         # free-text tag copied into every row (e.g. cpu-node)
REPS=0           # 0 = pick from N (below); >0 = force this many reps

while [ $# -gt 0 ]; do
    case "$1" in
        --bin)     BIN="$2"; shift 2 ;;
        --out)     OUT="$2"; shift 2 ;;
        --solvers) SOLVERS="$2"; shift 2 ;;
        --ns)      NS="$2"; shift 2 ;;
        --theta)   THETAS="$2"; shift 2 ;;
        --thetas)  THETAS="$2"; shift 2 ;;
        --eps)     EPS="$2"; shift 2 ;;
        --budget)  BUDGET="$2"; shift 2 ;;
        --reps)    REPS="$2"; shift 2 ;;
        --label)   LABEL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$BIN" ] || [ -z "$OUT" ]; then
    echo "usage: nbody_bench_sweep.sh --bin <exe> --out <csv> [--solvers ..] [--ns ..]" >&2
    exit 2
fi
if [ ! -x "$BIN" ]; then
    echo "error: not executable: $BIN" >&2
    exit 1
fi

THREADS="${OMP_NUM_THREADS:-1}"

echo "label,threads,solver,n,theta,order,ms_per_eval,mean_rel_err" > "$OUT"

IFS=',' read -r -a SOLVER_ARR <<< "$SOLVERS"
IFS=',' read -r -a N_ARR <<< "$NS"
IFS=',' read -r -a THETA_ARR <<< "$THETAS"

for solver in "${SOLVER_ARR[@]}"; do
    # Repetition count comes down as N grows: enough samples to average at
    # small N without spending minutes per point at the top of the ladder.
    for n in "${N_ARR[@]}"; do
        if [ "$REPS" -gt 0 ]; then
            reps=$REPS
        elif [ "$n" -le 8000 ];  then reps=10
        elif [ "$n" -le 64000 ]; then reps=5
        else                          reps=3
        fi

        retire=0
        for theta in "${THETA_ARR[@]}"; do
            start=$(date +%s.%N)
            line=$("$BIN" --bench "$solver" --bench-n "$n" --bench-theta "$theta" \
                          --bench-eps "$EPS" --bench-reps "$reps" 2>/dev/null)
            rc=$?
            end=$(date +%s.%N)
            elapsed=$(awk -v a="$start" -v b="$end" 'BEGIN{printf "%.1f", b-a}')

            if [ $rc -ne 0 ] || [ -z "$line" ]; then
                echo "  $solver N=$n theta=$theta -> FAILED (rc=$rc) after ${elapsed}s" >&2
                retire=1; break
            fi

            echo "$LABEL,$THREADS,$line" >> "$OUT"
            ms=$(echo "$line" | cut -d, -f5)
            err=$(echo "$line" | cut -d, -f6)
            printf "  %-14s N=%-7s th=%-4s reps=%-3s %12s ms  err=%-12s (%ss wall)\n" \
                "$solver" "$n" "$theta" "$reps" "$ms" "$err" "$elapsed" >&2

            over=$(awk -v e="$elapsed" -v b="$BUDGET" 'BEGIN{print (e>b)?1:0}')
            if [ "$over" = "1" ]; then
                echo "  $solver retired above N=$n (${elapsed}s > ${BUDGET}s budget)" >&2
                retire=1; break
            fi
        done
        [ "$retire" = 1 ] && break
    done
done

echo "wrote $OUT ($(( $(wc -l < "$OUT") - 1 )) rows)" >&2
