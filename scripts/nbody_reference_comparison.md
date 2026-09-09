# N-body: comparing our solvers to open-source references

The Studies scaling sweep showed our mutual FMM is slow -- it is dominated
by our *own* Barnes-Hut at every N up to at least 64k (BH is both faster
and more accurate), and only crosses Direct near N ~ 14k. This directory
adds a like-for-like comparison against well-optimised open-source solvers
to quantify how much of that is implementation vs. algorithm.

## Pieces

| file | what |
|---|---|
| `02_nbody_gravity --dump-ic <f> --bench-n N [--bench-eps h]` | write the byte-identical Plummer sphere `--bench` uses (raw f32 `x,y,z,m,vx,vy,vz`) so a reference solver benchmarks the same problem |
| `02_nbody_gravity --bench <solver> --bench-eps h` | our side; `--bench-eps 0` gives the near-unsoftened `1/r` regime the references use |
| `nbody_bench_reference.py` | drives **FMM3D / fmm3dpy** (Flatiron: Greengard-Gimbutas adaptive spherical-harmonic Laplace FMM) and **pytreegrav** (numba Barnes-Hut); emits the `nbody_bench_sweep.sh` CSV schema |
| `nbody_bench_sweep.sh --thetas a,b,c --eps h` | our sweep, now with an opening-angle *ladder* per (solver, N) and a softening knob, so we have an accuracy/cost curve to match against |
| `plot_nbody_vs_reference.py` | for each (pair, N, target error), log-log-interpolates each side's cost at that error and reports `ms_ours / ms_ref` |
| `perlmutter_reference_bench.sbatch` | runs all of the above on a Perlmutter CPU node -- **the run that produces meaningful timing** |

Pairings: `fmm -> fmm3d` (FMM vs FMM), `spherical_fmm -> fmm3d` (both
spherical-harmonic), `barnes_hut -> pytreegrav` (tree vs tree).

## Methodology

- **Same problem.** The exact Plummer sphere `ngrav::ic::Plummer<3>` seed 1,
  fed to the reference via `--dump-ic`. The dump is float32 (the repo's
  `ic_file` format); the f32 quantisation is ~1e-6 relative, negligible next
  to the ~1e-3-1e-2 force errors being compared.
- **Unsoftened.** FMM3D's kernel is `1/r`; run `--bench-eps 0` on our side
  and `--soft-eps 0` on the reference. The Plummer softening barely moves
  our FMM's error anyway -- that error is set by the linear-shift local
  expansion, not the softening (verified: 4.73e-2 at eps=1e-4 vs 4.75e-2 at
  eps=0.02, theta=0.5, N=8000). `--soft-eps > 0` is supported (a KD-tree
  near-field correction to FMM3D's unsoftened field) but has its own ~1e-3
  accuracy floor at a practical cutoff radius, so it is not the default.
- **Match accuracy, not parameters.** Each solver sweeps its own knob
  (opening angle for ours + pytreegrav, requested precision `eps` for
  FMM3D) and the plot interpolates cost to a common target error. The
  honest comparison band is where both ladders overlap -- roughly 1e-3
  for our FMM (FMM3D is *already* ~1e-3 accurate at its loosest `eps=0.1`,
  so it cannot be made as fast-and-inaccurate as our theta=0.5 point).
- **Same node, same `OMP_NUM_THREADS`.** All on one Perlmutter CPU node
  (dual EPYC 7763), full-node and 1-thread, matching the `new_cpu_*` sweep.
- **Direct is the cross-check.** Both harnesses' direct sums must agree.

## Why not just run it here

`fmm3dpy`'s Windows wheel appears to be single-threaded / unoptimised --
it measures ~130 ms for N=2000 where a threaded FMM3D on Linux does
single-digit ms. Windows is fine for validating correctness (the accel
formula `a = 4*pi*G * lfmm3d(...).grad`, verified vs direct to ~8e-16) and
the plumbing; **run `perlmutter_reference_bench.sbatch` for real numbers.**

## Running on Perlmutter

```sh
source scripts/linux-env.sh
cmake --preset linux-release && cmake --build --preset linux-release --target 02_nbody_gravity
sbatch scripts/perlmutter_reference_bench.sbatch
# then, with the CSVs it drops in $SCRATCH/nbody-bench/<jobid>/:
python scripts/plot_nbody_vs_reference.py \
    --ngrav-csv .../ngrav_pareto_full.csv --ref-csv .../ref_pareto_full.csv
```

## Next: falcON (NEMO)

The definitive comparison is against **falcON** (`getgravity` from NEMO) --
Dehnen's own implementation of the *exact* algorithm we ported (mutual /
symmetric dual-tree FMM). That isolates implementation quality completely,
because the algorithm is held fixed. Not yet wired up; it needs a NEMO
build on Perlmutter.
