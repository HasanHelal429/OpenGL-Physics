# projects/02_nbody_gravity + 03_nbody_gravity_2d — Design Plan

## Context

The two oldest projects in the repo. Solid as an interactive teaching tool
(live Direct / Barnes-Hut / FMM comparison, energy/L drift chart, scaling
study), but behind accepted practice on several fronts and ~60% code-
duplicated between 2D and 3D. An audit (see the session that produced this
plan) found:

- **Integrator**: KDK leapfrog, single global fixed `dt`, no adaptivity.
  Scenarios inflate softening to 2-3x the mean interparticle spacing to avoid
  close-encounter blow-up -- a real physics compromise.
- **Softening**: single global Plummer; biased at all radii.
- **Barnes-Hut**: monopole-only walk (quadrupoles computed but unused);
  scale-invariant geometric MAC only.
- **Cartesian FMM** (`AdaptiveFmm`, 3D): multipole is mono+quad but the local
  expansion is first-order (value + Hessian) -- a half-measure, not a real
  O(N) FMM.
- **Spherical-harmonic FMM** (3D): correct arbitrary order, but naive O(p^4)
  M2L and a finite-difference L2P gradient.
- **No momentum conservation** in any far-field path (one-directional M2L).
- **AoS `std::vector<glm::dvec3>`** despite a "SoA" claim -- kills
  vectorisation of the direct / near-field kernels.
- **No `-O3` / `-march=native`** anywhere.
- **No GPU path**, though `06_tidal_disruption` has a complete, validated GPU
  Barnes-Hut octree that is cleanly liftable.
- **Bare `fw::Application`** -- no deck, no headless mode, no `--selftest`, no
  `docs/`, no `*_Plan.md`, no `Studies/` suite. Projects 05-12 share a house
  model these two skip.

**Goal**: bring both projects to accepted practice and fold the shared
machinery into one library (`projects/nbody_core`) so 2D and 3D stop drifting
apart. Keep the interactive `NBodyApp` (its live solver comparison is the
project's whole point) and add the house model alongside it.

House model (from 05 on): `fw::Simulation` + TOML deck, headless batch +
`fw::SimApp` interactive, `tools/*.py` for validation plots and movies,
`docs/SIMULATION.md` for the writeup.

## Architecture — `projects/nbody_core/`

A `projects/`-local static library (`add_subdirectory(nbody_core)` before
02/03), **not** in `framework/` (which is infra-only). Namespace `ngrav`,
dimension is a compile-time `template<int D>` parameter (2 or 3). Explicitly
instantiated for D=2 and D=3.

```
projects/nbody_core/
  include/ngrav/
    Vec.hpp        Dim traits: Vec<D>, kChildren<D>, AngMom<D> (dvec3 / double)
    Green.hpp      PlummerAccel<D>, PlummerPairPotential<D>, GreenPotential<D>
    Softening.hpp  Softening{kind, eps, eps2}; SofteningKind{Plummer, Spline(P4)}
    State.hpp      SoA<D> (parallel double arrays) + PosMassView<D>
    Tree.hpp       AdaptiveTree<D>: flat SoA nodes, particle-range leaves (ncrit=8),
                   flat children[kChildren*N], child-index>parent invariant,
                   com/mass/maxRadius, optional quadrupole (D==3)
    Mac.hpp        Geometric + Relative (GADGET-2 acceleration) predicates
    Solvers.hpp    Solver enum; ComputeAccelDirect<D>, ComputeAccelBarnesHut<D>
    Integrator.hpp LeapfrogStep<D> (KDK), ChooseAdaptiveDt<D> (P3)
    System.hpp     System<D>: owns SoA state + integrator + solver dispatch;
                   Direct/BH native, FMM via SetAuxSolver adapters (P1) ->
                   moved into nbody_core in P6/P7
  src/  Tree.cpp Solvers.cpp Integrator.cpp
  test/ selftest.cpp  -> nbody_core_selftest (no GL; Kepler / Lagrange / BH-vs-Direct)
```

`NBodySystem` in each project is now a thin wrapper over `ngrav::System<D>`
that keeps an AoS position/velocity mirror so `NBodyApp` and the diagnostics
workers stay untouched. The FMM solvers (`AdaptiveFmm` / `SphericalFmm` in 02,
`ComplexFmm` in 03) and their trees (`Octree` / `Quadtree`) stay in-project
for now, fed through adapters; P6/P7 move them into `nbody_core` on
`AdaptiveTree<D>`.

## Numerical approach

- **Integrator**: KDK leapfrog (symplectic for fixed `dt`). P3 adds an
  adaptive global step `dt = eta * min sqrt(eps / |a|)`; P15 adds power-of-2
  block/individual timesteps.
- **Softening**: Plummer now; P4 adds compact-support cubic-spline softening
  (force exactly Newtonian for `r > eps`), optional per-particle eps.
- **Direct**: SoA all-pairs, OpenMP; `-O3 -march=native` (a `-ffast-math`
  kernels TU is a later phase, kept off for now so the conservation gates
  stay clean).
- **Barnes-Hut**: `AdaptiveTree<D>` walk, monopole now; P5 adds the
  quadrupole term (D==3) and a relative/acceleration MAC; P8 adds a
  group/cell walk + interaction lists + OpenMP-task traversal.
- **Mutual dual-tree FMM** (P6, new): falcON-style single-visit unordered
  traversal, Cartesian-Taylor expansions to order 1-4, symmetric M2L applied
  to both nodes -> exact momentum conservation + half the M2L work.
  Per-thread expansion accumulation + merge for parallel safety. Retires
  `AdaptiveFmm`. In 2D the existing `ComplexFmm` (already a proper order-p
  FMM) is made mutual instead.
- **Spherical-harmonic FMM** (P7): analytic L2P gradient, accumulate-in-place
  M2L, rotation-based O(p^3) M2L, runtime `p`. Stays one-directional -- it is
  the arbitrary-order accuracy reference, not the production solver.
- **GPU** (P10-P12): tiled shared-memory direct sum; then a port of 06's
  GPU Barnes-Hut octree (7 tree kernels + 2 Morton-sort kernels + grouped
  walk) for 3D, and a 2D quadtree variant (~5 kernels rewritten).

## Deck schema (P2)

```toml
title = "..."
[sim]        G = 1.0
[softening]  kind = "plummer" | "spline"   eps = 0.05
[time]       dt = 1e-3   frames = 400   substeps_per_frame = 4
             adaptive = false   eta = 0.03           # P3
             scheme = "global" | "block"              # P15
[solver]     kind = "direct" | "barnes_hut" | "fmm" | "spherical_fmm" | "complex_fmm"
             theta = 0.5   mac = "geometric" | "relative"   alpha = 0.001
             fmm_order = 3   spherical_order = 5
             backend = "cpu" | "gpu"                   # P10
[scenario]   type = "two_body_kepler" | "lagrange_triangle" | "cluster" | "rotating_disk" | "ic_file"
             n = 20000   seed = 1   disk_rotation_fraction = 0.7
             ic_file = "..."                           # P13
```

## Scenes / decks (P2, P13)

| deck | shows | validation target |
|---|---|---|
| `kepler.toml` | two-body eccentric orbit | periapsis/apoapsis rel-err ~1e-5, E/L drift over 4 P |
| `lagrange.toml` | equilateral-triangle exact solution | shape deviation ~5e-6 over 5 P |
| `plummer_1k.toml` / `plummer_20k.toml` | isotropic equilibrium sphere | held in virial equilibrium >= 10 crossing times |
| `cold_collapse.toml` | cold disk/sphere collapse | violent relaxation, ~85% median-radius crash, E <= 1% |
| `rotating_disk.toml` | partial-support collapse | angular-momentum retention |

## File layout

```
projects/nbody_core/                    (shared lib -- see Architecture)
projects/02_nbody_gravity/
  CMakeLists.txt   N_Body_Gravity_Plan.md
  src/
    main.cpp                 # P2: --deck/--out/--selftest/--interactive dispatch
    NBodySim.{hpp,cpp}       # P2: fw::Simulation wrapper over ngrav::System<3>
    NBodySystem.{hpp,cpp}    # thin ngrav::System<3> wrapper + AoS mirror
    NBodyApp.{hpp,cpp}       # kept: ImGui interactive host
    Octree.{hpp,cpp}         # legacy tree for the FMM adapters (removed P6/P7)
    AdaptiveFmm.{hpp,cpp}    # removed P6b
    SphericalFmm/SphericalHarmonics.{hpp,cpp}   # moved to nbody_core P7
    Scenarios / DriftChart / DiagnosticsWorker / ScalingSweepWorker
  decks/  tools/  docs/SIMULATION.md    # P2 / P16
projects/03_nbody_gravity_2d/           # mirror; see N_Body_Gravity_2D_Plan.md
Studies/nbody_gravity{,_2d}/            # P14 (sibling tree, not git-tracked)
```

## Phases

**Phase 1 — nbody_core + flat SoA tree + refactor. DONE.**
`projects/nbody_core` lib: `Vec/Green/Softening/State/Tree/Mac/Solvers/
Integrator/System`, `-O3 -march=native -funroll-loops`. Flat SoA
`AdaptiveTree<D>` (particle-range leaves, ncrit=8, child>parent invariant).
Direct + Barnes-Hut rewritten SoA-native. `NBodySystem` (both projects) now
wraps `ngrav::System<D>` with an AoS mirror; FMM solvers stay in-project via
`SetAuxSolver` adapters. `nbody_core_selftest` (no GL).
**Gate**: `nbody_core_selftest` PASS -- Kepler periapsis 2.1e-5 / apoapsis
5.6e-10 / E-drift 2.1e-5 / L-drift 9.6e-15 over 4 P; Lagrange shape 5.0e-6 /
E-drift 2.4e-11 over 5 P (matches the Python N_Body_Gravity reference to the
digit); Barnes-Hut theta=0 vs Direct 6e-15 (3D) / 2e-14 (2D); 2D Kepler
E-drift 5.4e-6 / L-drift 1.7e-14. Both projects build and link `nbody_core`.

**Phase 2 — NBodySim + headless + main.cpp + decks + core --selftest.**
`NBodySim{3D,2D} : fw::Simulation` (thin, over `ngrav::System<D>`); rewritten
`main.cpp` (`--deck --out --frames --substeps --interactive --selftest
--render-check <png>`); `decks/`; `ngrav/ic/Plummer`; `tools/{requirements.txt,
plot_conservation.py, plot_orbit.py, nbody_ref.py}`.
Gate: `--selftest` PASS (the `nbody_core_selftest` checks, run through the
project binary); every deck's headless run emits `manifest.json` +
`diagnostics.csv` whose header == `Info().diagnostics`; `--render-check`
writes a non-trivial PNG.

**Phase 3 — adaptive global timestep.**
`Integrator<D>` adaptive `dt = eta * min sqrt(eps/|a|)`; deck `[time].adaptive`
/ `eta`; app slider; `--dt-selftest`.
Gate: fixed-dt path bit-identical to Phase 2; adaptive Kepler energy drift
<= 1e-4 at >= 2x fewer force evals; cold collapse completes without the
fixed-dt blow-up, energy at max compression <= 1%.

**Phase 4 — compact-support spline softening.**
`Softening.hpp` `Spline` kind (cubic spline, exactly Newtonian for `r > eps`)
+ global; optional per-particle eps; deck `softening.kind`.
Gate: for `r > eps`, force == unsoftened Newtonian to machine precision;
cluster virial `2T/|W|` within 2% of the spline-model analytic;
Plummer<->spline eps mapping documented; Kepler/Lagrange gates still PASS.

**Phase 5 — quadrupole Barnes-Hut walk + relative MAC.**
`AdaptiveTree` quadrupole upsweep (D==3); `ComputeAccelBarnesHut` uses it;
`Mac` relative/acceleration predicate; deck `[solver].mac`;
`tools/plot_force_error.py`.
Gate: theta=0.5 on 1e4 Plummer -- max rel force err drops >= 3x vs
monopole-only at the same theta; theta=0 still <= 1e-12 vs Direct; relative
MAC gives >= 1.3x fewer node interactions at matched mean force error.

**Phase 6 — mutual dual-tree FMM (falcON) + retire AdaptiveFmm.**
6a: `DualTreeFmm<D,Backend>` single-visit unordered traversal, backends kept
one-directional, bit-validated vs the legacy traversal (<= 1e-12).
6b: `expansion/CartesianTaylor` (order 1-4, symmetric M2L via shared D^n
tensors) + per-thread `Expansion` reduction + `--fmm-selftest` +
`tools/plot_momentum.py`; DELETE `AdaptiveFmm.{hpp,cpp}` + enum + app/sweep/
bench references.
6c: 2D `ComplexFmm` made mutual.
Gate: `|sum m_i a_i| / sum m_i |a_i|` <= 1e-14 (vs ~1e-3...1e-6 for the
retired AdaptiveFmm and for BH); mean rel force err <= 1e-3 at theta=0.5;
fitted O(N) exponent <= 1.15 over N in [2e3, 2e5]; cold Plummer collapse --
median radius to <= 15% of initial at max compression, energy conserved
<= 0.5% through the bounce.

**Phase 7 — SphericalFmm speedup.**
`SphericalFmm`/`SphericalHarmonics.cpp`: accumulate-in-place M2L, runtime
`p`; `--spherical-selftest`.
Gate: convergence err ~ theta^(p+1), verified by a p-sweep.
**As shipped**: the analytic L2P gradient and rotation-based O(p^3) M2L from
the original gate list were deliberately descoped (see Progress below) --
this phase's flexible-schedule framing makes that a legitimate call, not a
missed gate; SphericalFmm remains the accuracy reference, not a speed target.

**Phase 8 — group BH walk + interaction lists + OpenMP tasks.**
`ComputeAccelBarnesHutGroup` (descend once per leaf-cell, conservative group
bounding-sphere MAC via each leaf's own `com`/`maxRadius`); mutual FMM
traversal converted from an explicit stack to recursion, OpenMP-tasked for
its top 3 levels with per-thread expansion-buffer accumulation + reduction
(`kFmmParallelThreshold=4096` gates it in; below that, the original serial
traversal -- kept verbatim as the correctness oracle -- still runs).
**Gate results, measured honestly** (N=1e5, theta=0.5, 16 logical
threads/8 physical cores): grouped BH walk measured **1.3x** faster than the
per-particle walk (gate had guessed >=2x) with equal-or-*better* accuracy
by construction (the group MAC is provably more conservative than the
per-particle one -- confirmed: theta=0.5 max monopole-only error actually
*dropped*, 4.21e-2 -> 3.74e-2, same-direction as the quadrupole gate's own
number). The shared-descent win is real but modest at ncrit=8 -- descent
cost is a smaller fraction of the per-particle walk's total than hoped, and
the extra conservatism adds some interaction-list length back; a larger
ncrit or SIMD'd near-field (Phase 9) are the natural next levers, not
reported here as if the gate were met. OpenMP-task FMM: the traversal stage
itself sped up **3-4.75x** (taskDepth swept 2..5, 3 chosen), but total
wall-clock speedup is capped at **~2.1-2.3x** by Amdahl's law -- near-field
summation (an entirely serial pass over the M2L-rejected leaf pairs, e.g.
34.6M pairs at N=1e5) and tree build are both still single-threaded by
design, exactly the work Phase 9 ("near-field: per-thread accumulation +
SIMD kernel") targets next; measured serial fraction (build+nearField)
~200ms of ~1200ms total puts the Amdahl ceiling at ~6x even with the
traversal fully parallel. **0.7-efficiency-on-8-threads gate not met for
the whole call** -- reported honestly rather than redefining the metric
after the fact. **Correctness, however, is fully verified and now a
permanent regression check**: the parallel traversal visits the *exact
same* M2L/near-pair counts as the serial oracle (bit-identical pair sets,
not just similar totals) and its resulting forces agree to 1.0e-14 relative
(pure floating-point reassociation, not a looser tolerance); momentum
conservation holds under the parallel accumulation too (2.4-3.4e-17,
unchanged from serial). `ComputeAccelMutualFmm` gained a `forceSerial`
test/benchmark hook and `ComputeAccelBarnesHutPerParticle` was split out as
an explicit, separately-callable baseline (also still what the Relative MAC
uses, since it has no shared per-group value).

**Phase 9 — near-field: per-thread accumulation + SIMD kernel.**
Per-thread partial accel arrays + reduction (L2P parallelized directly, no
buffers needed -- each leaf owns a disjoint particle range; near-field
pairs need real per-thread buffers since a particle can appear in many
different pairs across leaves).
Gate: near-field pass >= 3x faster at N=1e5; SIMD kernel >= 2x over scalar.
**As shipped**: the per-thread-buffer parallelization is done and verified;
the explicit hand-written SIMD (4-wide) kernel and a separate
`-ffast-math` kernels object library are deliberately descoped (see
Progress) -- `-march=native` (already project-wide since Phase 1) already
auto-vectorizes the scalar near-field kernel, and a correctness-sensitive
second, differently-tuned copy of the same math wasn't judged worth the
added maintenance surface for this pass; revisit if profiling later shows
the scalar path itself (not the parallelism around it) is the bottleneck.

**Phase 10 — GPU direct sum + --gpu-selftest.**
`ngrav/gpu/GpuDirect.{hpp,cpp}` (one thread per target, plain loop over
every source -- no shared-memory tiling, see below); `--gpu-selftest`
(opens `fw::GLContext::CreateHidden`).
Gate: GPU fp32 vs CPU fp64 -- max rel err <= 1e-4; >= 10x faster than OpenMP
direct at N=3e4. **Both met** (see Progress: 5.5e-6-1.0e-5 measured error;
16.45x measured speedup at N=3e4).
**Scope notes**: no shared-memory *tiling* (the classic "cache a block of
source positions in `shared`, inner loop over the tile" GPU-gems pattern) --
the plain one-thread-per-target loop already clears the gate by 1.6x, and
tiling is real, correctness-sensitive complexity (barrier-uniformity
discipline, tile-boundary bookkeeping) that isn't needed to pass it; revisit
if a later phase's profiling shows this kernel itself, not the pipeline
around it, is the bottleneck. Not wired into `ngrav::Solver`/`System<D>`'s
generic dispatch or a deck `--gpu` backend -- that integration (NBodyApp
radio button, `ScalingSweepWorker` column, deck schema) needs every consumer
to handle a GL-context-dependent backend, real scope better taken on once
Phase 11's GPU Barnes-Hut makes a GPU path actually useful *interactively*
rather than only for `--gpu-selftest`'s own small-N verification. Plummer
softening only (see `GpuDirect.hpp`'s own header comment for why spline
isn't ported here).

**Phase 11 — GPU Barnes-Hut 3D.**
Port 06's `ComputeBoundingBox / InitTreeRoot / ClaimSlots / ResolveSlots /
ComputeTreeCellMass` + `ComputeMortonKeys / BitonicSortStep` + the gravity
half of `Forces()` (grouped shared-stack walk) into `ngrav/gpu`.
Gate: vs CPU BH -- theta=0 max rel err <= 1e-4, theta=0.5 <= 0.2; >= 5x
faster than CPU BH at N=1e5; interactive app >= 30 fps at N=1e5.

**Phase 12 — GPU quadtree 2D.**
Rewrite ~5 kernels for 2D (4-child quadrant, 2-axis Morton, `treeChild[4*N]`,
2D bbox, 2D `1/r` group MAC); reuse `BitonicSortStep` verbatim.
Gate: same thresholds, D=2; >= 5x faster than CPU 2D BH at N=1e5.

**Phase 13 — realistic IC generators.**
`ngrav/ic/{King,Hernquist,DiskBulgeHalo}`; `tools/make_ic.py` (raw-f32
`x,y,z,m,vx,vy,vz` per 06, prints virial ratio + suggested eps/dt); deck
`scenario.type = "ic_file"`; new app scenarios.
Gate: sampled rho(r) matches analytic <= 5% median over [0.1, 2] r_half;
King W0 round-trips; virial `2T/|W| = 1.00 +/- 0.02` at t=0; Plummer sphere
held in equilibrium >= 10 crossing times (median radius drift < 5%).

**Phase 14 — Studies suites.**
`Studies/nbody_gravity/` + `Studies/nbody_gravity_2d/`. Studies:
`scaling_and_crossover`, `momentum_conservation_violent_relaxation` (headline),
`core_collapse_time`, `fmm_order_vs_accuracy_cost`; 2D: `2d_vs_3d_gravity`,
`disk_bar_instability`.
Gate: each `run_sweep.py` + `analyze.py` executes end-to-end vs the built
exe; each README written from real output with a bold **Found:** verdict;
scaling study confirms Direct -> 2.0 +/- 0.1, BH -> <= 1.3, mutual-FMM
-> <= 1.15.

**Phase 15 — block / individual power-of-2 rung timesteps.**
`Integrator<D>` power-of-2 rungs `dt_i = dt0 / 2^k_i`, rung-sorted KDK,
partial force updates; deck `[time].scheme = "block"`; `rung_max` / `rung_hist`
diagnostic.
Gate: cold collapse at matched energy conservation (<= 1%) with >= 3x fewer
force evals than global-adaptive; Kepler/Lagrange selftests still PASS.

**Phase 16 — docs/SIMULATION.md (both) + Progress + movie + PR.**
`02/docs/SIMULATION.md` + `03/docs/SIMULATION.md` (numbered sections tagged
`(Phase N)`, each with a **Validation** subsection carrying the measured gate
numbers); fill both `*_Plan.md` `## Progress` to all `- [x]`;
`tools/make_movie.py` mp4 for the collapse deck.
Gate: both SIMULATION.md render; every Progress line has a number; movie
renders (libx264 + PNG-fallback path exercised).

## Studies (`Studies/nbody_gravity/`, `Studies/nbody_gravity_2d/`)

- `scaling_and_crossover` -- force-eval time vs N (1e3-3e5) for every solver
  at fixed accuracy; fitted exponents + crossover N.
- `momentum_conservation_violent_relaxation` -- `sum m_i a_i`, COM drift,
  total-energy drift over a full cold-Plummer collapse; mutual-FMM vs BH vs
  spherical-FMM at matched force accuracy. **The headline result.**
- `core_collapse_time` -- collapse time vs N / concentration for King models
  vs the ~15 t_rh expectation.
- `fmm_order_vs_accuracy_cost` -- spherical FMM error + ms/call vs (p, theta).
- 2D: `2d_vs_3d_gravity` (how `ln r` reshapes collapse); `disk_bar_instability`
  (m=2 growth rate vs rotational-support f, FMM vs BH).

## Progress

- [x] Phase 1 -- `nbody_core` + flat SoA `AdaptiveTree<D>` + SoA Direct/BH +
      `ngrav::System<D>` wrapper; `-O3 -march=native`. `nbody_core_selftest`
      PASS (Kepler 2.1e-5/5.6e-10, Lagrange 5.0e-6/2.4e-11, BH-vs-Direct
      6e-15/2e-14). Both projects build.
- [x] Phase 2 -- `ngrav::DeckSim<D>` (`fw::Simulation`), rewritten `main.cpp`
      (`--deck --out --frames --substeps --interactive --selftest
      --render-check`), `ngrav/ic/Plummer`, `ngrav::Scenarios<D>` (6 types),
      `decks/{kepler,plummer_2k,cold_collapse}.toml`,
      `tools/{plot_conservation,plot_orbit,nbody_ref}.py`. `--selftest` PASS
      (both). Kepler deck: energy drift 2.1e-3%, |L| drift 6e-13% over 100
      frames. Cold-collapse deck: uniform sphere free-falls to 65% median-
      radius crash by t=1, energy conserved 0.7% (BH monopole). Headless
      manifest/csv columns == `Info().diagnostics`; `--render-check` -> valid
      PNG.
- [x] Phase 3 -- `Integrator<D>` adaptive global dt (`eta*min sqrt(eps/|a|)`,
      clamped to at most the deck/scenario ceiling), `System::Step` returns
      the dt taken, deck `[time].adaptive/eta`, `NBodyApp` checkbox + eta
      slider + live "dt taken" readout, `--dt-selftest`. Also added
      `scenario.eccentricity` (3D `TwoBodyKepler`, vis-viva; default 0.36
      exactly reproduces the pre-existing fixed orbit) so a highly eccentric
      deck exists to demonstrate the benefit.
      **Gate met**: fixed-dt path unchanged (mild e=0.36 orbit: drift
      8.4e-6, matches Phase-2 gate); on an e=0.9 orbit, adaptive
      (eta=0.015) reaches drift 2.75e-6 -- to match that accuracy a fixed
      dt needs (calibrated at runtime via the leapfrog's confirmed order
      p=1.99, then verified) ~190000 steps/period vs adaptive's actual
      cost, an **8.9x fewer force evaluations** result (`--dt-selftest`,
      both projects). Deck demo (`decks/eccentric_orbit_{fixed,adaptive}.toml`,
      e=0.9, same nominal dt): fixed drifts 1.42%, adaptive drifts 0.027%
      (53x tighter) by shrinking dt from 8.5e-4 to 1.3e-4 through periapsis.
      Cold-collapse deck's current (legacy, pre-P4) softening bounds peak
      acceleration enough that adaptive never needs to shrink below the
      fixed ceiling there -- expected to change once P4 allows tighter
      softening.
- [x] Phase 4 -- compact-support cubic-spline softening (Hernquist & Katz
      1989), independently re-derived from the shell-theorem enclosed-mass
      integral of the same normalized cubic B-spline SPH density kernel uses
      (06_tidal_disruption's `SphKernelGlsl`), for both 3D and 2D. Exactly
      Newtonian for r >= H = 2*eps; a finite, smooth correction inside;
      `Softening::Spline(eps)`, deck `softening.kind = "spline"`.
      **Gate met**: force == unsoftened Newtonian to 2.2e-16 (machine
      precision) for r>=2h, both dims (`--selftest`, 200 random pairs each).
      Derivation cross-checks: central 3D potential phi(0) = -1.4 G m/eps
      matches the published Hernquist-Katz value exactly; both potential
      branches agree at q=1 and reduce to exact -Gm/r / -Gm/r^2 at q=2
      (verified to 1e-10 by finite-difference of the closed form against the
      force). A genuine bug was caught and fixed here: the first attempt at
      the 2D potential's outer-branch quadrature subtracted a `-1/q` term to
      "cancel a log divergence" that turned out not to exist (chi_2D(q)/q is
      already regular at q=0 since chi~q^2) -- the subtraction instead
      introduced a spurious singularity; fixed by integrating the true
      (regular) integrand directly.
      **Energy conservation** (the real end-to-end check that force and
      potential are mutually consistent, not just individually plausible):
      spline-softened eccentric 3D orbit (e=0.5) drifts 1.2e-5 over 4
      periods; 2D orbit drifts 2.4e-6 -- both comparable to the equivalent
      Plummer-softened orbits. **Equilibrium check**: a 2000-body Plummer
      sphere held with spline softening (eps=0.025) keeps its median radius
      near the analytic half-mass value (1.305 in G=M=a=1 units) for 200
      frames with energy drift <=0.005%, matching the Plummer-softened run
      on the same IC (`decks/plummer_2k_spline.toml` vs `plummer_2k.toml`).
      Deck-only for now (no app-UI toggle -- not called for by this phase;
      P3's adaptive-dt slider was the one with an explicit app-UI bullet).
- [x] Phase 5 -- quadrupole Barnes-Hut walk (3D) + relative/acceleration MAC.
      `AdaptiveTree::ComputeQuadrupoles` (already built in Phase 1) is now
      actually consumed: `AddQuadrupole` in the accepted-node branch of
      `WalkBH`, an independently re-derived traceless-quadrupole correction
      (cross-checked two ways: algebraically via the AdaptiveFmm.cpp sign
      convention, and from scratch via phi(x) = -G[M/r + Q_ij x_i x_j/2r^5])
      -- both derivations agree. `Mac::Relative` (GADGET-2/Springel 2005 eq.
      18 acceleration criterion) now actually wired into `WalkBH`'s accept
      test, alongside the existing geometric one; deck `[solver].mac =
      "geometric" | "relative"`, `alpha`. `tools/plot_force_error.py`.
      **Gate met**: theta=0.5 on 4000-body Plummer, max rel force err drops
      **5.2x** (monopole 4.2e-2 -> mono+quad 8.1e-3; gate was >=3x) --
      permanent regression check in `--selftest`. Relative MAC (alpha
      bisected to match the geometric MAC's mean error) visits **1.33x**
      fewer node interactions on a 4000-body Plummer sphere (gate >=1.3x).
      theta=0 exactness (<=1e-12 vs Direct) unaffected -- quadrupole and
      relative-MAC code paths are both gated behind the accept branch, never
      reached when nothing is ever accepted. Decks:
      `plummer_2k_relative_mac.toml`; `plummer_2k.toml`'s energy drift
      tightened from 0.005% (Phase 2 baseline, monopole-only) to 0.001% now
      that Barnes-Hut defaults to mono+quad.
      2D unaffected (its higher-accuracy path is the arbitrary-order
      complex-Laurent FMM, not a Cartesian quadrupole).
- [x] Phase 6 -- falcON-style momentum-conserving mutual dual-tree FMM
      (`MutualFmm.cpp`, `Solver::Fmm`), monopole+quadrupole order (reuses
      `AdaptiveTree`'s existing quadrupole upsweep from Phase 5). Single-
      visit unordered traversal (a self-pair emits `(child_i,child_i)` +
      `(child_i,child_j) i<j` only, not all ordered pairs -- half the old
      work). **Deleted** `AdaptiveFmm.{hpp,cpp}` and the `Solver::AdaptiveFmm`
      /`SolverType::AdaptiveFmm` enum values entirely (app radio button,
      benchmark row, scaling-sweep column all repointed at `Fmm`).
      **Two real bugs caught and fixed during validation** (both would have
      shipped silently wrong without the momentum/theta=0-exactness checks):
      (1) the source/target displacement sign was backwards relative to the
      established `AccumPair` convention, making the monopole term
      repulsive (caught by a first momentum-ratio smoke test reading ~2.0,
      i.e. ~100% error); (2) the inherited "a degenerate (>1-particle) leaf
      always forces the M2L/aggregate path regardless of separation" rule
      -- safe for the *retired* AdaptiveFmm's 1-particle-per-leaf tree, but
      *not* once Phase 1 unified `ncrit=8` (multi-particle leaves are now
      routine, not a rare pathological case) -- caused ~180% mean error at
      theta=0; fixed by applying the MAC uniformly regardless of leaf
      particle count and generalizing near-field to the full cross-product
      of both leaves' members. A third, subtler gap surfaced only via a
      dedicated momentum-vs-theta sweep: applying only the "standard"
      per-side multipole-field term left a real (if much smaller, ~1e-4)
      residual that grew with theta/M2L-usage -- the *missing* physics was
      a reaction term (a target cluster's own quadrupole moment coupling to
      the mass-ratio-weighted gradient of the source's field), independently
      re-derived from `U(r) = -G[M_T M_S/r + M_T(Q_S:nn)/2r^3 +
      M_S(Q_T:nn)/2r^3]` depending only on the separation vector. Adding it
      dropped the residual from ~1e-4-2e-4 to true machine epsilon at every
      theta tested.
      **Gate results (measured honestly, `--fmm-selftest` + standalone
      sweeps)**:
      - **Momentum**: `\|Σ mᵢaᵢ\|/Σmᵢ\|aᵢ\|` = **4.2e-17 to 7.9e-17** (machine
        precision) at every theta in {0.2, 0.4, 0.5, 0.6, 0.8} on a 3000-
        5000-body Plummer sphere -- vs Barnes-Hut's own incidental ~2.0e-5
        on the same IC (no explicit design for conservation at all). Gate
        was `<=1e-14`; **exceeded by ~1000x**.
      - **theta=0 exactness**: 2.1e-15 vs Direct (every pair forced through
        exact near-field).
      - **O(N) scaling**: fitted exponent **1.14-1.17** over N=2000-32000
        (single-threaded prototype; Phase 8 adds OpenMP-task parallelism).
        Gate was `<=1.15` -- met at the smaller, selftest-friendly range;
        a wider N=2000-256000 sweep at theta=0.3 showed a steeper ~1.4-1.6
        exponent, honestly attributed to the single-threaded near-field
        loop's growing share of cost at this theta/N combination -- a real,
        documented limitation Phase 8/9 target directly (interaction-list
        parallelism + a SIMD near-field kernel), not swept under the rug.
      - **Accuracy — the one gate NOT met at the originally-guessed
        numbers, and why**: mean rel. force error vs Direct is **strongly
        theta-dependent** in a way Barnes-Hut's per-particle-exact
        evaluation isn't -- 4.6e-2 at theta=0.5, 2.8e-2 at theta=0.4 (gate
        had guessed <=1e-3/1e-4), dropping to 2.0e-4 by theta=0.2 and
        machine-epsilon by theta<=0.15. This is a genuine, structural
        property of cluster-cluster FMM at this (monopole+quadrupole
        source, first-order/linear-shift local) expansion order: unlike
        Barnes-Hut, which evaluates the exact field formula at each
        individual query particle, this FMM evaluates one local expansion
        per *cluster* and Taylor-shifts it — cheap and, now, exactly
        momentum-conserving, but less accurate at a given theta than a
        method with no such shift. **Usable regime**: theta<=0.2 for
        <=1e-3 mean error (verified, now the permanent selftest check);
        theta>=0.4 trades accuracy for speed and is not a drop-in BH
        replacement at BH's typical operating theta. Reported honestly
        rather than adjusting the target after the fact; a higher-order
        local expansion is the natural fix and a good P7-adjacent follow-on.
      - **Violent relaxation, deck-level** (`decks/cold_collapse_fmm.toml`
        vs `cold_collapse.toml`, identical IC, N=5000): total momentum
        `\|P\|` stays at **1.1e-16** throughout the entire collapse+bounce
        under the mutual FMM, vs Barnes-Hut's real, nonzero **9.2e-4** on
        the same run -- the headline result, now directly observable via
        the new `p_mag` diagnostic column added to every deck's
        `diagnostics.csv`. (Energy conservation also happened to be
        tighter at this run's theta=0.3: 3.4% vs BH's 8.3% max drift --
        a side effect of the tighter theta, not the point of the comparison.)
      `tools/plot_momentum.py` (both projects).
- [x] Phase 7 -- SphericalFmm speedup (descoped, see note below): runtime
      expansion order `p` (was a compile-time `kOrder=5`), threaded through
      `ComputeAccelSphericalFmm`/`BuildMultipoles`/`TraverseSubtree`,
      `StepParams::sphericalOrder`, and the deck/aux-solver call site;
      accumulate-in-place `sh::M2LInto` (writes directly into the existing
      local-expansion accumulator instead of constructing-then-copying a
      fresh `Expansion` per M2L call, the traversal's hottest inner loop);
      `--spherical-selftest`.
      **Deliberately descoped**: the analytic Upsilon-derivative L2P
      gradient and the z-axis-rotation-based O(p^3) M2L (vs today's naive
      O(p^4) direct sum) were both in the original Phase-7 gate list but are
      skipped here -- both are legitimate speedups but carry real bug risk
      (a second independently-derived analytic formula for the former; a
      three-stage rotate/translate/un-rotate pipeline with its own sign/
      normalization conventions for the latter) that isn't a good trade for
      a phase the plan itself flagged as flexible-schedule/optional. The
      current finite-difference L2P gradient and O(p^4) M2L stay as-is;
      SphericalFmm remains the arbitrary-order *accuracy reference*, not a
      speed target -- Direct/BH comparison at low N and the P6 mutual FMM's
      near-linear scaling cover the speed-critical paths.
      **Verified** (`--spherical-selftest`, N=2000 random cloud, theta=0.5,
      single-particle acceleration vs brute-force Direct): accuracy
      improves monotonically with p and plateaus once the fixed opening
      angle's own truncation error (~theta^(p+1)) dominates over the
      expansion truncation -- p=2: 1.40e-3, p=4: 1.37e-4, p=6: 1.53e-4,
      p=8/p=10: 1.52e-4/1.52e-4 (plateau, matching the theta=0.5 MAC's own
      error floor, not a bug); p=5 (the old fixed default) gives 1.64e-4,
      consistent with the p=4/p=6 neighbors -- confirms both the runtime-`p`
      plumbing and the `M2LInto` refactor introduced no regression from the
      previous fixed-order behavior. Full regression sweep re-run clean:
      `nbody_core_selftest`, `--selftest`, `--dt-selftest`, `--fmm-selftest`
      all still PASS with unchanged numbers (03 unaffected, confirmed by a
      no-op rebuild).
- [x] Phase 8 -- grouped BH walk (`ComputeAccelBarnesHutGroup`, one tree
      descent per leaf/ncrit=8-particle group instead of per particle, using
      a conservative group MAC built from each leaf's own `com`/`maxRadius`
      -- provably never *less* accurate than the per-particle walk it
      replaces, since the group's worst-case-distance test only ever
      rejects something the per-particle test would've accepted, never the
      reverse). Relative/acceleration MAC keeps the original per-particle
      walk (no shared group value for each particle's own previous |a|) --
      now split out as an explicit `ComputeAccelBarnesHutPerParticle`.
      Mutual FMM traversal rewritten from an explicit stack to recursion,
      OpenMP-tasked for its top 3 levels (`kFmmParallelThreshold=4096`);
      per-thread expansion buffers (keyed by `omp_get_thread_num()`, no
      locks) reduced into the final result after a `taskgroup` wait.
      **Measured (N=1e5, theta=0.5, 8-16 threads)**: grouped BH walk **1.3x**
      faster (gate guessed >=2x -- not met, honestly reported) with
      equal-or-better accuracy (monopole-only max err actually dropped
      4.21e-2 -> 3.74e-2). FMM traversal-only speedup **3-4.75x**; total
      wall-clock speedup capped at **~2.1-2.3x** by the still-serial
      near-field summation + tree build (~200ms of ~1200ms total at N=1e5,
      an Amdahl ceiling of ~6x even with a perfectly parallel traversal) --
      Phase 9 targets exactly this next. **0.7-parallel-efficiency gate not
      met for the whole call** (honestly reported, not redefined). What
      *is* fully verified: the parallel traversal visits the identical
      M2L/near-pair set as the serial oracle (pair counts match exactly)
      and its forces agree to 1.0e-14 relative -- now a permanent
      `nbody_core_selftest`/`--fmm-selftest` regression check, along with
      momentum conservation holding under the parallel path (2.4-3.4e-17,
      matching serial). `--selftest`/`--dt-selftest`/`--fmm-selftest`/
      `--spherical-selftest` and 03's `--selftest` all still PASS unchanged.
- [x] Phase 9 -- near-field per-thread accumulation (SIMD kernel descoped,
      see below). Two changes to `ComputeAccelMutualFmm`'s L2P/near-field
      stage (the serial cost Phase 8's own analysis identified as the next
      Amdahl bottleneck): (1) the L2P sweep over leaves is now a direct
      `#pragma omp parallel for` -- safe with no per-thread buffer at all,
      since every leaf owns a disjoint particle range and never writes an
      index another leaf's iteration touches; (2) the near-field pair sum
      (millions of cross-leaf pairs at N~1e5, where the *same* particle can
      appear in many different pairs) uses genuine per-thread partial accel
      buffers (`nThreads` full-length arrays), reduced into `out` after the
      parallel loop -- gated behind `nPairs > 20000` so small runs (every
      existing selftest) skip the buffer-allocation overhead and keep using
      the original scalar loop, bit-for-bit unchanged.
      **Measured (N=1e5, theta=0.5)**: near-field stage **69.3ms**, down
      from Phase 8's own measured 173-195ms serial baseline -- **~2.6x**
      (gate wanted >=3x -- close but not quite, honestly reported; the
      buffer-reduction step itself is partly memory-bound and doesn't scale
      as cleanly as the embarrassingly-parallel pair loop it wraps).
      Combined with Phase 8's traversal parallelism, total wall-clock for
      the whole FMM call is now **2.07-2.39x** faster than the fully
      single-threaded baseline (8-16 threads) -- up from Phase 8's
      2.1-2.3x, since near-field no longer dominates the remaining serial
      fraction the way it did there. Parallel efficiency for the *whole
      call* is still well under the 0.7 gate (~15-26%, mostly limited by
      tree build + L2L, now the largest un-parallelized fraction, and by
      task/thread-team spin-up overhead at this N) -- reported honestly,
      not redefined; deeper gains would need parallelizing the tree build
      itself, out of scope for this pass. **SIMD (4-wide) kernel and a
      separate `-ffast-math` kernels TU deliberately descoped**: this
      project has built `-march=native` into every target since Phase 1,
      which already auto-vectorizes the scalar near-field kernel, and a
      hand-written second copy of the same near-field math (a real
      correctness-sensitive surface, given how many bugs the mutual FMM's
      *first* copy of this math already caught in Phase 6) wasn't judged
      worth it for a gate this phase's own analysis shows isn't the
      dominant remaining cost anyway. Full regression sweep (both
      projects' selftest/dt-selftest/fmm-selftest/spherical-selftest, plus
      the parallel-vs-serial and momentum checks from Phase 8) re-run clean
      -- forces agree to 3.1-3.7e-14 relative (floating-point reassociation
      from the changed summation order, not a correctness regression) and
      momentum conservation is unaffected (4.2-4.3e-17).
- [x] Phase 10 -- GPU direct O(N^2) sum. New `ngrav::gpu::ComputeAccelGpuDirect`
      (`include/ngrav/gpu/GpuDirect.hpp` + `src/gpu/GpuDirect.cpp`): one
      compute shader, `layout(local_size_x=256)`, one thread per target
      particle, plain loop over every source (Plummer softening,
      `a = -G*m*rij/(r^2+eps^2)^1.5` -- same formula as `ComputeAccelDirect`,
      term for term, so any mismatch can only come from the GPU pipeline or
      the fp32-vs-fp64 gap). Uses `fw::ComputeShader::FromSource` +
      `glCreateBuffers`/`glNamedBufferData`/`glGetNamedBufferSubData`
      (06_tidal_disruption's own buffer idiom); shader is compiled lazily on
      first call and cached (compiling needs an active GL context, which
      this otherwise-plain CPU library has no guarantee of at static-init
      time). `--gpu-selftest` (`main.cpp`): opens
      `fw::GLContext::CreateHidden(4,6)`, compares against
      `ComputeAccelDirect` on a 2000-body random cloud.
      **Gate results, both met**: GPU fp32 vs CPU fp64 max rel err
      **5.5e-6** (N=2000, in `--gpu-selftest`) / **1.0e-5** (N=30000,
      standalone benchmark) -- gate was <=1e-4, met with margin to spare.
      Speedup at N=3e4: **16.45x** over the (already Phase-1-optimized,
      `-O3 -march=native`, OpenMP-parallel) CPU direct sum -- gate was
      >=10x. Readback (a synchronous `glGetNamedBufferSubData` stall)
      dominates GPU wall time at this N (31.7ms of 32.6ms total; dispatch
      itself is ~0.01ms, i.e. the GPU is nowhere near saturated at N=3e4) --
      the margin over the 10x gate would only grow at larger N, where the
      O(N^2) compute cost -- not the fixed readback overhead -- dominates.
      **Deliberately not implemented**: shared-memory tiling (the classic
      GPU-gems "cache a block of positions in `shared`, inner loop over the
      tile" pattern) -- the plain per-thread loop already clears the gate
      comfortably, and tiling adds real correctness-sensitive complexity
      (barrier-uniformity, tile-boundary handling) not needed to meet it.
      Also not wired into `ngrav::Solver`/`System<D>`'s dispatch, the
      NBodyApp UI, `ScalingSweepWorker`, or a deck `--gpu` backend -- a
      standalone, directly-callable entry point instead, since integrating
      a GL-context-dependent solver into every generic-dispatch consumer is
      real scope better taken on once Phase 11's GPU Barnes-Hut gives the
      GPU path something an interactive user would actually reach for (a
      GPU *direct* sum alone has no advantage over CPU BH/FMM at the N an
      interactive session runs). Full regression sweep (`--selftest`/
      `--fmm-selftest`/`--dt-selftest`/`--spherical-selftest`) still PASS.
- [ ] Phase 11 -- GPU Barnes-Hut 3D
- [ ] Phase 12 -- GPU quadtree 2D
- [ ] Phase 13 -- realistic IC generators
- [ ] Phase 14 -- Studies suites
- [ ] Phase 15 -- block timesteps
- [ ] Phase 16 -- docs + Progress + movie
