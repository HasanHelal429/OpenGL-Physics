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
`expansion/SolidHarmonics` (moved from 02): analytic L2P gradient,
accumulate-in-place M2L, rotation-based O(p^3) M2L, runtime `p`;
`--spherical-selftest`.
Gate: analytic vs FD L2P grad <= 1e-6, L2P >= 4x faster; rotation M2L ==
naive O(p^4) to <= 1e-11, M2L cost fit exponent <= 3.3 in p over p in [3,10];
at p=10, mean rel force err <= 1e-7 on 1e4 Plummer; convergence err ~
theta^(p+1).

**Phase 8 — group BH walk + interaction lists + OpenMP tasks.**
`BarnesHut<D>` group/cell walk (descend once per leaf-cell, group bounding-
sphere MAC); `DualTreeFmm` seeding via OpenMP `taskgroup` recursion.
Gate: N=1e5, theta=0.5 -- BH force pass >= 2x faster than the per-particle
walk at equal accuracy; OpenMP-task FMM >= 0.7 parallel efficiency on 8
threads.

**Phase 9 — near-field: per-thread accumulation + SIMD kernel.**
Per-thread partial accel arrays + reduction; SIMD (4-wide) softened near-
field kernel (a `-ffast-math` kernels object library lands here).
Gate: near-field pass >= 3x faster at N=1e5 in the rotating-disk scenario;
SIMD kernel >= 2x over scalar.

**Phase 10 — GPU direct sum + --gpu-selftest.**
`ngrav/gpu/Kernels.hpp` (tiled shared-memory direct); `GpuDirect`; `--gpu`
backend; `--gpu-selftest` (opens `fw::GLContext::CreateHidden`).
Gate: GPU fp32 vs CPU fp64 -- max rel err <= 1e-4; >= 10x faster than OpenMP
direct at N=3e4.

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
- [ ] Phase 6 -- mutual dual-tree FMM + retire AdaptiveFmm
- [ ] Phase 7 -- SphericalFmm speedup
- [ ] Phase 8 -- group BH walk + OpenMP tasks
- [ ] Phase 9 -- near-field per-thread + SIMD
- [ ] Phase 10 -- GPU direct
- [ ] Phase 11 -- GPU Barnes-Hut 3D
- [ ] Phase 12 -- GPU quadtree 2D
- [ ] Phase 13 -- realistic IC generators
- [ ] Phase 14 -- Studies suites
- [ ] Phase 15 -- block timesteps
- [ ] Phase 16 -- docs + Progress + movie
