# projects/03_nbody_gravity_2d — Design Plan

The 2D companion to `02_nbody_gravity`. **The full plan, phase list, gates and
`nbody_core` architecture live in
[`../02_nbody_gravity/N_Body_Gravity_Plan.md`](../02_nbody_gravity/N_Body_Gravity_Plan.md)**
-- both projects are upgraded in parity on one feature branch. This file only
records what is genuinely 2D-specific.

## What differs from 3D

- **Green's function**: 2D gravity uses the true 2D fundamental solution --
  force ~ 1/r, potential +ln r (not a projected 3D law). Circular speed is
  separation-independent; a 1/r force is not one of Bertrand's closed-orbit
  laws, so 2D orbits precess. `ngrav::PlummerAccel<2>` / `PlummerPairPotential<2>`
  handle this via `if constexpr`.
- **Angular momentum** is a scalar (the out-of-plane z-component
  `r_x v_y - r_y v_x`), not a 3-vector. `ngrav::AngMom<2> = double`.
- **Tree**: quadtree (4 children) instead of octree (8). One
  `ngrav::AdaptiveTree<D>` template, `kChildren<2> == 4`.
- **High-accuracy FMM**: `ComplexFmm` (complex-Laurent series, order p ~ 10) is
  already a proper order-p dual-tree FMM -- 2D's `ln r` kernel makes each
  degree one complex power and M2L O(p) per coefficient. It is **not** retired;
  Phase 6c makes it mutual (momentum-conserving) rather than adding a separate
  Cartesian-Taylor 2D solver. It doubles as both the production O(N) and the
  accuracy-reference solver for 2D.
- **GPU** (Phase 12): a 2D quadtree variant of 06's octree -- ~5 kernels
  rewritten (4-child quadrant test, 2-axis Morton with 15 bits/axis,
  `treeChild[4*cell]`, 2D bounding box, group MAC with the 2D 1/r accel). The
  bitonic sort and the grouped-walk host loop port unchanged.
- **Studies**: `Studies/nbody_gravity_2d/` -- `2d_vs_3d_gravity` (how `ln r`
  reshapes collapse: no `1/r^2` focusing, solid-body rotation in a uniform
  disk) and `disk_bar_instability` (m=2 bar growth rate vs rotational-support
  f, FMM vs BH -- needs the mutual FMM's momentum conservation to not
  spuriously heat the disk).
- **App**: simpler than 3D -- orthographic pan/zoom camera, synchronous O(N^2)
  diagnostics (no background `DiagnosticsWorker`), no `ScalingSweepWorker`
  (the offline `tools/plot_scaling.py` covers it).

## Progress

- [x] Phase 1 -- `NBodySystem` wraps `ngrav::System<2>`; Direct/BarnesHut via
      `nbody_core`'s flat quadtree; `ComplexFmm` kept in-project via adapter.
      `nbody_core_selftest` 2D checks PASS (Kepler E-drift 5.4e-6 / L-drift
      1.7e-14; BH theta=0 vs Direct 2e-14). Builds and links `nbody_core`.
- [x] Phase 3 -- adaptive global dt (shared `ngrav::Integrator<D>`), deck
      `[time].adaptive/eta`, `NBodyApp` checkbox/slider, `--dt-selftest`
      (runs the same 3D eccentric-orbit validation as 02 -- 2D's `1/r`
      force law has no closed-form vis-viva orbit; the integrator code
      itself is dimension-generic). No 2D-specific eccentric-orbit deck:
      2D's `Kepler()` intentionally keeps its fixed `0.8*vCirc` convention
      (2D orbits precess rather than closing, so "eccentricity" isn't the
      same well-defined quantity -- see `Kepler()`'s own comment in
      `Scenarios.cpp`).
- [x] Phase 4 -- compact-support spline softening, shared `ngrav::Softening`.
      2D uses its own closed-form force (`SplineChi2D`, enclosed-"mass"/r
      law) and a once-built quadrature table for the potential (the log
      outer branch has no elementary closed form -- see 02's Progress entry
      for the bug this caught). `--selftest` PASS: force == Newtonian to
      2.2e-16 for r>=2h; spline-softened 2D orbit energy drift 2.4e-6.
- [x] Phase 5 -- relative/acceleration MAC (shared `ngrav::Mac`) is available
      here too (deck `[solver].mac = "relative"`), but the **quadrupole**
      correction is 3D-only by design: 2D's higher-accuracy path is the
      arbitrary-order complex-Laurent `ComplexFmm`, not a Cartesian
      quadrupole term (`AddQuadrupole` in `Solvers.cpp` is a no-op for
      D==2). No 2D-specific gate beyond the shared selftest's own Barnes-Hut
      θ=0 exactness (unaffected, still 2e-14).
- [ ] Phase 6c -- make `ComplexFmm` mutual (momentum-conserving), matching
      02's Phase 6 for the 2D solver. **Not yet done** -- 02's Phase 6
      (the 3D Cartesian-Taylor mutual FMM + retiring AdaptiveFmm) is
      complete and validated; 2D's own `ComplexFmm` is untouched by this
      phase (it's already a proper order-p FMM, just not yet mutual). `--dt-
      selftest` here still exercises 02's shared 3D eccentric-orbit check
      (see Phase 5's note -- the integrator code is dimension-generic, but
      2D has no analogous closed-form orbit to validate against directly).
- [ ] Phases 7-16 -- in parity with 02 (see that plan's Progress).
