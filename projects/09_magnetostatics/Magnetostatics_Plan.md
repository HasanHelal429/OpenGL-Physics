# projects/09_magnetostatics — Design Plan

## Context

`Physics Simulations/Electromagnetism/Poisson_Solver/poisson.py` solves the
2D electrostatic Poisson equation by a **sparse direct solve** — ideal for a
one-shot static field, with nothing to gain from a straight port.
Magnetostatics is the same elliptic problem with a different source and a
curl post-step, but two things make it worth a C++/GPU project rather than a
second Python notebook:

1. An **iterative GPU solve** (red-black Gauss-Seidel, then geometric
   multigrid) makes the field update real-time, so current-carrying
   conductors become draggable and the field re-solves live.
2. **3D Biot-Savart** over arbitrary coil geometry is an embarrassingly
   parallel field-point sum — a natural compute-shader workload with no
   linear solve at all.

On top of the field solver, a **relativistic Boris pusher** traces test
charges through the computed **B** (and an optional static **E**), so the
project has motion to watch: cyclotron orbits, E×B drift, magnetic mirroring
and bottling. Same pattern as `08_compressible_fluid` folding immersed
obstacles into `Euler2D` rather than spinning up a separate project.

House model (from `05` on): `fw::Simulation` + TOML deck, headless batch +
`fw::SimApp` interactive, `tools/*.py` for validation plots and movies,
`docs/SIMULATION.md` for the writeup.

## Numerical approach

### 2D field solver (`MagnetostaticsField`, out-of-plane current)

Planar current `J = J_z(x,y) ẑ` ⇒ vector potential `A = A_z(x,y) ẑ` obeys

```
∇²A_z = -μ₀ J_z
```

the identical 5-point Laplacian as `poisson.py`. Solved iteratively on the
GPU:

- **Red-black Gauss-Seidel** as the baseline smoother — two compute-shader
  passes per sweep, checkerboard update, no data race.
- **Geometric multigrid V-cycle** (restrict / prolong compute shaders) once
  RBGS convergence is the bottleneck — target ~1e-6 residual drop in O(10)
  V-cycles, iteration count independent of grid size.
- Boundary: `A_z = 0` on the domain edge (field from compact currents
  decays; box taken large enough), or a deck-selectable homogeneous Neumann
  `∂A_z/∂n = 0`.

Then `B = ∇×A = (∂A_z/∂y, -∂A_z/∂x, 0)` by central differences (same
convention as `poisson.py::efield`, one-sided at the edges).

Conductors are set cells of `J_z` — a wire is one cell (or a small disk)
carrying total current `I`, so `J_z = I / cell_area`.

### 3D Biot-Savart (`BiotSavartField`)

For polylines (coils) carrying currents `I_k`, the field at grid point `r`:

```
B(r) = Σ_k (μ₀ I_k / 4π) Σ_seg  dl × (r − r') / |r − r'|³
```

one compute-shader thread per field point, inner loop over all segments
(endpoints in an SSBO). No solve; exact up to the polyline discretization of
the coil. Used wherever the 2D out-of-plane idealization is too crude —
Helmholtz pairs, finite solenoids, single loops.

### Boris pusher (`TestCharge`)

Standard relativistic Boris algorithm (half electric kick / magnetic
rotation / half electric kick). `q/m` and initial `(x, v)` from the deck,
field sampled by bi/trilinear interpolation from whichever field source is
active. Fixed `dt` set to a fraction of the local cyclotron period. Exact
kinetic-energy conservation in a static magnetic field is a built-in
per-step invariant to monitor.

## Deck schema

```toml
[grid]      nx, ny, lx, ly                 # 2D solver domain
[solver]    method = "multigrid" | "rbgs" | "biot_savart"
            tol, max_cycles
[[wire]]    x, y, current                  # 2D out-of-plane line current
[[coil]]    shape = "loop" | "solenoid" | "helmholtz"
            radius, axis, center, turns, current, length   # 3D Biot-Savart
[[charge]]  q_over_m, x0, v0, color        # optional test charges
[render]    field = "B_mag" | "Bx" | "By" | "A_z"
            show_field_lines, show_particles
```

## Scenes / decks

| deck | shows | validation target |
|---|---|---|
| `wire.toml` | single out-of-plane wire | `B = μ₀I/2πr`, field direction |
| `solenoid.toml` | finite solenoid cross-section | `B ≈ μ₀nI` inside, ~0 outside; end falloff |
| `helmholtz.toml` | coil pair at spacing = radius (3D) | on-axis `d²B/dz² = 0` at centre |
| `magnetic_bottle.toml` | two coils + trapped charge | particle mirrors; `μ = v⊥²/B` conserved |
| `cyclotron.toml` | uniform B + test charge | radius `r = mv/qB`, period `T = 2πm/qB` |
| `exb_drift.toml` | crossed E, B + charge | drift velocity `v_d = E×B/B²` |

## File layout

```
projects/09_magnetostatics/
    CMakeLists.txt
    Magnetostatics_Plan.md
    src/
        main.cpp                       # dispatch RunHeadless / SimApp
        MagnetostaticsSim.{hpp,cpp}    # fw::Simulation: field + charges + render
        FieldSolver.{hpp,cpp}          # RBGS / multigrid host orchestration
        kernels_field.hpp              # GPU smoother / restrict / prolong source
        kernels_biot_savart.hpp
        BorisPusher.hpp
    decks/
    tools/
        plot_wire.py  plot_solenoid.py  plot_helmholtz.py
        plot_orbit.py  make_movie.py
    docs/SIMULATION.md
```

## Phases

**Phase 1 — 2D field solver + validation plots.**
`MagnetostaticsField` (RBGS first), `B = ∇×A`, headless output of `A_z`,
`Bx`, `By`. Gate: wire field within ~1% of `μ₀I/2πr` over the mid-domain;
manufactured-solution 2nd-order convergence (same `sin·sin` technique as
`poisson.py`'s check); solenoid interior within a few % of `μ₀nI`.

**Phase 2 — Multigrid + interactive view.**
Geometric multigrid V-cycle; `fw::SimApp` live heatmap (B magnitude /
components / `A_z`, cycle with `M`), field-line (streamline) overlay,
draggable wires re-solving live. Gate: multigrid iteration count flat in
grid size (≈ const from 128² to 1024²); interactive re-solve under ~16 ms at
512².

**Phase 3 — 3D Biot-Savart.**
`BiotSavartField` compute shader, coil primitives (loop / solenoid /
Helmholtz). Gate: single loop on-axis vs `μ₀I R²/[2(R²+z²)^{3/2}]`
(polyline-limited precision); Helmholtz centre `d²B/dz²` zero to
discretization; solenoid interior vs analytic.

**Phase 4 — Boris pusher.**
Relativistic Boris integrator, field interpolation, deck-driven test
charges, trajectory output + render. Gate: cyclotron `r`, `T` exact to
integrator order; energy drift < 1e-6 over 1e5 steps in static B; E×B drift
speed within tolerance; magnetic-bottle reflection with adiabatic invariant
`μ` conserved to a few %.

**Phase 5 — `docs/SIMULATION.md` + website media.**
Full writeup; render the deck movies and validation figures for
`_projects/electrodynamics.md`.

## Studies (`Studies/magnetostatics/`)

- `helmholtz_uniformity_vs_spacing` — centre-field flatness vs coil
  separation / radius, confirming the `s = R` optimum and quantifying the
  usable-volume tradeoff.
- `multigrid_scaling` — V-cycles to convergence vs grid size (should be
  flat), wall-time vs RBGS.
- `bottle_confinement_vs_mirror_ratio` — confined fraction of an isotropic
  charge population vs mirror ratio `B_max/B_min`, against the loss-cone
  prediction `sin²θ_loss = B_min/B_max`.

## Progress

- [x] Phase 1 — 2D field solver + validation (RB-GS/SOR; `--selftest` MMS
      ratio 4.000; wire 0.3% vs `μ₀I/2πr`; solenoid 1.9% vs `μ₀K_s`, Neumann BC)
- [x] Phase 2 — geometric multigrid (W-cycle flat at 8-10 across 64²→1024²,
      `--mg-scaling`) + interactive `fw::SimApp` view + `--render-check`
- [x] Phase 3 — GPU Biot-Savart (`--biot-selftest`: loop on-axis 3e-4,
      Helmholtz centre 3e-5, `d²B/dz²`≈0); slice planes; loop/solenoid/helmholtz
- [x] Phase 4 — relativistic Boris pusher (`--boris-selftest`: cyclotron r
      3e-7, `dγ/γ` 6e-16, E×B 1e-5); cyclotron / E×B / magnetic-bottle decks
- [x] Phase 5 — `docs/SIMULATION.md`; `Studies/magnetostatics/`
      (`multigrid_scaling`, `helmholtz_uniformity_vs_spacing`); `make_movie.py`

All phases complete.
