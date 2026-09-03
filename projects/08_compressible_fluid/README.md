# 08 — Compressible fluid: 1D Euler, HLLC/MinMod/RK2 finite volume

A conservative finite-volume solver for the compressible Euler/Navier-Stokes
equations — the flat-spacetime, non-relativistic limit of
[`07_grhd`](../07_grhd/)'s Valencia formulation (see this project's own
[`docs/SIMULATION.md`](docs/SIMULATION.md) for the full formalism-mapping
walkthrough): conserved variables evolved by flux differencing, MinMod-limited
piecewise-linear reconstruction, an HLLC approximate Riemann solver, RK2
(Heun) method-of-lines time integration. Unlike the incompressible solvers in
[`Physics Simulations/Fluid Mechanics`](../../../Physics%20Simulations/Fluid%20Mechanics/)
(Stable Fluids, a MAC-grid projection method, LBM), this is a genuine
shock-capturing conservation-law scheme — it can resolve a real discontinuity
(a shock) without producing spurious oscillations or losing mass/momentum/
energy across it.

**Status** — 1D (CPU + GPU), 2D inviscid (CPU, via Strang dimensional
splitting), 2D **viscous** (Newtonian shear stress + Fourier conduction),
and flow past an **immersed obstacle** (Brinkman-style cell masking + a
prescribed inflow boundary). Validated against the exact Sod shock tube, a
GPU-vs-CPU cross-check, an exact 2D isentropic-vortex advection solution, a
qualitative match to the published Kurganov & Tadmor 2D Riemann
"Configuration 3" density pattern, the exact plane-Poiseuille velocity
profile (0.08% error), the exact Taylor-Green vortex decay rate (1.6%
error), and cylinder vortex shedding (a real von Kármán street, Strouhal
number within 13% of Roshko's correlation — see "Flow past an obstacle"
below for the honest read on that gap). Used in
[`Studies/compressible_fluid`](../../../Studies/compressible_fluid/) for a
viscosity-vs-vorticity-decay sweep.

## Physics

**Equations.** 1D compressible Euler, ideal Gamma-law gas (`P=(Γ-1)ρε`):
conserved state `(ρ, ρu, E)`, `E = ρε + ½ρu²`.

**Primitive recovery.** Closed-form — `u=ρu/ρ`, `P=(Γ-1)(E-½ρu²)` — no
Newton iteration needed (unlike `07_grhd`'s con2prim: there's no Lorentz
factor to solve for once velocities aren't relativistic).

**Reconstruction.** Piecewise-linear, MinMod-limited (same TVD limiter as
`07_grhd`), applied to primitives, extrapolated a half-cell to each face.

**Riemann solver.** HLLC (Toro ch. 10) — an upgrade over `07_grhd`'s HLLE:
it resolves the middle (contact/shear) wave explicitly, so a contact
discontinuity stays sharp instead of smearing over many cells. Wave-speed
bounds use the simple Davis estimate (`min/max` of `u±c` on each side).

**Time integration.** RK2 (Heun's method), identical in structure to
`07_grhd`.

**What's absent, on purpose:** geometric source terms (flat metric — they're
identically zero, so the whole term simply doesn't appear), and a genuinely
adaptive per-step CFL controller — `dt` is fixed once at `Configure()` from
the initial condition's max wave speed (`Euler1D::MaxWaveSpeed()`), safe for
a single self-similar Riemann problem since a shock/rarefaction/contact's
speeds are always bounded by the data (see `CompressibleSim.cpp`'s comment)
but not something a later time-varying flow can rely on in general (the
Poiseuille deck below sidesteps this a different way — see its own comment).

## Validation

`decks/sod_shocktube.toml` — the classic ρ/P ratio-10 Sod problem. Compare
against the exact self-similar solution:

```sh
cmake --build --preset release --target 08_compressible_fluid
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe \
    --deck projects/08_compressible_fluid/decks/sod_shocktube.toml \
    --out projects/08_compressible_fluid/out/sod_shocktube
python projects/08_compressible_fluid/tools/plot_shocktube.py projects/08_compressible_fluid/out/sod_shocktube
```

Every 2D scenario below runs through the same `--scene` flag and the same
`CompressibleSimScene` class — see "The generic scene schema" for what
that means and how the schema is structured.

`decks/riemann2d_config3.toml` — the 2D four-quadrant Riemann problem:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --scene \
    --deck projects/08_compressible_fluid/decks/riemann2d_config3.toml \
    --out projects/08_compressible_fluid/out/riemann2d_config3
python projects/08_compressible_fluid/tools/plot_riemann2d.py projects/08_compressible_fluid/out/riemann2d_config3
python projects/08_compressible_fluid/tools/make_movie.py projects/08_compressible_fluid/out/riemann2d_config3 --field density
```

This is the deck to watch if you want to see a shock actually *form*: the
sharp initial quadrant boundaries curve and steepen into two genuine shock
fronts, meeting a mixing/vortical region at the center (see `--field
density` below for why density, not the default vorticity, is the field to
render for this one). It's also the one `--scene` deck small and fast
enough to watch live: `--interactive --scene --deck
projects/08_compressible_fluid/decks/riemann2d_config3.toml` (density is
already the default field in the interactive view, no `--field` flag
needed there).

`decks/poiseuille_channel.toml` — plane Poiseuille flow; see "Viscosity" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --scene \
    --deck projects/08_compressible_fluid/decks/poiseuille_channel.toml \
    --out projects/08_compressible_fluid/out/poiseuille_channel
python projects/08_compressible_fluid/tools/plot_poiseuille.py projects/08_compressible_fluid/out/poiseuille_channel
```

`decks/taylor_green.toml` — 2D Taylor-Green vortex decay; see "Viscosity" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --scene \
    --deck projects/08_compressible_fluid/decks/taylor_green.toml \
    --out projects/08_compressible_fluid/out/taylor_green
python projects/08_compressible_fluid/tools/plot_taylor_green.py projects/08_compressible_fluid/out/taylor_green
```

`decks/cylinder_re100.toml` — cylinder vortex shedding; see "Flow past an
obstacle" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --scene \
    --deck projects/08_compressible_fluid/decks/cylinder_re100.toml \
    --out projects/08_compressible_fluid/out/cylinder_re100
python projects/08_compressible_fluid/tools/plot_strouhal.py projects/08_compressible_fluid/out/cylinder_re100
python projects/08_compressible_fluid/tools/make_movie.py projects/08_compressible_fluid/out/cylinder_re100
```

`decks/airfoil_wind_tunnel.toml` — a NACA0012 airfoil at a fixed angle of
attack; see "Wind tunnel: airfoil at angle of attack" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --scene \
    --deck projects/08_compressible_fluid/decks/airfoil_wind_tunnel.toml \
    --out projects/08_compressible_fluid/out/airfoil_wind_tunnel
python projects/08_compressible_fluid/tools/make_movie.py projects/08_compressible_fluid/out/airfoil_wind_tunnel
```

`tools/make_movie.py` works for any `--scene` run, rendering vorticity by
default (`--field density` instead for a shock-dominated run, e.g. the
Riemann deck above, where vorticity is dominated by any mixing region and
a shock only shows as a faint thin line) frame-by-frame, drawing every deck
obstacle -- a circle as a solid disk, an airfoil as its rotated NACA00xx
outline — the flow is far more legible as a movie than any single static
frame.

## Interactive mode

Any `--scene` deck can also be watched live instead of run headless to disk:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --interactive --scene \
    --deck projects/08_compressible_fluid/decks/cylinder_re100.toml
```

This opens a real window (`fw::SimApp`) with the usual HUD (play/pause,
step, speed, reset, record-to-disk, `F12` screenshot) and a live heatmap of
one scalar field at a time — `CompressibleSimScene::Render()` re-packs the
chosen field from `Euler2D::PrimAt` into a small GPU buffer every frame
(the solver's state lives on the CPU, so there's no persistent GPU copy to
just re-bind) and draws it with a `magma`-colormapped fullscreen-quad
shader, the same technique `05_tdse_gpu`'s live view uses. Obstacle cells
are overlaid as a flat color so the cylinder itself is always visible
regardless of which field is selected. A small text label in the top-left
corner (via `fw::Font`/`fw::TextRenderer`, the same text-rendering pieces
`04_molecular_dynamics`'s `MDSim` uses for its own on-screen readout) always
names the field currently on screen, so the mode is visible without
already knowing the `M` keybinding.

Keys:

| Key | Effect |
|---|---|
| `M` | Cycle the displayed field: density -> speed -> vorticity -> tracer |
| `[` / `]` | Decrease / increase colormap gain |
| `-` / `=` | Decrease / increase the gamma curve (lower lifts faint detail) |
| `0` | Reset zoom, pan, gain, and gamma |
| `Up` / `Down` | If the deck has an airfoil obstacle: increase / decrease its angle of attack live |
| drag / scroll | Pan / zoom the view |

`Up`/`Down` rewrite `Euler2D`'s obstacle mask on the fly (`CompressibleSimScene::RebuildObstacleMask`) without touching the flow field or resetting anything, so you can sweep the angle mid-run and watch the wake respond — a cell that becomes solid gets its velocity zeroed from the next step on (`Euler2D::ApplyObstacleMask`, unchanged), and a cell that becomes fluid again just continues from rest. `R` (reset) restores the flow to its initial condition but deliberately leaves the angle wherever you left it — angle of attack is a live control knob, not part of the deck's "initial condition."

Bare `--interactive` (no `--scene`) opens the 1D shock tube instead
(`CompressibleSim`), which still has full play/pause/step/record controls
but currently draws nothing — only `CompressibleSimScene` overrides
`Render()` so far (see `framework/Simulation.hpp`: rendering hooks are
optional and default to a no-op, the same fallback `07_grhd`'s own sim
classes currently rely on).

`--selftest` (no deck needed) runs five fast, deck-independent checks:
HLLC-flux self-consistency (`F_HLLC(s,s) == F(s)` exactly), a
conservation-identity check (mass and energy exactly conserved, momentum
matching the analytic boundary-pressure-forcing prediction, both before
either wave reaches the domain edge — see `main.cpp`'s comment for why
these are exact identities, not approximate physics), a 1D GPU-vs-CPU
cross-check (the same Sod IC run 50 RK2 steps on both `Euler1D` and the
`kernels_euler1d.hpp` compute-shader port, comparing recovered primitives —
matches to ~1e-7 relative, the expected float32 precision floor), the 2D
isentropic-vortex advection check described below, and a 2D GPU-vs-CPU
cross-check (`kernels_euler2d.hpp` vs. `Euler2D` on that same vortex
scenario — see "Performance" below).

## The generic scene schema

Every 2D scenario above is the **same C++ class**, `CompressibleSimScene`,
reading a common deck schema — not four separate hardcoded classes each
with its own scenario-specific setup logic (that *was* the design through
this project's earlier phases: `CompressibleSim2D`/`CompressibleSimChannel`/
`CompressibleSimTaylorGreen`/`CompressibleSimCylinder`, one per `--flag`).
Any *combination* of the pieces below is expressible as pure deck data,
with no new C++ needed:

- **`[grid]`** — raw `nx`, `ny`, `x_min`, `x_max`, `y_min`, `y_max`. Always
  physical values, not scenario-specific convenience parameters (e.g. no
  `diameter`/`blockage`/`reynolds` fields) — matching this project's
  existing convention (every other deck, e.g. `decks/sod_shocktube.toml`,
  already specifies raw numeric values with a derivation comment, not a
  higher-level physical parameterization the C++ then reverse-engineers).
  A deck that wants Reynolds-number-style parameters computes them itself
  into the raw fields, with a comment showing the arithmetic (see
  `decks/cylinder_re100.toml`'s `mu = ...` comment for an example).
- **`[physics]`** — `gamma`, `mu`, `conductivity`, `tracer_diffusivity`,
  `body_force_x` (all default to 0 except `gamma=1.4`).
- **`[boundary.left/right/bottom/top]`** — each side's `type`
  (`outflow`/`periodic`/`no_slip`/`free_slip`/`inflow`), defaulting to
  `outflow` if the table is omitted entirely. An `inflow` side also reads
  `rho`/`u`/`v`/`p`/`tracer`, and can carry a nested
  `[boundary.<side>.profile]` table selecting a named, position-dependent
  variation of that base state — currently just `type = "tracer_stripes"`
  (`stripe_width`), the one pattern this project has actually needed
  (continuous alternating dye bands at an inlet, `decks/cylinder_re100.toml`).
  A new pattern is a new named case in `CompressibleSimScene.cpp`, same as
  `WallBC` itself already being a fixed enum of named cases.
- **`[[obstacles]]`** — an array of tables (any number, combined by logical
  OR into one solid-cell mask), each either `{shape="circle", center=[x,y],
  radius}` or `{shape="airfoil", center=[x,y], chord, thickness, angle_deg}`
  (`center` is the leading edge; `thickness` is the NACA00xx thickness
  fraction, e.g. `0.12` for NACA0012; `angle_deg` is the angle of attack,
  positive nose-up, also adjustable live in `--interactive` -- see
  "Interactive mode"). Another shape is a new named case away, same
  philosophy as `WallBC` and the inflow profile above.
- **`[initial_condition]`** — a `type` (`uniform`, `riemann_quadrants`, or
  `taylor_green_vortex`, each reading its own type-specific keys — see the
  four decks above for a worked example of each) plus an optional
  **`[initial_condition.perturbation]`** modifier applied on top
  (currently just `type = "antisymmetric_gaussian"`: a one-time,
  Gaussian-localized, sign-flipped-across-`center.y` velocity bump on
  `component` — the belt-and-suspenders symmetry-breaker
  `decks/cylinder_re100.toml` needs, see "Flow past an obstacle" below).
- **`[[probes]]`** — named points (`name`, `x`, `y`); each writes four
  diagnostics `<name>_rho`/`_u`/`_v`/`_p` every frame.
- **`[output] write_tracer`** — whether to write the `tracer` frame field
  (default `false`).

Diagnostics `mass_total`, `energy_total`, `kinetic_energy`, and `u_max` are
always written regardless of scenario — cheap to compute in the same pass
that writes the frame fields, so every deck's `diagnostics.csv` is a
superset covering whichever of these any particular analysis script
actually reads (`plot_poiseuille.py` reads `u_max`, `plot_taylor_green.py`
reads `kinetic_energy`, neither cares that the other's column is also
present).

The 1D shock tube (`CompressibleSim`, `Euler1D`, `--deck` with no `--scene`
flag) is deliberately **not** folded into this schema — boundary "sides"
(plural), 2D obstacles, and probes are 2D-specific concepts with no natural
1D analog, and forcing them into one shared schema would only obscure both.

## 2D: dimensional splitting

`Euler2D` extends the 1D scheme to 2D via **Strang splitting**
(X half-step / Y full-step / X half-step per timestep, LeVeque ch. 17):
each fractional step reuses the exact same MinMod+HLLC+RK2 line update as
`Euler1D`, applied row-by-row (X sweeps) or column-by-column (Y sweeps), so
a flow aligned with a grid axis reduces to running `Euler1D` on every line.
The only genuinely new physics is the HLLC extension to a passively
advected transverse momentum component (`HllcFluxX`/`HllcFluxY` in
`Euler2D.cpp`, Toro sec. 10.4): the velocity component *along* the sweep
direction sees the usual three-wave HLLC structure, while the transverse
velocity is simply carried unchanged into whichever star state (left or
right of the contact) the flux point falls in.

Boundary condition: like `Euler1D`, zero-gradient/transmissive — but here
implemented via **clamped neighbor indices** rather than explicit ghost
cells (see `kernels_euler1d.hpp`-style reasoning, restated in `Euler2D.cpp`'s
anonymous namespace): clamping forces the boundary cell's own MinMod slope
to exactly zero, which is provably identical to `Euler1D`'s ghost-copy
approach, not an approximation of it.

**Validation.** `--selftest`'s `SelfTestVortexAdvection` — a smooth 2D
isentropic vortex (an exact traveling solution of the 2D Euler equations)
advected diagonally, checked against the exact solution shifted by the
background flow. Unlike Sod/Config 3 (both have shocks), this is the check
that dimensional splitting didn't introduce a splitting-direction bias or
degrade the scheme's smooth-flow accuracy — relative L2 density error
~4% at a deliberately coarse resolution (~10 cells per vortex core radius).
Separately, `decks/riemann2d_config3.toml` + `tools/plot_riemann2d.py`
reproduce Kurganov & Tadmor (2002)'s 2D Riemann "Configuration 3" (four
quadrant states, no exact solution, but a widely published reference
density pattern to compare against qualitatively).

## Viscosity: Newtonian stress + Fourier conduction

`Euler2D::SetViscosity(mu, conductivity)` adds a Navier-Stokes viscous term
on top of the inviscid Strang-split sweeps, via an extra explicit-diffusion
sub-step each `Step()`: `mu*Laplacian(u)` added to x-momentum, `mu*Laplacian(v)`
to y-momentum, each Laplacian assembled from one row-local `d²/dx²` pass
(`DiffuseX`) plus one column-local `d²/dy²` pass (`DiffuseY`), so every
viscous term stays a 1D operation like the inviscid sweeps. This is the
exact incompressible-limit reduction of the full Newtonian viscous stress
divergence — dropping only Stokes' hypothesis' bulk-viscosity correction
and a mixed `d²/dxdy` cross term that cancels a compensating factor of 2
elsewhere in the full tensor (see `Euler2D.cpp`'s `LineDiffuse` comment for
the derivation) — exact for divergence-free flow, and a good approximation
at the low Mach numbers this project's validation targets use.

**A bug this caught, worth naming.** The first version of this diffusion
term used coefficient `2*mu` on the direction-of-sweep ("normal") second
derivative and `mu` on the transverse one — correct in isolation for a
flow with *no dependence on the swept direction* (Poiseuille: `u=u(y)`
only, so the erroneous factor multiplies an identically-zero term), but
silently 1.5x too strong for a genuinely 2D velocity field. It passed
Poiseuille validation (0.08% error, unaffected either way) but decayed a
Taylor-Green vortex 2x too fast — caught only once a second, independent
viscous-flow test existed. See docs/SIMULATION.md's Phase 4 section for
the full derivation of why the fix is exactly `mu*Laplacian`, and the
lesson generalizes: **a simplification validated by only one test case can
be wrong in a way that test can't see** — same shape of lesson as
`07_grhd`'s docs/SIMULATION.md section 8.

Explicit diffusion has its own (usually tighter than the hyperbolic CFL's)
parabolic stability limit, so `Step()` sub-cycles `DiffuseX`/`DiffuseY`
against `0.4*dx_min²/max(mu, conductivity)` automatically — a deck's chosen
`time.cfl` never needs to know about it.

**No-slip / periodic boundaries.** `WallBC::NoSlipReflective` (ghost
momentum negated, so velocity linearly extrapolates to exactly zero at the
wall face) and `WallBC::Periodic` (ghost wraps to the far interior cell) are
new alongside the existing `Outflow` (Phases 1-3's zero-gradient, unchanged
default). `decks/poiseuille_channel.toml` uses periodic-x/no-slip-y; a
constant body force (`SetBodyForceX`) drives the flow, the standard way to
sustain a periodic channel without an actual streamwise pressure drop.
`decks/taylor_green.toml` instead uses periodic in *both* directions — no
walls at all.

**Validation, two ways.**
- **Steady state**: started from rest, `decks/poiseuille_channel.toml`
  spins up to the analytic steady parabolic profile
  `u(y) = (f/(2*mu))*y*(H-y)` — `tools/plot_poiseuille.py` measured
  **0.08% max error** against it (residual: compressibility,
  `u_max`/soundspeed ≈ 0.1, not zero).
- **Transient decay**: `decks/taylor_green.toml`'s doubly-periodic vortex
  pair has an exact viscous-decay solution, kinetic energy
  `KE(t) = KE(0)*exp(-4*nu*k²*t)` — `tools/plot_taylor_green.py` fits the
  measured log-slope and found **1.6% error** against the analytic rate
  (residual: the inviscid HLLC/MinMod scheme's own numerical dissipation,
  on top of the physical viscosity, plus the same small compressible
  correction as Poiseuille — expected to be larger here than Poiseuille's
  steady-state residual, since a genuinely time-evolving flow keeps
  accumulating that numerical dissipation rather than converging past it).

## Flow past an obstacle: cylinder vortex shedding

`Euler2D::SetObstacleMask(isSolid)` adds an immersed-boundary solid region
via **cell masking**: every step, cells inside the mask have their
velocity forced to exactly zero (energy reduced by exactly the removed
kinetic energy, so pressure is preserved rather than spiking) — the hard
(instantaneous) limit of Brinkman penalization, a standard, simple
immersed-boundary technique that needs no changes to the Riemann solver or
line-sweep structure at all. Two new boundary conditions support external
flow: `WallBC::Inflow` (ghost cells fixed to a prescribed upstream state,
`SetInflowState`) and `WallBC::FreeSlipReflective` (mirrors only the
boundary-normal velocity component — a symmetry/no-penetration wall, for a
domain edge that's a truncation, not a physical wall).

`decks/cylinder_re100.toml` mirrors the geometry and boundary conditions of
[`MAC_Grid_Solver`'s `Cylinder_Vortex_Shedding.ipynb`](../../../Physics%20Simulations/Fluid%20Mechanics/MAC_Grid_Solver/Cylinder_Vortex_Shedding.ipynb)
(that project's own incompressible validation of the same physics):
`D=1, Re=100, U_inf=1`, blockage `D/H=0.125`, 5D upstream / 20D downstream,
inflow-left / outflow-right / free-slip-top-bottom. `mach=0.2` (soundspeed
`=U_inf/mach=5`, baked into the deck's raw `p` value with a derivation
comment — see "The generic scene schema" above) keeps this weakly
compressible without the acoustic stiffness a much lower Mach number would
need a preconditioner to handle well. `nx=200, ny=64` (`cells_per_d=8`
equivalent) is a first-pass resolution compromise — coarser even than the
Python reference's own flagged-as-suboptimal 12 — chosen for a single
validated run's compute budget (~1.5 minutes on this project's usual
hardware for 130 time units, after the OpenMP parallelization in
"Performance" below).

**A real gotcha, worth naming**: this domain, obstacle mask, and boundary
conditions are symmetric about the channel centerline to floating-point
precision. A first attempt with an exactly-centered cylinder and a uniform
impulsive-start initial condition **never sheds at all** — the wake
settles into an exactly symmetric, steady (if linearly unstable) state and
sits there for the entire run, because nothing in a perfectly symmetric
simulation breaks the symmetry vortex shedding requires breaking. Real
cylinders always shed because real flows always carry some asymmetric
disturbance (free-stream turbulence, structural vibration); the "cleaner"
perfectly symmetric numerical case is the artificial one. The fix: the
`[[obstacles]]` circle's `center` is placed slightly off the exact
centerline (`y=4.02` vs. the geometric `4.0`) — a *permanent* geometric
asymmetry, not just a transient perturbation, so the instability has
something to keep growing from — plus a smaller, belt-and-suspenders
one-time antisymmetric velocity bump via
`[initial_condition.perturbation]` to seed it faster than waiting on the
offset's much smaller steady-state asymmetry alone.

**Validation.** `tools/plot_strouhal.py` FFTs the downstream velocity
probe (4D behind the cylinder, on the centerline) and finds the shedding
frequency, giving `St = f*D/U_inf`. The default deck's run produces a real
von Kármán street (alternating vorticity lobes, visually unmistakable) and
measures **St=0.138** against Roshko's correlation's `St=0.159` at Re=100
(literature: `St~0.166`) — a real, if visible, **13% gap**, not hidden:
- The probe signal was still in its exponential-growth phase at the end of
  this run's 130 time units (visible in the probe timeseries), not yet
  saturated to a full periodic limit cycle — the growing oscillation's
  frequency should already closely track the eventual shedding frequency
  for a standard supercritical-Hopf-type onset, but "closely" isn't
  "exactly," and a longer run would settle this.
- `cells_per_d=8` is coarser than even the Python reference's own
  already-conservative 12.
- This project's own numerical dissipation (quantified directly in
  [`Studies/compressible_fluid/vorticity_decay_vs_viscosity`](../../../Studies/compressible_fluid/vorticity_decay_vs_viscosity/README.md))
  acts like added viscosity at this coarser resolution — and lower
  Strouhal number is exactly the direction that predicts (Roshko's own
  correlation gives a lower `St` at lower `Re`; solving backward from the
  measured `St=0.138` implies an effective `Re~65`, a real and sizable
  reduction from the nominal `Re=100` fully consistent with numerical
  dissipation at this resolution being the dominant cause).
- The flow is weakly compressible (`mach=0.2`), unlike the incompressible
  reference.

None of these are surprising or hidden — they're exactly the tradeoffs a
first validated pass at a coarser resolution and shorter run predicts, and
the measurement is otherwise clean (a single, sharp FFT peak, no secondary
structure). A finer grid and/or a longer run to reach saturation is the
natural follow-up to close this gap, not attempted here.

## Wind tunnel: airfoil at angle of attack

`decks/airfoil_wind_tunnel.toml` is the cylinder deck's obstacle swapped
for a symmetric NACA0012 airfoil (`shape="airfoil"`, see "The generic scene
schema" above) — same channel geometry style, same `mach=0.2`/`rho0`/`U_inf`
convention, `Re=rho0*U_inf*chord/mu=100` scaled to the airfoil's own chord
instead of a cylinder diameter. `IsInsideAirfoil` (`CompressibleSimScene.cpp`)
evaluates the standard symmetric NACA00xx thickness distribution
(`y_t(x) = 5*t*c*(0.2969*sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)^2 +
0.2843*(x/c)^3 - 0.1015*(x/c)^4)`, Abbott & von Doenhoff's open-trailing-
edge coefficients) in the airfoil's own body frame, rotated by the angle of
attack — otherwise it's exactly the same cell-masking immersed-boundary
technique the cylinder deck uses, with no changes needed to the Riemann
solver, viscosity, or obstacle machinery at all: a NACA section is just
another pointwise inside/outside test, like a circle's `dx²+dy²<=r²`.

Unlike the cylinder, this deck needs no artificial symmetry-breaking
perturbation — a nonzero angle of attack already makes the geometry
top/bottom-asymmetric before the solver takes a single step, unlike a
perfectly centered cylinder in a symmetric channel (see the gotcha above).
At the deck's default 8° angle of attack, the vorticity field shows the
expected lifting-airfoil signature: a negative-vorticity sheet along the
suction (upper) side and a positive-vorticity sheet along the pressure
(lower) side, visible in a `make_movie.py` render.

The angle of attack is also the one obstacle parameter exposed live in
`--interactive` (`Up`/`Down`, see "Interactive mode" above) — sweeping it
mid-run and watching the wake respond is the whole point of a wind tunnel,
and rewriting `Euler2D`'s cell mask on the fly needed no new machinery
either: `CompressibleSimScene::RebuildObstacleMask()` is the same
mask-construction logic `Configure()` already ran once at startup, just
callable again whenever the angle changes.

## Tracing the flow: a passive scalar dye

`Prim2D::tracer` / `Cons2D::rhoTracer` is a passive scalar (concentration,
dye-like) carried by the flow — it exerts no force and has no pressure of
its own, so it's dynamically inert: adding it changes nothing about the
solver's physics (confirmed: the cylinder deck's measured Strouhal number
is bit-identical with and without the tracer enabled). It's carried
through the HLLC Riemann solver exactly like the transverse momentum
component already was (Toro sec. 10.4's "passive variable" extension —
only jumps at the contact, carried unchanged through the star state) and
reconstructed by the same MinMod limiter as everything else, so its
interfaces stay as sharp as the scheme allows without any special-casing.

`decks/cylinder_re100.toml` feeds it in as **alternating 1D-wide stripes**
at the inflow (`[boundary.left.profile] type="tracer_stripes"`, see "The
generic scene schema" above), fed *continuously* rather than as a one-time
initial-condition pulse — a pulse would wash downstream and out of the
domain long before a 130-time-unit run is over, exactly the same problem
the shedding-symmetry perturbation faced (see above). Continuous injection
needed a new capability, `Euler2D::SetInflowProfile(function<Prim2D(coord)>)`:
a position-dependent inflow state, evaluated per-row/per-column instead of
once for the whole boundary.

`tools/make_movie.py` renders the tracer as a second panel under vorticity
whenever a run wrote one. The result is directly legible: the stripes stay
perfectly straight and undisturbed upstream of the cylinder, then get
visibly rolled up and interleaved by each shed vortex downstream — the
swirl pattern in the tracer panel lines up exactly with each vorticity
lobe above it, showing *what the vortices are actually doing to the fluid*
(mixing/entrainment), which vorticity alone (a measure of local rotation)
doesn't directly show.

## Performance

`Euler2D`'s row/column sweeps are OpenMP-parallelized (each line is fully
independent of every other line within one sweep) with per-thread reused
scratch buffers (no heap allocation once warmed up, no allocator
contention between threads). Measured on the cylinder-shedding deck (16
logical cores): **~2.4× wall-clock speedup** (~3m17s → ~1m22s) over the
single-threaded, allocation-heavy original, validated to reproduce every
existing result bit-for-bit (no data races). See
[`docs/SIMULATION.md`](docs/SIMULATION.md#8-performance-complexity-the-bottleneck-that-measurement-actually-found-and-what-fixed-it)
for the full complexity analysis — including a real surprise (heap
allocation, despite ~169 million calls in one run, turned out to cost only
~6-7%, not the dominant fraction a naive estimate suggested) and a
known-but-currently-inactive scaling risk in the explicit viscous-diffusion
sub-stepping (`O(N²)` at fine-enough resolution/low-enough viscosity).

`Euler2D`'s core inviscid method (HLLC/MinMod/RK2, Strang-split X/Y sweeps)
also has a GPU port, `kernels_euler2d.hpp`, scoped exactly like `Euler1D`'s:
core method only, no viscosity/obstacle/tracer, validated only via
`--selftest`'s GPU-vs-CPU cross-check on the isentropic-vortex scenario, not
wired into any deck-driven run. Matches the CPU reference to ~1e-5 relative
— inside the ~1e-3 float32-vs-double tolerance floor with headroom to
spare, despite Strang splitting running 3x more flux evaluations per step
than the 1D case. See
[`docs/SIMULATION.md`](docs/SIMULATION.md#84-a-gpu-port-of-euler2d-scoped-exactly-like-section-3s)
for the dispatch design (one compute pass per fractional step over the
*whole* grid, rather than one per row/column).

## File map

| File | Role |
|---|---|
| `src/Euler1D.{hpp,cpp}` | 1D physics: conserved/primitive conversion, HLLC flux, MinMod reconstruction, RK2 step. Pure C++, no GL dependency — the reference `kernels_euler1d.hpp` is cross-checked against. |
| `src/kernels_euler1d.hpp` | GLSL compute-shader port of `Euler1D` (same formulas, transliterated) — string-builder style matching `07_grhd`'s kernels files. |
| `src/Euler2D.{hpp,cpp}` | 2D physics: Strang-split X/Y sweeps reusing `Euler1D`'s line update, the transverse-momentum HLLC extension, the viscous diffusion sub-step, per-side `WallBC` boundary conditions (`Outflow`/`Periodic`/`NoSlipReflective`/`FreeSlipReflective`/`Inflow`) with an optional position-dependent inflow profile, the obstacle cell-masking pass, and the passive scalar tracer field. CPU only (OpenMP-parallelized, see "Performance"). |
| `src/kernels_euler2d.hpp` | GLSL compute-shader port of `Euler2D`'s core inviscid method (same formulas, transliterated; axis-parameterized kernels covering both X and Y sweeps) — exercised only by `--selftest`, see "Performance". |
| `src/CompressibleSim.{hpp,cpp}` | `fw::Simulation` wrapper for the 1D shock tube. CPU (`Euler1D`) path only — the GPU kernels are exercised by `--selftest`, not yet a second deck-driven `Simulation`. |
| `src/CompressibleSimScene.{hpp,cpp}` | The single, generic `fw::Simulation` for every 2D scenario (see "The generic scene schema" above) — grid/physics/boundary/obstacles/initial-condition/probes/tracer, all parsed from deck tables. Also the only sim in this project with a live view: overrides `Render`/`OnViewInput`/`OnKey` to draw a `magma`-colormapped heatmap of density/speed/vorticity/tracer (see "Interactive mode" above). |
| `src/main.cpp` | CLI entry point (`--scene` selects `CompressibleSimScene`; bare `--deck` is the 1D `CompressibleSim`; `--interactive` opens either live via `fw::SimApp` instead of running headless) + `--selftest` (HLLC consistency, conservation identities, 1D GPU-vs-CPU cross-check, vortex advection, 2D GPU-vs-CPU cross-check). |
| `tools/exact_riemann_newtonian.py` | Toro's exact 1D Riemann solver (same one `07_grhd` uses to validate its Newtonian limit). |
| `tools/plot_shocktube.py` | Final-frame ρ/u/P vs. exact solution, plus the conservation-identity plot (1D). |
| `tools/plot_riemann2d.py` | Final-frame density/pressure heatmaps (2D). |
| `tools/plot_poiseuille.py` | Final velocity profile vs. the analytic parabolic solution, plus the spin-up-to-steady-state curve. |
| `tools/plot_taylor_green.py` | Measured kinetic-energy decay rate vs. the analytic `4*nu*k²`; exposes `measure_decay_rate()` for `Studies/compressible_fluid`. |
| `tools/plot_strouhal.py` | FFTs the downstream velocity probe, measures the shedding Strouhal number vs. Roshko's correlation. |
| `tools/make_movie.py` | Vorticity- or density-field (`--field`) movie for any 2D run (obstacle drawn as a disk or rotated NACA00xx outline when present; adds a tracer panel when a run wrote one). |
