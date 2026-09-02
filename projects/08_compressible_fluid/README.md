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

**Status: Phase 4 (this phase)** — 1D (CPU + GPU), 2D inviscid (CPU, via
Strang dimensional splitting), and 2D **viscous** (Newtonian shear stress +
Fourier conduction). Validated against the exact Sod shock tube, a
GPU-vs-CPU cross-check, an exact 2D isentropic-vortex advection solution, a
qualitative match to the published Kurganov & Tadmor 2D Riemann
"Configuration 3" density pattern, the exact plane-Poiseuille velocity
profile (0.08% error), and the exact Taylor-Green vortex decay rate (1.6%
error). Used in [`Studies/compressible_fluid`](../../../Studies/compressible_fluid/)
for a viscosity-vs-vorticity-decay sweep.

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

`decks/riemann2d_config3.toml` — the 2D four-quadrant Riemann problem
(`--2d` selects the 2D sim):

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --2d \
    --deck projects/08_compressible_fluid/decks/riemann2d_config3.toml \
    --out projects/08_compressible_fluid/out/riemann2d_config3
python projects/08_compressible_fluid/tools/plot_riemann2d.py projects/08_compressible_fluid/out/riemann2d_config3
```

`decks/poiseuille_channel.toml` — plane Poiseuille flow (`--channel`
selects the viscous channel sim); see "Viscosity" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --channel \
    --deck projects/08_compressible_fluid/decks/poiseuille_channel.toml \
    --out projects/08_compressible_fluid/out/poiseuille_channel
python projects/08_compressible_fluid/tools/plot_poiseuille.py projects/08_compressible_fluid/out/poiseuille_channel
```

`decks/taylor_green.toml` — 2D Taylor-Green vortex decay (`--taylor-green`
selects the doubly-periodic sim); see "Viscosity" below:

```sh
./build/release/projects/08_compressible_fluid/08_compressible_fluid.exe --taylor-green \
    --deck projects/08_compressible_fluid/decks/taylor_green.toml \
    --out projects/08_compressible_fluid/out/taylor_green
python projects/08_compressible_fluid/tools/plot_taylor_green.py projects/08_compressible_fluid/out/taylor_green
```

`--selftest` (no deck needed) runs four fast, deck-independent checks:
HLLC-flux self-consistency (`F_HLLC(s,s) == F(s)` exactly), a
conservation-identity check (mass and energy exactly conserved, momentum
matching the analytic boundary-pressure-forcing prediction, both before
either wave reaches the domain edge — see `main.cpp`'s comment for why
these are exact identities, not approximate physics), a GPU-vs-CPU
cross-check (the same Sod IC run 50 RK2 steps on both `Euler1D` and the
`kernels_euler1d.hpp` compute-shader port, comparing recovered primitives —
matches to ~1e-7 relative, the expected float32 precision floor), and the
2D isentropic-vortex advection check described below.

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

## File map

| File | Role |
|---|---|
| `src/Euler1D.{hpp,cpp}` | 1D physics: conserved/primitive conversion, HLLC flux, MinMod reconstruction, RK2 step. Pure C++, no GL dependency — the reference `kernels_euler1d.hpp` is cross-checked against. |
| `src/kernels_euler1d.hpp` | GLSL compute-shader port of `Euler1D` (same formulas, transliterated) — string-builder style matching `07_grhd`'s kernels files. |
| `src/Euler2D.{hpp,cpp}` | 2D physics: Strang-split X/Y sweeps reusing `Euler1D`'s line update, the transverse-momentum HLLC extension, and (Phase 4) the viscous diffusion sub-step + `WallBC` boundary conditions. CPU only. |
| `src/CompressibleSim.{hpp,cpp}` | `fw::Simulation` wrapper for the 1D shock tube. CPU (`Euler1D`) path only — the GPU kernels are exercised by `--selftest`, not yet a second deck-driven `Simulation`. |
| `src/CompressibleSim2D.{hpp,cpp}` | `fw::Simulation` wrapper for the 2D four-quadrant Riemann problem. |
| `src/CompressibleSimChannel.{hpp,cpp}` | `fw::Simulation` wrapper for the viscous Poiseuille channel flow. |
| `src/CompressibleSimTaylorGreen.{hpp,cpp}` | `fw::Simulation` wrapper for the doubly-periodic Taylor-Green vortex decay. |
| `src/main.cpp` | CLI entry point (`--2d`/`--channel`/`--taylor-green` select `CompressibleSim2D`/`CompressibleSimChannel`/`CompressibleSimTaylorGreen`) + `--selftest` (HLLC consistency, conservation identities, GPU-vs-CPU cross-check, vortex advection). |
| `tools/exact_riemann_newtonian.py` | Toro's exact 1D Riemann solver (same one `07_grhd` uses to validate its Newtonian limit). |
| `tools/plot_shocktube.py` | Final-frame ρ/u/P vs. exact solution, plus the conservation-identity plot (1D). |
| `tools/plot_riemann2d.py` | Final-frame density/pressure heatmaps (2D). |
| `tools/plot_poiseuille.py` | Final velocity profile vs. the analytic parabolic solution, plus the spin-up-to-steady-state curve. |
| `tools/plot_taylor_green.py` | Measured kinetic-energy decay rate vs. the analytic `4*nu*k²`. |
