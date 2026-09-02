# 08 — Compressible fluid: 1D Euler, HLLC/MinMod/RK2 finite volume

A conservative finite-volume solver for the compressible Euler equations —
the flat-spacetime, non-relativistic limit of
[`07_grhd`](../07_grhd/)'s Valencia formulation (see that project's
`docs/SIMULATION.md`): conserved variables evolved by flux differencing,
MinMod-limited piecewise-linear reconstruction, an HLLC approximate Riemann
solver, RK2 (Heun) method-of-lines time integration. Unlike the incompressible
solvers in [`Physics Simulations/Fluid Mechanics`](../../../Physics%20Simulations/Fluid%20Mechanics/)
(Stable Fluids, a MAC-grid projection method, LBM), this is a genuine
shock-capturing conservation-law scheme — it can resolve a real discontinuity
(a shock) without producing spurious oscillations or losing mass/momentum/
energy across it.

**Status: Phase 3 (this phase)** — 1D (CPU + GPU) and 2D (CPU, via Strang
dimensional splitting), validated against the exact Sod shock tube, a
GPU-vs-CPU cross-check, an exact 2D isentropic-vortex advection solution,
and a qualitative match to the published Kurganov & Tadmor 2D Riemann
"Configuration 3" density pattern. See the top-level plan for the phases
after this one (viscous terms + Poiseuille flow, then a design-doc writeup).

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

**What's absent, on purpose, at this phase:** geometric source terms (flat
metric — they're identically zero, so the whole term simply doesn't
appear), viscosity (Phase 4), and a genuinely adaptive per-step CFL
controller — `dt` is fixed once at `Configure()` from the initial
condition's max wave speed (`Euler1D::MaxWaveSpeed()`), safe for a single
self-similar Riemann problem since a shock/rarefaction/contact's speeds are
always bounded by the data (see `CompressibleSim.cpp`'s comment) but not
something a later time-varying flow can rely on.

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

## File map

| File | Role |
|---|---|
| `src/Euler1D.{hpp,cpp}` | 1D physics: conserved/primitive conversion, HLLC flux, MinMod reconstruction, RK2 step. Pure C++, no GL dependency — the reference `kernels_euler1d.hpp` is cross-checked against. |
| `src/kernels_euler1d.hpp` | GLSL compute-shader port of `Euler1D` (same formulas, transliterated) — string-builder style matching `07_grhd`'s kernels files. |
| `src/Euler2D.{hpp,cpp}` | 2D physics: Strang-split X/Y sweeps reusing `Euler1D`'s line update, plus the transverse-momentum HLLC extension. CPU only. |
| `src/CompressibleSim.{hpp,cpp}` | `fw::Simulation` wrapper for the 1D shock tube. CPU (`Euler1D`) path only — the GPU kernels are exercised by `--selftest`, not yet a second deck-driven `Simulation`. |
| `src/CompressibleSim2D.{hpp,cpp}` | `fw::Simulation` wrapper for the 2D four-quadrant Riemann problem. |
| `src/main.cpp` | CLI entry point (`--2d` selects `CompressibleSim2D`) + `--selftest` (HLLC consistency, conservation identities, GPU-vs-CPU cross-check, vortex advection). |
| `tools/exact_riemann_newtonian.py` | Toro's exact 1D Riemann solver (same one `07_grhd` uses to validate its Newtonian limit). |
| `tools/plot_shocktube.py` | Final-frame ρ/u/P vs. exact solution, plus the conservation-identity plot (1D). |
| `tools/plot_riemann2d.py` | Final-frame density/pressure heatmaps (2D). |
