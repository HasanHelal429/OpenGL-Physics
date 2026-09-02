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

**Status: Phase 1 (this phase)** — 1D only, CPU only, validated against the
classic Sod (1978) shock tube's exact solution. See the top-level plan for
the phases after this one (GPU port + cross-check, 2D via dimensional
splitting, viscous terms + Poiseuille flow, then a design-doc writeup).

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

`--selftest` (no deck needed) runs two fast, deck-independent checks:
HLLC-flux self-consistency (`F_HLLC(s,s) == F(s)` exactly), and a
conservation-identity check (mass and energy exactly conserved, momentum
matching the analytic boundary-pressure-forcing prediction, both before
either wave reaches the domain edge — see `main.cpp`'s comment for why
these are exact identities, not approximate physics).

## File map

| File | Role |
|---|---|
| `src/Euler1D.{hpp,cpp}` | The physics: conserved/primitive conversion, HLLC flux, MinMod reconstruction, RK2 step. Pure C++, no GL dependency — this is the Phase 2 GPU-port target. |
| `src/CompressibleSim.{hpp,cpp}` | `fw::Simulation` wrapper: deck parsing, `Step`/`Snapshot`/`Info`. |
| `src/main.cpp` | CLI entry point + `--selftest`. |
| `tools/exact_riemann_newtonian.py` | Toro's exact Riemann solver (same one `07_grhd` uses to validate its Newtonian limit). |
| `tools/plot_shocktube.py` | Final-frame ρ/u/P vs. exact solution, plus the conservation-identity plot. |
