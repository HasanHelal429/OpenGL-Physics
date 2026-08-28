# 06 — Tidal disruption event (Tier 1: Newtonian self-gravitating SPH star)

The long-term goal of this project is a simulation of a star being tidally
shredded by a black hole. That is too much to build in one step, so it is
being built up tier by tier, in increasing order of physical realism:

| tier | physics |
|---|---|
| **1 (this phase)** | Newtonian self-gravitating SPH star + external point-mass / Paczynski-Wiita black hole |
| 2 | fluid on a fixed analytic Kerr/Schwarzschild background (relativistic effects, no dynamical spacetime) |
| 3 | + magnetic fields (GRMHD, real disk formation/accretion) |
| 4 | + radiative transfer (observable light curves) |
| 5 | full dynamical numerical relativity -- likely unnecessary here, since M_star << M_BH |

**Current state: the star, no black hole yet.** Smoothed-particle
hydrodynamics (SPH): equal-mass particles feel mutual softened self-gravity
plus a polytropic pressure/artificial-viscosity force, integrated by
kick-drift-kick leapfrog, entirely on the GPU. Initial conditions are a
Lane-Emden polytrope (`tools/lane_emden.py`) Monte-Carlo-sampled into
particles (`tools/make_star_ic.py`). The milestone for this phase is a star
that sits in approximate hydrostatic equilibrium instead of collapsing or
flying apart -- validated below -- which has to hold before a black hole's
tidal field is ever added.

## Physics

Governing equations, per SPH particle *i*:

    rho_i     = sum_j  m_j W(|r_i - r_j|, h)                       (density, cubic spline kernel)
    P_i       = K rho_i^Gamma                                       (polytropic EOS)
    a_i       = -sum_j G m_j (r_i-r_j) / (|r_i-r_j|^2 + eps^2)^1.5  (softened self-gravity)
              -  sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) grad_i W_ij   (SPH pressure + Monaghan artificial viscosity)

`K` and `Gamma = 1+1/n` come directly from the Lane-Emden solve for
polytropic index `n`, so a correctly-sampled star starts close to
hydrostatic equilibrium (`dP/dr = -G m(r) rho / r^2`) by construction.

**Known simplifications of this phase**, to be revisited before the actual
disruption run needs them: a single **fixed** smoothing length `h` (no
adaptive `h` yet -- fine for a star sitting still, but the tidal stream's
huge density range will need it); a **barotropic** EOS (no separately
evolved internal energy/entropy, so artificial-viscosity heating isn't
tracked as heat -- fine for the quiescent case validated here, a source of
slow energy drift once real shocks show up).

## GPU compute shaders (`src/kernels.hpp`)

One thread per particle, brute-force O(N^2) (fine at the few-thousand-particle
scale used so far -- see `docs/BOUNDARIES.md`-style honesty: revisit with a
tree code only if profiling actually says so):

- `Density()` -- kernel-weighted mass sum -> rho, then P and sound speed (both are local once rho is known, so folded into the same pass).
- `Forces()` -- fused self-gravity + SPH pressure/viscosity acceleration (one pairwise loop does both).
- `Kick()` / `Drift()` -- leapfrog `v += halfDt*a` / `x += dt*v`.

`--selftest` cross-checks `Density`/`Forces` against an independent
double-precision CPU implementation of the same formulas (37 random
particles): max relative error `rho=2.2e-7`, `accel=3.2e-6` -- PASS.

## Build & run

```sh
cmake --preset release
cmake --build --preset release --target 06_tidal_disruption

# GPU-vs-CPU compute shader cross-check
06_tidal_disruption --selftest

# batch: run a deck, write raw data
06_tidal_disruption --deck decks/star_relax_n1.5.toml --out out/star_relax_n1.5

# interactive: window + HUD + orbit camera, particles colored by density
06_tidal_disruption --interactive --deck decks/star_relax_n1.5.toml
```

Then, from `tools/` (needs `numpy` + `matplotlib`):

```sh
# regenerate initial conditions for a given (n, N, mass, radius) -- prints the
# K/Gamma/suggested-h to copy into the deck, and a radial-density sampling
# check
python tools/make_star_ic.py --n 4000 --index 1.5 --out ic/star_n1.5.bin

# validation panels: energy/momentum conservation + radial density profile
# vs. the analytic Lane-Emden curve
python tools/plot_star.py out/star_relax_n1.5 --index 1.5
```

## Deck format

```toml
title = "..."
[star]      ic_file = "ic/star_n1.5.bin"     # raw binary, N x (x,y,z,mass,vx,vy,vz) float32
[gravity]   G = 1.0   softening = 0.063
[sph]       h = 0.157   K = 0.424217   gamma = 1.666667   visc_alpha = 1.0   visc_beta = 2.0
[time]      dt = 0.005   substeps_per_frame = 10   frames = 200
```

`K`/`gamma`/the suggested `h`/`softening` must come from the same
`make_star_ic.py` invocation that produced `ic_file` (they encode the
specific polytrope the particles were sampled from) -- regenerate both
together if `n`/`N`/mass/radius change. Output fields (all `(N,4)` float32,
extra column unused/padding): `pos_mass` (x,y,z,mass), `vel` (vx,vy,vz,-),
`rho_press` (rho,pressure,soundspeed,-). Diagnostics (fixed list, not yet
deck-selectable like `05_tdse_gpu`'s): `kinetic`, `thermal`, `potential`,
`energy` (=sum of those three), `virial_2T_over_W`, `com_x/y/z`, `com_speed`.

## Validation (`decks/star_relax_n1.5.toml`: n=1.5 polytrope, N=4000, G=M=R=1, run 10 dynamical times)

| check | result |
|---|---|
| momentum conservation | `com_speed` max `1.7e-9` (machine precision -- the pairwise force sum is exactly antisymmetric, as it must be) |
| energy conservation | `3.2%` drift over 10 dynamical times |
| hydrostatic virial residual `3(Gamma-1) U_thermal + W` (should be ~0 at equilibrium for a static, pressure-supported polytrope -- **not** `2T/|W|`, which is the wrong check here since bulk kinetic energy stays near zero throughout) | `+0.021` vs. `\|W\|=0.843` (~2.5%) at t=0, similar throughout the run |
| radial density profile vs. analytic Lane-Emden | bulk (`r<0.6R`) stays within ~10% of analytic across the full run, no collapse or runaway expansion; outer envelope (`r>0.7R`) systematically over-dense (ratio grows to ~1.6x at `r=0.875R`) -- the expected fixed-`h` SPH free-surface artifact (undersamples the steep low-density edge), not a bug; matches the "known simplifications" above |

Net: the star holds its structure -- it does not collapse or fly apart -- for
10 dynamical times, with the surface artifact and slow energy drift being the
two concrete things adaptive smoothing length (and, later, tracked internal
energy) should fix before the actual disruption run needs the resolution.

## Progress

- [x] `tools/lane_emden.py` -- polytropic stellar structure, validated against tabulated Lane-Emden benchmarks (n=0,1,1.5,3, ~1e-7) and the hydrostatic-equilibrium ODE itself (~2e-4)
- [x] `tools/make_star_ic.py` -- Monte-Carlo IC sampling, validated radial density recovery against the analytic profile
- [x] `src/kernels.hpp` self-gravity + SPH compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] `src/TdeSim` leapfrog integration + `decks/star_relax_n1.5.toml` relaxation run, validated per the table above
- [ ] adaptive smoothing length (fix the free-surface density excess)
- [ ] external black hole potential (point-mass, then Paczynski-Wiita) + the actual encounter
- [ ] fallback-rate diagnostics (dM/dt vs. t) compared to the classic t^(-5/3) law
