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

    h_i       = eta*(m_i/rho_i)^(1/3)                                (adaptive smoothing length, fixed-point solve)
    rho_i     = sum_{j!=i}  m_j W(|r_i - r_j|, h_i)                  (density, cubic spline kernel, "gather" form)
    P_i       = K rho_i^Gamma                                        (polytropic EOS)
    a_i       = -sum_j G m_j (r_i-r_j) / (|r_i-r_j|^2 + eps^2)^1.5   (softened self-gravity)
              -  sum_j m_j (P_i/rho_i^2 + P_j/rho_j^2 + Pi_ij) grad_i W_ij(r_ij, h_ij)   (SPH pressure + Monaghan artificial viscosity, h_ij=0.5(h_i+h_j))

`K` and `Gamma = 1+1/n` come directly from the Lane-Emden solve for
polytropic index `n`, so a correctly-sampled star starts close to
hydrostatic equilibrium (`dP/dr = -G m(r) rho / r^2`) by construction.

**Adaptive smoothing length** (`eta`, `sph.h_iters` in the deck): a single
fixed `h` under-resolves the star's sparse outer envelope relative to its
dense core (the free-surface excess in an earlier validation pass). `h_i` is
solved per particle via `h_i = eta*(m_i/rho_i)^(1/3)` (Springel & Hernquist
2002's ansatz, `eta~1.2` -> ~40-60 effective neighbors in 3D), iterated a few
times per step, warm-started from the previous step's `h_i`. Two real bugs
surfaced building this, both worth remembering for any future adaptive-h
SPH work:
- **Self-term bias.** `W(0,h)` is nonzero (unlike the force kernel's
  gradient, which vanishes at `r=0`), so a naive self-inclusive density sum
  adds a spurious `m_i/(pi h_i^3)` term -- negligible at a large fixed `h`
  (~1.5% of the total at the old `h=0.157`), but 15-30% of the total once
  `h_i` adapts down toward the true local spacing (measured: bin-averaged
  bulk density running 25-30% high with the self-term included). Fixed by
  excluding `j==i` from the density sum.
- **Asymmetric h_i, h_j.** The force kernel uses the *symmetrized* pair
  length `h_ij = 0.5*(h_i+h_j)` for both particles in a pair (not each
  particle's own `h_i`), so the same `W(r,h_ij)` is used in both directions
  and momentum conservation stays exact to machine precision despite
  `h_i != h_j` in general.

**Relaxation needs damping, and damping alone doesn't prove relaxation
worked.** A Monte-Carlo-sampled star isn't in exact numerical equilibrium,
and switching the density estimator (adding adaptive `h`) shifts that
equilibrium again -- either alone excites the star's radial breathing mode,
which artificial viscosity barely damps (it targets convergent/shock flows,
not a smooth global oscillation), so the mode rings indefinitely without
growing *or* decaying. `time.damping` (deck key, 1/time units) adds an
exponential velocity decay in the `Kick` pass for relaxation-only runs. But
a damped run only shows the *ringing* was suppressed -- the actual test is
re-running the settled final state **without** damping
(`tools/extract_ic_from_frame.py` + `decks/star_verify_n1.5.toml`) and
confirming it stays put on its own.

**Known simplifications of this phase**, to be revisited before the actual
disruption run needs them: a **barotropic** EOS (no separately evolved
internal energy/entropy, so artificial-viscosity heating isn't tracked as
heat); and a real, still-open calibration gap documented in Validation
below (the settled equilibrium's bulk density runs ~10-30% below the exact
Lane-Emden target at N=4000 -- a finite-resolution/EOS-calibration mismatch
between the discrete SPH equilibrium and the continuum profile it was
initialized from, not something reduced softening fully closed).

## GPU compute shaders (`src/kernels.hpp`)

One thread per particle, brute-force O(N^2) (fine at the few-thousand-particle
scale used so far -- see `docs/BOUNDARIES.md`-style honesty: revisit with a
tree code only if profiling actually says so):

- `Density()` -- the adaptive-`h` fixed-point loop, then kernel-weighted mass sum -> rho, then P and sound speed (all local once rho is known, so folded into the same pass).
- `Forces()` -- fused self-gravity + SPH pressure/viscosity acceleration (one pairwise loop does both), using the symmetrized `h_ij`.
- `Kick()` -- leapfrog `v += halfDt*a`, then optional relaxation damping. `Drift()` -- `x += dt*v`.

`--selftest` cross-checks `Density`/`Forces` (including the adaptive-`h`
solve) against an independent double-precision CPU implementation of the
same formulas (37 random particles): max relative error `h=1.5e-7`,
`rho=2.4e-7`, `accel=4.7e-7` -- PASS.

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

Then, from `tools/` (needs `numpy`; `matplotlib` for `plot_star.py` --
**use a plain Python/matplotlib install, not the `physics-sims` conda env**,
whose `savefig` is broken on this machine, see the memory note on this repo's
host):

```sh
# regenerate initial conditions for a given (n, N, mass, radius) -- prints the
# K/Gamma/suggested-h_init/softening to copy into the deck, and a
# radial-density sampling check
python tools/make_star_ic.py --n 4000 --index 1.5 --out ic/star_n1.5.bin

# validation panels: energy/momentum conservation + radial density profile
# vs. the analytic Lane-Emden curve
python tools/plot_star.py out/star_relax_n1.5 --index 1.5

# settle-then-verify: extract a damped run's settled final frame as a new IC,
# then re-run it undamped (decks/star_verify_n1.5.toml) as the real check
python tools/extract_ic_from_frame.py out/star_relax_n1.5 --frame 0199 --out ic/star_n1.5_relaxed.bin

# movie: (x,y) scatter colored by density (log scale, fixed across the run)
# -- mp4 via imageio's bundled ffmpeg (not the system PATH), PNG sequence
# fallback if that import fails
python tools/make_movie.py out/star_relax_n1.5
```

## Deck format

```toml
title = "..."
[star]      ic_file = "ic/star_n1.5.bin"     # raw binary, N x (x,y,z,mass,vx,vy,vz) float32
[gravity]   G = 1.0   softening = 0.02
[sph]       h_init = 0.157   eta = 1.2   h_iters = 3   K = 0.424217   gamma = 1.666667   visc_alpha = 1.0   visc_beta = 2.0
[time]      dt = 0.005   substeps_per_frame = 10   frames = 200
            damping = 0.5   # relaxation runs only -- omit (or 0) once past settling
```

`K`/`gamma`/the suggested `h_init`/`softening` must come from the same
`make_star_ic.py` invocation that produced `ic_file` (they encode the
specific polytrope the particles were sampled from) -- regenerate both
together if `n`/`N`/mass/radius change. `softening` deliberately sits well
below `h_init`: it needs to stay smaller than the adaptive `h` the run
converges to in the core, or self-gravity gets measurably weakened at the
resolved scale (see Physics above). Output fields (all `(N,4)` float32,
extra column unused/padding except `h` which is `(N,1)`): `pos_mass`
(x,y,z,mass), `vel` (vx,vy,vz,-), `rho_press` (rho,pressure,soundspeed,-),
`h`. Diagnostics (fixed list, not yet deck-selectable like `05_tdse_gpu`'s):
`kinetic`, `thermal`, `potential`, `energy` (=sum of those three),
`virial_2T_over_W`, `com_x/y/z`, `com_speed`.

## Validation (n=1.5 polytrope, N=4000, G=M=R=1, eta=1.2, softening=0.02)

`decks/star_relax_n1.5.toml` (damping=0.5, 10 dynamical times, from the raw
Monte-Carlo IC) settles the radial breathing mode excited by switching to
adaptive `h` (energy drift oscillates but its amplitude visibly decays,
ending around -10% of `E0`, vs. never damping out at all with damping=0).
The real check is `decks/star_verify_n1.5.toml`: the settled frame re-run
for a further 10 dynamical times with **no** damping.

| check | result (`star_verify_n1.5`) |
|---|---|
| momentum conservation | `com_speed` max `1.2e-8` (all pairwise sums are exactly antisymmetric; this is finite-precision floor, not physical drift) |
| energy conservation | `0.20%` drift over 10 dynamical times (vs. `3.2%` before adaptive `h` -- a real improvement, and no longer oscillating: the settled state is a genuine equilibrium of the *undamped* equations) |
| radial density profile, t=0 vs. t=end (10 t_dyn later) | **time-independent to a few percent everywhere** -- the settled configuration does not evolve once damping is removed |
| radial density profile vs. analytic Lane-Emden | **still open**: bulk (`r<0.7R`) runs a systematic `10-30%` *below* the analytic target (e.g. ratio `0.71` at `r=0.125R`, `0.91` at `r=0.708R`); outer envelope (`r>0.79R`) still over-dense (ratio `~1.7` at `r=0.875R`), the same free-surface excess as the fixed-`h` version, not fixed by adaptive `h` alone |

Net: adaptive `h` (with the self-term fix) converts the star from
"oscillates forever around approximately the right structure" into "settles
to a genuine, time-independent equilibrium" -- a real improvement in
dynamical correctness -- but that equilibrium's absolute density
normalization doesn't exactly match the continuum Lane-Emden target it was
initialized from. Tried and ruled out as the (sole) cause: gravitational
softening comparable to the adaptive `h` reached in the core (reducing
`softening` from `0.063` to `0.02` gave a small improvement, not a fix).
Likely a genuine finite-`N` discrete-equilibrium-vs-continuum mismatch;
candidate fixes for a future session: iteratively recalibrate `K` against
the settled state rather than the raw analytic profile, or increase `N`.
Does not block adding the black hole -- the star is a stable, non-oscillating
object either way -- but is worth closing before precision fallback-rate
comparisons (Phase after next) depend on knowing the pre-disruption profile
exactly.

## Progress

- [x] `tools/lane_emden.py` -- polytropic stellar structure, validated against tabulated Lane-Emden benchmarks (n=0,1,1.5,3, ~1e-7) and the hydrostatic-equilibrium ODE itself (~2e-4)
- [x] `tools/make_star_ic.py` -- Monte-Carlo IC sampling, validated radial density recovery against the analytic profile
- [x] `src/kernels.hpp` self-gravity + adaptive-h SPH compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] `src/TdeSim` leapfrog integration + damped relaxation + undamped verification, validated per the table above
- [ ] close the ~10-30% bulk density calibration gap (see Validation) -- lower priority than the items below, doesn't block them
- [ ] external black hole potential (point-mass, then Paczynski-Wiita) + the actual encounter
- [ ] fallback-rate diagnostics (dM/dt vs. t) compared to the classic t^(-5/3) law
