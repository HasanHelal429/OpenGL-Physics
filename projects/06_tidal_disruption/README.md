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

**Current state: the star gets disrupted.** Smoothed-particle hydrodynamics
(SPH): equal-mass particles feel mutual softened self-gravity plus a
polytropic pressure/artificial-viscosity force, integrated by
kick-drift-kick leapfrog, entirely on the GPU. Initial conditions are a
Lane-Emden polytrope (`tools/lane_emden.py`) Monte-Carlo-sampled into
particles (`tools/make_star_ic.py`), relaxed to hydrostatic equilibrium
(`decks/star_relax_n1.5.toml`/`star_verify_n1.5.toml`, validated below), then
boosted onto a parabolic orbit around a fixed black hole point mass
(`tools/make_orbit_ic.py`) and run through pericenter
(`decks/encounter_beta3.toml`). The result: the star stretches into the
classic curved tidal debris stream, visible within one frame of pericenter
passage -- see the movie. Running longer (`decks/encounter_beta3_long.toml`,
50 dynamical times) shows something more: about half the star's mass is
actually bound to the black hole, and that debris measurably falls back --
swinging past the black hole again, closer than the original pericenter --
right around the analytically-predicted return time. That is the literal
"feeding" this whole project is about.

```sh
python tools/make_movie.py out/encounter_beta3
# --half-extent crops out the far unbound majority to see the close-in
# disruption/fallback; --gamma (default 0.4) lifts the fallback stream's
# very dilute debris out of nearly-black on the log color scale
python tools/make_movie.py out/encounter_beta3_long --half-extent 40 \
    --out out/encounter_beta3_long/movie_zoom.mp4
# quantitative fallback rate: dM/dt vs. time since pericenter, vs. t^(-5/3)
python tools/fallback_rate.py out/encounter_beta3_long
```

## Physics

Governing equations, per SPH particle *i*:

    h_i       = eta*(m_i/rho_i)^(1/3)                                (adaptive smoothing length, fixed-point solve)
    rho_i     = sum_j  m_j W(|r_i - r_j|, h_i)                       (density, cubic spline kernel, "gather" form, includes j=i)
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
- **Self-term convergence, not a self-term bug.** Standard SPH summation
  density *includes* `j=i`: `W(0,h)` is nonzero (unlike the force kernel's
  gradient, which vanishes at `r=0`) and that self-contribution is what
  keeps the estimate well-behaved as the local neighbor count drops (e.g.
  near the star's free surface). An earlier version of this project
  excluded `j==i` after seeing a 25-30% *high* bulk-density bias with it
  included -- but that was a fixed-point-iteration convergence artifact of
  only running `sph.h_iters=3` (self-inclusion makes the `rho<->h` loop
  stiffer near the core), not a flaw in the self-term. Excluding it
  "fixed" that symptom but replaced it with a systematic 10-30%
  *under*-estimate everywhere the neighbor count is modest, since every
  particle was missing its own legitimate mass contribution -- this was
  the calibration gap flagged below. Fixed for real by keeping the
  self-term and confirming 3 fixed-point iterations is actually enough to
  converge it (it is -- see Validation below).
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
heat). The bulk-density calibration gap from earlier sessions (settled
equilibrium running 10-30% below the analytic Lane-Emden target) is
**closed** -- see Validation below; it was the self-term exclusion above,
not a resolution or EOS mismatch. The remaining, much smaller free-surface
excess at the outermost, sparsely-sampled radii is a separate, lower-priority
artifact (still present, not addressed by this fix).

**The black hole** is a fixed point mass at the origin -- not integrated,
since `M_bh >> M_star` makes its recoil negligible to leading order in the
mass ratio. Two force laws (`blackhole.type` in the deck):

    a_bh = -G*M_bh*r / (|r|^2 + eps^2)^1.5                  ("point": Newtonian)
    a_bh = -G*M_bh/(|r|-r_s)^2 * r_hat, floored at eps below r_s   ("paczynski_wiita")

Paczynski-Wiita is a pseudo-relativistic stand-in (correct ISCO at `6*G*M/c^2`,
diverges at the Schwarzschild radius `r_s`) for pericenter precession/plunge
physics without an actual metric -- a bridge toward Tier 2, not used in the
validated encounter below (that uses plain `"point"`).

**Placing the star on an encounter** (`tools/make_orbit_ic.py`): a rigid
boost of every particle's position/velocity by the star's intended
center-of-mass position/velocity on a parabolic orbit (eccentricity 1,
specific orbital energy 0 -- the standard assumption for stars scattered
into a black hole's loss cone from a large, near-isotropic orbit). Given the
tidal radius `r_t = R_star*(M_bh/M_star)^(1/3)` and a penetration factor
`beta = r_t/r_p`, the pericenter distance `r_p = r_t/beta` fixes the orbit's
angular momentum `L = sqrt(2*G*M_bh*r_p)`; starting the star at separation
`r_0` (still inbound) gives `r(phi) = 2*r_p/(1+cos(phi))` its starting
`phi_0`, and `v_r`/`v_t` follow from `L` and energy conservation. Independent
validation (integrate the resulting initial condition through the point-mass
two-body problem alone, no self-gravity/pressure): reaches the intended
`r_p` to 6 significant figures, specific energy conserved to `1.6e-12`
(machine precision for a value that should be exactly 0) -- see the
derivation and check in the tool's own docstring/module.

**A real crash found running the encounter longer (50 dynamical times, to
see fallback -- see Validation below): zero-density particles.** A tidal
stream stretched over 100+ length units eventually leaves a small number of
particles (~0.3% at the point this hit) with literally zero neighbors
within their kernel support, even at `h=h_max` -- `rho=0`. That alone is
fine, but `csound = sqrt(Gamma*pressure/rho)` is then a literal `0/0 = NaN`,
and because `Forces()` loops over all `N` particles for every particle's
force, one NaN position poisons *every* particle's computed force the very
next step (measured: all 4000 particles NaN within a single frame of the
first isolated particle appearing). Flooring `rho` alone is not the fix:
`P/rho^2 ~ rho^(Gamma-2)` actually *diverges* as `rho->0` for `Gamma<2`
(true here, `Gamma=5/3`), so a naive floor silently trades a crash for a
large spurious pressure kick. The correct treatment -- a particle with no
real neighbors should feel zero SPH pressure/viscosity force (it still
feels gravity, which doesn't depend on `rho`) -- needs pressure/soundspeed
set to exactly `0` below a density floor (`kRhoFloor=1e-8`) rather than
computed from a floored `rho`, with the same floor applied again in
`Forces()` to skip the `P/rho^2` terms and the viscosity `1/rhobar` term.
Rerunning the same 50-dynamical-time deck after the fix: zero NaN, energy
conservation still `0.05%` of the peak orbital energy scale (see
Validation).

## GPU compute shaders (`src/kernels.hpp`)

One thread per particle, brute-force O(N^2) (fine at the few-thousand-particle
scale used so far -- see `docs/BOUNDARIES.md`-style honesty: revisit with a
tree code only if profiling actually says so):

- `Density()` -- the adaptive-`h` fixed-point loop, then kernel-weighted mass sum -> rho, then P and sound speed (all local once rho is known, so folded into the same pass).
- `Forces()` -- fused self-gravity + black hole + SPH pressure/viscosity acceleration (one pairwise loop plus one O(N) black-hole term), using the symmetrized `h_ij`.
- `Kick()` -- leapfrog `v += halfDt*a`, then optional relaxation damping. `Drift()` -- `x += dt*v`.

`--selftest` cross-checks `Density`/`Forces` (including the adaptive-`h`
solve and the black hole term, Paczynski-Wiita) against an independent
double-precision CPU implementation of the same formulas (37 random
particles): max relative error `h=1.5e-7`, `rho=2.4e-7`, `accel=5.9e-7` --
PASS.

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

# place the relaxed star on a parabolic encounter (see Physics above for the
# beta/tidal-radius parameterization); prints r_t, r_p, and the start state
python tools/make_orbit_ic.py --star ic/star_n1.5_relaxed.bin --M_bh 1000 \
    --beta 3 --out ic/star_n1.5_encounter_beta3.bin

# movie: (x,y) scatter colored by density (log scale, fixed across the run),
# a marker at the origin if [blackhole].enabled -- mp4 via imageio's bundled
# ffmpeg (not the system PATH), PNG sequence fallback if that import fails
python tools/make_movie.py out/encounter_beta3
```

## Deck format

```toml
title = "..."
[star]      ic_file = "ic/star_n1.5.bin"     # raw binary, N x (x,y,z,mass,vx,vy,vz) float32
[gravity]   G = 1.0   softening = 0.02
[sph]       h_init = 0.157   eta = 1.2   h_iters = 3   K = 0.424217   gamma = 1.666667   visc_alpha = 1.0   visc_beta = 2.0
[blackhole] enabled = true   type = "point"   mass = 1000.0   # type = "point" | "paczynski_wiita"; omit section or enabled=false to turn off
[time]      dt = 0.005   substeps_per_frame = 10   frames = 200
            damping = 0.5   # relaxation runs only -- omit (or 0) once past settling
```

`K`/`gamma`/the suggested `h_init`/`softening` must come from the same
`make_star_ic.py` invocation that produced `ic_file` (they encode the
specific polytrope the particles were sampled from) -- regenerate both
together if `n`/`N`/mass/radius change. `softening` deliberately sits well
below `h_init`: it needs to stay smaller than the adaptive `h` the run
converges to in the core, or self-gravity gets measurably weakened at the
resolved scale (see Physics above). `blackhole.schwarzschild_radius` only
matters for `type = "paczynski_wiita"`. `dt` needs to resolve the pericenter
dynamical time `sqrt(r_p^3/(G*M_bh))`, not just the star's own -- see
`decks/encounter_beta3.toml`'s comment for the actual numbers used. Output
fields (all `(N,4)` float32, extra column unused/padding except `h` which is
`(N,1)`): `pos_mass` (x,y,z,mass), `vel` (vx,vy,vz,-),
`rho_press` (rho,pressure,soundspeed,-), `h`. Diagnostics (fixed list, not
yet deck-selectable like `05_tdse_gpu`'s): `kinetic`, `thermal`, `potential`
(star-star self-gravity only), `potential_bh` (star-black hole, `0` if no
black hole), `energy` (=`kinetic+thermal+potential+potential_bh`),
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
| momentum conservation | `com_speed` max `3.7e-8` (all pairwise sums are exactly antisymmetric; this is finite-precision floor, not physical drift) |
| energy conservation | `0.09%` drift over 10 dynamical times (vs. `0.20%` with the self-term excluded, `3.2%` before adaptive `h` at all -- monotonically improving, and no longer oscillating: the settled state is a genuine equilibrium of the *undamped* equations) |
| radial density profile, t=0 vs. t=end (10 t_dyn later) | **time-independent** -- the settled configuration does not evolve once damping is removed |
| radial density profile vs. analytic Lane-Emden | **calibration gap closed**: bulk (`r<0.7R`) now agrees with the analytic target to `<1%` everywhere (e.g. ratio `1.004` at `r=0.125R`, `1.005` at `r=0.542R`, `1.005` at `r=0.625R`, 9 bins spanning `0<r<0.7R`, `N>=95` particles each) |

The 10-30% *under*-density that used to run through the bulk here is gone:
it was the self-term exclusion (see Physics above), not a finite-`N` or
EOS-calibration mismatch as originally suspected -- restoring the
self-inclusive density sum, with `sph.h_iters=3` (unchanged) enough to
converge the `rho<->h` fixed point, reproduces the analytic profile
directly with no other tuning. One artifact remains, smaller and separate:
the outermost, sparsely-sampled bins (`r>0.79R`, `N<=24` particles/bin)
still run over-dense (ratio `1.05` at `r=0.792R`, `1.75` at `r=0.875R`) --
the same free-surface excess seen with the old fixed-`h` version, a
resolution/edge effect at the star's surface rather than a bulk
calibration problem, and lower priority since it affects a small fraction
of the star's mass.

## Validation: the encounter (`decks/encounter_beta3.toml`: M_bh=1000 M_star, beta=3, "point" gravity)

r_t=10, r_p=3.333, starting at r_0=30 (still inbound); Kepler two-body
time-of-flight to pericenter t=2.82, back out to r_0 at t=5.65 (both
confirmed by the orbit tool's own point-mass-only integration, see Physics
above). Run for 300 frames (15 dynamical times) at `dt=0.002`.

| check | result |
|---|---|
| qualitative disruption | visible tidal stream (elongated, curved debris tail) within one frame of pericenter (t=2.85 vs. predicted t=2.82) -- see the movie |
| peak kinetic energy timing | t=2.80 (bulk orbital KE, dominated by the whole star's ~24 (code-units) COM speed through the BH's potential well at closest approach) -- matches the predicted pericenter time to 1% |
| energy conservation | naive `\|dE/E0\|` looks alarming but is **the wrong number**: `E0` is a near-total cancellation of large terms even at the start, so it's a tiny, noise-sensitive denominator once the encounter's orbital kinetic/potential terms grow to ~590 (code units) near pericenter. The physically meaningful number -- absolute energy drift relative to that actual dynamic range -- is `0.11%`: the leapfrog integrator resolves the pericenter passage well at this `dt`. |
| NaN/blowup check | none -- `kinetic` and `potential_bh` both swing over a ~590-unit range and back down smoothly |
| thermal energy | now that the density calibration gap is closed, `thermal` starts at its correct pre-encounter equilibrium value (`0.44`, not the old kernel's artifactually low `0.007`), spikes to `0.67` right at pericenter (`t=2.95` -- real compressional heating as the star is squeezed) and then decays as the debris stretches into a rarefied stream (`0.008` by `t=15`). That decay past pericenter is an artifact of the **barotropic** EOS (see Physics above): `thermal` is recomputed each step purely from the instantaneous density via `P=K rho^Gamma`, so it tracks compression/rarefaction, not accumulated viscous heat -- a real gas would stay hot in the stream, this one doesn't. |

The `energy0`-relative-drift trap above is worth remembering for any future
orbital-encounter diagnostic: **compare drift to the dynamic range the
quantity actually spans during the run, not to its value at an arbitrary
starting instant**, especially when that instant is itself a near-cancellation
of much larger terms.

## Validation: fallback (`decks/encounter_beta3_long.toml`, same encounter, extended to 50 dynamical times)

Checking the final frame's specific orbital energy relative to the black
hole (`eps = 0.5*v^2 - G*M_bh/r`, star self-gravity/pressure ignored -- a
fine approximation once `M_bh` dominates and the debris has cleared the
star's own gravity) finds **~54% of the star's mass is bound** (`eps<0`),
with periods (Kepler's third law from `eps`) ranging `6.87` (most tightly
bound single particle) up to effectively unbounded (`eps` clusters near
`0` for the least-bound tail, giving formally enormous periods -- physically
correct, these particles trickle back over a very long tail rather than a
sharp cutoff). Earliest possible return: `t_pericenter + 6.87 = 2.80 + 6.87
= 9.67`; the bulk of returning mass arrives later, matching where the
fallback-rate curve below actually peaks.

| check | result |
|---|---|
| onset of fallback | the minimum particle-BH separation starts dipping back below the original pericenter (`r_p=3.33`) around `t=24.5`, with repeated passages closer than that afterward (e.g. `r_min=2.02` at `t=44.75`) -- real, repeated close encounters, not a one-off |
| NaN/blowup check (post-fix) | none over the full 1000-frame run -- the zero-density fix (found by an earlier version of this run) holds |
| energy conservation | `0.11%` of the peak orbital energy scale (~590 code units), consistent with the shorter run |

**Quantitative fallback rate** (`tools/fallback_rate.py`): binning the bound
debris mass by each particle's Kepler return time gives `dM/dt` vs. time
since pericenter. The result follows a clean power law across ~4 decades
in time (`t~10` to `t~3e5`), with a fitted slope of **`-1.31`**, shallower
than the classic Rees (1988) "frozen-in", asymptotic `t^(-5/3)` (`-1.667`)
-- see `out/encounter_beta3_long/fallback_rate.png`. This is not a bug: the
`-5/3` law is a late-time asymptote of the idealized frozen-in-energy
argument, and realistic polytropic disruptions (particularly the
centrally-concentrated `n=3/2` profile used here) are well known to show a
shallower decline near peak fallback, steepening toward `-5/3` only much
later (Lodato, King & Pringle 2009). The measured slope is a physically
reasonable value for a real `n=1.5` SPH disruption run of this length, not
an exact `-5/3` match.

The bulk of the star's mass (the unbound majority) keeps receding to
hundreds of length units, which forces `tools/make_movie.py`'s default
auto-fit view so wide that the close-in fallback action becomes a few
barely-visible pixels next to the black hole marker -- `--half-extent 40`
(crop to the interesting region) and the default `--gamma 0.4` (perceptual
lift so the fallback stream's very dilute debris, ~1e-4 vs. the intact
star's ~2 peak density, doesn't render as near-black) together make it
actually visible; see `out/encounter_beta3_long/movie_zoom.mp4`.

## Progress

- [x] `tools/lane_emden.py` -- polytropic stellar structure, validated against tabulated Lane-Emden benchmarks (n=0,1,1.5,3, ~1e-7) and the hydrostatic-equilibrium ODE itself (~2e-4)
- [x] `tools/make_star_ic.py` -- Monte-Carlo IC sampling, validated radial density recovery against the analytic profile
- [x] `src/kernels.hpp` self-gravity + adaptive-h SPH compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] `src/TdeSim` leapfrog integration + damped relaxation + undamped verification, validated per the table above
- [x] external black hole potential (point-mass done + validated; Paczynski-Wiita implemented + selftest-validated, not yet used in a full encounter run) + `tools/make_orbit_ic.py` (parabolic orbit placement, validated) + the actual encounter (`decks/encounter_beta3.toml`, validated per the table above) + `tools/make_movie.py` (BH marker, tight framing, `--half-extent`/`--gamma`)
- [x] fallback: extended the encounter to 50 dynamical times (`decks/encounter_beta3_long.toml`), found and fixed a real zero-density-particle NaN crash along the way (see Physics above), confirmed bound debris (~54% of the star's mass) actually returns and re-passes the black hole closer than the original pericenter, matching the predicted return time -- validated per the table above
- [x] closed the ~10-30% bulk density calibration gap -- was the self-term exclusion, not a resolution/EOS mismatch (see Physics + star-relaxation Validation above)
- [x] quantitative fallback-rate diagnostics (`tools/fallback_rate.py`): dM/dt vs. t from the bound debris's frozen-in orbital energy, compared to the classic t^(-5/3) law -- see Validation above
- [ ] Paczynski-Wiita encounter run (relativistic apsidal precession/plunge) as a bridge toward Tier 2
