# 07 — General-relativistic hydrodynamics (Tier 2, Phases 0-2b done; Kerr-Schild + non-equilibrium demos follow-on)

This is Tier 2 of the tidal-disruption project (see
`06_tidal_disruption/README.md`'s tier table): fluid on a fixed
Schwarzschild/Kerr background, no self-gravity, no dynamical spacetime.
Real GR hydro codes are grid-based finite-volume solvers in conservative
variables, not particle methods -- a genuinely different numerical method
from Tier 1's SPH -- so this project starts from scratch rather than
extending `06_tidal_disruption`.

**For the physics derivations and implementation walkthrough of the
current Kerr-Schild torus solver, see `docs/SIMULATION.md`** -- a
from-scratch, physics-first writeup (metric -> Valencia formulation ->
con2prim -> HLLE -> the Kerr-Schild reformulation -> implementation) that
supersedes the phase-by-phase narrative below as the reference for how
the solver actually works today. This file remains the chronological
development log.

**Phase 0 (done): validate the fluid solver before adding a curved
metric.** A relativistic hydro code has two pieces that are easy to get
subtly wrong and hard to debug once a black hole's metric is also in the
mix: the conservative-to-primitive variable recovery, and the Riemann
solver. Both are built and checked here on flat (Minkowski) spacetime
first, where trustworthy answers exist -- the classic relativistic shock
tube, and the low-velocity limit where the equations must reduce to
ordinary Newtonian hydrodynamics -- exactly the same "validate the
foundation before adding the black hole" approach Tier 1 used (Lane-Emden
hydrostatic equilibrium before adding gravity).

**Phase 1 (done): a fixed Schwarzschild background.** Radial-only flow,
spherical symmetry, in *Schwarzschild* coordinates (not the originally
planned Kerr-Schild -- see Physics below for why that turned out to be the
better choice for this phase specifically). Validated against the exact
analytic relativistic Bondi accretion solution: initialize the grid to it
and confirm the solver holds it steady over many settling times, the same
"settle then verify it's a genuine equilibrium" methodology Tier 1 used for
its SPH star. Found and fixed one real bug along the way -- an ill-posed
outer boundary condition that let mass pile up and eventually diverge (see
Validation below) -- and, in deriving the curved-spacetime equations from
first principles, caught a mis-remembered sonic-point formula before it
ever reached code (see Physics below).

**Phase 2a (done): Kerr, restricted to the equatorial plane.** Full Kerr
(nonzero spin) is a bigger jump than Phase 1 was: it's only axisymmetric,
not spherically symmetric, and the eventual Fishbone-Moncrief equilibrium
torus is a genuinely 2D (r,theta) structure with real thickness away from
the equatorial plane. Rather than take that whole jump at once, this phase
restricts to the equatorial plane (theta=pi/2) -- an exact invariant
submanifold of Kerr (reflection symmetry keeps v^theta=0 fluid exactly
there), which keeps the numerics "1D in r" like Phases 0-1, at the cost of
not yet being able to represent a torus's finite thickness (the actual
Fishbone-Moncrief validation is deferred to Phase 2b). Validated against
exact circular geodesic orbits, derived independently via the
effective-potential double-root condition rather than trusting a recalled
closed form (Bardeen-Press-Teukolsky give one, but see Phase 1's own
sonic-point mistake above for why that habit changed). Kerr's off-diagonal
metric made this derivation considerably more error-prone than
Schwarzschild's -- it caught two further mistakes before they reached
code (a sign error in the frame-dragging shift, a missing factor in the
specific-energy formula), both described in Physics below, both caught by
cross-checking against independently-verified results rather than by
proofreading the algebra.

## Physics

1D special-relativistic hydrodynamics (flat metric, `c=G=1`), Valencia
conservative formulation, ideal Gamma-law EOS `P=(Gamma-1)*rho*eps`:

    D   = rho*W                          (relativistic mass density)
    S   = rho*h*W^2*v                    (momentum density)
    tau = rho*h*W^2 - P - D              (energy density minus rest mass)

with Lorentz factor `W=1/sqrt(1-v^2)` and specific enthalpy
`h=1+Gamma*P/((Gamma-1)*rho)`. Flux: `F_D=D*v`, `F_S=S*v+P`,
`F_tau=(tau+P)*v`. No source terms in flat 1D Cartesian coordinates --
those appear in Phase 1 once a curved metric is added.

**Primitive recovery** (`ConsToPrim`, per cell): given `(D,S,tau)`, a 1D
Newton-Raphson iteration on `f(P)=P_new(P)-P` (Marti & Muller's living
review, sec. 4.2), warm-started from the previous step's converged
pressure. The derivative is taken numerically (central difference) rather
than analytically, to avoid deriving -- and risking a sign error in -- the
closed form; this problem's `f(P)` is smooth enough for the mildly
relativistic regimes this phase targets that a numeric derivative costs
only two extra evaluations per iteration and converges reliably.

**Riemann solver**: HLLE, the simplest genuinely relativistic Riemann
solver (two-wave estimate bounding the true Riemann fan, no explicit
contact wave -- unlike HLLC -- so contact discontinuities smear over a few
cells at this first-order-in-space resolution; see Validation below).
Signal speeds from the relativistic sound speed `cs^2=Gamma*P/(rho*h)` and
relativistic velocity addition, `lambda_pm=(v+-cs)/(1+-v*cs)`.

**Time integration**: RK2 (Heun's method, SSP) method-of-lines -- an Euler
predictor step, another Euler step from the predicted state, then
`U_next=0.5*(U0+U2)`. First-order-in-space reconstruction (piecewise
constant, i.e. plain Godunov) -- no slope limiter yet.

**Timestep**: `dt=CFL*dx` with no adaptive wave-speed readback, since in
special relativity no signal exceeds `c=1` -- this is always a safe CFL
bound, just not always the tightest one. A documented simplification, not
a bug: an adaptive `dt` (from the fastest cell's actual `lambda_pm`) would
let bigger steps be taken whenever the flow is far from `c`, at the cost
of a GPU->CPU reduction each substep.

**Boundaries**: outflow (zero-gradient) -- the two domain-edge interfaces
use the nearest interior cell as both "left" and "right" state, which
makes HLLE degenerate to that cell's own physical flux, exactly a
zero-gradient condition.

## Physics: Phase 1 (Schwarzschild)

Schwarzschild coordinates `(t,r,theta,phi)`, `ds^2=-f dt^2+dr^2/f+r^2 dOmega^2`,
`f=1-2M/r` (`G=c=1`), radial-only flow (spherical symmetry, `v^theta=v^phi=0`).
**Chose Schwarzschild coordinates over the originally planned Kerr-Schild**:
Schwarzschild is diagonal and static with zero shift, which makes the
conserved-variable/flux/source-term derivation meaningfully lower-risk to
get right from first principles than Kerr-Schild's off-diagonal `g_tr`
term -- worth the trade for this phase, at the cost of not being
horizon-penetrating (the grid's inner edge has to stay outside `r=2M`,
fine for a Bondi-flow test with the sonic point well outside the horizon).
Kerr forces genuinely more general coordinates anyway (Boyer-Lindquist is
also off-diagonal), so Phase 2 was always going to need to handle that
regardless -- this just defers it rather than taking it on twice.

Conserved variables (same `v`/`W`/`h` definitions as flat SRHD --
`v` is the physical, *locally* orthonormal-frame radial velocity measured
by a static observer, not a coordinate velocity; only `D`,`S`,`tau` and the
flux/source pick up the metric's `r`, `f`, `alpha=sqrt(f)` dependence):

    D   = r^2 * rho * W / alpha
    S   = r^2 * rho * h * W^2 * v
    tau = r^2 * (rho*h*W^2 - P) - D

    d_t D   + d_r(f*v*D)                 = 0
    d_t S   + d_r(f*(v*S + r^2*P))       = -M*rho*h + 2*M*P + 2*f*P*r
    d_t tau + d_r(f*(S - v*D))           = 0

Derived from `nabla_mu T^munu = 0` (perfect fluid, projected onto the `r`
and `t` directions with the standard Schwarzschild Christoffel symbols) and
`nabla_mu(rho u^mu)=0` (baryon number, exactly source-free always). Cross-
checked three independent ways before trusting it -- worth doing given how
easy curved-spacetime source terms are to get subtly wrong, and how hard
that is to notice from the numerics alone:

1. **Newtonian limit**: as `M/r->0`, `h->1`, the momentum source
   `-M*rho*h+2*M*P+2*f*P*r` reduces to `-rho*GM+2*P*r`, exactly the
   classical spherical Euler momentum source (`-rho*GM/r^2` gravity plus
   the `r^2*dP/dr` geometric pressure term), derived independently.
2. **Killing-vector energy conservation**: the derivation initially gave
   an awkward `tau` equation with an explicit `1/f` factor that didn't
   match the clean conservative form the other two equations have; working
   through it algebraically, the source term canceled to exactly zero.
   That's not a coincidence to shrug off -- it has to be exactly zero,
   because a stationary spacetime's timelike Killing vector guarantees an
   exactly conserved current by Noether's theorem, with no source ever.
   Getting a clean zero (rather than something merely small) was the
   confirmation the algebra was actually right, not the risk that it
   wasn't.
3. **Flat-space limit**: stripping the `r^2` geometric factors (setting
   `f=alpha=1`) from the `tau` flux reduces it to exactly Phase 0's
   already-validated flat SRHD flux form -- a direct check against code
   that was itself checked against an exact Newtonian solution.

**The sonic-point condition was wrong on the first pass, and the fix is a
good general lesson.** A remembered formula (`v_c^2=M/(2r_c)`,
`cs_c^2=v_c^2/(1-3v_c^2)`) gave `v_c != cs_c` at the "sonic" point, which
contradicts the very definition of a sonic point. Rather than trust the
recollection, the critical-point condition was re-derived directly from
the conserved-quantity equations (write `du/dr` as a ratio `N/D`; a finite
`du/dr` at a point where `D=0` requires `N=0` there too) -- yielding the
correct, self-consistent condition `u_c=cs_c` with
`cs_c^2=M/(2*r_c-3*M)` (checked to reduce to the standard Newtonian
`cs_c^2=M/(2*r_c)` as `r_c->infinity`). Using the wrong formula produced a
profile with a visible, discontinuous jump exactly at `r_c` when sampled
finely there; the corrected one is smooth and monotonic straight through
it (see `tools/bondi_analytic.py`'s docstring for the full derivation).
**Lesson: a formula recalled from memory that produces an internally
inconsistent result (here, `v_c != cs_c` at a point defined by `v=cs`) is
worth re-deriving from the equations you already trust, not patching or
re-guessing.**

**Outer boundary must be Dirichlet, not outflow.** The domain's outer edge
(`r_max`) sits in the *subsonic* part of the flow, where one characteristic
family is incoming from outside the domain -- a zero-gradient/outflow
ghost state there is ill-posed (confirmed the hard way: it let mass
visibly pile up at the outer edge over ~100 time units, then diverge --
see Validation). Fixed by pinning the outer ghost state to the exact
analytic solution at `r_max` (`tools/make_bondi_ic.py` appends it as an
`N+1`-th IC record). The inner edge (`r_min<r_c`) stays plain outflow,
correctly, since the flow is supersonic there and all characteristics
already point outward.

## Physics: Phase 2a (Kerr, equatorial plane)

Boyer-Lindquist coordinates, restricted to `theta=pi/2`:

    g_tt=-(1-2M/r)   g_tphi=-2Ma/r   g_rr=r^2/Delta   g_phiphi=r^2+a^2+2Ma^2/r
    Delta=r^2-2Mr+a^2

Frame dragging (`g_tphi != 0`) means the fluid now has two physical
(ZAMO-frame) velocity components, `v_r` and `v_phi`,
`W=1/sqrt(1-v_r^2-v_phi^2)`. Conserved variables, mixed-index
`T^mu_nu=rho*h*u^mu*u_nu+P*delta^mu_nu` (still "areal", `r^2`-inclusive):

    D   = r^2 * rho * u^t                     (baryon current -- always source-free)
    Sr  = r^2 * T^t_r   = r^2*rho*h*u^t*u_r    (radial momentum -- the only one with a source)
    L   = r^2 * T^t_phi = r^2*rho*h*u^t*u_phi  (angular momentum -- source-free: phi is a Killing direction too)
    tau = -r^2*T^t_t - D                       (source-free: t-Killing, same argument as Phase 1)

with (from the general 3+1 relation `u^mu=W(n^mu+v^mu)`, worked out fully
in `tools/kerr_equatorial_ref.py`):

    u^t=W/alpha   u_r=W*sqrt(gamma_rr)*v_r   u_phi=W*sqrt(gamma_phiphi)*v_phi
    E:=-u_t = W*(alpha - g_tphi*v_phi/sqrt(gamma_phiphi))

**Avoided hand-derived Christoffel symbols entirely.** Kerr's off-diagonal
metric makes the standard symbolic Christoffel-symbol algebra considerably
more error-prone than Schwarzschild's diagonal case -- borne out in
practice (see the two bugs below, both found in derivations that used
closed-form algebra, neither in the numerical parts). So the one place a
geometric source term is actually needed (the radial momentum equation)
uses the standard symmetric-tensor identity
`Gamma^lambda_{mu r} T^mu_lambda = 0.5*T^ab*d(g_ab)/dr`, with the metric
derivative taken by **central finite difference of the metric components
themselves** -- which are the textbook Kerr metric definition, not a
derived quantity, so there is nothing left to get subtly wrong. The exact
same numerical approach is used in `tools/kerr_metric_check.py` (Python
reference) and `src/kernels_kerr.hpp` (the actual shader), so the two are
provably doing the same computation, not just two independent
transcriptions of one formula that could silently diverge.

**Two more mistakes caught before reaching code** (on top of Phase 1's
sonic-point one), both while deriving the primitive<->conserved relations
above, both caught by cross-checking against the independently-derived
circular-orbit solution (`tools/kerr_orbits.py`) rather than by
proofreading the algebra:
- **Wrong sign in the shift.** The standard ADM relation is
  `beta_phi=g_tphi` (no extra minus sign), but an early derivation used
  `beta^phi=-g_tphi/gamma_phiphi`. The giveaway wasn't a failed check --
  it was that a simplification that *should* have collapsed cleanly
  (`u_phi` reducing to the same simple form as `u_r`) kept leaving behind
  an extra term instead. That "this should be simpler than it is" feeling
  turned out to be correct; fixing the sign made the extra term cancel
  exactly.
- **Missing factor of `W`.** The first version of the `E` formula was
  `W*alpha - g_tphi*v_phi/sqrt(gamma_phiphi)` (missing a `W` on the second
  term). This one passed a self-consistency round-trip test (build
  conserved variables from primitives, recover primitives, compare) to
  within ~0.1-0.3% for nonzero spin -- close enough to look like normal
  solver-tolerance noise, *not* an error, unless directly compared against
  the independently-trusted circular-orbit `E`. Lesson: a round-trip test
  where both directions share the same bug doesn't catch that bug -- it
  takes an independent reference to catch a self-consistent-but-wrong
  formula.

**HLLE wave speed: a deliberately conservative simplification.** Getting
the *tight* multi-dimensional characteristic speed right (the actual
flux-Jacobian eigenvalues for this coupled 4-variable system) needs the
correct local-ZAMO-frame-to-coordinate-frame conversion factor for a
signal moving partly radially and partly azimuthally -- and unlike
Schwarzschild, where that factor was a single clean multiplier (`f`) that
came directly out of the flux derivation, Kerr's shift breaks that
simplicity (worked through two candidate conversion factors that
disagreed with each other, without a clean way to tell which was right
without further risky derivation). Rather than accept that risk for a
Phase-2a validation target, this uses a manifestly safe bound instead: the
exact coordinate speed of a purely radial photon,
`sqrt(-g_tt/g_rr)` (from the null condition with `dphi=0` -- exact, not an
approximation, and `g_tphi`/`g_phiphi` drop out entirely since there's no
azimuthal motion to couple to). No physical signal exceeds this, so it's
always a stable bound -- at the cost of more numerical diffusion than a
tight bound would give, a known, visible-in-the-results trade-off (see
Validation below), not a hidden one.

**Known simplification: no self-consistent equilibrium density/pressure
profile.** A circular orbit's *velocity* is exactly determined by geodesic
motion regardless of density or pressure, but a real equilibrium disk
additionally needs its pressure gradient to exactly balance the residual
radial force at every radius -- that is exactly the Fishbone-Moncrief
construction, deferred to Phase 2b (it also needs actual 2D structure).
This phase's initial condition uses arbitrary constant density/pressure
instead, which is not that equilibrium -- see Validation below for what
that costs quantitatively.

## GPU shaders (`src/kernels.hpp`, `src/kernels_schwarzschild.hpp`, `src/kernels_kerr.hpp`)

Buffer layout (std430 SSBOs; `Cons0`/`Cons1` ping-pong the "current" state
across substeps, `Stage1`/`Stage2` hold the RK2 intermediate states):

    binding 0/1  Cons0/Cons1  vec4(D, S, tau, _)
    binding 2/3  Stage1/Stage2  vec4(D, S, tau, _)
    binding 4    Prim          vec4(rho, v, P, _)
    binding 5    Flux          vec4(F_D, F_S, F_tau, _), N+1 entries (interfaces)

Each substep (`GrhdSim::RunOneRk2Step`): `ConsToPrim` -> `Fluxes` ->
`EulerStep` (predictor, current->Stage1) -> `ConsToPrim` -> `Fluxes` ->
`EulerStep` (Stage1->Stage2) -> `Combine` (`0.5*(current+Stage2)` ->
the other ping-pong slot).

`kernels_schwarzschild.hpp` (`SchwarzschildSim`) reuses this exact buffer
layout and RK2 pipeline (`ConsToPrim`/`Fluxes`/`EulerStep`/`Combine`) --
only the per-cell/per-interface physics changes: `ConsToPrim` and `Fluxes`
take extra `uM`/`uRmin`/`uDr` uniforms to compute `r`, `f`, `alpha` per
cell/interface; `EulerStep` adds the momentum source term (reading `Prim`,
an extra binding it didn't need in the flat case); `Fluxes` also takes a
`uGhostHi` uniform for the outer Dirichlet boundary (see Physics above).

## Build & run

```sh
export PATH="/c/msys64/mingw64/bin:$PATH"; export VCPKG_ROOT="/c/vcpkg"
cmake --build build/release --target 07_grhd

./build/release/projects/07_grhd/07_grhd.exe --selftest

# Phase 0: flat-spacetime shock tubes
./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/shocktube_relativistic.toml --out out/shocktube_relativistic
python tools/plot_shocktube.py out/shocktube_relativistic

./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/shocktube_newton_limit.toml --out out/shocktube_newton_limit
python tools/plot_shocktube.py out/shocktube_newton_limit --newtonian

# Phase 1: Schwarzschild Bondi accretion
python tools/make_bondi_ic.py --M 1.0 --r_c 8.0 --gamma 1.4 \
    --n 400 --r_min 3.0 --r_max 40.0 --out ic/bondi_rc8.bin
./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/bondi_schwarzschild.toml --out out/bondi_schwarzschild
python tools/plot_bondi.py out/bondi_schwarzschild
# movie: revolves the 1D radial rho(r) profile into a 2D density image
# (the flow is genuinely spherically symmetric, so this is an honest
# visualization, not an artistic liberty), event horizon marked
python tools/make_bondi_movie.py out/bondi_schwarzschild

# Phase 2a: Kerr equatorial circular orbit
python tools/make_kerr_orbit_ic.py --M 1.0 --a 0.7 --gamma 1.333333 \
    --n 400 --r_min 6.0 --r_max 20.0 --rho 1.0 --p 0.01 --out ic/kerr_ring_a07.bin
./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/kerr_circular_orbit.toml --out out/kerr_circular_orbit
python tools/plot_kerr_orbit.py out/kerr_circular_orbit
python tools/make_kerr_orbit_movie.py out/kerr_circular_orbit

# Phase 2b, part 1: Fishbone-Moncrief torus (analytic construction only --
# see Physics/Progress for the dynamical solver's current status)
python tools/plot_fishbone_moncrief.py --M 1.0 --a 0.9 --gamma 1.333333 \
    --r_in 6.0 --r_center 10.0 --out fishbone_moncrief.png

# Phase 2b, part 2: dynamical 2D torus run -- no longer crashes (runs the
# full t=897 duration with zero NaN), but still loses ~78% of the torus's
# mass over that time (see Physics above) -- not yet a validated
# multi-orbit result
python tools/make_fm_torus_ic.py --M 1.0 --a 0.9 --gamma 1.333333 \
    --r_in 6.0 --r_center 10.0 --nr 96 --ntheta 64 \
    --r_min 4.0 --r_max 30.0 --theta_min 0.5 --out ic/fm_torus_a09.bin
./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/kerr_torus.toml --out out/kerr_torus
python tools/make_torus_movie.py out/kerr_torus --fps 12
# --frame_start/--frame_end zoom a movie into a specific window (e.g. to
# inspect a crash) instead of rendering the whole run -- useful with a
# fine-grained diagnostic run (--frames <N> --substeps 1) that lands many
# frames right around a moment of interest, e.g.:
#   ./build/release/projects/07_grhd/07_grhd.exe --deck decks/kerr_torus.toml \
#       --out out/kerr_torus_finegrained --frames 2350 --substeps 1
#   python tools/make_torus_movie.py out/kerr_torus_finegrained \
#       --frame_start 2200 --frame_end 2315 --fps 12
```

## Deck format

```toml
[grid]      n = 400          # cells
            length = 1.0     # domain [0, length]
[physics]   gamma = 1.666667
[riemann]   rho_l = 1.0  v_l = 0.0  p_l = 1.0
            rho_r = 0.125  v_r = 0.0  p_r = 0.1
            x0 = 0.5        # discontinuity location, fraction of length
[solver]    cons_to_prim_iters = 25
[time]      cfl = 0.4  substeps_per_frame = 8  frames = 50
```

No external IC-generation tool is needed for this phase (unlike Tier 1's
Lane-Emden sampling) -- a Riemann problem's initial condition is just two
constant states, built directly in `GrhdSim::Configure` from the deck.

Phase 1 (`decks/bondi_schwarzschild.toml`, `SchwarzschildSim`):

```toml
geometry = "schwarzschild"          # selects SchwarzschildSim instead of GrhdSim
[grid]      n = 400  r_min = 3.0  r_max = 40.0   # r_min must be > 2*M
[physics]   M = 1.0  gamma = 1.4
[bondi]     ic_file = "ic/bondi_rc8.bin"   # tools/make_bondi_ic.py
            r_c = 8.0                      # sonic radius (for tools/plot_bondi.py's reference overlay)
[solver]    cons_to_prim_iters = 30
[time]      cfl = 0.2  substeps_per_frame = 100  frames = 100
```

`bondi.ic_file` is `grid.n` cell records plus one extra outer-ghost record
(`tools/make_bondi_ic.py`, see Physics' Dirichlet-boundary note above) --
`grid.n`/`r_min`/`r_max`/`physics.M`/`gamma` must match what the IC was
generated with, the same convention as Tier 1's `ic/*.bin`.

Phase 2a (`decks/kerr_circular_orbit.toml`, `KerrEquatorialSim`):

```toml
geometry = "kerr_equatorial"
[grid]      n = 400  r_min = 6.0  r_max = 20.0   # r_min must clear the ISCO/photon region for this spin
[physics]   M = 1.0  a = 0.7  gamma = 1.333333
[kerr]      ic_file = "ic/kerr_ring_a07.bin"      # tools/make_kerr_orbit_ic.py
[solver]    cons_to_prim_iters = 40
[time]      cfl = 0.2  substeps_per_frame = 400  frames = 200
```

`kerr.ic_file` is `grid.n` records of `(rho, v_r, v_phi, P)` -- no outer
ghost record needed here (unlike Phase 1's Bondi flow), since there's no
bulk radial accretion in this test to make the boundary treatment matter
the same way; plain outflow at both ends. Pick `time.substeps_per_frame *
time.frames * (time.cfl*dr)` to cover several orbital periods at
`r_min` (the fastest, most dynamically demanding radius) -- the deck's own
comment shows the estimate used here.

Phase 2b, part 2 (`decks/kerr_torus.toml`, `KerrTorusSim` -- see Physics
above for the current, not-yet-fully-stable status):

```toml
geometry = "kerr_torus"
[grid]      nr = 96  ntheta = 64  r_min = 4.0  r_max = 30.0  theta_min = 0.5  # theta_max fixed at pi-theta_min
[physics]   M = 1.0  a = 0.9  gamma = 1.333333
[torus]     ic_file = "ic/fm_torus_a09.bin"          # tools/make_fm_torus_ic.py
[solver]    cons_to_prim_iters = 40  rho_floor = 1e-8  p_floor = 1e-11
[time]      cfl = 0.15  substeps_per_frame = 300  frames = 150
```

`torus.ic_file` is `grid.nr*grid.ntheta` records of
`(rho, v_r, v_theta, v_phi, P)`, row-major `i*ntheta+j`
(`tools/make_fm_torus_ic.py`, sampling the validated analytic torus from
Phase 2b part 1). `theta_min` sets a safety margin away from both poles
(`theta_max=pi-theta_min`) -- deliberately not attempting genuine polar
coordinate-singularity handling, since the torus is already known to taper
to zero well before either pole (see Physics: Phase 2a's domain-choice
note, which applies here too).

## Validation

`--selftest` runs all three geometries' GPU-vs-CPU cross-checks (independent
double-precision reference for `ConsToPrim` and `Fluxes`, 40 random
mildly-relativistic cells each -- not a physics validation, just a check
the GPU shaders implement the documented formulas correctly):

    selftest: max relative error  prim_vs_cpu=3.4e-07  prim_vs_truth=3.6e-07  flux=2.3e-06
    selftest: PASS
    selftest (schwarzschild): max relative error  prim_vs_cpu=1.9e-07  prim_vs_truth=2.0e-07  flux=8.6e-07
    selftest (schwarzschild): PASS
    selftest (kerr equatorial): max relative error  prim_vs_cpu=2.9e-07  prim_vs_truth=3.1e-07  flux=3.8e-06
    selftest (kerr equatorial): PASS
    selftest (kerr torus 2D): max relative error  prim_vs_cpu=2.3e-07  prim_vs_truth=2.2e-07  flux=1.4e-06
    selftest (kerr torus 2D): PASS

`prim_vs_truth` recovers the *original* `(rho,v,P)` the conserved
variables were built from, not just agreement between two implementations
that could share a bug -- both are essentially at float32 precision.

**Newtonian limit** (`decks/shocktube_newton_limit.toml`: same 10x
density/pressure ratio as the classic Sod problem, but pressures scaled
~1000x below the rest-mass-energy scale so `v<<1` and `eps<<1`
everywhere): compared against the exact Newtonian Riemann solution
(`tools/exact_riemann_newtonian.py`, the standard iterative solver from
Toro's textbook -- itself checked against the universally-quoted classic
Sod reference values, `p*=0.30313`, `u*=0.92745`, before being trusted
here). Density/velocity/pressure profiles at `t=0.392` are visually
indistinguishable from the exact solution (`tools/plot_shocktube.py
--newtonian`'s overlay) -- the only real difference is the expected
first-order-scheme smearing of the shock and contact discontinuities over
a few cells, not a physics mismatch. This is a genuine, non-trivial check:
the SRHD conservative variables, HLLE flux, and primitive recovery all
correctly reduce to ordinary Newtonian hydrodynamics as `v,eps -> 0`, which
would fail if any of that machinery had a real bug (a sign error, a
mis-transcribed formula) that just happened not to matter at `v=0`.

**Relativistic shock tube** (`decks/shocktube_relativistic.toml`, `Gamma
=5/3`, our own choice for consistency with 06's polytrope -- not claiming
an exact match to any specific published post-shock table): the correct
qualitative wave structure appears cleanly -- a rarefaction fan
(`0.2<x<0.4`), a genuine contact discontinuity (density jumps from `~0.49`
to `~0.24` at `x~0.63` while velocity and pressure stay continuous, exactly
what a contact should do), and a shock (`0.8<x<0.85`) -- with peak velocity
`~0.44c`, a real, checkable relativistic effect. No oscillations, no
NaN/blowup.

**Conservation** (both decks): total mass and energy conserved to
`~1e-8` (absolute; both `O(0.1-1)` in these units) over the full run.
Total momentum is *not* conserved, correctly: this is an open (outflow)
boundary problem with `p_L != p_R`, so the boundary cells (still at their
unperturbed initial pressure, since no wave has reached them by the run's
end time) exert a net force the whole time, exactly
`dS_total/dt = -(p_R-p_L)`. Predicted momentum
(`-(p_R-p_L)*t = -(0.1-1.0)*0.392 = 0.3528` for the relativistic deck)
matches the measured value (`0.352800...`) to 5 significant figures -- an
independent, non-trivial consistency check (not just "conserved", which
would be the wrong thing to check here), and a good general lesson: don't
assume every conserved-in-a-closed-system quantity should look conserved
in an open-boundary test, work out what the boundary flux actually
predicts first.

**Known simplifications**, to be addressed as needed in later phases:
first-order-in-space reconstruction (a slope limiter, e.g. minmod/MC, would
sharpen the contact/shock -- also the likely source of Phase 1's ~4%
accretion-rate spread, see below); no adaptive timestep (see Physics
above); HLLE rather than HLLC (no resolved contact wave); 1D only (Phase 2a
handled Kerr by staying in the equatorial plane rather than going 2D --
see its own Physics/Validation sections -- and the actual Fishbone-Moncrief
torus will need real 2D structure, deferred to Phase 2b).

## Validation: Phase 1 (Schwarzschild Bondi accretion)

`decks/bondi_schwarzschild.toml`: `M=1`, sonic radius `r_c=8` (`Mdot=16`,
`Be=1.116`, `K=0.068` at the sonic point), grid `r in [3,40]` (`N=400`),
evolved to `t=183` (~100+ settling/free-fall times at the inner edge).

The analytic solution itself (`tools/bondi_analytic.py`) is self-checked
before being trusted: `Mdot`, `Be`, `K` evaluated along the constructed
profile agree with their sonic-point values to `~1e-16` (machine
precision) at every sampled radius, and the profile is smooth and strictly
monotonic through `r_c` (not true of the first, wrong sonic-point formula
-- see Physics above).

| check | result |
|---|---|
| profile match, `t=0` (IC) vs. `t=183` | visually indistinguishable from the analytic Bondi solution across the whole domain (`tools/plot_bondi.py`'s overlay) -- density spans a factor of ~10.7 from `r=3` to `r=40` |
| final-frame max relative error vs. analytic | `rho: 0.63%`  `v: 1.9%` |
| accretion-rate uniformity (`f*v*D`, exactly constant in `r` in the true steady state) | spread `(max-min)/mean` rises from `0` to `~4%` within the first ~20 time units, then stays flat (`3.9%-4.0%`) for the remaining ~160 time units -- a small, stable first-order-discretization offset, not a growing instability |
| NaN/blowup | none (after the boundary fix below) |

**Found and fixed a real bug**: the first version of this test used plain
outflow at both domain edges and mass visibly piled up at the outer edge
over ~100 time units (density there rising monotonically, `mdot` spread
growing past `200%` and eventually diverging to `O(100-1000)` -- a real,
reproducible blowup, not a subtle drift). Root cause and fix: see Physics
above (the outer edge is subsonic, needs a Dirichlet ghost state, not
outflow). After the fix, `mdot` spread the first frame is `2e-7`
(essentially the exact IC) and settles to a flat `~4%`, confirming the fix
addressed the actual mechanism rather than just delaying the symptom.

## Validation: Phase 2a (Kerr equatorial circular orbit)

`decks/kerr_circular_orbit.toml`: `M=1`, `a=0.7M`, grid `r in [6,20]`
(`N=400`, comfortably outside the ISCO for this spin), every cell
initialized on its own exact circular geodesic (`v_r=0`, `v_phi` from
`tools/kerr_orbits.py`), constant `rho=1`, `P=0.01` (not a real
equilibrium profile -- see Physics above), evolved to `t=557` (~6 orbital
periods at the fast, demanding inner edge).

`tools/kerr_orbits.py` itself is validated before being trusted: the
circular-orbit solve reproduces published Kerr ISCO radii (e.g. `a=0.9M`:
prograde `2.32M`, matching standard tables) despite being derived from
scratch via the effective-potential double-root condition, not by
reproducing a textbook formula; `tools/kerr_metric_check.py` separately
confirms the numerically-differentiated Christoffels give exactly zero
radial 4-acceleration for these orbits (`~1e-12`, floating-point noise).

| check | result |
|---|---|
| NaN/blowup | none, over the full 6-orbit run |
| `v_phi` profile shape, `t=0` vs. `t=557` | tracks the exact circular-orbit curve closely at both times (`tools/plot_kerr_orbit.py`'s overlay); final-frame max relative error vs. exact `2.4%` |
| `max|v_r|` across the grid vs. time | rises from `0` to a peak `~0.034` within the first orbital period (the initial adjustment away from the non-equilibrium constant-density IC), then damps into a bounded oscillation around `~0.023-0.024` for the remaining ~5 orbits -- not growing, not diverging |
| radial velocity profile shape | smooth and monotonic, negative (slow inward drift) at the inner edge grading to positive (slow outward drift) at the outer edge -- consistent with the known-imperfect (non-equilibrium) density/pressure profile relaxing, not consistent with a numerical instability, which would not produce this smooth a shape |

The dominant source of the ~2-3% quantitative drift (as opposed to the
sharper Phase 0/1 results) is almost certainly the deliberately
conservative photon-speed HLLE bound (see Physics above) rather than the
physics itself: `v_phi` -- which the source term and flux need to get
right for the orbit to hold at all -- tracks the exact solution well
throughout, while the noisier quantity (`v_r`, which should be exactly
zero) is exactly the one a diffusive Riemann solver smears fastest. Tightening
that bound, and/or starting from a genuine equilibrium profile instead of
constant density, are the natural next steps if Phase 2b's precision needs
demand it.

## Physics: Phase 2b, part 1 (Fishbone-Moncrief torus -- analytic construction)

`tools/fishbone_moncrief.py`. A stationary, axisymmetric equilibrium with
purely toroidal motion (`u^r=u^theta=0`) and constant specific angular
momentum `l=-u_phi/u_t` everywhere. This needed the FULL (r,theta)-dependent
Kerr metric (`Sigma=r^2+a^2*cos^2(theta)`, etc. -- the textbook definition,
not derived), not just Phase 2a's equatorial slice.

**This construction is commonly quoted with a shortcut** ("potential
`W=ln|u_t|`, enthalpy `h=exp(W_in-W)`") that this project tried first and
which **failed its own consistency check**: the claimed pressure maximum
came out as a potential *maximum* (a pressure *minimum*) -- backwards.
Rather than hunt for a sign fix inside a shortcut that was already
producing the wrong physical picture, this instead:

1. Rederived the 4-acceleration from scratch:
   `a_mu = -0.5*(d(g^ab)/dx^mu)*u_a*u_b` (`ab` in `{tt,tphi,phiphi}`) --
   the correct index placement for this identity (inverse metric derivative
   contracted with *covariant* momenta) was itself only settled by trying
   the other natural-looking option (metric derivative with *contravariant*
   velocities) and rejecting it because it failed the check below, not by
   getting the tensor calculus right on paper first.
2. **Validated `a_mu` three independent ways** before using it for
   anything: (a) at `l` equal to the local geodesic value
   (`tools/kerr_orbits.py`), `a_r` comes out `~1e-11` -- exactly matching
   the independently-derived circular-orbit condition; (b)
   `d(a_r)/d(theta) == d(a_theta)/d(r)` to 5 significant figures -- the
   field is curl-free, i.e. really is a gradient (a mathematical
   requirement for the whole construction, not assumed); (c) integrating
   it gives `h` that actually *peaks* (not troughs) at `r_center`.
3. Found the Euler equation's sign was backwards in the first pass too:
   the correct relation is `d(ln h) = +a_r*dr` (not `-a_r*dr`) -- resolved
   empirically, by checking which sign produces the physically required
   shape, not by re-deriving the Euler equation a third time.
4. Given `a_mu` is curl-free, `h` is well-defined by any path integral
   from a reference point `(r_in, pi/2)` (`h=1` there): radially in to
   `(r,pi/2)`, then vertically to `(r,theta)` -- computed by direct
   numerical quadrature of the (already validated) `a_r`/`a_theta`, not
   by trying yet another closed-form shortcut.

**A fourth, unrelated bug** surfaced once the shape was qualitatively
right but a quantitative cross-check was run: `u_t_of_l`'s implementation
was `-sqrt(-(denom))` where it should have been `-1/sqrt(-(denom))` (the
docstring already had the right formula -- this was a plain transcription
slip, not a derivation error). It didn't break the qualitative torus shape
at all (the pressure-maximum/curl-free checks above are insensitive to an
overall, even position-dependent, rescaling of the acceleration's
magnitude in specific ways), so it was only caught by a targeted numeric
comparison: `u_t_of_l` evaluated at `r_center` with `l=l_geodesic(r_center)`
should equal `-E_geodesic(r_center)` exactly, and it didn't, until fixed.

`l` is fixed by requiring the pressure maximum sit exactly at the chosen
`r_center`: this project's earlier, independently-derived
`tools/kerr_orbits.py` gives that condition directly
(`l = L_geodesic(r_center)/E_geodesic(r_center)`) -- verified, not just
assumed, by confirming `a_r(r_center)~0` inside `FishboneMoncriefTorus`'s
own constructor. `r_in` (inner edge, `h=1`) is the second free shape
parameter, fixing the overall additive normalization of `ln(h)`; density
is normalized to `rho=1` at `r_center` (same convention as
`tools/bondi_analytic.py`'s `rho_c=1`).

## Validation: Phase 2b, part 1 (Fishbone-Moncrief torus construction)

`M=1`, `a=0.9M`, `r_in=6M`, `r_center=10M`, `Gamma=4/3`:
`l=3.6307`, `ln(h)` at the pressure maximum `=0.0233`.

| check | result |
|---|---|
| `a_r` at the local geodesic radius, 3 independent radii (`r=7,12,18M`) | `~1e-11` (floating-point noise) |
| curl-free (`d(a_r)/dtheta` vs. `d(a_theta)/dr`, 3 test points) | agree to 5 significant figures |
| `a_r` at `r_center` (the pressure-maximum condition itself) | `<1e-6` (checked inside the constructor; construction refuses to proceed otherwise) |
| `u_t_of_l(r_center)` vs. independently-computed `-E_geodesic(r_center)` | exact match after the transcription-bug fix (mismatched by ~10% before it) |
| torus shape (`tools/plot_fishbone_moncrief.py`) | a genuine, smooth, symmetric-about-the-equator crescent cross-section, density=0 at `r_in` and beyond a few tens of `M`, peak exactly at `r_center`, correctly excluding the horizon and polar regions -- see the poloidal-cross-section plot |

This is a real, validated equilibrium *solution* -- the piece Phase 2b
needs before there's anything meaningful to hand the dynamical solver.

## Physics: Phase 2b, part 2 (dynamical 2D solver -- crash fixed, mass conservation still open)

`src/kernels_kerr2d.hpp` / `src/KerrTorusSim.cpp` extend Phase 2a's
equatorial solver to a genuine 2D `(r,theta)` grid: a third velocity
component `v_theta`, a 5th conserved quantity `L` (kept in its own buffer
-- a `vec4` only fits `D,Sr,Stheta,tau`), 2D flux divergence (separate
`FluxesR`/`FluxesTheta` passes, each with their own HLLE bound), and
source terms on *both* `Sr` and `Stheta` now (neither r nor theta is a
Killing direction, unlike Phase 2a where only r needed one). Conserved
variables are "areal", weighted by `sqrt(-g)` (see the bug below for why
that specific quantity, not `sqrt(gamma)`).

**Found and fixed two real bugs while trying to validate this against the
Phase 2b part 1 torus** (neither caught by `--selftest`, since GPU and CPU
reference share the same formulas there -- both needed an actual physics
check against the analytic solution):

1. **Wrong volume-element weighting: `sqrt(gamma)` instead of `sqrt(-g)`.**
   The conservative GRHD formulation (Gammie, McKinney & Toth 2003,
   "HARM") weights conserved variables/fluxes/sources by `sqrt(-g)`, the
   *full 4-metric* determinant -- not `sqrt(gamma)=sqrt(g_rr*g_thth*g_phiphi)`,
   the 3-metric one, which an earlier version of this code used
   (`sqrt(-g)=alpha*sqrt(gamma)`, the standard ADM identity -- they differ
   by exactly the lapse). This was invisible in Phase 2a because
   `sqrt(-g)` happens to equal `r^2` exactly *at the equator* for Kerr
   (`sqrt(-g)=Sigma*sin(theta)`, and `Sigma=r^2`, `sin(pi/2)=1` there) --
   the same value Phase 2a's shorthand already used, for the right
   reason without the general formula ever being spelled out. Off the
   equator the two quantities genuinely diverge, and using the wrong one
   broke both mass conservation and the flux/source balance. Caught by a
   from-scratch divergence-theorem residual check
   (`d(F_r)/dr + d(F_theta)/dtheta` should equal the source term, for the
   *already-validated* analytic torus's own `rho`/`P`/`v_phi` fields) --
   not by any shape-level or `--selftest` check, both of which stayed
   green throughout. Fixing it brought short-time (`t<1`) mass
   conservation from a clearly-wrong result to `~5e-8` relative.
2. **Floor cells need to be actively re-floored every step, not just
   seeded once.** The region between the horizon and the torus's inner
   edge (`r_in`) starts at a tiny floor density/pressure at rest (see
   `tools/make_fm_torus_ic.py`), but flux exchange with the real torus
   material slowly perturbs these near-vacuum cells' conserved
   quantities, and dividing by their tiny `D` to recover a velocity
   amplifies that perturbation into a large, spurious velocity -- the
   same class of bug as `06_tidal_disruption`'s zero-density SPH fix.
   Traced by locating *where* `max|v_theta|` was largest (the innermost
   radial cells at near-floor density, not the torus material itself,
   which stayed small and well-behaved) rather than treating the whole
   grid as equally suspect. Fixed in `ConsToPrim`: whenever recovered
   `rho` falls below a floor, primitives are reset to vacuum-at-rest and
   the reset is written back to the conserved buffers (not just the
   displayed primitives), so it actually persists to the next step.

**Both fixes are real and confirmed necessary** (mass conservation and
short-time behavior measurably improved), but a further instability
remained after them: a growing perturbation, confirmed CFL-independent
(1/3 the timestep gave an identical onset time and location), that
eventually produced NaN around `t~46-47` (about a quarter of one orbital
period at `r_center`) -- see `out/*/crash_movie.mp4` (via
`tools/make_torus_movie.py --frame_start ... --frame_end ...`, rendered
from a `--substeps 1` diagnostic run) for a direct visualization: two
small white "NaN" wedges erupt right at the horizon, in the vacuum-floor
gap between it and the torus's inner edge -- exactly the region this
debugging round traced the failure to.

**Three more real, root-cause fixes closed this out:**

3. **The Newton solve for specific enthalpy `h` had no upper bound and no
   step damping.** Only a lower clamp existed (`h>=1+1e-9`), and that
   epsilon was itself broken: float32 has ~1.19e-7 ULP spacing near 1, so
   `1.0+1e-9` silently rounded to exactly `1.0`, making pressure exactly
   `0.0` (not just small) whenever a cell's iteration converged there --
   confirmed by replaying saved simulation frames through the exact
   iteration and watching `P` sit at literal `0.0` for many frames before
   a subsequent step's flux update left no valid solution nearby and the
   unguarded step sent `h` to `2.36e8` in one jump. Fixed with a
   float32-safe floor (`1.0+1e-4`) and a damped Newton step (at most a 50%
   change in `h` per iteration) -- a standard primitive-recovery
   safeguard (c.f. HARM's "u2p" routines).
4. **That alone didn't fix it**: raising `cons_to_prim_iters` from 40 to
   200 changed nothing (same crash, same time), which is the tell that
   the conserved state itself sometimes has **no** valid `h>=1` solution
   at all -- a genuine "failed inversion," not a slow-converging one.
   Derived the exact solvability condition (`f(h)` is monotonically
   negative as `h->infinity`, so a root exists only if
   `f(h=1)=(tau+D)/D + g_tphi*L/(D*gamma_phiphi) >= sqrt(1+K)*alpha`;
   verified against synthetic solvable/unsolvable cases before use).
   First attempt: full reset to vacuum-at-rest on violation (mirroring
   the density-floor branch) -- this ran the whole `t=897` duration with
   no NaN, but lost **97% of the total mass**, because the violation
   turns out to be common at the torus's own low-density surface (any
   equilibrium's outer layers are the most marginal, easily perturbed
   across this boundary by a first-order scheme's diffusion every step),
   and discarding all momentum there every time was wholesale erasing
   real torus material, not fixing a rare corner case.
5. **Replaced the full reset with a minimal correction**: on violation,
   rescale the momentum (`Sr,Sth,L`) down by the smallest factor (found
   by bisection) that restores physicality, instead of discarding it
   entirely -- a standard GRMHD "conserved-variable fixup" for a failed
   inversion. This dropped the mass loss from 97% to 78% over the same
   run and confirmed (via a tolerance experiment: allowing a `1e-4`
   margin before triggering the fixup changed the result by nothing
   measurable) that these are genuine, substantial violations, not
   float-noise sitting right at the boundary.

**Net result of this round**: the original NaN crash is fixed -- the full
`t=897` (~4.4 orbits) run now completes with no NaN at all, using
principled, validated corrections rather than papering over the symptom.
What is NOT yet fixed: quantitative fidelity. The torus still loses ~78%
of its mass over that run (`865.8 -> 187.4`).

**Follow-up investigation (mass conservation) -- the initial hypothesis
above (that the conserved-variable fixup itself was the driver) was
tested directly and disproven.** Added two GPU atomic-counter diagnostics
(`fixup_count`, `floor_count` in `KerrTorusSim`'s per-frame output) to
measure, rather than guess, how often each `ConsToPrim` branch actually
fires:

- **Doubling resolution (96x64 -> 192x128) made mass loss *worse*** (93.8%
  vs. 78% over the same run), which already ruled out "insufficient
  resolution / needs a slope limiter" as the primary story -- a
  genuinely under-resolved diffusive scheme should *improve* with more
  cells, not degrade.
- **`fixup_count` and the mass-loss rate are anti-correlated, not
  correlated.** Over a 20-frame (`t=0..114`) run, `fixup_count` fell
  130x (423,642 -> 3,157 per frame) while the mass-loss rate *rose* then
  plateaued high (0.25 -> ~19 per frame); loss-per-fixup-event grew
  4 orders of magnitude (7e-9 -> 5e-3) over the same window. A true
  driver should track the loss rate, not move opposite to it -- this
  rules the momentum fixup out.
- **`floor_count` (the vacuum density-floor branch, which *does* overwrite
  `D` directly) fires hard only in the first substep (2,194 of 6,144
  cells, i.e. `--substeps 1 --frames 2` from the pristine analytic IC)
  and then drops to exactly zero for the rest of a normal run** -- a
  one-time discretization-settling transient (the analytic torus sampled
  onto a finite grid isn't a bit-exact discrete equilibrium at its sharp
  vacuum/torus surface), not an ongoing sink. Algebraically, both floor
  branches also reconstruct `D` from the *same* `D` they started from
  (`D_new = sqrtg*rho_new*u^t_new` reduces to `D` exactly given how
  `rho_new`/`u^t_new` are derived) -- so neither is a mass-creating or
  mass-destroying operation by construction, consistent with the
  measured near-zero (`~3e-9` relative) mass change over a single RK2
  substep from the pristine equilibrium.
- **Tried scaling down the HLL wave-speed bound** (`fPhoton`, currently
  the *exact photon speed* `sqrt(-g_tt/g_rr)` -- a deliberately
  conservative, but far larger than the fluid's actual `v_r,v_theta`
  (~0.1-0.5) or sound speed, choice that makes this an effectively
  Rusanov/local-Lax-Friedrichs flux) by an ad-hoc `x0.3`, expecting less
  artificial dissipation to *reduce* mass loss. It did the opposite:
  mass loss got worse (55.9% remaining vs. 64.5% at the same `t=114.4`
  in the unmodified run) and both `fixup_count` and `floor_count` stopped
  decaying to zero and stayed persistently nonzero for the whole run.
  This rules out "too much artificial viscosity" as the story, and
  instead points to the *opposite*: the wide photon-speed bound is
  currently damping/suppressing a real draining or (numerically-seeded)
  instability process in this axisymmetric torus, and reducing that
  damping -- exactly like increasing resolution -- lets more of it
  through. (Change reverted; this was a diagnostic-only experiment,
  confirmed via `--selftest` afterward.)

**Current understanding**: the sustained mass loss is not attributable to
either `ConsToPrim` branch, and is not simple excess numerical diffusion
in the flux scheme (less of it made things worse, not better). The two
independent facts that *do* point the same way -- finer resolution loses
more mass, and less flux dissipation loses more mass -- are the signature
of a real (possibly numerically-seeded, since there is no explicit
viscosity in ideal GRHD and this axisymmetric 2D setup cannot capture the
true, non-axisymmetric Papaloizou-Pringle instability of constant-l
tori) draining/accretion process through the inner vacuum boundary that
the flux scheme's dissipation happens to be partially suppressing, not a
discrete implementation bug of the kind found and fixed above. This is a
harder, more open-ended problem (disk stability/transport, not a
one-line fix) than the five bugs above -- see Progress below.

**Second follow-up (mass conservation) -- resolved, not a mystery
process after all.** The investigation above correctly ruled out the
`ConsToPrim` branches and "too much dissipation," but stopped short of
actually measuring where the mass was going. A later session built a
genuine accretion-rate diagnostic (`inner_boundary_flux_cum`/
`outer_boundary_flux_cum`, a direct time-integral of the same HLLE flux
`EulerStep` already uses, RK2-consistent) and checked it against the
actual total-mass change. They match to 0.02-0.05% for a=0.9, 0.5, and
0.998: **the mass loss is real accretion through the inner boundary**,
not numerical diffusion, not a fixup artifact, and not an unexplained
draining process -- the "resolution/dissipation both make it worse"
result above makes sense in hindsight too, since both changes let more
of the *real* inflow through rather than diffusing across the boundary.
This was independently re-derived and re-confirmed after also
reformulating the whole solver in Kerr-Schild coordinates (see
`docs/SIMULATION.md`), so it holds for the current solver, not just the
Boyer-Lindquist version this investigation was run on originally.

## Progress

- [x] Valencia conservative variables, HLLE flux, Newton-Raphson primitive recovery, RK2 time integration -- all GPU compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] Newtonian-limit validation against an exact Riemann solver (`tools/exact_riemann_newtonian.py`, itself checked against the classic Sod reference values)
- [x] relativistic shock tube: correct qualitative wave structure (rarefaction/contact/shock), no NaN, conservation validated including the open-boundary momentum-forcing check
- [x] Phase 1: fixed Schwarzschild metric (Schwarzschild coordinates, not the originally planned Kerr-Schild -- see Physics above for why), conserved variables/flux/momentum source term derived from first principles and cross-checked three ways, validated against the analytic Bondi accretion solution per the table above; found and fixed a real outer-boundary instability and a wrong sonic-point formula along the way
- [x] Phase 2a: Kerr restricted to the equatorial plane (an exact invariant submanifold, keeping the grid 1D-in-r), conserved variables/flux/source rederived for frame dragging and cross-checked against an independently-derived circular-orbit solution; validated per the table above; caught two further mistakes (a shift sign error, a missing factor in the specific-energy formula) before they reached code
- [x] Phase 2b, part 1: Fishbone-Moncrief equilibrium torus analytic construction (`tools/fishbone_moncrief.py`), rederived from the Euler equation after a commonly-quoted shortcut failed its own consistency check; validated per the table above (curl-free acceleration field, zero radial force at the geodesic/pressure-maximum radius, correct torus shape); caught two further mistakes (a wrong potential shortcut/Euler-equation sign, a `u_t_of_l` transcription bug) before trusting it
- [x] Phase 2b, part 2 (crash fixed): dynamical solver extended to genuine 2D `(r,theta)` structure (3-component velocity primitive recovery, 2D flux divergence, source terms on both `Sr`/`Stheta`) -- GPU compute shaders, validated against an independent CPU reference (`--selftest`). Found and fixed FIVE real bugs total in getting a full `t=897` (~4.4-orbit) run to complete with zero NaN: (1) wrong volume-element weighting `sqrt(gamma)` vs. the correct `sqrt(-g)`; (2) floor cells not being re-enforced every step; (3) the primitive-recovery Newton solve had no upper bound/step damping and a float32-broken epsilon; (4) confirmed via an iteration-count experiment that some conserved states have no valid solution at all (a genuine failed inversion, not a convergence problem) and derived the exact solvability condition; (5) a full vacuum reset on failed inversion avoided the crash but erased 97% of the torus's mass, so replaced it with a minimal conserved-variable rescue (bisection to the smallest momentum reduction that restores physicality) -- see Physics above for the full story, including the crash visualization command
- [x] Phase 2b, part 2 (mass-loss question RESOLVED, not a bug): built a genuine horizon/boundary accretion-rate diagnostic (`KerrTorusSim`'s `inner_boundary_flux_cum`/`outer_boundary_flux_cum`, a direct RK2-consistent time-integral of the same HLLE flux `EulerStep` already uses) and checked actual total-mass change against it. For the a=0.9 torus (and a=0.5, a=0.998) the match is 0.02-0.05% -- **the mass loss is real accretion through the inner boundary, not a numerical leak.** (a=0.0 is the one exception: its `total_mass` is dominated by a genuine floor-injection artifact and should not be trusted directly -- plausibly because a=0's `r_in=6` sits exactly at the Schwarzschild ISCO, a degenerate Fishbone-Moncrief choice the other spins don't share.) See `out/accretion_balance*.png` (regenerable, gitignored) and `docs/SIMULATION.md` sec. 9 for the current, still-open item this reframes (the NaN blowup itself, not conservation).
- [x] Kerr-Schild reformulation: the whole torus solver (metric, ADM 3+1 split, primitive velocity convention, con2prim, source terms, HLLE wave speeds, IC transform) rederived and reimplemented in horizon-penetrating Kerr-Schild coordinates, replacing Boyer-Lindquist's coordinate singularity at the horizon (which forced an artificial inner boundary well outside it). Two more real bugs caught the same adversarial way as the earlier five: a missing `beta_i*u^t` term in the covariant velocity and a wrong con2prim shortcut, both formulas that are exactly correct in the simpler BL special case (`beta_r=0`) but silently wrong in general -- caught by a 4D-normalization check passing while a round-trip check failed, not by rereading the algebra. Full derivation story in `docs/SIMULATION.md`.
- [x] Spin scan (a=0, 0.5, 0.9, 0.998) and a family of non-equilibrium demos built on the validated solver -- a free-falling blob, under/over-rotating detuned tori, Bondi accretion ported onto Kerr-Schild (cross-validated at a=0 against the independent 1D Schwarzschild Bondi solver to ~2%), and detuned narrow "ring" tori (prograde vs. retrograde comparison, a two-ring collision) built from the same Fishbone-Moncrief construction rather than an ad-hoc profile, after an ad-hoc Gaussian-blob ring was found to launch its own far-from-equilibrium transient and fail well before periapsis dynamics could be observed. See `docs/SIMULATION.md` and `tools/make_*_ic.py`/`decks/kerr_*.toml` for the full set.
- [ ] Still open: a genuine near-horizon numerical instability, confirmed via three independent setups (a torus with `r_min` inside the horizon, and a free-falling blob with `r_min` inside the horizon, twice) to fail almost immediately (t~9-36) whenever the grid's inner boundary sits inside the horizon, regardless of rotation or what physical flow is happening -- a general property of the floor/vacuum region there, not yet root-caused. Also open: the equilibrium torus itself still eventually destabilizes (NaN) near `r_min` even when `r_min` is safely outside the horizon (r=4M), just later than before the Kerr-Schild reformulation -- not yet root-caused either.
- [ ] Phase 3: an actual fluid blob disrupted near/inside a Kerr black hole's tidal field -- the Tier 2 payoff
