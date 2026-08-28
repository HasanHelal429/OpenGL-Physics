# 07 — General-relativistic hydrodynamics (Tier 2, Phases 0-1)

This is Tier 2 of the tidal-disruption project (see
`06_tidal_disruption/README.md`'s tier table): fluid on a fixed
Schwarzschild/Kerr background, no self-gravity, no dynamical spacetime.
Real GR hydro codes are grid-based finite-volume solvers in conservative
variables, not particle methods -- a genuinely different numerical method
from Tier 1's SPH -- so this project starts from scratch rather than
extending `06_tidal_disruption`.

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

## GPU shaders (`src/kernels.hpp`, `src/kernels_schwarzschild.hpp`)

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

## Validation

`--selftest` runs both geometries' GPU-vs-CPU cross-checks (independent
double-precision reference for `ConsToPrim` and `Fluxes`, 40 random
mildly-relativistic cells each -- not a physics validation, just a check
the GPU shaders implement the documented formulas correctly):

    selftest: max relative error  prim_vs_cpu=3.4e-07  prim_vs_truth=3.6e-07  flux=2.3e-06
    selftest: PASS
    selftest (schwarzschild): max relative error  prim_vs_cpu=1.9e-07  prim_vs_truth=2.0e-07  flux=8.6e-07
    selftest (schwarzschild): PASS

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
above); HLLE rather than HLLC (no resolved contact wave); 1D only (Phase 2
will need at least 2D for a non-axisymmetric Kerr test, or an assumption of
equatorial-plane confinement).

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

## Progress

- [x] Valencia conservative variables, HLLE flux, Newton-Raphson primitive recovery, RK2 time integration -- all GPU compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] Newtonian-limit validation against an exact Riemann solver (`tools/exact_riemann_newtonian.py`, itself checked against the classic Sod reference values)
- [x] relativistic shock tube: correct qualitative wave structure (rarefaction/contact/shock), no NaN, conservation validated including the open-boundary momentum-forcing check
- [x] Phase 1: fixed Schwarzschild metric (Schwarzschild coordinates, not the originally planned Kerr-Schild -- see Physics above for why), conserved variables/flux/momentum source term derived from first principles and cross-checked three ways, validated against the analytic Bondi accretion solution per the table above; found and fixed a real outer-boundary instability and a wrong sonic-point formula along the way
- [ ] Phase 2: extend to Kerr (`a != 0`), genuinely off-diagonal/non-static coordinates this time (Boyer-Lindquist or Kerr-Schild); validate against a Fishbone-Moncrief equilibrium torus sitting still (no spurious drift)
- [ ] Phase 3: an actual fluid blob disrupted near/inside a Kerr black hole's tidal field -- the Tier 2 payoff
