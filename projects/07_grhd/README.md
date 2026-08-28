# 07 — General-relativistic hydrodynamics (Tier 2 Phase 0: flat spacetime)

This is Tier 2 of the tidal-disruption project (see
`06_tidal_disruption/README.md`'s tier table): fluid on a fixed
Schwarzschild/Kerr background, no self-gravity, no dynamical spacetime.
Real GR hydro codes are grid-based finite-volume solvers in conservative
variables, not particle methods -- a genuinely different numerical method
from Tier 1's SPH -- so this project starts from scratch rather than
extending `06_tidal_disruption`.

**Phase 0 (this phase, done): validate the fluid solver before adding a
curved metric.** A relativistic hydro code has two pieces that are easy to
get subtly wrong and hard to debug once a black hole's metric is also in
the mix: the conservative-to-primitive variable recovery, and the Riemann
solver. Both are built and checked here on flat (Minkowski) spacetime
first, where trustworthy answers exist -- the classic relativistic shock
tube, and the low-velocity limit where the equations must reduce to
ordinary Newtonian hydrodynamics -- exactly the same "validate the
foundation before adding the black hole" approach Tier 1 used (Lane-Emden
hydrostatic equilibrium before adding gravity).

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

## GPU shaders (`src/kernels.hpp`)

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

## Build & run

```sh
export PATH="/c/msys64/mingw64/bin:$PATH"; export VCPKG_ROOT="/c/vcpkg"
cmake --build build/release --target 07_grhd

./build/release/projects/07_grhd/07_grhd.exe --selftest

./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/shocktube_relativistic.toml --out out/shocktube_relativistic
python tools/plot_shocktube.py out/shocktube_relativistic

./build/release/projects/07_grhd/07_grhd.exe \
    --deck decks/shocktube_newton_limit.toml --out out/shocktube_newton_limit
python tools/plot_shocktube.py out/shocktube_newton_limit --newtonian
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

## Validation

`--selftest` (GPU vs. an independent CPU double-precision reference, for
both `ConsToPrim` and `Fluxes`, on 40 random mildly-relativistic cells --
not a physics validation, just a check the GPU shaders implement the
documented formulas correctly):

    selftest: max relative error  prim_vs_cpu=3.4e-07  prim_vs_truth=3.6e-07  flux=2.3e-06
    selftest: PASS

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
sharpen the contact/shock -- worth adding if Phase 1's curved-spacetime
tests need it); no adaptive timestep (see Physics above); HLLE rather than
HLLC (no resolved contact wave); 1D only (Phase 1 will need at least 2D for
a non-trivial curved-spacetime test problem).

## Progress

- [x] Valencia conservative variables, HLLE flux, Newton-Raphson primitive recovery, RK2 time integration -- all GPU compute shaders, validated against an independent CPU reference (`--selftest`)
- [x] Newtonian-limit validation against an exact Riemann solver (`tools/exact_riemann_newtonian.py`, itself checked against the classic Sod reference values)
- [x] relativistic shock tube: correct qualitative wave structure (rarefaction/contact/shock), no NaN, conservation validated including the open-boundary momentum-forcing check
- [ ] Phase 1: add a fixed Schwarzschild metric (Kerr-Schild coordinates, horizon-penetrating) -- geometric source terms, metric-dependent fluxes -- validated against the analytic steady-state relativistic Bondi accretion solution
- [ ] Phase 2: extend to Kerr (`a != 0`); validate against a Fishbone-Moncrief equilibrium torus sitting still (no spurious drift)
- [ ] Phase 3: an actual fluid blob disrupted near/inside a Kerr black hole's tidal field -- the Tier 2 payoff
