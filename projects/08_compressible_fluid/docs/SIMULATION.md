# How this solver relates to 07_grhd, and how it works

This document exists for the same reason
[`07_grhd`'s `docs/SIMULATION.md`](../../07_grhd/docs/SIMULATION.md) does: to
write down the physics and design decisions in one place, in enough detail
that "why is it built this way" doesn't have to be re-derived from the code.
It assumes you've read (or will read) that document — this one is written
as a *diff* against it, since `08_compressible_fluid` is deliberately the
same numerical formalism with general relativity subtracted out, not a
fresh design.

---

## 1. The formalism, and what changes in flat spacetime

`07_grhd` derives the **Valencia formulation**: write the conservation laws
as an exact divergence of *conserved* quantities (`D, S_i, tau`), so a
finite-volume scheme can handle even a genuine shock correctly via flux
differencing. That derivation goes through the 3+1 (ADM) split of a curved
spacetime — lapse `alpha`, shift `beta^i`, spatial metric `gamma_ij` — and
everything downstream (conserved variables, fluxes, wave speeds, primitive
recovery) carries those metric factors around.

Set the metric to flat Minkowski: `alpha=1`, `beta^i=0`, `gamma_ij=delta_ij`
(flat, diagonal, unit spatial metric), `sqrt(-g)=1`. Every `07_grhd` formula
in sections 2-5 of its document specializes as follows:

| Quantity | `07_grhd` (curved, relativistic) | `08_compressible_fluid` (flat, Newtonian) |
|---|---|---|
| Conserved mass | `D = sqrt(-g)*rho*u^t = sqrt(-g)*rho*W/alpha` | `rho` (the Lorentz factor `W -> 1` as `v -> 0`; `sqrt(-g)/alpha -> 1`) |
| Conserved momentum | `S_i = sqrt(-g)*rho*h*u^t*u_i` | `rho*u` (`h := 1+eps+P/rho -> 1` as `eps,P/rho -> 0` relative to rest-mass energy; `u_i -> u^i` in a flat, orthonormal frame) |
| Conserved energy | `tau = sqrt(-g)*rho*h*u^t*E - sqrt(-g)*P - D` | `E - rho = (rho*eps + 1/2 rho*u^2)`, i.e. `tau` is exactly "energy minus rest mass" in both cases — `08`'s `E` in `Euler1D.hpp` is `07`'s `tau + D` |
| Flux | `F^r_D=sqrt(-g)*rho*u^r`, etc. | `F_rho=rho*u`, `F_mom=rho*u^2+P`, `F_E=u*(E+P)` (`Euler1D.cpp`'s `Flux`) — the same expressions with every `sqrt(-g)`/Lorentz factor set to 1 |
| Source terms | `Source_Si = sqrt(-g)*0.5*sum T^{mu nu} dg_{mu nu}/dx^i` (section 3.4) | **identically zero** — a flat metric doesn't depend on position, so every metric derivative in that sum is exactly zero, not just small. The whole term drops out of the code, not just numerically |
| Primitive recovery | 1D Newton iteration on specific enthalpy `h` (section 4) — genuinely nonlinear, no closed form | **closed-form**: `u=(rho u)/rho`, `P=(Gamma-1)(E-1/2 rho u^2)` (`Euler1D.cpp`'s `ToPrim`) — there's no Lorentz factor to solve for, so the same algebra that made `07`'s con2prim need 25 Newton iterations and multiple numerical safeguards (floors, physicality pre-checks, damped steps) here just... inverts |
| Riemann solver | HLLE (needs only the outermost signal speeds) | **HLLC** (Toro ch. 10) — once GR's coordinate/frame complexity is gone, resolving the middle (contact) wave explicitly is affordable and worth doing: a contact discontinuity stays sharp instead of smearing over many cells the way HLLE's numerical diffusion would spread it |
| Wave speeds | Marti & Muller / Anile multi-dimensional relativistic acoustic formula, converted from a local orthonormal frame to a coordinate speed via the lapse/shift | Newtonian sound speed `c=sqrt(Gamma*P/rho)`, Davis' simple estimate `S_L=min(u_L-c_L,u_R-c_R)`, `S_R=max(u_R+c_R,u_L+c_L)` — no frame conversion needed since the coordinate frame already *is* the physical rest frame |
| Reconstruction | Piecewise-linear, MinMod-limited | **Identical** — this piece of the formalism doesn't reference the metric at all, so it carries over completely unchanged |
| Time integration | RK2 (Heun's method) | **Identical**, same reason |

The practical upshot: everything `07_grhd` had to fight to get right —
non-diagonal spatial metrics, horizon-penetrating coordinates, con2prim
safeguards, a from-scratch symbolic derivation checked against independent
numerical invariants at every step — either doesn't exist in flat spacetime
or reduces to something with a closed-form answer. What's left is the part
of the formalism that was never about curvature in the first place:
conservative variables, a limited reconstruction, an approximate Riemann
solver, method-of-lines time integration. That's `08_compressible_fluid`.

---

## 2. Phase 1: 1D, validated against an exact answer

`Euler1D` (`src/Euler1D.{hpp,cpp}`) implements exactly the table above:
`ToPrim`/`ToCons`/`Flux`/`HllcFlux` are the closed-form primitive recovery,
conserved-variable construction, physical flux, and HLLC Riemann solver;
`Step` is the MinMod-reconstruct → HLLC-flux → RK2 update sequence, with
2 ghost cells per side (`kGhost=2`, enough for a MinMod slope to reach the
domain's outermost interior face) implementing zero-gradient (transmissive)
boundaries.

The validation target, `decks/sod_shocktube.toml`, is the classic Sod
(1978) Riemann problem — the standard first test for *any* compressible
solver, because it has an exact self-similar solution
(`tools/exact_riemann_newtonian.py`, Toro ch. 4's iterative pressure
solver) and exercises all three wave families (rarefaction, contact,
shock) at once. This is also the one benchmark none of the incompressible
Python solvers (`Simple_Fluid`, `MAC_Grid_Solver`, `Lattice_Boltzmann`) can
even attempt.

Two deck-independent identities back this up (`main.cpp`'s
`SelfTestHllcConsistency`/`SelfTestConservation`): HLLC must reduce to the
plain physical flux when fed identical left/right states (a basic sanity
check on the Riemann solver's algebra), and — because the classic Sod IC
starts both sides at rest — mass and energy are *exactly* conserved and
momentum grows at an *exactly* predictable rate for as long as neither
wave has reached the domain boundary, a consequence of the conservative
flux-differencing form itself, not approximate physics. Both check to
machine precision (`~1e-15`).

## 3. Phase 2: the GPU port, and why it needed no ghost cells

`kernels_euler1d.hpp` transliterates `Euler1D` to GLSL compute shaders —
same formulas, same reconstruction, same HLLC — cross-checked against the
CPU reference by `main.cpp`'s `SelfTestGpu` (matches to `~1e-7` relative,
the float32 precision floor, same reasoning as `04_molecular_dynamics`'s
GPU-vs-CPU force check).

One simplification: the GPU buffers carry no ghost cells at all. Every
pass instead **clamps** its neighbor/interface index to `[0, N-1]`. At the
boundary this makes a cell's own "neighbor" itself, which forces MinMod's
slope there to exactly zero (one of the two differences it minmods is
exactly zero by construction) — which is *provably* the same outcome
`Euler1D`'s explicit ghost-copy produces (its ghost cells are an exact copy
of the boundary cell, so the same zero-difference argument applies there
too). Clamping isn't an approximation of the CPU's boundary condition; it's
a different, simpler way of arriving at the identical answer.

## 4. Phase 3: 2D via Strang splitting, and the one genuinely new piece of physics

`Euler2D` extends to 2D via **Strang splitting** — `[X half-step][Y
full-step][X half-step]` per timestep (LeVeque, *Finite Volume Methods for
Hyperbolic Problems*, ch. 17) — where each fractional step reuses
`Euler1D`'s exact line update (`AdvanceLineRK2` in `Euler2D.cpp`) applied
row-by-row or column-by-column. A flow aligned with a grid axis reduces
*exactly* to running `Euler1D` on every line; this is deliberate, since it
means Phase 1's validation is inherited rather than needing to be redone.

The one piece that's genuinely new: **HLLC with a passive transverse
variable** (Toro sec. 10.4). Sweeping in `x`, the `y`-momentum is carried
along for the ride — its flux is `F_{rho*v} = rho*u*v`, and in the HLLC
star-state construction the transverse velocity `v` is simply inherited
unchanged from whichever side (`L` or `R`) the sampled point falls on. This
is exact: only the *normal*-direction fields (density, normal momentum,
pressure) actually jump across the two non-linear (shock/rarefaction)
waves in a 1D Riemann problem; the tangential velocity only jumps across
the contact, and HLLC already resolves the contact exactly by construction.

Boundary conditions generalize the clamped-index trick from Phase 2 (see
`Euler2D.cpp`'s `ApplyBoundaryLine`), extended with two new options needed
for Phase 4 (`Periodic`, `NoSlipReflective`) alongside the original
`Outflow`.

**Validation, and why Sod alone wasn't enough.** A pure-shock test like Sod
(replicated in 2D as the `decks/riemann2d_config3.toml` four-quadrant
Kurganov & Tadmor (2002) problem) can't by itself catch a splitting-order
or transverse-momentum bug, because MinMod's limiter tends to mask small
errors near a discontinuity anyway. The real check is a **smooth** 2D flow:
`main.cpp`'s `SelfTestVortexAdvection` advects an isentropic vortex (an
exact traveling solution of the 2D Euler equations) diagonally — exercising
both sweep directions equally — and compares against the exact solution
shifted by the background flow. A splitting-direction bug would show up
here as an asymmetric smearing of the vortex along one axis; it doesn't
(relative L2 density error ~4%, consistent with MinMod's known diffusivity
at the deliberately coarse ~10-cells-per-core-radius resolution used, not
with any directional artifact).

## 5. Phase 4: viscosity, one wrong simplification, and how a second test caught it

Everything in Phases 1-3 is the compressible **Euler** equations (inviscid).
Phase 4 adds the missing viscous terms of compressible **Navier-Stokes**: a
Newtonian stress tensor and Fourier heat conduction, via an explicit
diffusion sub-step appended to each `Euler2D::Step()`.

### 5.1 The simplification, done correctly

The full Newtonian viscous stress divergence for the `x`-momentum equation
is `d(tau_xx)/dx + d(tau_xy)/dy`, with `tau_xx=2*mu*du/dx` and
`tau_xy=mu*(du/dy+dv/dx)` (Stokes' hypothesis' bulk-viscosity correction
dropped throughout — a genuine, separate simplification, exactly zero for
incompressible flow since it's proportional to `div(v)`). Expanding:

```
d(tau_xx)/dx + d(tau_xy)/dy = 2*mu*d^2u/dx^2 + mu*d^2u/dy^2 + mu*d^2v/dxdy
```

The `d^2v/dxdy` term is a genuine mixed derivative, needing values from
neighboring lines — incompatible with this project's per-line (`DiffuseX`/
`DiffuseY`) structure. Simply dropping it, keeping the `2*mu` and `mu`
coefficients as written, is **wrong in general**: for divergence-free flow
(`du/dx=-dv/dy`, differentiating: `d^2u/dx^2=-d^2v/dxdy`), the correct
reduction substitutes this identity into the mixed term instead of dropping
it outright:

```
2*mu*d^2u/dx^2 + mu*d^2u/dy^2 + mu*(-d^2u/dx^2) = mu*d^2u/dx^2 + mu*d^2u/dy^2 = mu*Laplacian(u)
```

— the mixed term doesn't vanish; it exactly **cancels one of the two
factor-of-2 normal-derivative terms**. The correct per-line implementation
is therefore a *uniform* coefficient `mu` in both directions (`DiffuseX`
contributes `mu*d^2u/dx^2`, `DiffuseY` contributes `mu*d^2u/dy^2`, summing
to `mu*Laplacian(u)`), not `2*mu` on the "normal" direction and `mu` on the
"transverse" one.

### 5.2 The bug, and why one validation target wasn't enough to catch it

The first implementation used the naive (uncancelled) form —
coefficient `2*mu` on the swept-direction second derivative, `mu` on the
other — reasoning that dropping "just the cross term" was the only
simplification being made. **It passed Poiseuille validation** (0.08%
error, see below) because Poiseuille flow has `u=u(y)` only: the swept-in-x
second derivative `d^2u/dx^2` is *identically zero* regardless of its
coefficient, so the erroneous `2*mu` multiplies nothing. Poiseuille flow
genuinely cannot distinguish the correct coefficient from the wrong one.

The bug surfaced building this project's second viscous validation target,
a doubly-periodic **Taylor-Green vortex** (`decks/taylor_green.toml`,
`u=U0*cos(kx)*sin(ky)`) — a flow with real second derivatives in *both*
directions, where the miscounted factor doesn't cancel: the wrong
coefficients give `d(tau_xx)/dx + d(tau_xy)/dy = 2*mu*(-k^2 u) + mu*(-k^2 u)
= -3*mu*k^2*u` where the correct answer is `mu*Laplacian(u) = -2*mu*k^2*u`
— a real, `1.5`x error in the velocity decay rate (`3`x in kinetic energy,
since `KE∝velocity²`). The measured kinetic-energy decay rate came out at
almost exactly double the (initially, also miscalculated — see below) exact
prediction, which is what led to finding both errors.

**The general lesson** (stated because it recurred in `07_grhd`'s own
development too, section 8 there): a simplification validated against only
one test case can be silently wrong in a way that test case cannot reveal,
because the very thing that makes the target simple enough to have a clean
exact answer (here, `u`'s complete lack of `x`-dependence) is often exactly
what makes it blind to the specific error. The fix, both times, was the
same: get a second, independently-chosen validation target whose structure
is different enough to exercise the part of the formula the first target
couldn't reach.

### 5.3 Stability and boundary conditions

Explicit diffusion has a parabolic stability limit (`dt <~
dx²/(2*coefficient)`) that's typically *tighter* than the hyperbolic CFL
limit once viscosity is resolved on a reasonably fine grid — the opposite
scaling from advection, since the diffusive limit shrinks quadratically
with resolution while advection's shrinks linearly. Rather than requiring
a deck's `time.cfl` to somehow already account for this, `Euler2D::Step`
sub-cycles the diffusion update on its own, computing however many smaller
sub-steps its own stability limit requires within whatever `dt` it's
handed.

Poiseuille flow needs two BC types neither prior phase required: periodic
in the streamwise (`x`) direction (so an infinite channel can be
represented on a finite domain), and no-slip at the two walls (`y=0` and
`y=H`). `WallBC::NoSlipReflective` negates ghost-cell momentum
(density/energy are copied unchanged, since energy's kinetic term
`1/2 rho (u²+v²)` is invariant under a velocity sign flip) — a standard
Godunov-code technique that makes velocity extrapolate to exactly zero at
the wall face without needing to special-case the Riemann solver itself. A
constant body force (`SetBodyForceX`) replaces what would otherwise be an
explicit streamwise pressure gradient, which a periodic domain has no way
to sustain on its own. Taylor-Green needs no walls at all — periodic in
*both* directions.

### 5.4 Validation

**Steady state.** `decks/poiseuille_channel.toml` starts from rest and
spins up under a constant body force; the exact steady solution is the
textbook parabolic profile `u(y) = (f/(2*mu))*y*(H-y)`.
`tools/plot_poiseuille.py` measures the final-frame profile against it:
**0.08% max relative error**, attributable to the flow's small but nonzero
compressibility (`u_max`/soundspeed ≈ 0.1) rather than any bug.

**Transient decay.** `decks/taylor_green.toml`'s vortex pair has an exact
viscous-decay solution. `u`'s Laplacian is `-2*k²*u` (wavenumber `k` in
*both* `x` and `y`, each contributing `-k²*u`), so velocity decays as
`exp(-2*nu*k²*t)` and kinetic energy (quadratic in velocity) as
`exp(-4*nu*k²*t)` — getting this factor of 2 right was itself a second,
smaller mistake made and caught while writing the validation script (the
first draft assumed a single-direction wavenumber's `-k²*u`, off by exactly
the same kind of "which directions actually contribute" error as the
solver bug above, underscoring the same lesson from a different angle).
`tools/plot_taylor_green.py` fits the measured `log(KE(t))` slope and
found **1.6% error** against `4*nu*k²` — larger than Poiseuille's 0.08%
because a genuinely time-evolving flow keeps accumulating the inviscid
scheme's own numerical dissipation on top of the physical viscosity, rather
than converging past it the way a steady state does.

---

## 6. Flow past an obstacle: the immersed boundary and a symmetry gotcha

The original plan for this project explicitly scoped cylinder vortex
shedding *out*, on the reasoning that a compressible HLLC scheme run at
the very low Mach numbers an incompressible-solver comparison would want
suffers severe acoustic stiffness without a low-Mach preconditioner. That
concern is real for matching the Python reference's effectively-zero Mach
number exactly, but doesn't rule out a *weakly* compressible run
(`mach=0.2`, still well subsonic even for the ~2x-U_inf peak surface speed
potential flow predicts near a cylinder) — and vortex shedding turned out
to be tractable at that Mach number without any preconditioner.

### 6.1 The immersed boundary: cell masking

Adding a solid obstacle to a Cartesian-grid finite-volume code has several
standard approaches, ranging from cut cells (geometrically exact, a lot of
extra bookkeeping at every solid-adjacent face) to ghost-fluid methods
(exact-ish, needs per-face special-casing in the Riemann solver) to the
simplest: **volume penalization** (Brinkman penalization) — treat the
solid region as a porous medium with a friction coefficient so large it
forces velocity to zero there, without changing anything else. This
project uses the *hard* (instantaneous) limit of that idea: every step,
`Euler2D::ApplyObstacleMask` forces velocity to exactly zero inside the
masked region (energy reduced by exactly the removed kinetic energy, so
pressure — and hence the physically real force the obstacle exerts on
the surrounding fluid — is preserved rather than spiking). This needed
zero changes to `LineRhs`, the HLLC solver, or the line-sweep structure:
the masked region simply participates in ordinary flux exchange with its
neighbors as "fluid currently at rest," and the existing viscous machinery
(section 5) diffuses the resulting no-slip condition into the surrounding
flow exactly the way it would diffuse any other velocity discontinuity.

The tradeoff, honestly: the obstacle boundary is jagged (Cartesian
staircase, not a smooth circle) at any finite resolution, and this
technique doesn't produce a sharp boundary layer the way a body-fitted
grid or a cut-cell method would. For a global, wake-timing quantity like
the Strouhal number this is a reasonable trade (see section 6.3); it would
matter more for something resolution-sensitive right at the surface, like
a drag coefficient.

### 6.2 The BCs an external flow needs, and a genuine bug they exposed

`WallBC::Inflow` (ghost cells fixed to a prescribed upstream state,
independent of the interior) and `WallBC::FreeSlipReflective` (mirrors
only the boundary-normal velocity component, keeping the tangential one —
a symmetry/no-penetration condition, appropriate for a domain edge that's
a truncation rather than a physical wall) round out the `WallBC` set
already built for Poiseuille flow (`NoSlipReflective`, `Periodic`) and the
Riemann problems (`Outflow`). Supporting a *different* condition on each
side of the domain (inflow left, outflow right, free-slip top and bottom)
required promoting `Euler2D::SetBoundaryConditions` from one `WallBC` per
axis to four independent per-side values — a refactor validated by
re-running all five prior targets (Sod, GPU-vs-CPU, vortex advection,
Config3, Poiseuille, Taylor-Green) and confirming every one reproduced its
prior result exactly, before trusting the new capability on top.

**A genuine, caught-before-shipping bug, in the same spirit as section 5's
viscosity postmortem**: the deck's domain, obstacle mask, and boundary
conditions are symmetric about the channel centerline to floating-point
precision. The very first run — an exactly-centered cylinder, a uniform
impulsive-start initial condition, no perturbation of any kind — **never
sheds at all**: the wake settles into an exactly symmetric, steady state
(visually: two mirror-image recirculation lobes, no alternation) and sits
there for the entire simulated run. This isn't a solver bug in the usual
sense — the symmetric state genuinely is an exact solution of the
discretized equations — but it *is* a linearly *unstable* one, and nothing
in a perfectly symmetric simulation ever breaks the symmetry needed to
reach the (stable) shedding limit cycle instead. A real cylinder always
sheds because a real flow always carries some asymmetric disturbance;
the "clean" perfectly symmetric numerical case is the artificial one. The
fix has two parts, both expressed purely as deck data (see section 7 for
the generic scene schema all of this now runs through): a small
*permanent* geometric offset of the `[[obstacles]]` circle's `center` off
the centerline (2% of `D` in `decks/cylinder_re100.toml`) — persistent, so
the instability always has something to grow from, not just a one-time
nudge that convects away before the wake even forms — plus a smaller,
one-time antisymmetric velocity bump via `[initial_condition.perturbation]`
to seed the growth faster than waiting on the offset's much smaller
steady-state asymmetry alone.

### 6.3 Validation, and an honest read on a real discrepancy

`decks/cylinder_re100.toml` mirrors `MAC_Grid_Solver`'s
`Cylinder_Vortex_Shedding.ipynb` geometry and boundary conditions
(`D=1, Re=100, U_inf=1`, blockage `0.125`, 5D upstream / 20D downstream,
inflow/outflow/free-slip). `tools/plot_strouhal.py` FFTs a downstream
velocity probe and measures `St=f*D/U_inf`. The result: a genuine von
Kármán street (alternating vorticity lobes, visually unambiguous) at
**St=0.138**, against Roshko's correlation's `0.159` at this Reynolds
number (literature `~0.166`) — a real 13% gap, reported rather than
smoothed over, with three identifiable, non-mysterious contributors: the
probe signal hadn't yet saturated to a full periodic limit cycle within
this run's 130 time units (still visibly growing exponentially — the
frequency of a growing oscillation should already closely track the
eventual shedding frequency for this kind of onset, but not necessarily
exactly); the resolution (`cells_per_d=8`) is coarser than even the Python
reference's own flagged-as-suboptimal 12; and this project's own numerical
dissipation (quantified independently in
`Studies/compressible_fluid/vorticity_decay_vs_viscosity`) acts like added
viscosity at this resolution, and *lower* Strouhal number is exactly the
direction that predicts — solving Roshko's correlation backward from the
measured value implies an effective Reynolds number of roughly 65, a
sizable but physically coherent reduction from the nominal 100. None of
these individually or together are surprising; a finer grid and a longer
run are the natural next step to close the gap, not attempted here.

---

## 7. From four hardcoded scenarios to one data-driven scene schema

Through Phases 1-4 and sections 5-6, each new validation target got its
own C++ `fw::Simulation` class: `CompressibleSim2D` for the Riemann
problems, `CompressibleSimChannel` for Poiseuille, `CompressibleSimTaylorGreen`
for the vortex, `CompressibleSimCylinder` for the obstacle flow — each with
its own `Configure()` hardcoding that scenario's grid setup, boundary
conditions, initial condition, and (for the cylinder) obstacle geometry and
symmetry-breaking perturbation, with the deck supplying only the *scalar
parameters* within that fixed structure. This was a real, working design,
and it's also why the cylinder deck was noticeably *sparser* than it should
have been: `cylinder.y_offset_d`, `cylinder.perturb_amplitude`, and
`cylinder.tracer_stripe_width_d` all had silent C++-side defaults never
written into the deck at all — a reader opening the deck expecting it to
be the complete, reproducible specification of the run would get a
misleading picture of what was actually controlling the physics.

The fix chosen was the larger of two options considered: not just making
each existing scenario deck *complete* (listing every value it silently
defaulted), but replacing all four scenario classes with **one**,
`CompressibleSimScene`, driven entirely by deck content — grid, physics,
per-side boundary conditions (with an optional inflow profile), an
obstacle list, an initial-condition *type* (with an optional perturbation
modifier), named probes, and a tracer toggle (see the project README's
"The generic scene schema" section for the exact schema fields). Every
existing scenario is now a *deck* under this one schema, not a *class* —
adding a genuinely new combination of existing pieces (a different
obstacle position, a different boundary layout, Poiseuille flow with a
tracer) needs zero new C++. A genuinely new *shape* of initial condition,
inflow profile, or obstacle still needs one new named case in
`CompressibleSimScene.cpp` — the same scope of change `WallBC` already
represents (a fixed enum of named cases, not a full expression language),
deliberately chosen over building something more general than any actual
use case has needed so far.

**Validation.** Every one of the four existing decks was rewritten under
the new schema and re-run; every result reproduced *bit-for-bit* identical
output to the pre-refactor version — same final-frame density/pressure
ranges for the Riemann problem, same 0.08%/1.56% Poiseuille/Taylor-Green
errors, same St=0.1377 cylinder Strouhal number, same tracer field values
to the last float. This is a strong check specifically because the new
class's code path is written completely independently of the four it
replaces (parsing generic deck tables into closures, rather than reading
scenario-specific keys directly) — reproducing every prior numerical
result exactly means the generic parsing captures precisely the same
physics as each specialized class did, not an approximation of it.

---

## 8. Performance: complexity, the bottleneck that measurement actually found, and what fixed it

### 8.1 Asymptotic scaling

Let `n` be linear grid resolution (`nx≈ny≈n`, domain size fixed), `N=n²`
cells. The CFL-limited timestep shrinks as `dt~1/n`, so the number of
steps for a fixed simulated time scales as `O(n)`; work per step is
`O(N)=O(n²)`. Total inviscid cost is therefore `O(n)·O(n²)=O(n³)=O(N^1.5)` —
the standard, unavoidable scaling for an explicit hyperbolic scheme (2×
resolution in each direction → 8× cost). Nothing to fix here; it's inherent
to explicit time-stepping.

**The one real asymptotic risk**: `Euler2D::Step`'s diffusion sub-cycling
uses an *explicit* treatment with its own parabolic stability limit,
`dtDiffMax~h²/μ`. Since this shrinks as `h²` while the outer CFL `dt` only
shrinks as `h`, the number of diffusion sub-steps grows as `nSub~1/h~n`,
making the viscous part of the cost scale as `O(n)·O(n)·O(n²)=O(n⁴)=O(N²)`
— quadratic in cell count, not `N^1.5`. Checked against the actual cylinder
deck (`cells_per_d=8`, `μ=0.01`): `nSub` computes to exactly **1** today, so
this costs nothing currently, but it's a wall that would appear immediately
if resolution were increased (e.g. to close the 13% Strouhal gap, section
6.3) or Reynolds number raised (smaller `μ`). Not fixed here — the standard
remedy is an implicit or ADI (alternating-direction-implicit) treatment of
the viscous terms, a separate, larger change scoped only if actually needed.

### 8.2 What was actually slow, and the surprise in fixing it

Before any of the changes below, `LineRhs`/`AdvanceLineRK2`/`LineDiffuse`
were free functions that heap-allocated fresh `std::vector`s on every
single call — ~11 allocations per line, per `Step()`. For the validated
cylinder-shedding deck (200×64 grid, 38,400 steps), that works out to
**~169 million heap allocations** for one run. The naive expectation was
that this allocator churn was the dominant cost (typical allocator
overhead assumptions put this at potentially 50%+ of runtime).

**Measurement disagreed.** Converting these to `Euler2D` member functions
writing into per-object, reused scratch buffers (`Workspace`, sized once in
`Init()`) — eliminating essentially all of that allocation — gave a real
but modest **~6-7% wall-clock improvement** (cylinder deck: ~3m30s →
~3m17s, reproducible across repeat runs). Far smaller than the allocation
count would suggest: this allocator evidently already handles many
small, same-sized, short-lived allocations cheaply (a fast per-thread
free-list path is the likely explanation), so allocator overhead was never
the dominant fraction of runtime here. The actual dominant cost is the
arithmetic and branching in HLLC flux evaluation, the five-way `WallBC`
switch in `ApplyBoundaryLine`, and primitive recovery (each involving
several divisions and a `sqrt` per interface) — ordinary floating-point and
control-flow work, not allocator pressure.

This is worth stating plainly because it's a real, checkable lesson: a
plausible-sounding complexity argument ("169 million allocations must
dominate") turned out to be wrong once actually measured, and the fix
was still worth doing (it removes allocator-contention risk that *would*
have mattered once the next change made everything run concurrently — see
below) but for a different reason than originally assumed. One correctness
trap surfaced during this refactor, also worth naming: `LineDiffuse`'s
output buffer relied on a *fresh* vector's implicit zero-initialization for
`.rho` (never explicitly set) and `.rhoTracer` (only set when
`m_tracerDiffusivity>0`, not the default) — a *reused* buffer doesn't get
that for free, and missing this would have silently leaked a previous
call's diffusion increment into the tracer field under the default
configuration. Caught and fixed before merging, but a reminder that buffer
reuse refactors need an explicit audit of every field for implicit-zero-init
dependencies, not just a mechanical "stop allocating" pass.

### 8.3 Parallelization: the real win

Every row in a `SweepX`/`DiffuseX` call (and every column in
`SweepY`/`DiffuseY`) is fully independent of every other row/column within
that call — no cross-line data dependency anywhere in the reconstruction,
HLLC flux, or RK2 update. This makes the sweeps embarrassingly parallel,
and unlike the allocation question above, the benefit here didn't need
measuring to predict: `#pragma omp parallel for` over each sweep's outer
loop, with `Workspace` promoted to one-per-OpenMP-thread (indexed by
`omp_get_thread_num()`, sized to `omp_get_max_threads()` in `Init()` — a
direct extension of section 8.2's buffer-reuse structure, which already
separated "per-line scratch state" from "the object's own state" and made
this promotion a one-line change per sweep rather than a redesign).

**Measured result** (16 logical cores): the cylinder deck dropped from
~3m17s to **~1m22s — a reproducible ~2.4× speedup**. Well short of 16×
(expected: modest line lengths relative to thread count, per-call thread
spin-up/synchronization overhead, and the serial row/column
extraction-and-write-back either side of each parallel region all eat into
the theoretical ceiling), but a real, substantial, and free-going-forward
win — every future deck run benefits automatically, no per-deck tuning
needed. Validated identically to every prior change: `--selftest`'s four
checks, Poiseuille (0.08%), Taylor-Green (1.56%), and the cylinder's
Strouhal number all reproduce bit-for-bit identical results with
parallelism on, confirming no data races were introduced.

### 8.4 A GPU port of `Euler2D`, scoped exactly like section 3's

`kernels_euler2d.hpp` does for `Euler2D` what section 3's
`kernels_euler1d.hpp` did for `Euler1D`: the same algorithm (HLLC,
MinMod, RK2/Heun), transliterated to GLSL, validated only by a GPU-vs-CPU
cross-check (`main.cpp`'s `SelfTestGpu2D`) — not wired into any deck-driven
run. It is deliberately scoped down the same way: the core inviscid method
only, default (`Outflow`) boundaries via the same clamped-index trick
section 3 uses (unchanged by moving to 2D — it's a per-line, per-boundary
argument, and a 2D sweep is still just many independent 1D lines), no
viscosity, no obstacle mask, no tracer field.

The one real 2D-specific design choice is how much of Strang splitting to
push onto the GPU per dispatch. Section 4's CPU implementation parallelizes
*within* a sweep (one thread per row/column, via OpenMP in section 8.3) but
still issues one `AdvanceLineRK2` call per line. The GPU port instead
dispatches **one compute pass per fractional step for the whole grid at
once** — every kernel takes a `uAxis` uniform (0=X, 1=Y) selecting the
neighbor stride and interface layout, so a single GLSL definition of
`ComputeSlopes`/`Fluxes`/`EulerStep` covers both sweep directions instead of
duplicating each kernel per axis. This is exactly the same independence
argument section 8.3 uses to justify OpenMP, just carried one step further:
if every row is independent of every other row, there's no reason to even
loop over rows on the host side — the whole X sweep is one dispatch over
`nx*ny` cells (plus one over `(nx+1)*ny` interfaces for the flux pass), and
the whole Y sweep is the axis-1 equivalent over `nx*(ny+1)` interfaces.

Each of the three Strang sub-steps (X half, Y full, X half) runs as one
RK2/Heun fractional step — `B = A - dtFrac*div(F(A))`, `C = B -
dtFrac*div(F(B))`, result `= 0.5*(A+C)` — the identical arithmetic
`AdvanceLineRK2` performs per line, just applied to the whole grid buffer at
once; the sub-steps themselves still run in the CPU's exact
X-half/Y-full/X-half sequence, ping-ponging between two Cons buffers instead
of updating one grid in place.

Cross-checked against `Euler2D::Step` on the isentropic-vortex scenario
(`SelfTestVortexAdvection`'s exact IC, domain, and step count): **max
relative error rho=1.1e-05, u=5.8e-06, v=4.7e-06, p=9.5e-06** — comfortably
inside the `1e-3` float32-vs-double tolerance floor section 3's 1D check
already established, despite running roughly 3x more flux evaluations per
step (3 Strang sub-steps × 2 RK2 stages, vs. 1D's 2) over the ~250 steps
needed to reach `t=2`. Passed on the first attempt after the code compiled,
which — combined with section 7's identical experience porting the deck
schema — says more about how well `Euler2D`'s existing formulas were
already pinned down by the accumulated CPU test suite than about anything
special in the port itself: there was very little room left for the GPU
and CPU implementations to independently agree on a wrong answer.

---

## 9. What further extension would need

Not attempted here, but worth naming so the scope boundary is explicit
rather than implicit:

- **The full viscous stress tensor**, including the bulk-viscosity
  correction and the mixed `d²/dxdy` cross-derivatives section 5 drops —
  needed for a flow with genuine small-scale 2D velocity structure right at
  a surface (e.g. an accurate drag coefficient, not just the global wake
  timing a Strouhal number captures).
- **A body-fitted grid or cut-cell obstacle representation**, if the
  jagged Cartesian-staircase boundary of section 6's cell-masking approach
  ever needs to become resolution-independent rather than just resolution-
  convergent.
- **A genuinely adaptive per-step CFL controller.** Every deck in this
  project computes `dt` once from the *initial* condition and holds it
  fixed — safe here because each validation target either has provably
  bounded characteristic speeds (the Riemann problems), stays at low,
  roughly-constant Mach number throughout (Poiseuille, Taylor-Green), or
  has the sound speed dominate its wave-speed estimate by a wide-enough
  margin that a plausible velocity excursion barely moves it (the
  cylinder, backed by a lower CFL number for extra margin) — but not a
  general solution for a flow whose velocities or sound speed change
  substantially over the run.
- **Reaching a fully saturated shedding limit cycle** (section 6.3) and
  a resolution sweep, to see how much of the 13% Strouhal gap closes with
  each.
