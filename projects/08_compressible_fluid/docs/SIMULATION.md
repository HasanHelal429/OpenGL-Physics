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

## 5. Phase 4: viscosity, and the two simplifications that are exact for Poiseuille flow

Everything in Phases 1-3 is the compressible **Euler** equations (inviscid).
Phase 4 adds the missing viscous terms of compressible **Navier-Stokes**: a
Newtonian stress tensor and Fourier heat conduction, via an explicit
diffusion sub-step appended to each `Euler2D::Step()`.

Two simplifications are made, and both are worth being explicit about
because they are not generally valid — they are exact *for this project's
own validation target* (plane Poiseuille flow) and would need to be
revisited for a genuinely general viscous flow:

1. **Shear-only stress** — `tau_xx=2*mu*du/dx`, `tau_yy=2*mu*dv/dy`,
   `tau_xy=mu*(du/dy+dv/dx)`, dropping Stokes' hypothesis' bulk-viscosity
   correction (`-2/3*mu*div(v)` inside the normal stresses). This
   correction is a *compressibility* effect (it depends on `div(v)`, which
   vanishes for incompressible flow); Poiseuille flow's steady state has
   `div(v)=0` exactly, so the term this drops is exactly zero there.
2. **Same-line derivatives only** — the outer derivative of each stress
   term keeps only the part that's local to one row (`DiffuseX`) or one
   column (`DiffuseY`). Concretely, the `x`-momentum's viscous force is
   `d(tau_xx)/dx + d(tau_xy)/dy`; the `d(tau_xy)/dy` term is
   `d/dy[mu*du/dy] + d/dy[mu*dv/dx]`, and only the first (column-local)
   piece is kept — the second is a genuine mixed `d²v/dxdy` derivative that
   would need values from neighboring columns, breaking the
   independent-line structure every other operator in this project relies
   on. For Poiseuille flow, `v≡0` everywhere, so `dv/dx≡0` and the dropped
   term is exactly zero. (The symmetric term dropped from the `y`-momentum
   equation, `d/dx[mu*du/dy]`, is also exactly zero there: `u=u(y)` has no
   `x`-dependence, so `du/dy` doesn't vary in `x` either.)

Because both omissions are exact for the validation target, this is a
correctness-preserving simplification for that target, not an
approximation being passed off as one — but it is a real scope boundary: a
flow with actual cross-terms (a lid-driven cavity's corner vortices, say)
would need the full stress tensor and a genuinely 2D (not per-line)
diffusion operator.

**Stability.** Explicit diffusion has a parabolic stability limit
(`dt <~ dx²/(2*coefficient)`) that's typically *tighter* than the
hyperbolic CFL limit once viscosity is resolved on a reasonably fine grid —
the opposite scaling from advection, since the diffusive limit shrinks
quadratically with resolution while advection's shrinks linearly. Rather
than requiring a deck's `time.cfl` to somehow already account for this,
`Euler2D::Step` sub-cycles the diffusion update on its own, computing
however many smaller sub-steps its own stability limit requires within
whatever `dt` it's handed.

**Boundary conditions, generalized.** Poiseuille flow needs two BC types
neither prior phase required: periodic in the streamwise (`x`) direction
(so an infinite channel can be represented on a finite domain), and
no-slip at the two walls (`y=0` and `y=H`). `WallBC::NoSlipReflective`
negates ghost-cell momentum (density/energy are copied unchanged, since
energy's kinetic term `1/2 rho (u²+v²)` is invariant under a velocity sign
flip) — a standard Godunov-code technique that makes velocity extrapolate
to exactly zero at the wall face without needing to special-case the
Riemann solver itself. A constant body force (`SetBodyForceX`) replaces
what would otherwise be an explicit streamwise pressure gradient, which a
periodic domain has no way to sustain on its own.

**Validation.** `decks/poiseuille_channel.toml` starts from rest and spins
up under a constant body force; the exact steady solution is the textbook
parabolic profile `u(y) = (f/(2*mu))*y*(H-y)`. `tools/plot_poiseuille.py`
measures the final-frame profile against it: **0.08% max relative error**,
with the residual attributable to the flow's small but nonzero
compressibility (`u_max`/soundspeed ≈ 0.1) rather than any bug — a
literally-incompressible solver would be expected to do slightly better,
which is exactly the trade this project's "stronger, but still
compressible" formalism makes.

---

## 6. What a Phase 5+ would need, if extended further

Not attempted here, but worth naming so the scope boundary is explicit
rather than implicit:

- **The full viscous stress tensor**, including the bulk-viscosity
  correction and the mixed `d²/dxdy` cross-derivatives Phase 4 drops — needed
  for any flow with genuine 2D velocity structure (a lid-driven cavity, a
  cylinder wake), which is exactly why those two benchmarks were explicitly
  scoped *out* of this project from the initial plan (see the top-level
  conversation this project was proposed in): a compressible HLLC scheme
  run at the very low Mach numbers those benchmarks use suffers severe
  acoustic stiffness without a low-Mach preconditioner, a distinct
  numerical-methods problem from anything solved so far.
- **A genuinely adaptive per-step CFL controller.** Every deck in this
  project computes `dt` once from the *initial* condition and holds it
  fixed — safe here because each validation target either has provably
  bounded characteristic speeds (the Riemann problems) or stays at low,
  roughly-constant Mach number throughout (Poiseuille) — but not a general
  solution for a flow whose velocities or sound speed change substantially
  over the run.
