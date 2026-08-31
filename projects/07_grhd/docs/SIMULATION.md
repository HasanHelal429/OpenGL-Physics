# How the Kerr torus simulation works

This document explains the physics and implementation of `07_grhd`'s Tier 2
Phase 2b solver: a fluid orbiting a spinning (Kerr) black hole, evolved on a
fixed 2D (r, theta) grid. It starts from first principles (what equations
govern the fluid, why they take the form they do) and works up to the actual
GPU code. It assumes calculus and basic special relativity, but not prior
general relativity.

The concrete deliverable this solver targets: initialize a
**Fishbone-Moncrief torus** (an exact equilibrium solution — a donut of gas
orbiting the black hole, held up against gravity by a combination of
pressure and centrifugal support) and confirm it stays in equilibrium under
evolution. This is a standard test in the field for exactly this reason: a
correct code should show it sitting still (up to slow, small numerical
drift); a buggy one usually doesn't.

---

## 1. Why this is hard: general relativity in a nutshell

In Newtonian physics, "where a particle is" and "when" are separate,
absolute concepts. In special relativity they're unified into one 4D
spacetime, and different observers moving at different velocities disagree
about durations and distances (but agree on the spacetime interval `ds^2`
between events). General relativity goes one step further: spacetime itself
is curved by mass and energy, and that curvature *is* gravity — a particle
with no forces on it doesn't move in a straight line through space, it moves
along a **geodesic** (the straightest possible path) through curved
spacetime.

The curvature is encoded in the **metric** `g_mu_nu`, a 4x4 symmetric matrix
(Greek indices mu, nu run over t, r, theta, phi) that generalizes the flat
Minkowski metric. It defines the spacetime interval:

```
ds^2 = g_mu_nu dx^mu dx^nu    (sum over mu, nu = 0..3)
```

Everything below — how fast a fluid can flow, how pressure gradients push
it, how the black hole's spin drags spacetime around with it — is encoded
in the specific numerical values of `g_mu_nu` at each point, plus the rules
for how energy and momentum are conserved *given* that curvature.

### 1.1 The Kerr metric

A spinning black hole of mass `M` and spin `a` (with `0 <= a <= M`, `a` in
the same units as `M` since we use geometric units `G=c=1` throughout) is
described by the **Kerr metric**. In the most commonly quoted coordinates,
**Boyer-Lindquist (BL)** `(t, r, theta, phi)`, with:

```
Sigma = r^2 + a^2*cos^2(theta)
Delta = r^2 - 2*M*r + a^2
A     = (r^2+a^2)^2 - a^2*Delta*sin^2(theta)
```

the nonzero metric components are:

```
g_tt     = -(1 - 2*M*r/Sigma)
g_tphi   = -2*M*a*r*sin^2(theta)/Sigma
g_rr     = Sigma/Delta
g_thth   = Sigma
g_phiphi = A*sin^2(theta)/Sigma
```

Two things to notice, both consequences of the spin:

- **`g_tphi` is nonzero.** Time and azimuthal angle are coupled — this is
  *frame dragging*: spacetime itself is dragged around in the direction of
  the black hole's spin, and nothing (not even light) can stay at a fixed
  `phi` arbitrarily close to the hole.
- **`Delta = 0` at `r = r_+ := M + sqrt(M^2 - a^2)`** — the **event
  horizon**. `g_rr = Sigma/Delta` blows up there: this is a *coordinate*
  singularity (spacetime itself is perfectly smooth at the horizon; the BL
  coordinate system just breaks down there), but it means a simulation
  using BL coordinates cannot be evolved through or even very close to the
  horizon — see [section 6](#6-a-second-coordinate-system-kerr-schild) for
  why this matters and how it's fixed.

### 1.2 The 3+1 split

To evolve *anything* forward in time, it's useful to explicitly split
spacetime back into "space at each instant" plus "how instants are
stitched together" — the **3+1 (ADM) formalism**. Any metric can be
written as:

```
ds^2 = -alpha^2 dt^2 + gamma_ij (dx^i + beta^i dt)(dx^j + beta^j dt)
```

where Latin indices `i,j` run over the 3 spatial coordinates only. Three
new objects appear:

- **Lapse `alpha`**: how much *proper time* elapses for an observer who
  stays at fixed spatial coordinates, per unit of coordinate time `dt`.
  This observer is called the **ZAMO** (zero angular momentum observer) or
  **Eulerian/normal observer**.
- **Shift `beta^i`**: how much that same observer's spatial coordinates
  drift, per unit coordinate time, *relative to a "non-rotating" frame*
  (this is exactly where frame-dragging shows up: `beta^phi != 0` for
  Kerr).
- **Spatial metric `gamma_ij`**: the metric restricted to a single
  instant-in-time spatial slice — an actual 3D Riemannian metric, used to
  measure real physical distances and angles within that slice.

For Boyer-Lindquist Kerr, comparing to the metric above: `gamma_rr=g_rr`,
`gamma_thth=g_thth`, `gamma_phiphi=g_phiphi`, `beta^r=beta^theta=0`,
`beta^phi = g_tphi/g_phiphi`, and `alpha^2 = Sigma*Delta/A`. The spatial
metric is **diagonal** — `r`, `theta`, `phi` don't mix — which turns out to
matter a lot (see section 6).

---

## 2. Relativistic hydrodynamics: the physics of the fluid

### 2.1 The stress-energy tensor

A **perfect fluid** (no viscosity, no heat conduction — the standard
starting assumption, and what this solver models) is completely described
at each point by its rest-frame density `rho`, specific internal energy
`eps` (internal energy per unit rest mass), pressure `P`, and 4-velocity
`u^mu` (tangent to the fluid element's worldline, normalized `u^mu u_mu =
-1`). Its stress-energy tensor is:

```
T^mu_nu = rho*h*u^mu*u_nu + P*delta^mu_nu
```

where `h := 1 + eps + P/rho` is the **specific enthalpy**. This one tensor
encodes energy density, momentum density, and momentum flux (stress) all
at once — it's the source term on the right-hand side of Einstein's
equations, but here the spacetime is *fixed* (this is "test fluid"
hydrodynamics on a background metric, not full numerical relativity — the
fluid's own gravity is neglected relative to the black hole's), so we only
need the fluid's own conservation law:

```
grad_mu T^mu_nu = 0        (4 equations: energy + 3 momentum)
grad_mu (rho u^mu) = 0     (rest-mass/baryon-number conservation)
```

`grad` is the covariant derivative (the "correct", metric-aware
generalization of a partial derivative, which properly accounts for the
fact that basis vectors themselves twist and stretch as you move through
curved coordinates).

### 2.2 Equation of state

To close the system (5 unknowns — `rho`, 3 velocity components, `P` — need
one more relation beyond the 4 conservation laws + normalization), this
solver uses an ideal-gas **Gamma-law EOS**:

```
P = (Gamma-1) * rho * eps  <=>  h = 1 + Gamma*eps = 1 + Gamma*P/((Gamma-1)*rho)
```

with `Gamma=4/3` for the torus (a relativistic-gas-like adiabatic index).

### 2.3 Why "conservative" variables

Directly discretizing `grad_mu T^mu_nu=0` in the primitive variables
`(rho, v^i, P)` is possible but numerically fragile: shocks (genuine
discontinuities) don't have well-defined derivatives, so a scheme written
in terms of derivatives of the primitives can produce wrong jump
conditions or spurious oscillations at a shock. The fix, standard since
the 1950s for ordinary (non-relativistic) gas dynamics and carried over to
GR by the **Valencia formulation** (Banyuls et al. 1997; the specific
GR-with-source-terms version used here follows Gammie, McKinney & Toth
2003, "HARM"), is to write the SAME equations as an exact divergence of
some *conserved* quantity, which a finite-volume scheme can handle exactly
via flux differencing — even across a discontinuity — as long as you know
the flux.

---

## 3. The Valencia formulation, derived

Everything in this section is implemented in `kernels_kerr2d.hpp`, and
every formula was verified two ways before being trusted: (a) an
independent CPU (`main.cpp`) or Python (`tools/kerr_schild_ref.py`)
re-implementation cross-checked bit-for-bit against the GPU, and (b) at
least one physically-meaningful invariant (energy/angular-momentum
conservation, round-trip recovery, a known circular orbit) checked to
near machine precision. This mattered in practice — see
[section 8](#8-a-worked-example-of-getting-it-wrong-and-catching-it).

### 3.1 The Eulerian (3+1) velocity

Define `W := -n_mu u^mu` where `n^mu` is the ZAMO observer's own
4-velocity — `W` is exactly the special-relativistic Lorentz factor *as
measured by the ZAMO*. The primitive velocity `v^i` (what the code calls
`vr, vth, vphi`) is defined by:

```
v^i := u^i/(alpha*u^t) + beta^i/alpha  <=>  u^i = W*(v^i - beta^i/alpha)
u^t = W/alpha
```

Substituting into the normalization `u^mu u_mu = -1` gives the clean
result `W = 1/sqrt(1 - v^2)`, where `v^2 := gamma_ij v^i v^j` — literally
the special-relativistic formula, just with `v^2` computed via the (possibly
non-diagonal) *spatial* metric instead of a flat Euclidean one. This is
the sense in which `v^i` is "the physical velocity `u^i` measured by a
local, non-rotating, momentarily-static observer": in a diagonal metric
(true for Boyer-Lindquist), `v^i` further factors into a clean per-axis
"physical" velocity — but for a **non-diagonal** spatial metric (true for
Kerr-Schild — see section 6), it doesn't factor that way, and `v^i` must
be treated as a genuine 3-vector via the full quadratic form. Getting this
distinction wrong is exactly the bug described in section 8.

### 3.2 Conserved variables

Define the conserved variables as the ZAMO-observer's measured mass,
momentum, and energy density, each weighted by `sqrt(-g)` (the square root
of minus the determinant of the *full 4-metric* — not the 3-metric; see
the callout box below):

```
D    := sqrt(-g) * rho * u^t                              (rest mass)
S_i  := sqrt(-g) * T^t_i = sqrt(-g) * rho*h*u^t*u_i        (momentum, i=r,theta,phi)
tau  := -sqrt(-g)*T^t_t - D = sqrt(-g)*rho*h*u^t*E - sqrt(-g)*P - D   (energy, minus rest mass)
```

where `E := -u_t` is the specific energy-at-infinity (a fluid element's
energy per unit rest mass, as measured by a static observer at infinity —
conserved along a geodesic in any *stationary* spacetime, i.e. one where no
metric component depends on `t`). `sqrt(-g) = Sigma*sin(theta)` for Kerr,
in **both** Boyer-Lindquist and Kerr-Schild coordinates (this identity is
coordinate-independent).

> **A real, previously-made bug**: an earlier version of this code used
> `sqrt(gamma) = sqrt(g_rr*g_thth*g_phiphi)` (the 3-metric determinant, only
> valid for a diagonal spatial metric) instead of `sqrt(-g)`. The two differ
> by exactly the lapse (`sqrt(-g) = alpha*sqrt(gamma)`, a standard ADM
> identity) and happen to coincide *exactly at the equator* for Kerr in BL
> coordinates (where the discrepancy is invisible), but diverge off it —
> breaking mass conservation subtly enough that it needed a from-scratch
> divergence-theorem residual check against the analytic torus to catch, not
> any shape-level check.

`D`, `L := S_phi`, and `tau` are all **exactly source-free** — no numerical
approximation, an exact statement following from Noether's theorem: `t` and
`phi` are both **Killing vectors** of the Kerr metric (no metric component
depends on `t` or `phi`, in *either* coordinate system used here), so the
corresponding conserved currents (`D`, always; `L`, `tau`/`E`, whenever `t`
and `phi` are both Killing) pick up no source term at all. Only `S_r` and
`S_theta` need one (section 3.4) — `r` and `theta` are not symmetry
directions.

### 3.3 Fluxes

The conservation laws, in flux-conservative form (one radial and one polar
flux, since the grid is 2D in `r, theta`):

```
d/dt(D, S_r, S_theta, S_phi, tau) + d/dr F^r(...) + d/dtheta F^theta(...) = source
```

with (again, sourced directly from `T^mu_nu`, weighted by `sqrt(-g)`):

```
F^r_D      = sqrt(-g) * rho * u^r
F^r_{S_i}  = sqrt(-g) * (rho*h*u^r*u_i + P*delta^r_i)
F^r_tau    = sqrt(-g) * rho*u^r*(h*E - 1)
```

(and the analogous `F^theta` with `u^theta` in place of `u^r`). These are
implemented once, generically, in `sideState()` (`kernels_kerr2d.hpp`), and
called by both flux-direction passes.

### 3.4 Source terms

For the two non-Killing directions, the source term for the covariant
momentum equation reduces (a standard result for a *stationary* metric —
no explicit time dependence) to a remarkably compact form:

```
Source_{S_i} = sqrt(-g) * 0.5 * sum_{mu,nu} T^{mu nu} * d(g_mu_nu)/dx^i
```

— a sum over *every* independent metric component (5 for Boyer-Lindquist:
`tt, tphi, rr, thth, phiphi`; 7 for Kerr-Schild, which has two more nonzero
off-diagonal terms — see section 6) and its derivative in the `i`
direction, weighted by the corresponding contravariant stress-energy
component. The `r-theta` cross term some might expect from the fully
general formula is identically absent in both coordinate systems, for the
same reason in each: `g_r_theta = 0` everywhere, so its derivative (the
only way a `T^{r theta}` term could contribute) is identically zero, not
just small.

The code computes these metric derivatives by **numerical (central
finite) differencing of the metric itself** (`kernels_kerr2d.hpp`'s
`sourceTerms()`), not a hand-derived symbolic Christoffel-symbol formula.
This is a deliberate choice, carried over from earlier phases of this
project: Kerr's off-diagonal structure makes by-hand Christoffel algebra
unusually error-prone, and the metric formula itself is the one thing not
in dispute (a textbook definition, not a derived quantity) — so
differentiating it numerically removes an entire class of transcription
risk, at the cost of a small, well-controlled truncation error (step size
`~1e-5`, chosen well inside float32's precision floor without being so
small that cancellation error dominates).

---

## 4. Primitive recovery (con2prim)

The evolution advances the **conserved** variables; every other quantity
(pressure, velocity, the metric-dependent stuff needed for the next
flux/source evaluation) needs `rho, v^i, P` recovered from `D, S_i, tau`.
This inverse map has no closed form in general (it's a genuinely nonlinear
system) and is solved by a 1D Newton iteration on the specific enthalpy
`h`, following the same overall strategy as most production GRMHD codes'
"u2p" (conserved-to-primitive) solvers, e.g. Noble et al. (2006).

**Setup.** Define `kappa_i := S_i/D`. Since `S_i = sqrt(-g)*rho*h*u^t*u_i`
and `D = sqrt(-g)*rho*u^t`, this is exactly `kappa_i = h*u_i` — true
regardless of the metric's structure. Define
`K := gamma^ij kappa_i kappa_j` (contracting with the *inverse* spatial
metric — a diagonal sum of 3 terms for Boyer-Lindquist; a genuine 2x2
contraction for Kerr-Schild's non-diagonal `r-phi` block, section 6). A
short derivation from the normalization condition gives the clean relation
`W = sqrt(1 + K/h^2)` **only in the case `beta_r = 0`** (true for
Boyer-Lindquist) — see section 6 and 8 for the general case, which needs
more care.

**Iteration.** Given a trial `h`, get `W`, then `rho = D*alpha/(sqrt(-g)*W)`
(from the `D` definition), then `P` from the EOS, then a residual built
from the `tau` definition:

```
f(h) := (tau + D + sqrt(-g)*P) / (D*h) - E(h)
```

(using `tau+D+sqrt(-g)*P = D*h*E`, algebra directly from section 3.2's
definitions) and Newton-step `h` toward `f(h)=0`, using a centered finite
difference for `f'(h)` (cheap, avoids a second hand-derivation of the
analytic derivative) and a **damped step** (clamped to at most 50% of the
current `h` per iteration — a standard con2prim safeguard, since a bad
local derivative estimate can otherwise send `h` to a nonsensical value in
one step).

**Safeguards, each added in response to an actual observed failure, not
speculatively:**
- **Float32-safe `h` floor** (`h >= 1+1e-4`, not the mathematically tighter
  `1+1e-9`): float32 has ~1.19e-7 ULP spacing near `h=1`, so a smaller floor
  silently rounds to exactly `1.0`, making `P` exactly `0` — a degenerate
  state that was traced (replaying saved simulation frames) to precede
  every observed blowup in an earlier version.
- **Physicality pre-check.** As `h -> infinity`, the residual `f(h) ->
  -alpha/Gamma < 0` always (density saturates, so pressure grows too
  slowly to keep `f` positive) — so a root in `[1, infinity)` exists at all
  only if `f(h=1) >= 0`. When it doesn't (the conserved state is
  momentarily, genuinely unphysical — this happens, especially at a fluid's
  own low-density surface, where a first-order-accurate flux's diffusion
  can push a marginal state slightly over the line every step), the fixup
  is a **minimal correction**: rescale the momentum components down by the
  smallest factor (found by bisection) that restores physicality, rather
  than a full reset to vacuum (which was tried first and made things much
  worse — see section 8).
- **Two floors on the recovered primitive state**: a hard **density floor**
  (below which the cell is reset outright to vacuum-at-rest) and a softer,
  density-*scaled* **entropy floor** `P_floor = const * rho^Gamma` (as
  opposed to a fixed additive floor, which was tried first and caused an
  85% spurious mass *increase* — a floor that doesn't scale with density
  perturbs the physics badly wherever density is low).

---

## 5. The Riemann solver and wave speeds

### 5.1 HLLE

At each cell interface, the flux is computed with the **HLLE** approximate
Riemann solver (Harten-Lax-van Leer, Einfeldt's 1988 refinement) — cheaper
than an exact or characteristics-based solver, and standard for GRHD/GRMHD
codes precisely because it needs *only* an estimate of the leftmost and
rightmost signal speeds (`sL, sR`), not the full eigenstructure:

```
F_HLLE = (sR*F_L - sL*F_R + sL*sR*(U_R - U_L)) / (sR - sL)
```

### 5.2 The wave-speed bound: two choices, and why the tighter one matters

The simplest safe choice for `sL, sR` is the coordinate speed of light in
that direction — always an upper bound on any physical signal. This
solver briefly used exactly that, and it turned out to be the single
biggest problem: it introduces numerical dissipation *proportional to the
gap between the true (sound) speed and the bound used*, and light speed
can be dramatically larger than the fluid's actual sound speed. The fix is
the standard **Marti & Muller / Anile** multi-dimensional acoustic
characteristic-speed formula — still conservative (an upper bound on the
*fluid's* signal speed, not light speed), but far tighter:

```
cs^2 = Gamma*P/(rho*h)                                  (relativistic sound speed, ideal-gas EOS)

lambda_+- = [ v_x(1-cs^2) +- cs*sqrt((1-v^2)*[(1-v^2*cs^2) - v_x^2*(1-cs^2)]) ] / (1 - v^2*cs^2)
```

for a wave propagating in direction `x` through a fluid with velocity
component `v_x` along that direction and total speed `v^2` (transverse
components folded in only through the total, never individually — this
formula only ever needs `v_x` and `v^2`). This is valid **in a local
orthonormal (locally flat) frame** — which is exactly what the ZAMO-frame
velocity of section 3.1 already is, so the special-relativistic formula
applies with no further GR-specific derivation needed for this part. It
was verified (`tools/wave_speed_check.py`) against a from-scratch
*symbolic* flux-Jacobian eigenvalue computation (SymPy), independent of
this formula, matching to `1e-15`.

Converting the local-frame speed to a coordinate speed needs the lapse and
the appropriate metric scale factor — and, if the shift in that direction
is nonzero, a shift correction (section 6.3). The result is always clamped
to the (always-safe, if more dissipative) photon-speed bound as a
defensive backstop — the tight formula is trusted to be *tighter*, not
trusted to *never need the fallback*.

### 5.3 Reconstruction

Feeding raw cell-center values into the Riemann solver (no interpolation
to the interface at all) is formally first-order accurate and was, in
fact, the single largest departure from the reference algorithm this
scheme's source terms are modeled on (original HARM used piecewise-linear
reconstruction from its very first version). The fix is standard
**piecewise-linear reconstruction with a MinMod slope limiter**: each
primitive is linearly extrapolated a half-cell toward each face, using a
slope built from the **MinMod** limiter —

```
minmod(a, b) := 0                     if a*b <= 0  (a local extremum -- don't extrapolate through it)
              sign(a)*min(|a|,|b|)    otherwise
```

— applied to the two one-sided differences around each cell. MinMod is
the most diffusive of the standard limiters (more so than, say, MC or
PPM) but has the property that matters most here: it is **TVD**
(total-variation-diminishing) — the reconstructed face value is
mathematically guaranteed to never exceed the range spanned by the cells
it was built from, so it cannot manufacture a new negative density or
pressure at a steep edge by itself.

### 5.4 Time integration

A standard 2nd-order (Heun's method / midpoint) **RK2** predictor-corrector:
evaluate the full right-hand side (recover primitives, reconstruct,
compute fluxes and sources) once at the current state to get a
predicted state; evaluate it again at the predicted state; average the
two. Each of the two stages is a complete pass through every kernel
described above.

---

## 6. A second coordinate system: Kerr-Schild

### 6.1 The problem Boyer-Lindquist coordinates create

Section 1.1 noted that `g_rr` diverges at the horizon in BL coordinates —
a coordinate artifact, but a real practical one: a simulation grid using
BL coordinates cannot extend through, or even very close to, the horizon.
In practice this forces an artificial **inner radial boundary** somewhere
*outside* the horizon, well inside any accretion flow that's actually
happening there. Material genuinely flowing inward (falling toward the
black hole, exactly as it should) has nowhere to go once it reaches that
artificial cutoff — the boundary condition there (a standard
"outflow"/zero-gradient treatment) has to represent "this material
leaves the domain," but with an ordinary Eulerian grid and a coordinate
system that becomes pathological nearby, it can't drain it fast enough
once the flux scheme is accurate enough to actually let inflow reach that
edge. The practical symptom (see section 8): real, non-vacuum material
piling up at the artificial boundary until the scheme diverges.

The standard fix, used by essentially every modern GRMHD production code
(HARM among them) for exactly this reason, is a **horizon-penetrating**
coordinate system: one where the metric stays perfectly regular at (and
inside) the horizon, so the domain's inner edge can sit at or inside `r_+`
with no coordinate pathology nearby, and infalling material can be
followed smoothly across it rather than needing an artificial cutoff at
all.

### 6.2 The metric

**Kerr-Schild (KS)** coordinates keep `r` and `theta` **identical** to BL —
the existing grid, IC sampling radii, and diagnostics all carry over
unchanged — and redefine only time and azimuth via:

```
dt_KS   = dt_BL   + (2*M*r/Delta) dr
dphi_KS = dphi_BL + (a/Delta)     dr
```

Substituting this into the (already-trusted) BL metric and collecting
terms (done symbolically in `tools/kerr_schild_derive.py`, not by hand)
gives the KS metric:

```
g_tt     = -(1 - 2Mr/Sigma)                              g_tr = 2Mr/Sigma
g_tphi   = -2Mar*sin^2(theta)/Sigma
gamma_rr = 1 + 2Mr/Sigma                                 gamma_rphi = -a*sin^2(theta)*(1+2Mr/Sigma)
gamma_thth = Sigma                                        gamma_phiphi = sin^2(theta)*[Sigma+a^2*sin^2(theta)*(1+2Mr/Sigma)]
```

The key structural change: **two new nonzero off-diagonal terms**, `g_tr`
and `gamma_rphi` — KS's spatial metric is genuinely **non-diagonal**. The
3+1 split works out (also verified symbolically) to:

```
alpha^2 = Sigma/(Sigma+2Mr)          beta^r = 2Mr/(Sigma+2Mr)          beta^theta = beta^phi = 0
```

`alpha^2` is manifestly positive for **any** `r > 0` — Sigma and `2Mr` are
both non-negative — so this metric has no coordinate singularity anywhere
outside the true curvature singularity at `r=0`. (`beta^phi = 0` despite
both `g_tphi` and `gamma_rphi` being individually nonzero is a genuinely
famous, slightly surprising KS property — frame-dragging is carried
entirely by the metric components here, not by an azimuthal shift.)

### 6.3 What breaks, and how it's fixed

A non-diagonal spatial metric invalidates every formula in sections 3-5
that implicitly assumed a diagonal one:

- **Velocity convention.** BL's `v^i` used to factor into a clean
  per-axis "physical velocity", `v_i_hat = sqrt(gamma_ii)*v^i`. That
  factoring only works for a diagonal metric. KS instead uses the fully
  general coordinate-frame Valencia `v^i` of section 3.1 directly (the
  *definition* `u^i=W(v^i-beta^i/alpha)`, `v^2=gamma_ij v^i v^j`, was
  always general — only the *simplified, diagonal-only* special case
  needs to be abandoned).
- **Covariant velocity.** `u_i = gamma_ij u^j` is *also* only the
  diagonal-metric special case of the true relation
  `u_i = gamma_ij u^j + beta_i*u^t` — the `beta_i*u^t` term is
  identically zero for BL (`beta_r=0` there) but not for KS. This is
  exactly the bug in section 8.
- **con2prim's `K`.** Now a genuine 2x2 contraction with the inverse
  spatial metric (`gamma^rr, gamma^rphi, gamma^phiphi` all needed, not
  just three independent diagonal terms), and — more subtly — the
  shortcut `W=sqrt(1+K/h^2)` no longer holds (it implicitly assumed
  `beta_r=0`); the general case needs `u^t` solved from a genuine
  quadratic derived from the full normalization condition (worked out in
  `tools/kerr_schild_ref.py`, verified against an independent round-trip
  check to `1e-14`).
- **Source terms.** Same general formula (section 3.4), just summed over
  KS's 7 nonzero metric components instead of BL's 5.
- **Wave speeds.** `r` and `phi` are no longer orthogonal directions, so
  the local-frame `x`-velocity for the radial Riemann problem needs a
  genuine Gram-Schmidt projection (`v_r_hat = sqrt(gamma_rr)*v^r +
  (gamma_rphi/sqrt(gamma_rr))*v^phi`) rather than a simple per-axis
  scaling, and the coordinate-speed conversion picks up a `-beta^r` term
  (the ZAMO observer's own coordinate-r drift, absent when `beta^r=0`).
  The radial *photon* speed bound also becomes **asymmetric**
  (`g_rr*v^2 + 2*g_tr*v + g_tt = 0` has two different roots) — exactly
  the point of a horizon-penetrating slicing: ingoing and outgoing
  radial light rays genuinely travel at different coordinate speeds.
- **Initial data.** The validated Fishbone-Moncrief torus solution
  (`tools/fishbone_moncrief.py`) is naturally expressed in BL-orthonormal
  primitives. `r, theta` carry over unchanged (section 6.2), but the
  4-velocity must be explicitly transformed (`BlToKsVelocity()` in
  `KerrTorusSim.cpp`, mirroring `tools/kerr_schild_ref.py`'s
  `bl_to_ks_velocity()`) via the same differential relations above.
  Verified two ways: the transformed state satisfies the KS
  normalization condition exactly, and — a much stronger, independent
  check — the specific energy `E=-u_t` and angular momentum `L=u_phi` are
  **exactly** preserved by the transform, both to machine precision. This
  isn't a coincidence: `t` and `phi` are the *same* Killing vectors in
  both coordinate systems (only their labeling/simultaneity convention
  changed), so their conjugate conserved quantities *must* be identical
  before and after — a clean, independent invariant this is genuinely
  ***required*** to satisfy, not just observed to.

---

## 7. Implementation

### 7.1 File map

| File | Role |
|---|---|
| `src/kernels_kerr2d.hpp` | GLSL compute-shader source (as C++ string builders) for every pass: primitive recovery, reconstruction, fluxes, source terms, RK2 combine. This is the physics from sections 3-6, transliterated. |
| `src/KerrTorusSim.{hpp,cpp}` | The C++ driver: owns the GPU buffers, compiles the shaders, runs the RK2 step sequence each substep, uploads/transforms initial data, and reads back diagnostics. |
| `src/main.cpp` | CLI entry point; also hosts `SelfTestKerrTorus()`, an independent double-precision CPU re-implementation of every formula above, cross-checked against the GPU every `--selftest` run. |
| `tools/fishbone_moncrief.py` | The analytic Fishbone-Moncrief torus solution (independent of everything else — pure GR orbital mechanics). |
| `tools/make_fm_torus_ic.py` | Samples that analytic solution onto the simulation grid, writing the raw initial-condition binary. |
| `tools/kerr_schild_derive.py` | Symbolic (SymPy) derivation of the KS metric from BL via the coordinate transform, and the ADM 3+1 split — the *first* thing verified, since everything else depends on it. |
| `tools/kerr_schild_ref.py` | Double-precision Python reference for KS kinematics, con2prim, source terms, and the BL->KS IC transform — verified via round-trip, direct normalization, and known-circular-orbit checks, all before being trusted in the shader. |
| `tools/wave_speed_check.py` | Derivation and verification of the HLLE wave-speed bound, both the local-frame formula (against a symbolic flux Jacobian) and the coordinate-speed conversion (against a photon-speed safety bound). |

### 7.2 The GPU pipeline (one substep)

Each substep runs the full sequence **twice** (RK2's predictor and
corrector stages), reading from one conserved-variable buffer and writing
to another (double-buffered so a partially-updated grid never feeds itself
mid-pass):

1. **`ConsToPrim`** — recover `(rho, v^r, v^theta, v^phi, P)` from
   `(D, S_r, S_theta, tau, L)` at every cell (section 4).
2. **`ComputeSlopes`** — MinMod-limited reconstruction slopes in both `r`
   and `theta`, from the primitives `ConsToPrim` just wrote (section 5.3).
3. **`FluxesR`, `FluxesTheta`** — reconstruct interface states, compute the
   HLLE flux at every `r`- and `theta`-interface (sections 3.3, 5.1, 5.2).
4. **`EulerStep`** — flux-divergence update plus the two geometric source
   terms (section 3.4), producing the next stage's conserved state.
5. **`Combine`** (after both stages) — average the two stages'
   conserved states, per the RK2 rule (section 5.4).

Buffer bindings, exact GLSL, and the full derivation-to-code mapping are
documented in `kernels_kerr2d.hpp`'s own header comment and per-function
comments — this document explains *why* each piece exists; the header
comment is the precise, authoritative statement of *what* each buffer
holds and *which* binding number it lives at.

### 7.3 Validation methodology

Every nontrivial formula in this codebase followed the same three-step
process, in order, and this document's own claims were checked the same
way:

1. **Derive symbolically or from first principles** (SymPy where
   algebra is involved — e.g. the BL->KS metric substitution, the
   con2prim quadratic), rather than recalling or pattern-matching a
   formula from a paper.
2. **Verify against an independent numerical invariant** *before* writing
   any shader code — a from-scratch flux-Jacobian eigenvalue computation,
   a direct 4-velocity normalization check, a round-trip
   prim->cons->prim, a known circular orbit, an exactly-conserved
   quantity under a coordinate transform. Never "does this look right",
   always "does an independent computation agree to near machine
   precision."
3. **Port to GLSL and cross-check the GPU against an independently
   re-implemented CPU version** (`main.cpp`'s `SelfTestKerrTorus()`), run
   automatically via `--selftest`.

This process caught several real bugs during development (not just
typos — genuine misunderstandings of the relevant physics), two of which
are worth walking through in detail, because they're the kind of mistake
that *looks* right on inspection.

---

## 8. A worked example of getting it wrong, and catching it

**The bug.** An early draft of the Kerr-Schild covariant velocity used

```
u_r = gamma_rr*u^r + gamma_rphi*u^phi        (WRONG once beta^r != 0)
```

by direct, superficially reasonable analogy with "lower an index with the
metric." This is correct for Boyer-Lindquist. It is not correct in
general: the actual relation is `u_i = g_i_mu u^mu = g_it*u^t + g_ij*u^j =
beta_i*u^t + gamma_ij*u^j` — an extra term, `beta_i*u^t`, that happens to
vanish for BL (where `beta_r=0`) but not for KS.

**How it was caught.** Not by re-reading the derivation more carefully —
by two independent numerical checks disagreeing:

- A **direct 4-velocity normalization check**
  (`g_mu_nu u^mu u^nu = -1`, computed straight from the metric and the
  *contravariant* `u^mu` components, never going through `u_i` at all)
  **passed** at machine precision.
- A **round-trip check** (`prim -> cons -> prim`, which necessarily
  exercises the buggy `u_i` formula, since the conserved momentum `S_i`
  is built from it) **failed** badly — recovered primitives off by
  order-1 relative error.

The combination was the diagnosis: whatever was wrong lived specifically
in the covariant-velocity step, not in the more basic kinematics. Adding
the missing `beta_i*u^t` term fixed the round-trip check to `1e-14`.

**The lesson**, stated generally because it recurred (a second, related
bug — an incorrect simplification of the con2prim quadratic that also
implicitly assumed `beta_r=0` — was caught the same way): a formula that
is exactly correct in a simpler special case (here, zero shift in the
radial direction) can look identical to, and be trivially mistaken for,
the fully general formula. The only reliable way to tell them apart is a
numerical check built to be sensitive to *exactly* the term in question —
which is why this project insists on at least one independent invariant
per nontrivial formula, not just "the selftest passes" (a selftest built
from the same subtly-wrong assumption would not have caught either bug).

---

## 9. Current status

**Working and validated:**
- The full Kerr-Schild reformulation (sections 3-6): all four selftests
  pass, GPU matching an independent CPU reference to `~1e-6`.
- Piecewise-linear MinMod reconstruction and the tight (sound-speed-based)
  HLLE wave-speed bound: both independently derived and verified before
  use, and both measurably improve on the schemes they replaced.
- At the original inner boundary radius (`r_min=4M`, unchanged from the
  Boyer-Lindquist deck), switching to Kerr-Schild coordinates removed the
  most severe, earliest failure mode (a near-immediate, domain-wide
  blowup) — the torus now evolves stably for roughly an order of
  magnitude longer before the remaining issue below appears.

**Open issue:** the simulation still eventually loses mass and
destabilizes at the inner radial boundary, later than before but not
resolved outright. Moving the boundary *inward* — past the true horizon,
now possible precisely because of this reformulation — was tried and made
things *worse*: a much faster, much more severe (near-whole-grid) failure,
distinct in character from the original slow pileup. This points to a
different, not-yet-understood problem specific to operating very close to
or inside the horizon (candidates not yet ruled out: the vacuum floor's
"at rest" state, which for a nonzero shift means "comoving with a ZAMO
observer that is itself falling inward"; or the source terms' finite-
difference step size behaving badly in a region of much larger metric
curvature) — a genuinely separate investigation from everything resolved
so far, not yet undertaken.
