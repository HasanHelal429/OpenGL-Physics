# 11_retarded_fields: how it works

The Liénard-Wiechert fields of moving point charges -- the exact solution of
Maxwell's equations for a point source on an arbitrary worldline -- taken in
two directions the numba-parallel Python original (`Physics
Simulations/Electromagnetism/Retarded_Fields`) can't reach comfortably:

1. **GPU field-grid evaluation** (Phases 1-2). The per-point retarded-time
   root-find is an ideal compute-shader workload, so the radiated-field
   visualisation becomes interactive -- drag the charge, sweep its orbit
   frequency, watch the pattern reshape live.
2. **Self-consistent N-charge dynamics** (Phases 3-4). Charges move under
   each other's *retarded* fields, integrated with a relativistic pusher, so
   radiative inspiral, bremsstrahlung and beaming emerge from the dynamics.

Units: `c = eps0 = mu0 = 1`.

## 1. Retarded time and the Liénard-Wiechert fields (Phase 1)

For an observation event `(x, t)` and a source path `r(tau)`, the retarded
time solves `c (t - t_ret) = |x - r(t_ret)|`, `t_ret <= t`. `LwFields.hpp`
ports `retarded_fields.py`'s safeguarded solve line for line: Newton with
the analytic derivative `g'(t_ret) = -c + n.v`, a maintained bracket, and a
bisection step whenever Newton would leave it. The bracket's lower end is
grown geometrically until `g` changes sign.

The field is the standard velocity (near) + acceleration (radiation) split,
evaluated at `t_ret`:

```
E = q/(4 pi eps0) [ (n - beta)(1 - beta^2) / (kappa^3 R^2)
                    + n x ((n - beta) x betadot) / (c kappa^3 R) ],
kappa = 1 - n.beta ,   B = n x E / c .
```

`ChargePath.hpp` supplies the analytic prescribed paths (static, uniform,
circular, linear oscillator, figure-8) with the exact `r`, `v`, `a`.

**Validation** (`--selftest`, no GL context):

- **Static charge**: the field collapses to Coulomb `q rhat / 4 pi eps0 r^2`
  to `2e-16`, with `|B|` identically zero.
- **Uniform motion** (`beta = 0.3`): matches the boosted-Coulomb closed form
  `E = q(1 - beta^2) Rhat_now / [4 pi eps0 (1 - beta^2 sin^2 psi)^{3/2}
  R_now^2]` to `5e-16` -- the retarded solve and the present-position
  formula agree exactly.
- **Larmor**: a non-relativistic circular charge, Poynting flux over a
  Fibonacci sphere at 4 wavelengths, matches `P = q^2 a^2 / 6 pi eps0 c^3`
  to `8e-7`.

## 2. The GPU compute backend (Phase 1)

`kernels_lw.hpp` is one compute shader: per grid cell, loop the charges,
evaluate the analytic path in-shader, run the same Newton+bisection retarded
solve, accumulate `E`, `B`. The retarded time per (cell, charge) is written
back to an SSBO as a warm-start for the next frame (retarded time varies
slowly and continuously over a fixed grid, so this collapses the iteration
count after frame 0).

`--gpu-selftest` steps the CPU (float64) and GPU (float32) paths with the
same deck and compares `E` in the radiation zone (masking the near-field
`1/R^2` spike, where fp32 amplifies tiny retarded-time differences without
bound): relative L2 `~1e-3` -- fp32 shader vs fp64 reference.

## 3. The interactive prescribed view (Phase 2)

`--interactive` opens `fw::SimApp`: a fullscreen-triangle fragment shader
samples the chosen scalar from an SSBO -- `|E|` / `Ex` / `Ez` / radial
Poynting, magma or diverging map, log or linear -- with pan / zoom / gain /
gamma. Arrow keys drag the active charge; `,`/`.` sweep its orbit frequency,
`;`/`'` its radius; `Tab` cycles charges; a `gamma_max` readout tracks the
fastest charge.

**r-weight mode** (`R`, or `[render].r_weight`): multiply the displayed
field by distance from the source centroid and blank a small disk around
each charge, so the radiation pattern reads instead of the near-field spike.
Normalisation then uses the 99.5th field percentile rather than the raw max.
`--render-check <png>` renders one offscreen frame after warming the sim.

## 4. Trajectory buffer + self-consistent dynamics (Phase 3)

`TrajectoryBuffer.hpp` is a fixed-step ring buffer of `(r, v, a)` per charge
with cubic Catmull-Rom lookup in `t`, spanning at least the domain's
light-crossing time so every in-domain retarded query lands inside it. A
buffer-backed `PathFn` plugs straight into the same `LwFields()` used for
the prescribed paths.

**Startup**: `Prefill()` lays down an assumed steady past -- uniform drift,
a fixed point, or a Newtonian circular orbit -- so forces are defined from
`t = 0`. The `kepler` seed builds a two-body bound circular orbit from
`[mode].separation`: `omega^2 = k |q1 q2| (m1 + m2) / (d^3 m1 m2)`,
`k = 1/4 pi`, each charge on its COM-frame circle.

**Step**: for each charge, sum the *retarded* Liénard-Wiechert field of
every *other* charge (pairwise; the Abraham-Lorentz self-force is
pre-acausal / runaway and deliberately omitted), then a relativistic Boris
push (`Pusher.hpp`, state carried as proper velocity `u = gamma v`), then
append the new `(r, v, a)` sample. The relativistic Larmor power of each
charge is integrated for the energy budget.

`Snapshot` writes `KE`, `PE_interaction` (instantaneous Coulomb),
`E_radiated`, `E_total`, `energy_error`, `separation`, and per-charge
`x`/`y`/`gamma`/`speed` to `diagnostics.csv`.

**Gate** (`--dynamics-selftest`): a `kepler`-seeded equal-mass `+q / -q`
pair holds shape over 3 orbital periods -- separation stays in `[5.68,
6.04]` about `d0 = 6` (a monotonic radiative decay, not a buffer-pumped
oscillation) -- and the energy budget `KE + PE + radiated` closes to
`max |dE/E0| = 1.9e-3`.

The self-consistent field grid is evaluated on the CPU (the GPU kernel is
prescribed-path only). At `360^2` with a couple of charges that is `~5 s`
for 30 headless frames, `~6 fps` interactive.

## 5. Radiation-physics validation (Phase 4)

- **`--beaming-selftest`** -- a circular charge at `gamma in {2..12}`.
  Straight from the retarded-quantity angular distribution
  `dP/dOmega ~ |n x ((n - beta) x betadot)|^2 / (1 - n.beta)^5`: the
  forward-lobe half-max half-angle times `gamma` is constant to 8%
  (approaching `~0.31`), and the sphere-integrated power exponent is
  **4.000** (`P ~ gamma^4`).

- **`--brems-selftest`** -- a light charge deflected past a fixed like
  charge, self-consistent. Radiated energy vs impact parameter, log-log
  slope **-2.86** -- the weak-deflection Coulomb-bremsstrahlung `b^-3` law.

- **`--inspiral-selftest`** -- a `kepler` pair over 4 periods. The
  mechanical `dE/dt` tracks the integrated Larmor `dE_rad/dt` to **2.6%**
  (the budget closes -- the real test of the buffer + retarded force). The
  absolute rate is `~0.55 P_dipole`: the pairwise retarded interaction
  *without* the self-force carries exactly half of the symmetric pair's
  radiation reaction (the other half is the self-force each particle exerts
  on itself, which is out of scope). `tools/inspiral_ref.py --half` matches.

`tools/`: `larmor_ref.py` (analytic `dP/dOmega`, relativistic and not),
`array_factor_ref.py` (N-element array factor), `plot_pattern.py` (angular
pattern from headless frames -- reproduces the `sin^2 theta` doughnut and
the two-element array factor at a few wavelengths out), `plot_inspiral.py`
(trajectories, separation vs `inspiral_ref`, energy budget),
`make_movie.py`.

## 6. What further extension would need

- **GPU self-consistent grid**: upload the trajectory buffers as SSBOs and
  interpolate them in `kernels_lw.hpp` -- the retarded solve is already
  there; only the path sampling changes.
- **The Abraham-Lorentz-Dirac self-force**, in the Landau-Lifshitz reduced
  (non-runaway) form, would close the inspiral rate to the full dipole
  power and let a single charge spiral in on its own.
- **Higher-order history interpolation** (quintic, or Hermite on stored
  `v`) for very long-baseline retarded lookups -- `Studies/retarded_fields/
  history_convergence` fixes where cubic is enough.
