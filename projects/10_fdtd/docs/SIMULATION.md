# 10_fdtd: how it works

Full-wave Maxwell in the time domain on a Yee grid -- the canonical
computational-electrodynamics scheme, and the flagship EM solver of the
three new projects (`09_magnetostatics`, this, `11_retarded_fields`).

**Scope vs the future Optics page.** This is *vector* Maxwell -- radiation,
antennas, scattering, guided modes, lossy/dispersive media. Scalar methods
(Helmholtz / beam propagation / Fresnel-Kirchhoff for interference,
diffraction, lensing) are the Optics page's.

Units: `c = eps0 = mu0 = 1`. The deck's `courant` is the full Courant
number `S = c dt sqrt(1/dx^2 + 1/dy^2)`, stable for `S <= 1` (the plan doc
used the alternative `c dt/dx <= 1/sqrt(2)` convention; the code uses the
dimension-agnostic one).

## 1. The Yee scheme (Phase 1)

TM^z: `E_z` at the integer nodes, `H_x` offset half a cell in `y`, `H_y`
offset half a cell in `x`; leapfrog in time (`E` on integer steps, `H` on
half steps).

```
dE_z/dt = (1/eps)(dH_y/dx - dH_x/dy - sigma E_z)
dH_x/dt = -(1/mu) dE_z/dy
dH_y/dt =  (1/mu) dE_z/dx
```

The `E` update is kept in the standard `ca`/`cb` coefficient form from the
start (`ca = (1 - sigma dt/2eps)/(1 + sigma dt/2eps)`, `cb = (dt/eps)/(1 +
sigma dt/2eps)`), so lossy dielectric media plug in with no code change.

Boundaries: a closed **PEC** box (`E_z = 0` on the edge -- the edge cells
are simply never written by the interior update), a first-order **Mur**
absorbing condition, or **CPML** (section 3).

**Validation** (`--selftest`, no GL context):

- **CFL**: `S = 0.99` is stable, `S = 1.02` produces `NaN` within a few
  thousand steps -- the stability limit is exactly `S = 1`.
- **Energy**: the exact conserved discrete energy (`H` sampled at the two
  staggered half-steps, `0.5 sum[eps E_z^2 + mu H_x(n-1/2)H_x(n+1/2) + ...]`,
  cached each step) is constant to `2e-14` over 12000 steps in a PEC box.
  The naive `0.5 sum(E^2 + H^2)` ripples ~2% with the leapfrog -- that is
  the wrong quantity, not a drift.
- **Speed**: a Gaussian pulse's peak, timed between two radii (the
  difference cancels the source turn-on delay), travels at `0.995 c` -- the
  FDTD numerical-dispersion deficit at ~17 cells/wavelength.

## 2. The GPU compute backend (Phase 1b)

Five compute shaders (`kernels_fdtd.hpp`): H update, E update, copy-Ez-to-
prev (for Mur), source injection (via a small `(idx, value)` SSBO), Mur.
`solver.backend = "gpu"` or `--gpu` selects it; uniforms and buffer
bindings are hoisted out of the substep loop.

`--gpu-selftest` steps the CPU (float64) and GPU (float32) paths with the
same deck and compares `E_z`: PEC box, 1200 steps -> relative `1.6e-6`;
Mur -> `2.8e-5` (the Mur-boundary field is `~1e-4`, so float32 relative
error is larger while the absolute diff stays `~1e-9`). ~5.5x faster than
the CPU path at `1024^2` headless; the readback-free interactive loop
gains more. The GPU path is still plain-TM^z + Mur -- any CPML / material /
TFSF deck with `--gpu` warns and runs on the CPU.

## 3. CPML (Phase 2)

`Cpml.hpp` builds the Roden-Gedney coordinate-stretch profiles per axis at
both node sets (integer for `E_z`, half for `H`): `sigma(depth) ~ (d/L)^m`,
`kappa(depth)`, a CFS `alpha` tapering to zero at the wall, and the
recursive-convolution coefficients `b = exp(-(sigma/kappa + alpha) dt)`,
`a = sigma(b-1)/(sigma kappa + kappa^2 alpha)`. `UpdateH` / `UpdateE` carry
four `psi` auxiliary fields. Away from the PML layer the coefficients are
trivial (`b = 1, a = 0, kappa = 1`), so `psi` stays zero and the update
reduces *exactly* to the plain Yee scheme -- one code path, no interior
branch.

`--cpml-selftest`: a 12-cell CPML reflects a broadband pulse at **-81 dB**
(measured as the difference between a short CPML domain and a long
reference domain that share their near-source region).

**Dipole / 2D Green's function** (`tools/plot_dipole.py`): a CW point
source's steady-state `E_z` phasor (single-bin DFT), radially averaged,
against the 2D scalar Green's function. **Convention gotcha**: the FDTD
`sin(w t)` source with an `e^{+i w t}` phasor extraction is the `e^{-i w t}`
convention, so the outgoing wave is `H_0^{(1)}`, not the `H_0^{(2)}` the
plan wrote. Match: 2% median amplitude error, **0.9997** complex
(amplitude + phase) correlation over `kr` in `[12, 38]`, reproducing both
the `1/sqrt(r)` asymptote and the linear `+kr` phase.

## 4. Materials, and the TFSF plane wave (Phase 3)

`Materials.hpp` rasterises `[[material]]` (`eps_r` / `mu_r` / `sigma`) and
`[[pec]]` tables onto the grid (slab / halfspace / box / cylinder shapes).
PEC cells force `E_z = 0` after each `E` update.

**TFSF** (total-field / scattered-field) injects a clean plane wave across
a rectangular contour so the scattered field is isolated outside it. The
first attempt evaluated the incident field analytically -- and leaked at
`~-15 dB`, because the analytic wave propagates at exactly `c` while the
FDTD grid has numerical dispersion, so the TF/SF cancellation drifts across
the box. The fix (`IncidentWave.hpp`) is the standard **1D auxiliary FDTD
grid**: a 1D scheme running the incident wave along the propagation
direction, with its spacing `dxi` solved from the 2D dispersion relation at
the incidence angle so the 1D wave has the *same* numerical phase velocity
as the 2D grid. That took the leak to **-276 dB at normal incidence,
-50 dB at 20 degrees off-axis**.

Two more bits bit: all four *H*-side TF/SF corrections had flipped signs
(the *E*-side four were right), and the 1D grid's `H_y ~ -E_z` for a
forward-propagating wave, so `IncidentH` negates it.

**s-polarisation Fresnel** (`--fresnel-selftest`, `tools/plot_fresnel.py`):
a TFSF plane wave onto a dielectric half-space at a range of incidence
angles. The reflected field is isolated by a **with/without-slab phasor
subtraction** on a row just above the interface (the reflected beam still
fully overlaps the incident footprint there -- a single fixed probe misses
the specular ray at large angles, and a scattered-field strip is
contaminated by the CPML's vacuum-tuned impedance mismatch when a
transmitted wave in the dielectric reaches it). `R(theta)` matches the
Fresnel coefficient to **< 2% from 0 to 70 degrees**; near-grazing degrades
(staircased interface, finite TFSF beam).

TE^z (p-polarisation, Brewster angle) is Phase 3b, not yet built -- s-pol
is already a complete single-polarisation check.

## 5. PEC scatterers and near-to-far-field (Phase 4)

`output.h_fields` writes `H_x`, `H_y` frames alongside `E_z`.
`tools/plot_mie.py` runs `cylinder_scatter.toml` with and without the PEC
cylinder, isolates the scattered phasor field, and does a **2D
near-to-far-field transform** on a square contour: equivalent electric
current `J_z = (n x H)_z`, magnetic `M_phi = (n . phi_hat) E_z`, integrand
`J_z - M_phi` (the relative sign flips the forward and backward lobes --
`J_z + M_phi` gave the pattern mirrored).

The bistatic pattern matches the 2D PEC Mie series
`|sum_n J_n(ka)/H_n^{(1)}(ka) e^{i n phi}|^2` to a **0.995 pattern
correlation** at `ka ~ 3.14` -- the forward-scatter lobe, the sidelobe
plateau, and the nulls all reproduced.

`tools/plot_antenna.py`: a centre-fed half-wave PEC dipole radiates a
correct broadside figure-8 with deep axis nulls; `~0.89` correlation to the
idealised `|cos((pi/2) sin phi)/cos phi|^2` -- the gap is the real 2D
fat-strip vs 3D thin-filament difference, not a numerical error.

`--pec-selftest`: the PEC cylinder casts a forward shadow (`|E_z|` 0.15
behind vs 1.4 on the lit face) and leaves a scattered field in the SF
strip.

## 6. The interactive view (Phase 5)

`--interactive` opens `fw::SimApp`: a fullscreen-triangle fragment shader
samples the chosen scalar from an SSBO -- `E_z` (zero-centred diverging
map), `E`-energy, or `|S|` (Poynting magnitude, magma) -- cycled with `M`,
auto-scaled, with a material overlay (dielectric tint, PEC solid). It runs
on the **CPU backend**, which is fast enough (`~0.4 ms/step` at `256^2`,
`~38 steps/frame` inside a 60 fps budget), and a CPML / material / TFSF
deck can't use the GPU path yet anyway. `--render-check <png>` renders one
offscreen frame after warming the sim.

## 7. What further extension would need

- **GPU CPML + materials**: the four `psi` fields and per-cell `ca`/`cb`
  are all local -- direct compute-shader ports -- but the profile arrays
  and the reduced-branch structure need care. This is what a truly large
  interactive domain (several million cells) needs.
- **TE^z**: a second full set of update equations (`H_z, E_x, E_y`),
  needed for p-polarisation Fresnel and the Brewster angle.
- **Dispersive media** (Drude / Lorentz / Debye): an ADE or recursive-
  convolution term in the `E` update -- the `ca`/`ca` structure already
  isolates where it goes.
- **A 1D-aux-grid CPML tuned to `sqrt(eps)`** so a Fresnel / scattering
  measurement can run to full steady state without the transmitted wave's
  echo off a vacuum-tuned PML contaminating it.
