# projects/10_fdtd — Design Plan

## Context

Full-wave Maxwell in the time domain. The Yee-grid FDTD method is the
canonical computational-electrodynamics scheme and the archetypal GPU
stencil workload: every cell update is local, identical and explicit, so
large domains run in real time on compute shaders. This is the flagship EM
solver of the three new projects.

**Scope boundary with the future Optics page.** FDTD here covers *vector*
Maxwell — radiation, antennas, scattering, guided modes, lossy media. The
Optics page uses *scalar* methods (Helmholtz / beam propagation /
Fresnel-Kirchhoff) for interference, diffraction and lensing. Different
equations, different code; a shared GRIN-lens teaser deck is the only
deliberate overlap.

House model: `fw::Simulation` + TOML deck, headless batch + `fw::SimApp`
interactive, `tools/*.py` for analytic references and plots.

## Numerical approach

### Grid and update (`Fdtd2D`)

2D, **TMz** polarization first (`Ez`, `Hx`, `Hy` — one electric component,
the simplest working solver):

```
∂Ez/∂t = (1/ε)(∂Hy/∂x − ∂Hx/∂y − σ Ez)
∂Hx/∂t = −(1/μ) ∂Ez/∂y
∂Hy/∂t =  (1/μ) ∂Ez/∂x
```

Yee staggering in space, leapfrog in time (`E` on integer steps, `H` on
half steps). CFL: `dt ≤ 1 / (c √(1/dx² + 1/dy²))`; the deck carries a
Courant factor `S ≤ 1/√2`.

**TEz** (`Hz`, `Ex`, `Ey`) is added as a deck flag once TMz is validated —
needed for the full Fresnel picture (p-polarization, Brewster angle).

Materials: per-cell `ε_r(x,y)`, `μ_r(x,y)`, `σ(x,y)` textures; update
coefficients precomputed per cell in the standard `ca`/`cb` form. PEC
scatterers hold the tangential E components at zero in masked cells.

### Sources

- **Soft source** — additive `Ez += f(t)` at a point/line. Gaussian pulse
  (broadband, for spectra) or ramped sinusoid (single-frequency steady
  state).
- **TFSF** (total-field / scattered-field) — inject a clean plane wave
  across a rectangular contour so the scattered field is isolated outside
  it. Required for scattering / RCS work.

### Absorbing boundary

- **Phase 1** — first-order **Mur ABC**: cheap, gets a non-reflecting-ish
  demo running.
- **Phase 2+** — **CPML** (convolutional PML): stretched-coordinate,
  split-free, polynomial-graded conductivity profile. Needed before any
  radiation-pattern or scattering result is trustworthy (Mur leaves ~1–5%
  reflection).

### GPU

One compute-shader dispatch each for the `H` update, the `E` update, source
injection, and the PML auxiliary fields. Fields as `r32f` images.
Interactive domains of several million cells at 60 fps expected. Same
validation-only-first discipline as `07`/`08`'s GPU ports: cross-check
against a CPU reference before wiring a deck to it.

## Deck schema

```toml
[grid]     nx, ny, dx, courant
[time]     steps                  # or t_final
[boundary] type = "mur" | "cpml", pml_cells, pml_grade, pml_kappa
[[source]] kind = "gaussian" | "sine" | "tfsf"
           x, y, f0, bandwidth, amplitude, angle_deg
[[material]] shape = "slab" | "cylinder" | "box" | "lens"
             eps_r, mu_r, sigma, <geometry>
[[pec]]    shape = "cylinder" | "plate" | "strip", <geometry>
[probe]    points = [[x, y], ...]           # time series for spectra / coefficients
[render]   field = "Ez" | "E_energy" | "S_mag", gain, gamma
```

## Scenes / decks

| deck | shows | validation target |
|---|---|---|
| `dipole.toml` | point source, expanding ring waves | near-field vs 2D Green's function `−(i/4)H₀⁽²⁾(kr)` |
| `slab_fresnel.toml` | plane wave onto a dielectric slab, swept angle | `R(θ)`, `T(θ)` vs Fresnel (TE and TM) |
| `cylinder_scatter.toml` | plane wave on a PEC cylinder | scattered pattern / RCS vs the Mie (cylindrical-harmonic) series |
| `waveguide.toml` | parallel-plate / hollow guide | mode cutoff `f_c,n = nc/2a` |
| `dipole_antenna.toml` | driven half-wave PEC dipole | far-field pattern vs `cos((π/2)cosθ)/sinθ` |
| `grin_lens.toml` | graded-index slab focusing a beam | focal position vs ray-trace (Optics teaser) |

## File layout

```
projects/10_fdtd/
    CMakeLists.txt
    FDTD_Plan.md
    src/
        main.cpp
        Fdtd2D.{hpp,cpp}          # fw::Simulation
        kernels_fdtd.hpp          # E / H / source / PML compute-shader source
        Materials.{hpp,cpp}       # rasterize deck shapes -> ε/μ/σ/PEC textures
        Sources.hpp               # Gaussian / sine / TFSF injection
        Cpml.{hpp,cpp}
    decks/
    tools/
        green2d.py  fresnel.py  mie_cylinder.py     # analytic references
        plot_probe.py  plot_pattern.py  make_movie.py
    docs/SIMULATION.md
```

## Phases

**Phase 1 — TMz core + Mur, headless.**
Yee update, Gaussian/sine soft source, first-order Mur, `Ez` frame output.
Gate: propagation speed = `c` within numerical dispersion; instability
appears exactly when `S > 1/√2`; a closed PEC box conserves EM energy (no
PML) to round-off over a long run.

**Phase 2 — CPML + probes + dipole validation.**
CPML boundary, time-series probes, FFT spectra. Gate: CPML boundary
reflection < −40 dB across the pulse band; point-source near-field matches
the 2D Green's function in amplitude and phase over a radius range.

**Phase 3 — Materials + Fresnel.**
Per-cell `ε/μ/σ`, slab rasterization, TFSF plane wave, TEz mode. Gate: slab
`R(θ)`, `T(θ)` vs Fresnel for both polarizations (< 1% away from
grid-staircasing angles); Brewster null present in TM.

**Phase 4 — PEC scatterers + patterns.**
PEC masking, cylinder and dipole-antenna decks, a near-to-far-field
transform for radiation patterns. Gate: PEC cylinder scattered pattern vs
Mie series (correlation > 0.99 at a few `ka`); half-wave dipole pattern vs
analytic.

**Phase 5 — Interactive view.**
`fw::SimApp`: live `Ez` / energy / Poynting-magnitude heatmap; place and
drag PEC and dielectric blocks; move the source; adjustable frequency.
Offscreen-FBO-to-PNG render check (per `08_compressible_fluid`'s trick,
since OpenGL windows can't be inspected directly here).

**Phase 6 — `docs/SIMULATION.md` + website media.**

## Studies (`Studies/fdtd/`)

- `numerical_dispersion_vs_resolution` — measured phase velocity vs
  cells-per-wavelength and vs propagation angle, against the FDTD
  dispersion relation.
- `cpml_reflection_vs_params` — boundary reflection vs PML thickness /
  grading order / `κ_max`, to fix defaults.
- `cylinder_rcs_vs_ka` — scattering width vs size parameter, against the
  Mie series across the resonance region.

## Progress

- [ ] Phase 1 — TMz + Mur
- [ ] Phase 2 — CPML + dipole validation
- [ ] Phase 3 — materials + Fresnel
- [ ] Phase 4 — PEC scatterers + patterns
- [ ] Phase 5 — interactive
- [ ] Phase 6 — docs + media
