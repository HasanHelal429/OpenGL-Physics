# 10_fdtd

2D full-wave Maxwell on a Yee grid (FDTD), on the `fw::Simulation` / deck /
headless model. The flagship computational-electrodynamics solver of the
three EM projects; `FDTD_Plan.md` has the design and phase plan, and (later)
`docs/SIMULATION.md` the writeup.

**Scope vs the future Optics page**: this is *vector* Maxwell -- radiation,
antennas, scattering, guided modes, lossy/dispersive media. Scalar methods
(Helmholtz / beam propagation / Fresnel-Kirchhoff for interference,
diffraction, lensing) are the Optics page's.

TM^z now (`Ez, Hx, Hy`):

```
dEz/dt = (1/eps)(dHy/dx - dHx/dy - sigma Ez)
dHx/dt = -(1/mu) dEz/dy
dHy/dt =  (1/mu) dEz/dx
```

Yee staggering, leapfrog in time. Units `c = eps0 = mu0 = 1`; `courant` is
the full Courant number `S = c dt sqrt(1/dx^2 + 1/dy^2)`, stable for `S <= 1`.

## Build & run

```sh
cmake --build --preset release --target 10_fdtd
EXE=./build/release/projects/10_fdtd/10_fdtd.exe

$EXE --selftest         # CFL limit, PEC-box energy conservation, wave speed
$EXE --gpu-selftest     # GPU compute backend vs the CPU reference
$EXE --cpml-selftest    # CPML boundary reflection (short vs long domain)
$EXE --fresnel-selftest # s-pol Fresnel R(theta) via TFSF phasor subtraction
$EXE --pec-selftest     # PEC cylinder: forward shadow + scattered field

$EXE --deck projects/10_fdtd/decks/cylinder_scatter.toml --out out/cyl
python projects/10_fdtd/tools/plot_mie.py --exe <binary>      # vs the Mie series
python projects/10_fdtd/tools/plot_antenna.py out/antenna     # half-wave dipole

$EXE --deck projects/10_fdtd/decks/pulse_mur.toml --out out/pulse [--gpu]
python projects/10_fdtd/tools/plot_probe.py out/pulse
$EXE --deck projects/10_fdtd/decks/dipole.toml --out out/dipole
python projects/10_fdtd/tools/plot_dipole.py out/dipole   # vs the 2D Green's function
python projects/10_fdtd/tools/plot_fresnel.py --exe <binary>   # full angle sweep
```

## Status

- [x] Phase 1 -- TM^z Yee core, Gaussian/sine soft source, first-order Mur
      ABC / PEC box, `Ez` frames + energy diagnostic, `--selftest`
      (CFL exact at S=1; PEC-box exact energy to 2e-14; pulse speed 0.995c
      -- the FDTD numerical-dispersion deficit at this resolution)
- [x] Phase 1b -- GPU compute-shader port (5 kernels: H / E / copy-prev /
      inject / Mur). `--gpu-selftest` cross-check: Ez matches the CPU
      reference to rel 2e-5 over 1200 steps. `--gpu` opt-in; ~5.5x faster
      than CPU at 1024^2 headless (more in the readback-free interactive loop)
- [x] Phase 2 -- CPML (convolutional PML, Roden-Gedney; trivial coefficients
      away from the layer so one update runs everywhere) + the 2D
      Green's-function check. `--cpml-selftest`: -81 dB reflection with 12
      cells. `plot_dipole.py`: CW point source vs `(i/4)H0^(1)(kr)` --
      2% amplitude, 0.9997 complex (amplitude+phase) correlation
- [x] Phase 3a -- per-cell eps/mu/sigma + PEC mask (`Materials.hpp`: slab /
      halfspace / box / cylinder shapes), TFSF plane-wave source with a 1D
      auxiliary incident grid (`IncidentWave.hpp`, dispersion-matched dxi so
      the TF/SF cancellation stays clean at oblique incidence: -276 dB
      normal, -50 dB at 20 deg off-axis). `--fresnel-selftest` +
      `tools/plot_fresnel.py`: s-pol reflectance R(theta) matches Fresnel to
      < 2% from 0 to 70 deg
- [ ] Phase 3b -- TEz mode (p-polarisation, Brewster angle)
- [x] Phase 4 -- PEC mask + a 2D near-to-far-field transform. `plot_mie.py`:
      PEC-cylinder bistatic scattering vs the 2D cylindrical-harmonic (Mie)
      series -- 0.995 pattern correlation at ka ~ 3.1. `plot_antenna.py`: a
      centre-fed half-wave PEC dipole -- correct broadside pattern with deep
      axis nulls (~0.89 vs the idealised thin-wire formula; the rest is the
      2D fat-strip vs 3D-filament difference)
- [x] Phase 5 -- `--interactive` fw::SimApp view (CPU backend -- fast enough
      at 256^2): live Ez / E-energy / |S| heatmap (M cycles), dielectric tint +
      PEC solid overlay, pan/zoom/gain/gamma, `--render-check <png>`
- [ ] Phase 6 -- docs + Studies + website media

## Decks

| deck | shows |
|---|---|
| `pulse_mur.toml` | Gaussian point source, first-order Mur boundary |
| `pulse_box.toml` | same in a closed PEC box (energy-conservation check) |
| `dipole.toml` | CW line source in a CPML box (2D Green's-function check) |
| `slab_fresnel.toml` | TFSF plane wave onto a dielectric half-space |
| `cylinder_scatter.toml` | TFSF plane wave onto a PEC cylinder (Mie check) |
| `dipole_antenna.toml` | centre-fed half-wave PEC dipole |
| `playground.toml` | CW source + dielectric lens + PEC strip (`--interactive`) |
