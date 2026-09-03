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

$EXE --selftest        # CFL limit, PEC-box energy conservation, wave speed
$EXE --gpu-selftest    # GPU compute backend vs the CPU reference
$EXE --cpml-selftest   # CPML boundary reflection (short vs long domain)

$EXE --deck projects/10_fdtd/decks/pulse_mur.toml --out out/pulse [--gpu]
python projects/10_fdtd/tools/plot_probe.py out/pulse
$EXE --deck projects/10_fdtd/decks/dipole.toml --out out/dipole
python projects/10_fdtd/tools/plot_dipole.py out/dipole   # vs the 2D Green's function
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
- [ ] Phase 3 -- materials + Fresnel + TEz
- [ ] Phase 4 -- PEC scatterers + radiation patterns
- [ ] Phase 5 -- interactive view
- [ ] Phase 6 -- docs + Studies + website media

## Decks

| deck | shows |
|---|---|
| `pulse_mur.toml` | Gaussian point source, first-order Mur boundary |
| `pulse_box.toml` | same in a closed PEC box (energy-conservation check) |
| `dipole.toml` | CW line source in a CPML box (2D Green's-function check) |
