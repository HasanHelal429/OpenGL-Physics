# 05 — 2D TDSE on the GPU (split-step Fourier)

Time-dependent Schrödinger equation in 2D,

    i ∂ψ/∂t = [ -½∇² + V(x,y) ] ψ        (ħ = m = 1)

integrated by **Strang-split split-step Fourier**, entirely on the GPU:

1. half potential kick — `ψ ← ψ · exp(-i V Δt/2)`, fused with a border
   complex-absorbing potential `exp(-W Δt/2)` (one precomputed complex texture)
2. 2D forward FFT — row FFT → transpose → row FFT
3. exact kinetic phase — `ψ̂ ← ψ̂ · exp(-i (kx²+ky²) Δt/2)`
4. 2D inverse FFT
5. half potential kick

The 1D FFT is a hand-written radix-2 Cooley–Tukey compute shader
(`src/kernels.hpp`): one workgroup per row, the row staged in shared memory,
`log₂N` butterfly stages against a precomputed twiddle table. `--selftest`
checks it against a direct DFT and a round-trip before it is trusted.

`N` must be a power of two, ≤ 2048.

The interactive view domain-colors ψ (phase → hue). Brightness tracks the
probability density, **auto-scaled each frame to the current peak** (a GPU
max-reduction) so the whole distribution stays readable as the packet spreads
or is absorbed, with a γ≈0.5 (amplitude) curve so tails and fringe minima stay
visible rather than crushing to black. `m` cycles phase-colored density → a
magma density ramp → Re ψ. `tools/make_movie.py` uses the same mapping
(`--normalize frame|global`, `--gamma`, `--gain`).

## This project also introduces the reusable batch/interactive layer

New in `framework/`, meant for every future project:

| piece | role |
|---|---|
| `fw::Simulation` | the interface a sim implements once (`Configure`/`Step`/`Snapshot`/`Render`) |
| `fw::Deck` | TOML input deck, dotted-path access (vendored toml++) |
| `fw::OutputWriter` | `.npy` frames + `diagnostics.csv` + `manifest.json` |
| `fw::RunHeadless` + `fw::GLContext` | no-window batch runner |
| `fw::SimApp` + `fw::Hud` | window + a free-floating top-right control cluster (play/pause, step, reset, speed, record), pure OpenGL, no ImGui |
| `fw::ComputeShader` | compile/dispatch a `GL_COMPUTE_SHADER` |

## Build

```sh
cmake --preset release
cmake --build --preset release --target 05_tdse_gpu
```

## Run

```sh
# batch: run a deck, write raw data
05_tdse_gpu --deck decks/tunneling.toml --out out/tunneling [--frames N] [--substeps N]

# interactive: window + HUD
#   mouse drag = pan, wheel = zoom, 0 = reset view
#   m = cycle view (phase-colored density / magma density / Re ψ)
#   j = probability-current arrows,  k = momentum-space |ψ̂|²
#   [ ] = brightness gain,  - = = density gamma,  F12 = screenshot
05_tdse_gpu --interactive --deck decks/double_slit.toml

# imaginary-time relaxation: the K lowest eigenstates of -1/2 grad^2 + V
05_tdse_gpu --relax --deck <f.toml> --out <dir> --states 6 [--relax-steps N]

# self-consistent Poisson-Schrodinger
05_tdse_gpu --scf --deck <f.toml> --out <dir> --states 6 --electrons 3

# FFT kernel self-test
05_tdse_gpu --selftest
```

Then, from `tools/` (needs `numpy` + `matplotlib`):

```sh
python tools/make_movie.py out/tunneling --mode phase     # -> mp4 (or PNG seq if no ffmpeg)
python tools/plot_diagnostics.py out/tunneling            # -> validation panels
```

## Deck format

```toml
title = "..."
[grid]      n = 512   lx = 60.0   ly = 40.0        # n a power of two, <= 2048
[time]      dt = 0.002   substeps_per_frame = 10   frames = 400
[initial]   type = "gaussian"    # gaussian | hermite_gauss {nx,ny,omega} | superposition | file
            x0 = -14  y0 = 0  sigma = 2  kx = 5  ky = 0    # sigma_x / sigma_y also ok
            # type = "file": path = "psi.bin" -- raw complex64, interleaved re/im
            #   float32, row-major n*n (np.asarray(psi, np.complex64).tofile); renormalized
[boundary]  type = "cap"         # or "periodic"
            cap_width = 6  cap_strength = 4
[[potential]]  type = "barrier"  # free|harmonic|barrier|double_slit|well|coulomb|box
               x0 = 0  width = 1.0  height = 16.0
[magnetic]  B = 2.0              # uniform field, symmetric gauge (optional)
[[drive]]   type = "tilt"        # tilt | gate -- time-dependent V(t) (optional)
            amplitude = 1.0  omega = 3.0  dir_x = 1.0
[meanfield] enabled = true  coupling = 4.0    # Hartree self-repulsion (optional)
[scf]       tol = 3e-3  max_iter = 24  mix = 0.35     # --scf only
[output]    diagnostics = ["norm", "energy", "x_mean", "px_mean", "transmission"]
            transmission_x = 2.0
```

`[[potential]]` blocks superpose; `harmonic` also takes `omega_x`/`omega_y`.
Diagnostic column names: `norm`, `energy`, `kinetic`, `potential_energy`,
`x_mean`/`y_mean`, `x_var`/`y_var`, `px_mean`/`py_mean`, `Lz`, `transmission`,
`autocorr_re`/`autocorr_im`/`autocorr_abs`. Kinetic energy and momentum are
computed spectrally (an extra FFT per snapshot), so energy conservation is
exact to float precision.

## Validation (`decks/`)

| deck | check | result |
|---|---|---|
| `free_packet` | ⟨x⟩ = x₀ + kₓt; Var(x) = σ₀²/2 + t²/(2σ₀²) | matches analytic to plotting precision |
| `harmonic` | coherent state: ⟨x⟩ = x₀cos ωt, constant width, constant energy | period within <1%, energy drift 3e-6 |
| `tunneling` | steady transmission past a barrier above ⟨E⟩ | T ≈ 3% (wavepacket, > T(⟨E⟩) as expected from the energy spread) |
| `double_slit` | interference fringes past two slits | qualitatively correct |

Known limitation: norm drifts ~4e-4 over a long (~2000-step) periodic run — pure
float32 FFT round-off accumulation (the propagator is analytically unitary).
CAP runs lose norm on purpose as probability is absorbed at the boundary.
