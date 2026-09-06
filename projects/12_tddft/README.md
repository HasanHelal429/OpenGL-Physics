# 12 — Real-Time TDDFT on the GPU

The GPU port (Stage 2) of the rt-TDDFT solver whose Python reference is
`Physics Simulations/Quantum Mechanics/TDDFT/` — real-time propagation of the
Kohn–Sham orbitals for excited-state and light–matter response (absorption
spectra, high-harmonic generation), done entirely on the GPU by 3D split-step
Fourier.

```
i ∂ψ_j/∂t = [ -½∇² + v_ext(r,t) + v_H[ρ] + v_xc[ρ] ] ψ_j        (ħ = m = 1)
```

See `TDDFT_GPU_Plan.md` for the design, the architecture decision (a new project
reusing `05_tdse_gpu`'s FFT kernel rather than extending `05`, which is 2D), and
the phase plan.

## Status

| phase | what | state |
|---|---|---|
| 7 | 3D FFT + single-orbital propagator + selftest vs Python | **done** |
| 8 | ALDA `v_xc` + 3D Hartree + He δ-kick spectrum vs Python | **done** |
| 9 | interactive absorption / HHG demo | not started |

## Build & run

```sh
cmake --preset release
cmake --build --preset release --target 12_tddft
B=build/release/projects/12_tddft/12_tddft.exe
$B --selftest                       # 3D FFT + free/harmonic vs Python (8/8)
$B --relax-test                     # He Kohn-Sham ground state, imag. time (2/2)
$B --he-spectrum [--out DIR]        # He delta-kick absorption vs Python (3/3)
python tools/spectrum.py DIR        # -> S(omega) figure from --he-spectrum's dipole.csv
```

None of these need a window (hidden GL context).

### `--selftest` (8/8)

- **3D FFT** round-trip (fp32 machine precision) and forward-vs-direct-DFT.
- **free Gaussian wavepacket**: `σ(t) = σ₀√(1+(t/2σ₀²)²)` to 4e-6, `⟨x⟩ = k₀t`
  to 2e-6 — the split-step is exact for `V = 0` and the FFT is accurate.
- **harmonic coherent state**: `⟨x⟩(t) = x₀cos ωt` to 1e-5, constant width to
  8e-6.
- norm conserved to ~5e-4 over the run — fp32 FFT round-off accumulation (the
  propagator is analytically unitary), same character as `05_tdse_gpu`'s known
  ~4e-4/2000-step drift.

These are the exact closed-form checks the Stage-1 Python `validate.py --phase 1`
uses, now in fp32 on the GPU.

### `--relax-test` / `--he-spectrum` (Phase 8)

- **He Kohn-Sham ground state** by imaginary-time relaxation (FFT-only, ~1 s):
  eigenvalue **-0.7629 Ha, matching the Stage-1 Python
  `imaginary_time_ground_state` (-0.76291) to ~1e-5**; `∫ρ = 2.00000`.
- **He δ-kick absorption** (`ψ → e^{iκx}ψ`, 4000 ETRS steps, ~6 s): the
  **TRK f-sum rule recovers 97% of `N_e`** (identical to Stage-1 Python), the
  response is passive (`Im α ≥ 0` to -0.2% of peak), and the **lowest
  absorption line is 13.46 eV vs the Stage-1 Python 13.5 eV** — a 0.04 eV
  match. `tools/spectrum.py` reproduces the full `S(ω)` figure.

Same physical caveats as Stage 1 (softened nuclear cusp, small box,
periodic-FFT Poisson error) — the sum rule is geometry-exact and matches;
the absolute line position carries the shared grid limitation.

## Method

**3D FFT.** Three passes of *(FFT the contiguous axis; cyclically rotate the axes
`(z,y,x) → (x,z,y)`)*. Three rotations compose to the identity, so a full
forward/inverse 3D FFT ends back in the original `(z,y,x)` layout with every axis
transformed. The 1D FFT (`kernels3d::Fft1D`) is `05_tdse_gpu`'s proven radix-2
butterfly, dispatched over `N²` rows. `Rotate` is one ~15-line kernel.

**Split-step (Phase 7, fixed potential).** `Tddft3D::Configure` precomputes the
half-kick multiplier `vprop = exp(-i VΔt/2)` and the kinetic phase
`kprop = exp(-i k²Δt/2)` as `N³` complex buffers on the CPU. Each step:

```
CMul(ψ, vprop)  →  FFT₃D(ψ)  →  CMul(ψ, kprop)  →  iFFT₃D(ψ)  →  CMul(ψ, vprop)
```

`N` must be a power of two. Complex data is `vec2 (re, im)` in std430 SSBOs,
`(z,y,x)` layout with `x` contiguous.

**Phase 8** (`kernels_ks.hpp`, `Tddft3D::RelaxKS` / `EtrsStepKS` /
`KickAndRunKS`): the density-dependent Kohn-Sham path. Each ETRS substep
builds `V_eff = V_nuc + V_H[ρ] + v_xc[ρ]` — `V_H` by 3D FFT-Poisson
(`ρ → FFT → ×4π/k² → iFFT`), `v_xc` by an ALDA GLSL kernel (Slater exchange
at `α = 2/3` + Perdew-Zunger '81 correlation, both branches inline) — then
does a full-Strang predictor and the real ETRS step. The ground state comes
from imaginary-time relaxation (`dt → dτ` real, renormalize each step).
Reductions (norm, dipole, `<V>`, `<T>`) are shared-memory partial sums
finished on the CPU.

**Phase 9** (not started) adds the `fw::Simulation` + `SimApp` interactive
view with a live running-FFT harmonic-comb panel and `tools/make_movie.py`.
