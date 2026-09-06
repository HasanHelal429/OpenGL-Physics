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
| 8 | multi-orbital + ALDA `v_xc` + He δ-kick spectrum | not started |
| 9 | interactive absorption / HHG demo | not started |

## Build & run

```sh
cmake --preset release
cmake --build --preset release --target 12_tddft
build/release/projects/12_tddft/12_tddft.exe --selftest
```

`--selftest` needs no window (hidden GL context). 8/8:

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

**Phase 8+** adds `N_occ` orbitals, an `AccumRho` density pass, FFT-Poisson for
`V_H`, an ALDA `v_xc` GLSL kernel (Slater + PZ81), the ETRS predictor, an
imaginary-time `--relax` ground state, and the `--tddft` δ-kick deck mode —
validated against the Stage-1 Python He spectrum.
