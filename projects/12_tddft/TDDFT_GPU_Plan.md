# projects/12_tddft — rt-TDDFT on the GPU (Stage 2)

## Context

Stage 1 (`Physics Simulations/Quantum Mechanics/TDDFT/`, Python, complete) built
real-time TDDFT on `Molecular_DFT/`'s 3D FFT grid — a split-step / ETRS
propagator for the Kohn–Sham orbitals, validated across six phases (free/harmonic
propagator checks, the ground-state fixed point, He and H2 δ-kick absorption
spectra with the TRK sum rule, and hydrogen HHG). It is the reference.

Stage 2 is the GPU port: the same physics, every timestep on the GPU, fast enough
to watch live and to reach systems Stage 1's `numpy` propagator can't. The
Stage-1 plan flagged this as the one new-solver project where the C++/OpenGL
framework genuinely earns its place.

## Architecture decision — a new project, not an extension of `05_tdse_gpu`

`05_tdse_gpu` is **2D** and thoroughly 2D-coupled (an 1100-line `TdseSim.cpp`:
row-FFT + transpose, 2D k-space Poisson, 2D `BuildVprop`, a 2D interactive view,
`--relax`, `--scf`, a magnetic-field path). rt-TDDFT is 3D + multi-orbital +
exchange-correlation — a different enough problem that bolting it onto `05` would
double its size and risk its validated 2D paths.

So `12_tddft` is a **fresh project** that reuses:

- the **`Fft1D` compute shader** from `05_tdse_gpu/kernels.hpp`, verbatim — a
  proven radix-2 Cooley–Tukey butterfly. Here it is dispatched over `N*N` rows
  instead of `N`.
- the **`framework/` layer** — `ComputeShader`, `GLContext`, `Deck`, `SimApp`,
  `OutputWriter`, `HeadlessRunner` — like every project from `05` on.
- the **method** from `05` (SSBO ping-pong, precomputed twiddle table, split-step
  structure), lifted from 2D to 3D.

The 3D FFT is **three passes of "FFT the contiguous axis, then cyclically rotate
the axes `(z,y,x) → (x,z,y)`"**. Three rotations compose to the identity, so a
full forward or inverse 3D FFT returns the buffer to its original `(z,y,x)`
layout with all three axes transformed. This avoids a separate 3D-transpose
shader — one small rotate kernel does it.

## Phases

**Phase 7 — 3D FFT + single-orbital propagator + selftest vs Python.  ✅ DONE.**
- `kernels3d.hpp`: `Fft1D` (from `05`), `Rotate` (`(a,b,c)→(c,a,b)`), `CMul`
  (pointwise complex multiply), plus `AccumRho` / `BuildVprop` staged for
  Phase 8.
- `Tddft3D`: `Configure(N, L, dt)` builds the twiddle table and the precomputed
  `kprop = exp(-i k²Δt/2)` and `vprop = exp(-i VΔt/2)` multipliers; `Fft3D` is
  the 3-pass orchestration; `StrangStep` = `CMul(vprop) → FFT → CMul(kprop) →
  iFFT → CMul(vprop)`. Single orbital, fixed real potential.
- `main.cpp --selftest` (8/8): 3D FFT round-trip (fp32 machine precision) and
  forward-vs-direct-DFT; free Gaussian spreading matching
  `σ(t) = σ₀√(1+(t/2σ₀²)²)` to **4e-6** and `⟨x⟩ = k₀t` to 2e-6; harmonic
  coherent state `⟨x⟩ = x₀cos ωt` to 1e-5, constant width to 8e-6; norm
  conserved to ~5e-4 (fp32 FFT round-off, same character as `05`'s known
  ~4e-4/2000-step drift). These are the exact closed-form checks Stage-1
  Python Phase 1 uses, now in fp32 on the GPU.

**Phase 8 — ALDA `v_xc` + 3D Hartree + He δ-kick spectrum.  ✅ DONE.**
- `kernels_ks.hpp`: `Density` (rho = w·|ψ|²), `RealToComplex`, `PoissonK`
  (`×4π/k²`, k=0→0), `AssembleVeff` (`V_nuc + Re(V_H) + Slater(α=2/3) +
  PZ81 V_c` — PZ81's two branches inline), `HalfKick` / `HalfKickImag`,
  `MulRealK`, `Kick`, `Scale`, and two shared-memory partial-sum reductions
  (`ReduceMoment` for norm/dipole, `ReduceWeighted` for `<V>` and `<T>`).
- `Tddft3D::SetSoftenedNucleus`, `RelaxKS` (imaginary-time KS ground state,
  FFT-only — the GPU analogue of Stage-1's
  `propagate.imaginary_time_ground_state`), `BuildVeff` (density → FFT-Poisson
  → assemble), `EtrsStepKS` (V0 from ρ(t), a full-Strang predictor, V1 from
  ρ_pred, the real ETRS step), `KickAndRunKS` (`ψ → e^{iκx}ψ`, propagate,
  record the first moment every step).
- **He is one spatial orbital at occupation 2**, so no multi-orbital buffer
  management is needed for the first validation (a packed `N_occ`-orbital
  buffer + `AccumRho` loop is the natural extension for Be/LiH).
- `main.cpp --relax-test` (2/2): He KS eigenvalue **-0.7629 Ha, matching the
  Stage-1 Python `imaginary_time_ground_state` (-0.76291) to ~1e-5**;
  `∫ρ = 2.00000`. `--he-spectrum` (3/3): δ-kick, 4000 ETRS steps (~6 s total),
  a direct-DFT dipole analysis — **TRK sum rule 97% of `N_e` (identical to
  Stage-1 Python), passivity `Im α ≥ 0` to -0.2% of peak, lowest absorption
  line 0.495 Ha = 13.46 eV vs the Stage-1 Python 13.5 eV (0.04 eV)**.
- `tools/spectrum.py` reproduces the Stage-1 `response.py` analysis + figure
  from the written `dipole.csv`.

**Phase 9 — interactive absorption / HHG demo.**
- `Tddft3DSim : fw::Simulation` + `fw::SimApp`: a 3D density isosurface or a
  slice, the instantaneous dipole, and a live running-FFT `S(ω)` / harmonic-comb
  panel that fills in as the propagation runs. `[[drive]]` (a `sin²` or flat-top
  laser, ported from Stage-1's `perturb.py`) + `boundary_mask`.
- `tools/make_movie.py` records the comb building up — the website asset.
- **Validate:** the live-accumulated HHG spectrum converges to the Stage-1
  Python result (odd harmonics, the intensity-scaling cutoff).

## Layout

```
projects/12_tddft/
    TDDFT_GPU_Plan.md
    README.md
    CMakeLists.txt
    src/
        kernels3d.hpp     GLSL: Fft1D, Rotate, CMul (+ AccumRho, BuildVprop staged)
        Tddft3D.{hpp,cpp} the 3D split-step propagator
        main.cpp          --selftest (Phase 7); --relax / --tddft / --interactive to come
    decks/  tools/  docs/SIMULATION.md   (Phase 8+)
```

## Progress

- [x] Phase 7 — 3D FFT + single-orbital propagator + `--selftest` vs Python (8/8)
- [x] Phase 8 — ALDA v_xc + 3D Hartree + He δ-kick spectrum vs Python
      (`--relax-test` 2/2, `--he-spectrum` 3/3)
- [ ] Phase 9 — interactive absorption / HHG demo + MP4

## Verification

1. Phase 7: ✅ 3D FFT round-trips to fp32 machine precision and matches a direct
   DFT; the GPU propagator reproduces Stage-1's free-spreading and
   harmonic-coherent-state closed forms to ~1e-5 (fp32), norm to ~5e-4.
2. Phase 8: ✅ He KS eigenvalue -0.7629 Ha vs Stage-1 Python -0.76291 (1e-5);
   He δ-kick TRK sum rule 97% of `N_e` (matches Python); lowest line 13.46 eV
   vs Python 13.5 eV (0.04 eV).
3. Phase 9: live HHG spectrum converges to the Stage-1 result.
