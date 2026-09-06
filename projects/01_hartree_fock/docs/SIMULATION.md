# 01_hartree_fock: how it works

This is a straight numerical port of the atomic solver in
`Physics Simulations/Quantum Mechanics/HF_solver/` — the physics decisions,
derivations, and validation history live in that folder's `HF_Solver_Plan.md`.
This document is the condensed version: enough to read the code without
cross-referencing.

Atomic units throughout: `ħ = mₑ = e = 4πε₀ = 1`, energies in hartree, lengths
in bohr.

## 1. The problem

For an atom of nuclear charge `Z` we want the self-consistent Kohn-Sham (or
Hartree-Fock-Slater) ground state: a set of radial orbitals `R_nl(r)` and
energies `ε_nl` solving

```
[ -½∇² - Z/r + V_H[ρ](r) + V_xc[ρ](r) ] ψ_nlm = ε_nl ψ_nlm
```

with the density `ρ(r)` built from the occupied orbitals and fed back into
`V_H`, `V_xc` until nothing changes. Spherical symmetry (a
configuration-averaged central field) makes every quantity a function of `r`
alone, so the 3D problem separates into one 1D radial equation per angular
momentum `l`.

## 2. Radial grid and substitution (`Grid.hpp`, `RadialEigensolver.cpp`)

Core orbitals scale as `r ~ 1/Z`; valence shells reach `r ~ 10`. A uniform
grid cannot resolve both, so the grid is **logarithmic**:

```
x = ln r ,   x uniform with step h ,   r_i = r_min · e^{i h}
DefaultGrid(Z):  r_min = 1e-5 / Z ,  r_max = 150 ,  N = 4000
```

Write the radial function as `u(r) = r R(r)`, then `u(r) = √r · w(x)`. The
`√r` factor removes the first-derivative term the `x = ln r` change of
variables would otherwise introduce, and turns the centrifugal term
`l(l+1)/r²` into `(l+½)²` (a correct artifact of the substitution, **not**
`l(l+1)` — do not "fix" it). The radial Schrödinger equation becomes a
Sturm-Liouville generalized eigenproblem in `w(x)`:

```
H w = ε M w        per l-channel, tridiagonal
```

with, on the log grid,

```
main_diag(H)_i = 1/(2h²)·2 + (l+½)² + 2 r_i² V(r_i)      (see ScfSolver / RadialEigensolver for exact factors)
off_diag(H)_i  = -1/(2h²)   (finite-volume weighted, so H stays symmetric on the nonuniform-in-r grid)
M_diag_i       = 2 r_i²
```

## 3. Why a generalized eigenproblem, not a rescaled standard one

Dividing through by `r` to get a plain symmetric tridiagonal problem (solvable
by a stock `eigh_tridiagonal`) makes matrix entries span `~1/r_min²`, which
overruns float64's usable dynamic range once `r_min` is small enough to
resolve a heavy-atom core — silently producing garbage (sometimes positive)
eigenvalues. Keeping `H` and `M` at natural scale and solving
`H w = ε M w` directly avoids this.

`RadialEigensolver.cpp` uses `Spectra::SymGEigsShiftSolver` (shift-invert
Lanczos) with a **custom `O(N)` shift-invert operator**: it factorizes the
tridiagonal `H - σM` with the Thomas algorithm instead of forming an explicit
inverse. `σ` is seeded from `-r[0]·V[0]` (a proxy for the effective nuclear
charge). If Lanczos fails to converge — which happens for the crude seed
density on the first SCF iteration — it retries with a wider Krylov subspace
and a perturbed `σ` (up to 4 attempts), mirroring the ARPACK robustness fix
the Python solver needed.

Direct diagonalization (not shooting / Numerov) is deliberate: it returns the
whole ordered low-lying spectrum per `l` with no manual energy bracketing or
node counting, which is what makes the SCF loop automatable across arbitrary
`Z`.

## 4. Effective potential (`Potentials.cpp`)

**Hartree** — the electron-electron Coulomb potential, via the shell theorem
applied to the spherically averaged density (`O(N)`, not an `O(N²)` double
integral):

```
Q(r) = ∫₀^r 4π r'² ρ(r') dr'          (enclosed charge)
S(r) = ∫₀^r 4π r'  ρ(r') dr'
V_H(r) = Q(r)/r + [ S(r_max) - S(r) ]
```

`V_H(r)·r → N` (the electron count) at large `r`, as required.

**Slater / Xα exchange** — a purely local functional of the density:

```
V_x(r) = -3 α [ 3 ρ(r) / (8π) ]^{1/3}
```

with `α = 0.7` (Schwarz) the default, `α = 2/3` the theoretically exact LDA
value, `α = 1` classic Slater.

**PZ81 correlation** (`method = "lda"` only) — the Perdew-Zunger 1981
parametrization of the correlation energy per electron `ε_c(r_s)` and its
potential `V_c = ε_c - (r_s/3) dε_c/dr_s`, via the Wigner-Seitz radius
`r_s = [3/(4πρ)]^{1/3}`. Two independently curve-fit branches (`r_s < 1`
logarithmic, `r_s ≥ 1` a Padé form), fit to Ceperley-Alder QMC data for the
electron gas. `ρ` is floored at a tiny value before forming `r_s` so the
far-grid `ρ → 0` region does not produce `0·∞ = NaN`.

`method = "xalpha"` uses `V_xc = V_x(α)` only; `method = "lda"` uses
`V_xc = V_x(2/3) + V_c`.

## 5. Shell filling (`Shells.cpp`)

`GroundStateConfiguration(Z)` fills `(n,l)` shells in Madelung order
(increasing `n+l`, then `n`), capacity `2(2l+1)` per shell, then swaps in the
*valence* shells of a hardcoded exception when `Z` is one of the 20 well-known
Madelung anomalies (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac,
Th, Pa, U, Np, Cm, Lr), re-deriving the noble-gas core underneath by ordinary
Madelung filling. Occupations are **fixed for the whole SCF run** — a
configuration-averaged central field, spherically averaged density, integer or
fractional shell occupation, no term-symbol / multiplet structure.

## 6. The SCF loop (`ScfSolver.cpp`)

```
ρ ← crude exponential seed, normalized to Z electrons
repeat:
    V_H  ← HartreePotential(ρ)
    V_xc ← Slater/Xα  (+ PZ81 if lda)
    V    ← -Z/r + V_H + V_xc
    for each occupied l-channel:
        (ε_nl, u_nl) ← SolveRadialChannel(r, l, V)        # lowest states
    ρ_new ← (1/4πr²) Σ_nl N_nl u_nl(r)²
    E_total ← Σ N_nl ε_nl - E_H - E_x/3  [- E_c doublecount]
    converged if |ΔE_total| < tol_e AND ∫|ρ_new - ρ|·4πr² dr < tol_n
               for 2 consecutive iterations
    ρ ← (1 - β) ρ + β ρ_new          # linear mixing, β = mix
```

Convergence is clean and roughly geometric — `|ΔE|` and `∫|Δρ|` both drop
~10 orders of magnitude over ~40 iterations for a closed-shell atom (see
`tools/plot_diagnostics.py`).

## 7. Total energy (`ComputeTotalEnergy`)

```
E_total = Σ_nl N_nl ε_nl  -  E_H  -  E_x/3   [ + (E_c - ∫V_c ρ)  for lda ]

E_H = ½ ∫ V_H(r) ρ(r) 4π r² dr
E_x = ¾ ∫ V_x(r) ρ(r) 4π r² dr
```

Summing `N_nl ε_nl` double-counts electron-electron interaction: it counts the
Hartree term twice (hence the `½`-derived `-E_H`) and, because `E_x[ρ]` scales
as `ρ^{4/3}`, counts exchange `4/3` times — Euler's theorem then gives the
`-E_x/3` correction (not `-E_x/2`). Correlation has no such closed-form
scaling shortcut, so the general Kohn-Sham double-counting term
`E_c[ρ] - ∫ V_c ρ` is evaluated directly when `method = "lda"`.

## 8. What the headless run writes (`HFSim.cpp`, `main.cpp`)

`HFSim` implements `fw::Simulation` but the SCF is not a time-stepping
process, so `Configure()` runs the entire solve (buffering every iteration's
`ScfSnapshot`) and `Step()`/`Snapshot()` replay that history one iteration per
frame. `main.cpp` drives a custom frame loop of length `NumIterations()`
rather than `fw::RunHeadless` (whose fixed `[time].frames` count does not fit a
naturally-terminating solve — the same reason `05_tdse_gpu`'s `--scf` path is
custom). The per-iteration frame stream is exactly what
`tools/plot_scf_convergence.py` animates into the SCF-convergence movie.
