# 01 — Atomic Hartree-Fock-Slater / Kohn-Sham LDA

A from-scratch atomic electronic-structure solver: the self-consistent field
(SCF) ground state of any atom `Z = 1..103` on a logarithmic radial grid, with
either **Xα local exchange** or **genuine Kohn-Sham LDA** (Slater exchange +
Perdew-Zunger 1981 correlation).

It is a C++ port of
`Physics Simulations/Quantum Mechanics/HF_solver/` — same substitution, same
generalized-eigenproblem discretization, same shell-theorem Hartree potential,
same total-energy bookkeeping. The Python solver's `HF_Solver_Plan.md` carries
the full derivation and its own validation record; `docs/SIMULATION.md` here is
the condensed version. This project reproduces the Python energies to ~1e-6 Ha
(see `--scf-selftest`).

```
i∂ψ/∂t is not solved here — this is a stationary-state solver:
    [ -½∇² - Z/r + V_H[ρ] + V_xc[ρ] ] ψ_nl = ε_nl ψ_nl ,   ρ from the occupied ψ
iterated to self-consistency.                              (ħ = mₑ = e = 1)
```

## Build

```sh
cmake --preset release
cmake --build --preset release --target 01_hartree_fock
```

**Use the `release` preset.** The eigensolver leans on Eigen/Spectra
header-only template code, which is dramatically slower unoptimized — Argon's
SCF goes from seconds (Release) to minutes (Debug). Debug is for breakpoints,
not solves.

## Run

```sh
# eigensolver check: V(r) = -Z/r must reproduce E_nl = -Z²/(2n²) (CPU, no window)
01_hartree_fock --selftest

# SCF energies vs the Python reference numbers (He/Ne/Ar Xα + He LDA, ~20 s)
01_hartree_fock --scf-selftest

# headless SCF run -> per-iteration frames + diagnostics.csv + manifest.json
01_hartree_fock --deck decks/argon.toml --out out/argon

# the ImGui periodic-table explorer (needs a window)
01_hartree_fock --interactive
```

Then, from `tools/` (`pip install -r tools/requirements.txt`):

```sh
python tools/plot_scf_convergence.py out/argon    # -> scf_convergence.mp4 (4-panel movie)
python tools/plot_orbitals.py        out/argon    # -> orbitals.png (R_nl, radial density)
python tools/plot_diagnostics.py     out/argon    # -> diagnostics.png (E, convergence, barcode)
```

Parameter sweeps live in `Studies/hartree_fock/` (sibling to `Studies/fdtd/` etc.).

## Deck format

```toml
title = "Argon - Kohn-Sham LDA"     # ASCII only — non-ASCII does not survive the
                                     # JSON manifest round-trip cleanly

[atom]
Z      = 18
method = "lda"                       # "xalpha" (default) | "lda"
alpha  = 0.7                         # Xα only; ignored for lda (0.7 = Schwarz)

[grid]                               # all optional — DefaultGrid(Z) if omitted
r_min = 5.6e-7                        # default 1e-5 / Z
r_max = 150.0
n     = 4000

[scf]
max_iter = 200
mix      = 0.3                        # linear density mixing factor
tol_e    = 1e-6                       # Ha
tol_n    = 1e-5                       # integrated |Δρ|

[output]
diagnostics = ["e_total", "de", "dn"]
```

Output per SCF iteration (one "frame"): `rho`, `v_eff` (both `1×n`), `eps`
(`1×k` occupied orbital energies, sorted by `(n,l)`); frame 0 also has `r`
(the grid) and `eps_nl` (`2×k` int, the `(n,l)` labels); the last frame also
has `orbitals_final` (`k×n`, the converged `R_nl(r)`). `diagnostics.csv` has
`t,step,e_total,de,dn` per row.

`--selftest`, `--scf-selftest`, and `--deck` need **no GL context** — the
solver is CPU Eigen/Spectra. Only `--interactive` opens a window.

## Validation

| check | expectation | result |
|---|---|---|
| `--selftest` | `V=-Z/r` → `E_nl = -Z²/(2n²)`, `Z=1,10,50,90`, `l=0..3` | relErr < 4e-5, l-degeneracy holds |
| `--scf-selftest` He (Xα) | Python `-2.76637` Ha | `\|Δ\|` ~ 1e-6 |
| `--scf-selftest` Ne (Xα) | Python `-128.03490` Ha | `\|Δ\|` ~ 2e-6 |
| `--scf-selftest` Ar (Xα) | Python `-525.89503` Ha | `\|Δ\|` ~ 3e-7 |
| `--scf-selftest` He (LDA) | Python `-2.83418` Ha | `\|Δ\|` ~ 2e-6 |
| `decks/argon.toml` (LDA) | `-525.925` Ha, K/L/M shell structure | converges in ~45 iters |
| `decks/copper.toml` | config `... 3d10 4s1` (Madelung exception) | exception table used |
| `decks/hydrogen.toml` | Xα(0.7) ≈ `-0.416` Ha (self-interaction error, exact is `-0.5`) | as expected |

Literature non-relativistic HF (He `-2.86168`, Ne `-128.547`, Ar `-526.818` Ha)
is a trend check only — Xα is local exchange, not exact HF.

## Files

```
Hartree_Fock_Plan.md     the convention-cleanup plan + a record of what the solver does
docs/SIMULATION.md       condensed physics/numerics derivation
src/
  RadialEigensolver      generalized eigenproblem H w = E M w, Spectra shift-invert
  ScfSolver              the SCF loop + total energy
  Potentials             shell-theorem Hartree, Slater/Xα, PZ81 correlation
  Shells                 Madelung fill order + the 20-entry exception table
  HFSim                  fw::Simulation wrapper (replays the SCF history as frames)
  HFVisualizerApp        the ImGui explorer (--interactive)
  main.cpp               CLI dispatch
decks/  tools/  out/(gitignored)
```

## Not implemented

- **Emission-line spectra** (`spectra.py` in the Python solver) — the
  orbital-energy barcode is the only spectral output here.
- **Relativistic corrections** — `Z > 80` is qualitative only. A Dirac-Fock-
  Slater `--relativistic` mode is planned; see
  `Physics Simulations/Quantum Mechanics/Dirac_Solver/Dirac_Solver_Plan.md`.
- Configuration averaging, not full term-symbol multiplets (same as the Python
  solver).
