# projects/01_hartree_fock — Design Plan

## Context

`01_hartree_fock` is a near-complete C++ port of
`Physics Simulations/Quantum Mechanics/HF_solver/` (see that folder's
`HF_Solver_Plan.md` for the full physics derivation). It already implements:

- **Radial log-grid finite-difference eigensolver** (`RadialEigensolver`),
  generalized eigenproblem `H w = E M w` solved by a custom Spectra
  shift-invert operator — the same `u = √r · w`, `x = ln r` substitution and
  `(l+½)²` centrifugal form as the Python solver.
- **SCF loop** (`ScfSolver`) — Xα local exchange and genuine Kohn–Sham LDA
  (PZ81 correlation), `E_total = Σ N_nl ε_nl − E_H − E_x/3 [− E_c doublecount]`.
- **Madelung shell filling** with the 20-entry exception table (`Shells`).
- **Effective potential** (`Potentials`) — O(N) shell-theorem Hartree, Slater/Xα,
  PZ81 correlation.
- **Visualization** (`HFVisualizerApp`, ImGui) — 3D density view
  (`DensityView3D`), orbital-energy barcode (`SpectrumView`), SCF-convergence chart
  (`ConvergenceChart`), threaded solve (`ScfWorker`).
- **Validation in `main.cpp`** — the analytic hydrogen-like eigensolver check
  (gated) and a He/Ne/Ar SCF regression vs the Python reference numbers.

**The physics is done. What is missing is the packaging.** This project predates
the `fw::Simulation` + deck + headless conventions that `05` introduced and
`07`–`11` standardized on, so it has:

| convention (per `09`–`11`) | `01_hartree_fock` today |
|---|---|
| `<Name>_Plan.md` design doc | this file (new) |
| `README.md` | ❌ none |
| `docs/SIMULATION.md` physics writeup | ❌ none |
| `fw::Simulation` + `fw::Deck` + headless dispatch | ❌ ImGui app only, always opens a window |
| `decks/*.toml` per scenario | ❌ `Z` is an ImGui slider |
| `--selftest` CLI flags (validation on demand) | ⚠️ hardcoded in `main()`, runs every launch |
| `tools/*.py` + `requirements.txt` | ❌ none |
| `out/` + `.gitignore` | ❌ none |

Concretely: **the solver cannot run headless at all** — `main()` always calls
`app.Run()`, so it needs a window and a GL context, which blocks batch runs,
parameter sweeps, and CI. This plan fixes that, mirroring how
`04_molecular_dynamics` was retrofitted: keep the original ImGui app, add a
deck-driven `fw::Simulation` path alongside it.

## Scope

- **Not a rewrite.** No changes to `RadialEigensolver`, `ScfSolver`, `Shells`,
  `Potentials` numerics. Every existing number stays bit-for-bit.
- **Keep `HFVisualizerApp`** as the interactive periodic-table explorer
  (`04`'s `MDApp` precedent — ImGui is fine for `00`–`04`, and this app is
  genuinely useful for browsing).
- **Add** a headless deck path, CLI selftest flags, `tools/`, and docs.
- The convention target is `09`–`11` (the current best practice), not `05`.

## Numerical approach (already implemented — recorded here for completeness)

Atomic units (ħ = mₑ = e = 1). Per l-channel, `u_nl(r) = √r · w_nl(x)`,
`x = ln r` uniform step `h`, `r_i = r_min e^{ih}`:

```
main_diag_H[i] = 1/h² + (l+½)² + r_i² V(r_i) · 2      (see ScfSolver.cpp for the exact scaling)
off_diag_H[i]  = −1/(2h²)  ·  (finite-volume weighted)
M_diag[i]      = 2 r_i²
```

solved as `H w = E M w` via Spectra `SymGEigsShiftSolver` with a hand-rolled
shift-invert op (`RadialEigensolver.cpp`) — the C++ counterpart of the Python
solver's `scipy.sparse.linalg.eigsh(..., sigma=...)`, and the step most likely
to hide a sign/convention bug, so it is the first `--selftest`.

SCF: `V_eff = −Z/r + V_H[ρ] + V_xc[ρ]` → diagonalize occupied l-channels →
rebuild `ρ = (1/4πr²) Σ N_nl u_nl²` → linear density mix (`mixBeta`) → converge
when `|ΔE_total| < tolE` and `∫|Δρ| 4πr² dr < tolN` for 2 consecutive iterations
(200-iteration cap). Xα default `α = 0.7` (Schwarz); `method = "lda"` swaps in
exact-LDA exchange (`α = 2/3`) + PZ81 correlation.

The `docs/SIMULATION.md` written in Phase 1 carries the full derivation, lifted
from the Python `HF_Solver_Plan.md`.

## Deck format (new — Phase 2)

```toml
title = "Argon — Kohn-Sham LDA"

[atom]
Z = 18
method = "lda"          # "xalpha" (default) | "lda"
alpha  = 0.7            # Xα only; ignored for lda

[grid]                   # all optional — default_grid(Z) if omitted
r_min = 5.6e-7          # 1e-5 / Z
r_max = 150.0
n     = 4000

[scf]
max_iter = 200
mix      = 0.3
tol_e    = 1e-6
tol_n    = 1e-5

[output]
diagnostics = ["e_total", "de", "dn"]
```

The single-active-electron **emission spectrum** (`spectra.py` in the Python
solver) is **not** ported to C++ and is out of scope for this cleanup — the
orbital-energy barcode (`SpectrumView`) is the only spectral output here.

Headless run writes, per SCF iteration (one "frame"):

```
out/<name>/deck.toml
out/<name>/manifest.json
out/<name>/diagnostics.csv          t=iteration, step, e_total, de, dn, ...
out/<name>/frames/rho_0000.npy       radial density 4πr²ρ(r)      (1×n)
out/<name>/frames/v_eff_0000.npy     r·V_eff(r)                    (1×n)
out/<name>/frames/eps_0000.npy       occupied orbital energies      (1×k)
out/<name>/frames/r_0000.npy         the log grid (frame 0 only)
```

plus, on the final iteration, `orbitals_final.npy` (`R_nl(r)` per shell,
stacked, with a companion `orbitals_final_labels.json` naming the rows).

**No GL context is needed** for `--selftest`, `--scf-selftest`, or `--deck` —
the eigensolver is CPU Eigen/Spectra and `fw::OutputWriter` only writes files.
Only `--interactive` (the ImGui `HFVisualizerApp`) opens a window.

## Phases

Each phase is self-contained and leaves the project building and the existing
ImGui app working.

**Phase 1 — Docs** (`README.md`, `docs/SIMULATION.md`)
- `README.md` in the `09`–`11` shape: one-paragraph what/why, the method, build
  (`--preset release`, the Debug-is-too-slow note already in the top README),
  the run modes, the deck format, and a validation table (hydrogen exact, He/Ne/
  Ar/Cr/Cu vs Python + literature).
- `docs/SIMULATION.md` — the physics/numerics derivation ported from
  `HF_Solver_Plan.md` (substitution, generalized eigenproblem, shell-theorem
  Hartree, Slater/PZ81, total-energy double-counting, the ARPACK/Spectra
  robustness notes).
- Gate: both render correctly; no code touched.

**Phase 2 — `HFSim : fw::Simulation` + deck** (`HFSim.{hpp,cpp}`, `HFDeck` glue)
- `Configure(deck)` — parse the deck into `ScfParams`, run `RunScf` with the
  streaming `SnapshotCallback`, buffer every `ScfSnapshot` into a vector. (SCF
  for Ar is single-digit seconds in Release — running it all in `Configure` and
  *replaying* the history as frames is simpler than refactoring `RunScf` into an
  external stepper, and gives the convergence movie for free.)
- `Step(substeps)` / `Snapshot(writer)` — advance a replay index, write that
  iteration's `rho` / `v_eff` / `eps` / diagnostics.
- `Info()` — `frameFields`, `diagnostics` from the deck, `gridNx = n`.
- `Reset()` — replay index back to 0 (no re-solve).
- Gate: `HFSim` compiles and links; not yet wired into `main`.

**Phase 3 — `main.cpp` dispatch + CLI** (`main.cpp`, `CMakeLists.txt`)
- Argument parser (copy `05`'s `ParseArgs` shape):
  `--deck <f> --out <dir>` · `--interactive [--deck <f>]` · `--selftest` ·
  `--scf-selftest`.
- `--selftest` → the existing `ValidateEigensolver()` (hydrogen), CPU-only, no
  GL context. Exit 0/1.
- `--scf-selftest` → the existing `RunScfRegressionCheck()` extended to Cr/Cu
  (exception configs) and an LDA row. Exit non-zero if any atom is off its
  reference by more than a stated tolerance.
- `--deck ... --out ...` → no GL context; `HFSim sim; sim.Configure(deck);`
  then a custom frame loop writing `sim.NumIterations()` frames via
  `fw::OutputWriter` (like `05`'s dedicated `--scf` path — SCF has natural
  termination, so `fw::RunHeadless`'s fixed frame count does not fit).
- `--interactive` → `HFVisualizerApp` (optionally seeded with `[atom].Z` from
  the deck). Unchanged behaviour otherwise.
- Bare invocation → usage text and exit 2 (no more auto-opening a window / no
  more running validation on every launch).
- Gate: `01_hartree_fock --selftest` and `--scf-selftest` pass headless;
  `--deck decks/argon.toml --out out/argon` produces a readable results dir;
  `--interactive` still opens the explorer.

**Phase 4 — decks + tools** (`decks/*.toml`, `tools/*.py`, `tools/requirements.txt`)
- `decks/`: `hydrogen.toml` (Z=1, the analytic gate), `argon.toml` (closed
  shell, LDA), `iron.toml` (open d-shell), `copper.toml` (Madelung exception
  `3d10 4s1`), `gadolinium.toml` (f-shell, `lmax=3` end-to-end — note the long
  runtime), `krypton.toml` (Xα vs LDA comparison).
- `tools/plot_scf_convergence.py` — the 4-panel convergence movie
  (`4πr²ρ`, `r·V_eff`, per-l level diagram, `E_total` trace with a moving
  marker), MP4 via `imageio-ffmpeg`, GIF fallback — a direct port of the Python
  solver's `scf_movie.py`.
- `tools/plot_orbitals.py` — `R_nl(r)` and `4πr²R_nl²` from `orbitals_final.npy`;
  optional 2D `|ψ_nlm|²` cross-section.
- `tools/plot_diagnostics.py` — `E_total` / `dE` / `dn` vs iteration + the
  orbital-energy barcode from the final frame (the quick "did it converge"
  panel).
- Gate: each tool runs against a real `out/` dir and writes to `media/`.

**Phase 5 — `.gitignore` + top-README cross-links**
- `projects/01_hartree_fock/.gitignore` → `out/`, `media/`, `__pycache__/`.
- Update the repo `README.md` layout list entry for `01` to mention the deck
  path and `docs/SIMULATION.md` (it currently just names the solver).
- Gate: `git status` clean after a full batch + tool run.

## File layout (after cleanup)

```
projects/01_hartree_fock/
    Hartree_Fock_Plan.md            # this file
    README.md                        # Phase 1
    docs/SIMULATION.md               # Phase 1
    .gitignore                       # Phase 5
    CMakeLists.txt                   # + HFSim.cpp, main.cpp restructured
    decks/
        hydrogen.toml  argon.toml  iron.toml  copper.toml
        gadolinium.toml  krypton.toml
    src/
        main.cpp                      # restructured: ParseArgs + dispatch
        HFSim.hpp  HFSim.cpp          # NEW — fw::Simulation wrapper (replay SCF history)
        RadialEigensolver.{hpp,cpp}   # unchanged
        ScfSolver.{hpp,cpp}           # unchanged
        Potentials.{hpp,cpp}          # unchanged
        Shells.{hpp,cpp}              # unchanged
        Grid.hpp  Quadrature.hpp      # unchanged
        ScfWorker.{hpp,cpp}           # unchanged (used by HFVisualizerApp)
        RadialFieldSampler.{hpp,cpp}  # unchanged
        DensityView3D.{hpp,cpp}       # unchanged
        SpectrumView.{hpp,cpp}        # unchanged
        ConvergenceChart.{hpp,cpp}    # unchanged
        HFVisualizerApp.{hpp,cpp}     # +1 optional: accept a seed Z
    tools/
        requirements.txt
        plot_scf_convergence.py  plot_orbitals.py
        plot_diagnostics.py
    out/     media/                   # gitignored
```

## Verification

1. Phase 1: `README.md` and `docs/SIMULATION.md` render; no code changed;
   project still builds and the ImGui app still runs.
2. Phase 3: `01_hartree_fock --selftest` reproduces the analytic hydrogen
   energies (`relErr < 1e-3`, `Z = 1,10,50,90`, `l = 0..3`) with **no window
   and no GL context**; `--scf-selftest` reproduces He/Ne/Ar/Cr/Cu within
   tolerance of the Python reference numbers.
3. Phase 3: `--deck decks/argon.toml --out out/argon` writes a `manifest.json` +
   per-iteration `rho`/`v_eff`/`eps` frames + `diagnostics.csv` that
   `tools/plot_diagnostics.py` reads without error; final `E_total` matches the
   `--scf-selftest` value.
4. Phase 4: `tools/plot_scf_convergence.py out/argon` produces an MP4 whose last
   frame matches the ImGui app's converged view for Ar; `plot_orbitals.py` on
   `out/argon` reproduces the known Ar shell structure (1s/2s/2p/3s/3p radial
   densities).
5. Phase 5: a clean `git status` after `--deck` runs for every deck + all tools.

## Progress

- [ ] Phase 1 — README.md + docs/SIMULATION.md
- [ ] Phase 2 — HFSim : fw::Simulation + deck parsing
- [ ] Phase 3 — main.cpp dispatch + --selftest / --scf-selftest / --deck
- [ ] Phase 4 — decks/ + tools/
- [ ] Phase 5 — .gitignore + top-README cross-links

## Not in scope (future, tracked elsewhere)

- **Relativistic mode** (`--relativistic` → Dirac–Fock–Slater) — see
  `Physics Simulations/Quantum Mechanics/Dirac_Solver/Dirac_Solver_Plan.md`.
  This cleanup makes `01` the safe landing site for that port (decks +
  selftests in place).
- **A no-ImGui `fw::SimApp` live view** with an SCF heatmap in HUD style —
  possible later; not needed while `HFVisualizerApp` covers interactive use.
- **Refactoring `RunScf` into an external stepper** — the replay-the-history
  approach in Phase 2 avoids it; revisit only if a live per-iteration SimApp
  view is built.
