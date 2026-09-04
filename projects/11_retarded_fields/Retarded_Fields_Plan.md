# projects/11_retarded_fields — Design Plan

## Context

`Physics Simulations/Electromagnetism/Retarded_Fields/retarded_fields.py`
already has a validated Liénard-Wiechert solver with a numba-parallel grid
path (`lw_fields_grid_jit`). This project takes it in two directions the
Python version can't reach comfortably:

1. **GPU field-grid evaluation** — the per-point retarded-time root-find is
   an ideal compute-shader workload, making the radiated-field
   visualization *interactive* (drag the charge, sweep its orbit frequency,
   watch the pattern reshape live).
2. **Self-consistent N-charge dynamics** — charges move under each other's
   *retarded* fields, integrated with a relativistic pusher, so radiative
   inspiral, bremsstrahlung and beaming emerge from the dynamics rather
   than being prescribed.

These are independent. Build (a) first (direct port of validated math),
then (b).

House model: `fw::Simulation` + TOML deck, headless batch + `fw::SimApp`
interactive, `tools/*.py` for analytic references and plots.

## Numerical approach

### Retarded time

For observation event `(x, t)` and source path `r(τ)`, solve
`c(t − t_ret) = |x − r(t_ret)|` for `t_ret ≤ t`. Port
`retarded_fields.py`'s safeguarded Newton solve (analytic derivative
`g'(t_ret) = −c + n·v`, bisection fallback whenever a Newton step would
leave the maintained bracket). Warm-start from the neighbouring cell's or
previous frame's `t_ret` — retarded time varies slowly and continuously
over a fixed grid, so this collapses the iteration count after frame 0
(same trick the Python `t_ret_guess` argument already exploits).

### Liénard-Wiechert fields (`LwField`, GPU)

Standard velocity (near) + acceleration (radiation) split:

```
E = q/(4πε₀) [ (n−β)(1−β²) / (κ³ R²)  +  n × ((n−β) × β̇) / (c κ³ R) ] ,   κ = 1 − n·β
B = n × E / c
```

evaluated at `t_ret`. One compute-shader thread per grid point; the charge
path is sampled analytically (prescribed mode) from a small parameter set,
or from a trajectory buffer (self-consistent mode). Fields superpose over
multiple charges. This is a line-by-line port of `_lw_fields_jit_core`.

### Trajectory history (self-consistent mode)

Each charge keeps a ring buffer of `(τ, r, v, a)` samples spanning at least
the domain's maximum light-crossing time. Retarded-time lookups interpolate
(cubic in `τ`) within the buffer. Startup: pre-fill the buffer with an
assumed past — uniform motion, or the non-radiating Kepler / cyclotron
orbit — so forces are well-defined from `t = 0`; diagnostics flag the
settling window.

### Equations of motion

Relativistic Lorentz force from the summed retarded fields of *all other*
charges (pairwise; the self-force is omitted — the Abraham-Lorentz-Dirac
self-term is pre-acausal / runaway and out of scope). Momentum update by
the Boris pusher shared with `09_magnetostatics`, or a proper-time RK4.
Energy accounting: kinetic + field-interaction energy vs the time-integrated
radiated power (Larmor / Liénard) is the conservation check.

## Deck schema

```toml
[domain]   nx, ny, lx, ly, c, eps0
[mode]     "prescribed" | "self_consistent"
[[charge]] q, m
           # prescribed: path = "circular" | "linear_oscillator" | "figure8"
           #             radius, omega, phase, amplitude
           # self_consistent: x0, v0, seed_orbit = "kepler" | "cyclotron" | "uniform"
[history]  buffer_span, samples            # self-consistent only
[external] B0, E0                          # optional static background (traps, deflectors)
[time]     dt, steps
[render]   field = "E_mag" | "E_energy" | "S_mag" | "S_radial"
           log_scale, quiver
```

## Scenes / decks

| deck | mode | shows | validation target |
|---|---|---|---|
| `static_charge.toml` | prescribed | Coulomb field, `B = 0` | exact vs `retarded_fields.py` Phase-1 checks |
| `oscillating_charge.toml` | prescribed | dipole radiation lobes | matches Python `Oscillating_Charge.ipynb`; Larmor flux on a far circle |
| `dipole_array.toml` | prescribed | two-source interference | array factor vs `Dipole_Radiation.ipynb` (corr > 0.999) |
| `synchrotron.toml` | prescribed | ultra-relativistic circular motion | forward-beam half-angle `~1/γ`; power `∝ γ⁴` |
| `two_body_inspiral.toml` | self-consistent | bound charge pair spiralling in | orbit-averaged `dE/dt` vs analytic radiated power |
| `bremsstrahlung.toml` | self-consistent | light charge deflected by a heavy fixed charge | radiated energy vs impact parameter; pulse shape |

## File layout

```
projects/11_retarded_fields/
    CMakeLists.txt
    Retarded_Fields_Plan.md
    src/
        main.cpp
        RetardedFieldsSim.{hpp,cpp}   # fw::Simulation
        kernels_lw.hpp                # retarded-time + LW field compute shader
        TrajectoryBuffer.hpp
        ChargePath.hpp                # analytic prescribed paths (parity with the Python)
        Pusher.hpp                    # shared Boris / proper-time RK4
    decks/
    tools/
        larmor_ref.py  array_factor_ref.py  inspiral_ref.py
        plot_pattern.py  plot_inspiral.py  make_movie.py
    docs/SIMULATION.md
```

## Phases

**Phase 1 — GPU LW field grid, prescribed paths.**
Port retarded-time Newton + LW formula to a compute shader; analytic
circular / oscillator / figure-8 paths; frame output of `E`, `|E|`, radial
Poynting. Gate: agreement with `retarded_fields.py` on the static,
uniform-motion (`β = 0.3`) and Larmor-sphere-flux tests to the Python
solver's own tolerances; warm-start cuts mean Newton iterations to ≈2 after
the first frame.

**Phase 2 — Interactive prescribed view.**
`fw::SimApp`: live log-scale `|E|` / Poynting heatmap + direction quiver;
drag the charge; sliders for `ω` and orbit radius; `γ` readout. FBO-to-PNG
render check.

**Phase 3 — Trajectory buffer + self-consistent dynamics.**
Ring buffer, interpolated retarded lookups, pairwise retarded Lorentz
force, relativistic pusher, seed-orbit prefill. Gate: two equal charges
seeded on the non-radiating circular orbit hold position for many periods
before radiative decay is visible (buffer + interpolation inject no
spurious force); energy budget (kinetic + interaction + radiated) closes to
a few %.

**Phase 4 — Radiation-physics validation.**
`two_body_inspiral` decay rate vs analytic radiated power; `bremsstrahlung`
radiated-energy-vs-impact-parameter curve; `synchrotron` beaming angle and
`γ⁴` scaling. Gate: each within its stated tolerance.

**Phase 5 — `docs/SIMULATION.md` + website media.**
The inspiral is the page's Plotly interactive (3D trajectory + field
animation).

## Studies (`Studies/retarded_fields/`)

- `inspiral_rate_vs_separation` — measured orbital-decay rate vs initial
  separation and charge, against the two-body radiated-power law.
- `beaming_vs_gamma` — forward-lobe half-angle and peak power vs Lorentz
  factor, against `1/γ` and `γ⁴`.
- `history_convergence` — inspiral trajectory vs buffer span and sample
  density, to fix safe defaults.

## Progress

- [x] Phase 1 — GPU LW field grid
- [x] Phase 2 — interactive prescribed
- [x] Phase 3 — self-consistent dynamics
- [ ] Phase 4 — radiation validation
- [ ] Phase 5 — docs + media
