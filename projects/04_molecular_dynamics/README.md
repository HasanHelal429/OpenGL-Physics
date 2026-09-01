# 04 — Molecular dynamics: 3D periodic Lennard-Jones fluid

A classic all-pairs-within-cutoff LJ fluid: velocity-Verlet integration, a
linked-cell neighbor list with a Verlet skin, shifted-force cutoff
truncation, Berendsen/velocity-rescale/Nose-Hoover thermostats, and a
Berendsen NPT barostat. Reduced LJ units throughout (mass = sigma = epsilon
= k_B = 1), the standard convention that lets density/temperature/pressure
be quoted as the dimensionless rho\*, T\*, P\* used across the LJ
phase-diagram literature.

**Two ways to run this project**, and both stay supported (see the top-level
README's "Two ways a project can run"):

- **`MDApp`** (no arguments): the original app -- ImGui control panel, live
  `Chart2D` plots, a presets dropdown (Gas/Liquid/Solid/Custom). Good for
  interactively poking at parameters and watching charts update live.
- **`MDSim`** (`--deck`): implements `fw::Simulation`, driven by an input
  deck through either `fw::RunHeadless` (batch: writes `pos`/`vel`/
  diagnostics for `tools/*.py`) or `fw::SimApp` (interactive: orbit camera +
  free-floating HUD, no ImGui). This is the reproducible, deck-driven path
  the validation below actually uses.

## Physics

**Potential.** Lennard-Jones, shifted-force truncated at `cutoff` (sigma
units): `F(r)` and `U(r)` both have `F(cutoff)`/the matching linear term
subtracted so force AND energy go smoothly to zero exactly at the cutoff,
instead of the force jumping there. An unshifted hard cutoff doesn't just
lose accuracy -- it breaks energy conservation outright (the discontinuity
injects/removes energy every time a pair crosses the cutoff radius).

**Integration.** Velocity-Verlet, `dt` in reduced time units. Force
evaluation uses a linked-cell neighbor list (cell size = cutoff, so any pair
closer than cutoff must share a cell or one of its 26 neighbors -- O(N) at
fixed density instead of O(N^2)) with a Verlet skin so the list only needs
rebuilding every `time.neighbor_rebuild_every` steps; falls back to
brute-force all-pairs when the box is too small to fit >=3 cells per axis
(exercised directly by `--selftest`'s "small" case).

**Thermostats** (`thermostat.type`):
- `berendsen` -- weak velocity rescaling toward `target_t` over relaxation
  time `tau`. Fast, robust, but does not reproduce the true NVT kinetic-
  energy variance (it suppresses fluctuations rather than sampling them
  correctly) -- fine for equilibration, not for fluctuation statistics.
- `velocity_rescale` -- a hard rescale every `rescale_every` steps. Cruder
  than Berendsen, kept for comparison/legacy parity with the original
  `MDApp` presets.
- `nose_hoover` -- deterministic extended-Lagrangian thermostat (single
  friction variable `xi`, no chain): the `-xi*v` friction term is folded
  directly into the equations of motion via a half-step-split
  velocity-Verlet (Frenkel & Smit, *Understanding Molecular Simulation*,
  Sec. 6.1.2), rather than a post-hoc rescale -- this is what actually
  samples the canonical ensemble correctly. `Q = dof*target_t*tau^2`.
  `MDSystem::NoseHooverInvariant` tracks `KE + PE + 0.5*Q*xi^2 +
  dof*target_t*integral(xi dt)`, the quantity this scheme actually
  conserves (reduces to `TotalEnergy()` when this thermostat isn't active,
  since `xi` then stays exactly 0).
- `none` -- microcanonical (NVE). The primary check that the integrator +
  force law conserve energy at all (see Validation below).

**Barostat** (`barostat.enabled`): isotropic Berendsen pressure coupling --
`mu^3 = 1 - (dt/tau_p)*compressibility*(target_p - P)`, applied as a uniform
rescale of the box and every particle position. Weak coupling (same
fluctuation-suppression caveat as the Berendsen thermostat), and **couples
to the raw (cutoff) pressure**, not the tail-corrected one below -- the tail
correction is a reporting-only static addend that never feeds back into the
dynamics, so `target_p` should be read against the `pressure` diagnostic
column, not `pressure_tail` (see Validation).

**Tail corrections.** `cutoff` throws away every pair beyond it; the
standard mean-field correction (assumes `g(r)=1` beyond the cutoff -- Allen
& Tildesley eq. 2.98/2.120) is added for *reporting* only
(`potential_tail`, `pressure_tail` diagnostic columns) -- these do NOT
affect the dynamics, since the actual forces stay cutoff-only. NVE
conservation and the barostat both correctly use the raw, uncorrected
quantities that match the forces actually being integrated.

**Melting/freezing ramp** (`[ramp]`, optional): a piecewise-linear
`thermostat.target_t(t)` schedule (`[[ramp.segments]] { t0, t1, t_start,
t_end }`), applied fresh every substep in `MDSim::Step` regardless of which
thermostat is active -- Nose-Hoover's `Q = dof*target_t*tau^2` recomputes
from the current (ramped) `target_t` automatically, no extra plumbing
needed. Two order parameters track the resulting structural state, both
computed in `MDSim` (not `MDSystem` -- they're bookkeeping on top of the
physics, not physics themselves):
- `lindemann` -- rms displacement of each particle from its ORIGINAL FCC
  lattice site (the deck's IC starts exactly on the lattice), divided by
  the FCC nearest-neighbor distance at the deck's density. Reconstructed
  incrementally every substep via minimum-image frame-to-frame
  differencing (exact for a fixed box; not used together with the barostat
  above, whose uniform rescale isn't a pure translation). This is a sharp,
  reliable melting-*onset* detector (a stable solid stays near the classic
  Lindemann-criterion range ~0.10-0.15) but a one-way one: once melted,
  atoms diffuse and never return to their literal original coordinates even
  if the system recrystallizes elsewhere, so it keeps growing for the rest
  of any cooling branch regardless of what actually happens structurally.
- `rdf_peak` -- the tallest bin of `MDSystem::ComputeRDF`, recomputed fresh
  from the instantaneous configuration once per frame (cheap relative to a
  force evaluation, not run every substep). Unlike `lindemann` this depends
  on no history/reference position, so it's usable on both the heating
  and cooling branches -- see Validation below for what it actually showed.

**Initial conditions.** `Scenarios::BuildFccLattice`: particles placed on an
FCC lattice (never randomly -- an unstructured placement risks two atoms
overlapping, which the repulsive LJ core turns into a force blow-up on the
very first step) sized to hit the deck's `density` as closely as an integer
number of unit cells allows, velocities drawn from a Maxwell-Boltzmann
distribution at `temperature`, center-of-mass velocity zeroed, then rescaled
so the sampled instantaneous temperature lands exactly on the target. Kept
in C++ (not a Python `tools/make_*_ic.py` + `ic/` file, unlike the SPH/GRHD
projects) -- it's cheap, deterministic, and already seeded; there's no
sampling complexity here that would benefit from being inspectable/reusable
outside the binary.

## Deck format

```toml
title = "..."

[system]      target_n = 500       # actual N = 4*round(cbrt(target_n/4))^3 (FCC unit cell has 4 atoms)
              density = 0.8442     # rho* = N/V
              temperature = 1.44   # initial condition's T* (thermostat.target_t can differ -- see below)
              seed = 1

[potential]   cutoff = 2.5         # sigma units
              skin = 0.3           # Verlet-list skin beyond cutoff

[thermostat]  type = "berendsen"   # "none" | "berendsen" | "velocity_rescale" | "nose_hoover"
              target_t = 0.72
              tau = 1.0            # Berendsen relaxation time OR Nose-Hoover tau (whichever type is active)
              rescale_every = 20   # velocity_rescale only

[barostat]    enabled = false
              target_p = 0.05      # targets RAW pressure, not pressure_tail -- see Physics above
              tau_p = 2.0
              compressibility = 1.0

[time]        dt = 0.002
              substeps_per_frame = 25
              frames = 400
              neighbor_rebuild_every = 5
```

Output fields (batch mode, `(N,3)` float32): `pos`, `vel`. Diagnostics:
`kinetic`, `potential`, `potential_tail`, `total_energy`, `temperature`,
`pressure`, `pressure_tail`, `box_length`, `density`, `nh_xi`,
`nh_invariant` (the last two are 0 unless `thermostat.type = "nose_hoover"`),
`target_t` (the instantaneous, possibly-ramped thermostat target),
`lindemann`, `rdf_peak` (see the ramp/melting description above --
`lindemann` is 0 and `rdf_peak` is still meaningful even without `[ramp]`).

`[ramp]` (optional):

```toml
[ramp]
enabled = true
[[ramp.segments]]
t0 = 0.0     # simulated time, this MDSim instance's own clock (starts at 0 on Configure()/Reset())
t1 = 50.0
t_start = 0.10
t_end = 3.00
[[ramp.segments]]
t0 = 50.0
t1 = 100.0
t_start = 3.00
t_end = 0.10
```

## Build & run

```sh
cmake --preset release
cmake --build --preset release --target 04_molecular_dynamics

# original app: ImGui panel + live charts, no arguments
04_molecular_dynamics

# cross-check the neighbor list (both the brute-force-fallback and real
# linked-cell code paths) and the shifted-force LJ gradient against
# independent references
04_molecular_dynamics --selftest

# batch: run a deck, write raw data
04_molecular_dynamics --deck decks/liquid.toml --out out/liquid

# interactive: window + HUD + orbit camera, particles colored by speed
04_molecular_dynamics --interactive --deck decks/liquid.toml
```

Then, from `tools/` (needs `numpy`, `matplotlib`):

```sh
python tools/plot_diagnostics.py out/liquid       # energy/temperature/pressure vs time
python tools/plot_rdf.py out/liquid               # g(r) from the final frame
python tools/msd.py out/liquid                    # mean-squared displacement -> diffusion coefficient
python tools/eos_check.py out/liquid out/npt_liquid   # state-point table + NVT/NPT cross-consistency
python tools/plot_melting.py out/melting_ramp     # lindemann/rdf_peak vs target_t, heating vs cooling
```

## Validation

### `--selftest`

Two independent-reference cross-checks (a brute-force-force reference
re-derived directly from the shifted-force LJ formula, with no knowledge of
`MDSystem`'s neighbor list) plus a numerical-gradient check:

```
selftest (small (brute-force fallback)): max relative error  accel=2.016e-16  energy=5.919e-16  -- PASS
selftest (multicell (linked-cell)):      max relative error  accel=3.888e-17  energy=6.930e-16  -- PASS
selftest (numerical-gradient):           analytic=-2.172694  numerical=-2.172694  relerr=7.865e-11 -- PASS
```

`small` (N=40, box edge deliberately < 3*cutoff) exercises `BuildCellListPairs`'s
brute-force fallback branch; `multicell` (N=400, box several cells wide)
exercises the real 27-neighbor-cell linked-cell path. Both agree with the
independent reference to machine precision -- the neighbor list finds
exactly the right pairs in both regimes. The numerical-gradient case checks
that the analytic force is actually `-dU/dr` of the same shifted-force
potential `PotentialEnergy()` reports (central finite difference on a single
pair's separation), independent of literature agreement.

### NVE energy conservation (`decks/nve_equilibration.toml`, rho\*=0.8442, T\*=0.72)

| quantity | t=0 | t=20 (final) |
|---|---|---|
| kinetic | 538.920 | 294.893 |
| potential | -2846.639 | -2602.606 |
| total energy | -2307.719 | -2307.713 |
| instantaneous temperature | 0.7200 | 0.3940 |

**Energy conserved to `2.77e-6` relative** over 20 time units (10000
substeps) -- the core check that velocity-Verlet + the shifted-force cutoff
actually conserve energy. The temperature drop from 0.72 to ~0.39 is *not* a
bug: an FCC lattice is not the equilibrium liquid structure at this density,
so as the artificial ordered start relaxes into a genuine disordered liquid
arrangement, potential energy rises (less negative) by exactly the amount
kinetic energy falls (244 units each way, matching to 3 significant
figures) -- total energy stays fixed while its split shifts. This is exactly
why `decks/liquid.toml` deliberately starts hot (`temperature = 1.44`,
`thermostat.target_t = 0.72`): the thermostat needs to both absorb this same
structural-relaxation transient *and* remove the deliberate 2x initial
excess to land on the intended state point.

### Thermostat convergence

| deck | thermostat | initial T | target T | final/tail-mean T |
|---|---|---|---|---|
| `liquid.toml` | Berendsen | 1.44 | 0.72 | 0.7182 (final) |
| `solid_fcc.toml` | Berendsen | 0.30 | 0.30 | 0.2987 (final) |
| `gas.toml` | Berendsen | 2.00 | 2.00 | 1.9952 (final) |
| `nose_hoover_liquid.toml` | Nose-Hoover | 1.44 | 0.72 | 0.7282 +/- 0.0329 (tail mean +/- std) |

All four land within 1-2% of target. The Nose-Hoover run is reported as a
tail-averaged mean+std rather than a single final-frame value because a
correct NVT thermostat is *supposed* to leave real fluctuations in
instantaneous temperature (Berendsen deliberately suppresses them instead --
see Physics above) -- `0.033` standard deviation around a `0.728` mean is
the expected behavior, not noise to be minimized.

### Nose-Hoover invariant vs. raw energy (`decks/nose_hoover_liquid.toml`)

| quantity | relative drift over the run |
|---|---|
| raw total_energy | `1.328e-2` |
| `nh_invariant` (`KE+PE+0.5*Q*xi^2+dof*T0*integral(xi dt)`) | `4.839e-6` |

Raw energy drifts by more than 1% because the thermostat is deliberately
pumping/removing energy to hold temperature at the target through the same
lattice-relaxation transient described above -- expected, not an error. The
actual conserved quantity of the extended Lagrangian holds to `4.8e-6`,
matching the NVE case's `2.77e-6` to within the same order of magnitude:
the half-step-split Nose-Hoover integrator is implemented correctly.

### NPT barostat (`decks/npt_liquid.toml`, target_p\*=0.05, T\*=0.72)

Starts looser (rho\*=0.70) than `liquid.toml`'s rho\*=0.8442. Over 125 time
units (2500 frames) the density visibly overshoots (a fast initial
relaxation transient pushes it to ~0.89 by t=2, since the box also has to
absorb the same hot-start-then-cool transient as `liquid.toml`) then relaxes
back down, settling by ~t=60:

| quantity | tail mean (last 25% of run) | target |
|---|---|---|
| density | `0.7228 +/- 0.0044` | -- |
| **raw** pressure | `0.0499 +/- 0.0753` | `0.05` |
| tail-corrected pressure | `-0.5089` | (not what the barostat targets) |

The raw pressure converges to within `0.0001` of `target_p` -- the barostat
works. The large gap between raw and tail-corrected pressure at this state
point (`~0.56`) is exactly the analytic tail correction at this density/cutoff,
and is the reason the Physics section above calls out explicitly which
pressure column `target_p` actually means: comparing `target_p` against
`pressure_tail` instead (an earlier version of `tools/plot_diagnostics.py`
did exactly this) looks like the barostat converged to entirely the wrong
pressure, when it had actually converged correctly.

### Radial distribution function (`decks/liquid.toml`, final frame)

```
first peak: r=1.090  g(r)=3.073
g(r) at r=4.09 (largest bin): 1.014
```

A sharp first peak near `r ~= 1.1*sigma` with height in the 2.5-3.5 range,
decaying to `g(r) ~= 1` at large r, is the textbook signature of a Lennard-
Jones liquid near the triple point -- this is a qualitative structural
sanity check (see `tools/plot_rdf.py`'s docstring for why this project
doesn't attempt a decimal-precision literature comparison here).

### Self-diffusion coefficient (`decks/liquid.toml`)

```
max single-frame displacement / (0.5*L): 0.045   (unwrapping assumption comfortably holds)
diffusive-regime fit over t in [7.50, 14.95]
self-diffusion coefficient D = 0.03017  (reduced units, sigma^2/tau)
```

`tools/msd.py` reconstructs unwrapped trajectories from the wrapped
per-frame positions via minimum-image frame-to-frame differencing (valid as
long as no particle moves more than half the box between two saved frames --
the `0.045` ratio above confirms that assumption holds by a wide margin) and
fits the Einstein relation `MSD ~ 6*D*t` over the diffusive-regime tail,
skipping the initial ballistic (`t^2`) transient.

### Cross-consistency (`tools/eos_check.py`)

```
run                              T       rho    P (tail)    T/Tc  rho/rhoc
liquid                      0.7180    0.8442     0.72368   0.544     2.723
npt_liquid                  0.7163    0.7228    -0.50886   0.543     2.332
```

`liquid.toml` and `npt_liquid.toml` are independent decks with different
targets (fixed density vs. fixed pressure), not expected to land on the
same state point -- `eos_check.py` reports both for side-by-side comparison
and, for context only, each point's position relative to the LJ critical
point (`Tc*=1.32`, `rho_c*=0.31` -- Thol, Rutkai, Koester, Lustig, Span,
Vrabec, *Equation of State for the Lennard-Jones Fluid*, 2016, verified
directly from that paper rather than reimplementing its full multiparameter
EOS from transcribed coefficients, which would be a real transcription-risk
not worth taking for a demo validation script). Both points sit at
`T/Tc ~= 0.54`, deep in the liquid region, consistent with the RDF/MSD
results above actually describing a liquid and not a supercritical fluid.

### Melting/freezing hysteresis (`decks/melting_ramp.toml`, rho\*=0.95, heat 0.10->3.00 then cool back over t in [0,100])

```
melting detected: lindemann crosses 0.3 between target_t=1.617 (lindemann=0.298) and target_t=1.619 (lindemann=0.303)
rdf_peak: initial(t=0)=15.91  final(t=100.0)=4.66  liquid-plateau min=2.20
```

`lindemann` stays in `0.09-0.21` from `target_t=0.10` up through `~1.55`
(comfortably inside the classic ~0.10-0.15 Lindemann-criterion range for a
stable solid, allowing for this being a small, finite, thermally-vibrating
system rather than an infinite one), then rises sharply through `~1.6-1.9`
and keeps climbing for the rest of the run, reaching `4.97` by the end --
exactly the one-way behavior described in Physics above: a real melting
event around `target_t~1.6`, but `lindemann` cannot see whatever happens on
the cooling branch afterward, since molten atoms never return to their
original coordinates.

`rdf_peak` tells the rest of the story: it starts at `15.9` (the essentially
perfect starting lattice), collapses to `~4.6` within the first few time
units as thermal motion broadens the crystal's peak, continues falling
through melting to a liquid plateau around `2.2-2.6` through most of the
hot part of the run, then **rises again on cooling** -- back up to `4.66` by
`t=100` (`target_t=0.10`). That rise is a real (partial) re-ordering signal,
but `4.66` is far short of the original `15.9`: **this cooling rate outran
recrystallization** and the run ends in a disordered/glassy configuration
rather than cleanly refreezing into the original crystal. That is a
legitimate, well-known MD outcome for a fast quench (LJ systems are classic
glass formers under rapid cooling), reported here as what the run actually
produced rather than an idealized symmetric hysteresis loop -- see
`tools/plot_melting.py`'s docstring for the full reasoning, and its
`melting.png` for the two order parameters plotted against `target_t` on
both branches.

## Known simplifications

- **Single species only.** No mixtures yet (planned: a binary Kob-Andersen
  glass former, a later phase of this project).
- **Weak-coupling Berendsen thermostat/barostat don't sample correct
  ensemble fluctuations** (see Physics above) -- Nose-Hoover fixes this for
  temperature; there is no fluctuation-correct (e.g. Parrinello-Rahman)
  barostat yet.
- **CPU-only.** Unlike the GPU-compute-shader projects (`05`-`07`), the
  force/neighbor-list kernels here are OpenMP-parallel CPU code, not GPU
  compute shaders (planned as a later phase, porting the CPU implementation
  as the `--selftest` ground truth).
- **`lindemann` is a one-way melting-onset detector, not a hysteresis-loop
  diagnostic by itself** (see the melting/freezing Validation above) --
  `rdf_peak` covers the two-way case, but a proper bond-orientational order
  parameter (Steinhardt Q6, translation- *and* permutation-invariant) would
  be a more rigorous crystallinity measure than a single RDF-peak height.

## Progress

- [x] Migrated to `fw::Simulation`/`Deck`/headless model (`MDSim`) alongside
      the original ImGui app (`MDApp`), which stays as a separate build path
      -- see the top-level README's "Two ways a project can run"
- [x] `--selftest`: neighbor list (both code paths) + force-vs-numerical-
      gradient, both validated to machine precision against independent
      references
- [x] Nose-Hoover thermostat, validated via its conserved-invariant drift
      (`4.8e-6`, matching the NVE case) vs. raw energy drift (`1.3e-2`)
- [x] Analytic long-range tail corrections for energy/pressure (reporting
      only, does not feed back into the dynamics)
- [x] Berendsen NPT barostat, validated: converges raw pressure to within
      `0.0001` of `target_p`
- [x] `tools/plot_diagnostics.py`, `tools/plot_rdf.py`, `tools/msd.py`,
      `tools/eos_check.py` -- all validated against real runs, see above
- [x] Melting/freezing demo: `[ramp]` deck-driven `target_t(t)` schedule +
      `lindemann`/`rdf_peak` order parameters. Clean melting-onset detection
      (`target_t~1.6`); cooling branch shows partial re-ordering but not
      full recrystallization at this quench rate -- see Validation above
- [ ] Binary Kob-Andersen mixture
- [ ] GPU compute-shader port (with a CPU-vs-GPU `--selftest` cross-check,
      matching `06`/`07`'s pattern)
