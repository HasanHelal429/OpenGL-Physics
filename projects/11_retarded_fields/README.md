# 11_retarded_fields

Liénard-Wiechert fields of moving point charges -- the exact point-source
solution of Maxwell's equations -- on the `fw::Simulation` / deck / headless
model. `Retarded_Fields_Plan.md` has the design; `docs/SIMULATION.md` the
writeup.

Two modes:

- **prescribed** -- charges follow analytic paths (static / uniform /
  circular / linear oscillator / figure-8); a GPU compute shader evaluates
  `E`, `B` on a slice grid via the retarded-time root-find. Interactive.
- **self_consistent** -- charges move under each other's *retarded* fields
  (pairwise, no self-force), a trajectory ring buffer with cubic lookup
  feeds the retarded solve, a relativistic Boris pusher integrates the
  motion. Radiative inspiral / bremsstrahlung / beaming emerge from it.

Units `c = eps0 = mu0 = 1`.

```
E = q/(4 pi eps0) [ (n-beta)(1-beta^2)/(kappa^3 R^2)
                    + n x ((n-beta) x betadot)/(c kappa^3 R) ],
kappa = 1 - n.beta ,   B = n x E / c ,   evaluated at t_ret.
```

## Build & run

```sh
cmake --build --preset release --target 11_retarded_fields
EXE=./build/release/projects/11_retarded_fields/11_retarded_fields.exe

$EXE --selftest            # static Coulomb, uniform motion, Larmor sphere flux
$EXE --gpu-selftest        # fp32 compute shader vs the fp64 reference
$EXE --dynamics-selftest   # kepler pair: orbit holds, energy budget closes
$EXE --beaming-selftest    # synchrotron: 1/gamma half-angle, P ~ gamma^4
$EXE --brems-selftest      # Coulomb bremsstrahlung: E_rad ~ b^-3
$EXE --inspiral-selftest   # inspiral rate vs Larmor / dipole power

$EXE --deck projects/11_retarded_fields/decks/synchrotron.toml --interactive
$EXE --deck projects/11_retarded_fields/decks/oscillating_charge.toml --out out/osc
python projects/11_retarded_fields/tools/plot_pattern.py out/osc --dipole

$EXE --deck projects/11_retarded_fields/decks/two_body_inspiral.toml --out out/insp
python projects/11_retarded_fields/tools/plot_inspiral.py out/insp --q 1.5
python projects/11_retarded_fields/tools/make_movie.py out/insp
```

## Status

- [x] Phase 1 -- GPU Liénard-Wiechert field grid, prescribed analytic paths,
      retarded-time Newton+bisection with per-cell warm start. `--selftest`
      (Coulomb 2e-16, uniform motion 5e-16, Larmor 8e-7); `--gpu-selftest`
      (radiation-zone rel-L2 ~1e-3, fp32 vs fp64)
- [x] Phase 2 -- `fw::SimApp` interactive view: `|E|`/`Ex`/`Ez`/S_radial
      heatmap, log + r-weight display modes, drag the charge, sweep omega /
      radius, gamma readout. `--render-check <png>`
- [x] Phase 3 -- `TrajectoryBuffer` ring buffer + cubic lookup, pairwise
      retarded Lorentz force, relativistic Boris pusher, seed-orbit prefill
      (`kepler` / `uniform` / `static`). `--dynamics-selftest`: kepler pair
      holds shape over 3 periods, energy budget closes to 1.9e-3
- [x] Phase 4 -- `--beaming-selftest` (half-angle * gamma const to 8%,
      P ~ gamma^4 to 5 digits), `--brems-selftest` (E_rad ~ b^-2.86),
      `--inspiral-selftest` (dE_mech/dt tracks dE_rad/dt to 2.6%; rate is
      ~0.55 P_dipole -- the no-self-force half). tools: `larmor_ref.py`,
      `array_factor_ref.py`, `inspiral_ref.py`, `plot_pattern.py`,
      `plot_inspiral.py`
- [ ] Phase 5 -- `docs/SIMULATION.md` (done), `Studies/retarded_fields/`,
      `make_movie.py` (done), website Plotly inspiral

## Decks

| deck | mode | shows |
|---|---|---|
| `static_charge.toml` | prescribed | Coulomb field, `B = 0` |
| `oscillating_charge.toml` | prescribed | dipole radiation lobes (`sin^2 theta`) |
| `dipole_array.toml` | prescribed | two-source interference / array factor |
| `synchrotron.toml` | prescribed | ultra-relativistic beaming (`beta = 0.98`, gamma = 5) |
| `two_body_inspiral.toml` | self_consistent | bound `+q / -q` pair spiralling in |
