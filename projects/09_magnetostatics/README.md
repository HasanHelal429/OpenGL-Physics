# 09_magnetostatics

2D magnetostatics on the `fw::Simulation` / deck / headless model, a 3D
Biot-Savart field, and a relativistic Boris test-particle pusher.
`Magnetostatics_Plan.md` has the design and phase plan; `docs/SIMULATION.md`
is the full writeup (methods, the multigrid boundary gotcha, validation).

The out-of-plane current problem is the same 5-point Poisson stencil as
`Physics Simulations/Electromagnetism/Poisson_Solver/poisson.py`, but solved
by **iterative relaxation** (RB-GS/SOR, or geometric multigrid) rather than a
sparse direct solve -- an iterative smoother is what ports to the GPU.

```
laplacian(A_z) = -mu0 * J_z          B = curl(A_z zhat) = (dA_z/dy, -dA_z/dx)
```

Dirichlet (`A_z = 0` on a large box, right for compact sources) or Neumann
(natural far boundary, right for a solenoid's uniform field).

## Build & run

```sh
cmake --build --preset release --target 09_magnetostatics
EXE=./build/release/projects/09_magnetostatics/09_magnetostatics.exe

$EXE --selftest        # manufactured-solution convergence (2nd order)
$EXE --mg-scaling      # multigrid W-cycle count vs grid size (flat)
$EXE --biot-selftest   # GPU Biot-Savart vs analytic loop / Helmholtz
$EXE --boris-selftest  # relativistic pusher: cyclotron / E x B / energy

$EXE --deck projects/09_magnetostatics/decks/wire.toml --out out/wire
python projects/09_magnetostatics/tools/plot_wire.py out/wire
$EXE --deck projects/09_magnetostatics/decks/helmholtz.toml --out out/hh
python projects/09_magnetostatics/tools/plot_helmholtz.py out/hh

$EXE --interactive --deck projects/09_magnetostatics/decks/playground.toml
$EXE --deck .../playground.toml --render-check out/view.png   # one offscreen frame
```

Interactive keys: **M** cycle field (|B| / Bx / By / A_z), **L** field-lines,
**Tab** select wire, **arrows** move the active wire (re-solves live),
**[** **]** gain, **-** **=** gamma, mouse drag/scroll pan/zoom, **0** reset.

## Status

- [x] Phase 1 -- 2D field solver (RB-GS/SOR) + `--selftest` + `wire` / `solenoid`
- [x] Phase 2 -- geometric multigrid (grid-independent W-cycle) + interactive
      view (heatmap, field-lines, live wire editing) + `--render-check`
- [x] Phase 3 -- 3D Biot-Savart on the GPU (`[[coil]]` loop / solenoid /
      helmholtz, xy/xz/yz slice) + `--biot-selftest`
- [x] Phase 4 -- relativistic Boris test-particle pusher (`[background]` B/E,
      `[[charge]]`), trajectory trails + per-charge diagnostics, `--boris-selftest`
- [x] Phase 5 -- `docs/SIMULATION.md`, `Studies/magnetostatics/`
      (`multigrid_scaling`, `helmholtz_uniformity_vs_spacing`),
      `tools/make_movie.py`

All phases complete.

## Decks

| deck | shows | check |
|---|---|---|
| `wire.toml` | single out-of-plane line current | `\|B\| = mu0 I / 2 pi r` on an interior annulus (~0.3%) |
| `solenoid.toml` | opposite current sheets (Neumann BC) | uniform `B_x = mu0 K_s` between (~2%), ~0 outside |
| `playground.toml` | three wires, lighter grid | for `--interactive` |
| `loop.toml` | one 3D current loop, xz slice | on-axis `B_z = mu0 I R^2/2(R^2+z^2)^{3/2}` (<0.03%) |
| `helmholtz.toml` | Helmholtz pair, xz slice | centre field vs `(4/5)^{3/2} mu0 I/R` (3e-5), `d^2B/dz^2 ~ 0` |
| `cyclotron.toml` | one charge in uniform B | orbit radius / period / `KE` (pusher: r to 3e-7, `dgamma` 1e-15) |
| `exb_drift.toml` | two opposite charges, crossed E, B | drift together at `v_d = E x B / B^2` (1e-5) |
| `magnetic_bottle.toml` | charge in a mirror trap (two like-current loops) | bounces between throats; `mu` conserved bounce-to-bounce ~3.5% |
| `magnetic_cusp.toml` | same two coils as the bottle, opposite current | null/saddle topology instead of a mirror; trapped-bottle IC escapes through it on the first pass |
| `loss_cone.toml` | three pitch angles, same bottle field | critical angle found by bisection: trapped to ~34 deg, escaped by ~34.5 deg |
| `toroidal_ring.toml` | 8 loops arranged tokamak-style on a ring | qualitative -- no single symmetry axis for an analytic check |

## Solver notes

- **Multigrid** (`solver.method = "multigrid"`, default for Dirichlet): RB-GS
  smoother, transpose-compatible bilinear transfers (`R = P^T / 4`),
  rediscretised coarse operator, homogeneous reflective-ghost edge on every
  level, W-cycle. 8-10 W-cycles to 1e-9 relative residual from 64^2 to 1024^2.
- **RB-GS / SOR** (`solver.method = "rbgs"`): optimal-omega SOR, the Phase 1
  reference and the Neumann-boundary path.
