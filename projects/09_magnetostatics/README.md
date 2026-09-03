# 09_magnetostatics

2D magnetostatics on the `fw::Simulation` / deck / headless model, plus (later
phases) a 3D Biot-Savart field and a relativistic Boris test-particle pusher.
See `Magnetostatics_Plan.md` for the full design and phase plan.

The out-of-plane current problem is the same 5-point Poisson stencil as
`Physics Simulations/Electromagnetism/Poisson_Solver/poisson.py`, but solved
by **iterative relaxation** (red-black Gauss-Seidel now, GPU geometric
multigrid in Phase 2) rather than a sparse direct solve -- an iterative
smoother is what ports to the GPU.

```
laplacian(A_z) = -mu0 * J_z          B = curl(A_z zhat) = (dA_z/dy, -dA_z/dx)
```

with `A_z = 0` on the domain edge.

## Build & run

```sh
cmake --build --preset release --target 09_magnetostatics

# manufactured-solution convergence check (no GL context needed)
./build/release/projects/09_magnetostatics/09_magnetostatics.exe --selftest

# a physical deck -> A_z, Bx, By, Jz as one .npy frame each
./build/release/projects/09_magnetostatics/09_magnetostatics.exe \
    --deck projects/09_magnetostatics/decks/wire.toml --out out/wire
python projects/09_magnetostatics/tools/plot_wire.py out/wire
```

## Status

- [x] Phase 1 — 2D field solver (RBGS/SOR) + `--selftest` + `wire` / `solenoid`
      validation
- [ ] Phase 2 — GPU multigrid + interactive view
- [ ] Phase 3 — 3D Biot-Savart coils
- [ ] Phase 4 — Boris pusher
- [ ] Phase 5 — docs + Studies + website media

## Decks

| deck | shows | check |
|---|---|---|
| `wire.toml` | single out-of-plane line current | `\|B\| = mu0 I / 2 pi r` on an interior annulus |
| `solenoid.toml` | opposite current sheets (solenoid cross-section) | uniform `B_x = mu0 K_s` between, ~0 outside |
