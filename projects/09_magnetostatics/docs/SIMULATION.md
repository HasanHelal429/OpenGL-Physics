# 09_magnetostatics: how it works

This project takes the 2D electrostatic Poisson solver from
`Physics Simulations/Electromagnetism/Poisson_Solver/poisson.py` and asks
what actually benefits from moving it into the C++/OpenGL framework. The
answer is not "the same static solve, faster" -- a single sparse direct
solve is already excellent in Python. It is the things a real-time,
GPU-friendly, interactive code can do that a one-shot notebook cannot:

- **an iterative solver** (SOR, then geometric multigrid) so a
  current-carrying conductor becomes a draggable object and the field
  re-solves between frames;
- **3D Biot-Savart** over arbitrary coil geometry -- an embarrassingly
  parallel field-point sum with no linear solve at all, a natural
  compute-shader workload;
- **a relativistic test-particle pusher** so the field has something moving
  through it: cyclotron orbits, E x B drift, magnetic-mirror confinement.

The 2D field problem is the same 5-point stencil as `poisson.py`, with a
different source and a curl post-step:

```
laplacian(A_z) = -mu0 J_z          B = curl(A_z zhat) = (dA_z/dy, -dA_z/dx, 0)
```

for a current `J = J_z(x,y) zhat` out of the plane.

## 1. The 2D field solver

### 1.1 Red-black Gauss-Seidel / SOR (Phase 1, the reference)

`FieldSolver.cpp` relaxes the interior cells in two checkerboard passes per
sweep (`(i+j)&1`), with the optimal over-relaxation factor for the Poisson
problem on an `n x n` grid, `omega = 2 / (1 + sin(pi/n))`. Convergence is
judged by the L-infinity residual `rhs - laplacian(u)` over the free cells,
relative to `max|rhs|`.

Two boundary conditions are supported:

- **Dirichlet** (`A_z = 0` on the domain edge) -- the "field decays to zero
  on a large box" condition, right for a compact source like a single wire.
  Edge cells are held fixed and only the interior is relaxed.
- **Neumann** (`dA_z/dn = 0`) -- the natural far boundary, right when the
  physical field should *not* be dragged back to zero at the wall. The
  solenoid deck needs this: a finite Dirichlet box forces the solenoid's
  return flux through the walls and pulls the between-sheets field ~25%
  below `mu0 K_s`; with a natural boundary it lands at 1.9% (finite-width
  end effect). Neumann makes the discrete problem singular, so the rhs mean
  is projected out and the solution's additive constant is pinned to zero
  mean every few sweeps.

**Validation.** The manufactured solution `A(x,y) = sin(pi u/Lx) sin(pi
v/Ly)` (`u,v` measured from the domain corner, zero on the boundary) with
its analytic Laplacian fed in as the rhs gives interior L2 error ratios of
4.000 / 4.000 / 4.001 across 32^2 -> 256^2 -- clean second order. The
infinite-wire deck matches `mu0 I / 2 pi r` to a median 0.3% on an interior
annulus (the square `A_z = 0` boundary distorts the far field near the edge,
the same finite-box error `poisson.py`'s point-charge test showed).

### 1.2 Geometric multigrid (Phase 2, the default)

SOR needs O(n) iterations for a fixed accuracy, so a 512^2 solve is ~60 ms
single-threaded -- too slow to re-solve while dragging a wire. `Multigrid.cpp`
is a CPU geometric multigrid: RB-GS smoother, bilinear prolongation,
full-weighting restriction, rediscretised (not Galerkin) coarse operator.

Getting a **grid-independent** convergence rate out of it took three fixes
over the textbook-from-memory version, and the failures are worth recording
because each one still *converged to the right answer* -- only slowly:

1. **The restriction must be the scaled transpose of the prolongation**
   (`R = P^T / 2^dim`). A plain 2x2 average of the fine residual is a valid
   restriction but is not transpose-compatible with bilinear prolongation,
   and the two-grid rate degrades. Fixed by scattering each fine residual
   into the coarse rhs with exactly the weights prolongation would use to
   read it back, times 1/4.

2. **Every level must use the same boundary treatment, not just the finest.**
   This one cost the most time. A cell-centred hierarchy where only level 0
   clamps its edge cells has a subtle problem: the coarse grid's "edge cell"
   is a phantom half-cell *outside* the domain. Restriction dumps the
   near-boundary fine residual into those phantom cells, the coarse smoother
   never touches them, and the correction that comes back to the fine
   near-boundary layer is ~zero. The boundary-layer error then only shrinks
   at the (slow) smoother rate, and since every coarsening step re-introduces
   it, the V-cycle count grew roughly linearly with the number of levels
   (27 -> 47 across 64^2 -> 1024^2). Fixed by giving *every* level the same
   homogeneous reflective-ghost edge, `u_ghost = -u_edge` (so the zero of
   the correction sits on the domain edge itself), and relaxing all cells
   `[0, n-1]`. Magnetostatics on a large box has `A_z -> 0` at the edge
   anyway, so the homogeneous condition is the physical one; non-homogeneous
   Dirichlet data would need lifting into the rhs, which no current deck
   uses.

3. **A W-cycle, not a V-cycle.** With the two fixes above the V-cycle was
   down to 11 -> 16 cycles across the same range -- close, but still a mild
   grid-size dependence from the non-Galerkin coarse operator. `gamma = 2`
   (recurse twice on each coarse level) flattens it: **8 / 9 / 9 / 9 / 10
   W-cycles** to a 1e-9 relative residual from 64^2 to 1024^2, with the
   interior L2 error still `~h^2` (confirming it reaches the same discrete
   solution RB-GS does). 256^2 solves in ~14 ms, 512^2 in ~57 ms.

OpenMP on the smoother was tried and reverted: the W-cycle calls the
smoother hundreds of times per solve on many small levels, and the
per-`#pragma omp parallel` fork/join overhead on Windows dwarfed the work
-- a 64^2 solve went from 1 ms to 38 ms. Doing it right needs one outer
parallel region with `#pragma omp for` inside; deferred, and the GPU port
is the better answer for large interactive grids anyway.

## 2. 3D Biot-Savart (Phase 3)

`[[coil]]` tables (`shape = loop | solenoid | helmholtz`) are discretised to
closed polylines of straight current elements, each carrying `(I dl)`.
`kernels_biot_savart.hpp` is a compute shader: one thread per cell of a
slice plane, inner loop over every segment, accumulating

```
B(r) = (mu0 / 4pi) sum_seg  (I dl) x (r - mid) / |r - mid|^3
```

The slice plane (`grid.plane = xy | xz | yz`, `grid.plane_offset`) maps the
2D grid into 3D space: `xz` for a coil whose axis is `z`, so the on-axis
field and the uniform region are both in view. The full 3-vector `B` is
stored as two in-plane components (`Bx`, `By`) plus the out-of-plane one
(`Bz`); the Poisson path zero-fills `Bz`. `BiotSavartAt()` is a scalar CPU
version of the same sum, used as the pusher's on-the-fly field and as the
selftest's cross-check.

**Validation** (`--biot-selftest`, needs a GL context):

- single loop, GPU vs the analytic on-axis field `mu0 I R^2 / [2
  (R^2+z^2)^{3/2}]`: max relative error 3e-4 (the CPU reference: 0);
- Helmholtz pair (`spacing = R`): centre `|B|` vs `(4/5)^{3/2} mu0 I / R` to
  2e-5; `d^2 B/dz^2 R^2 / B0 = 2.6e-4` (i.e. ~0, the Helmholtz condition);
  on-axis uniformity 6e-4 over `|z| < 0.15 R`. `tools/plot_helmholtz.py`'s
  `|B - B0|/B0` map shows the classic four-lobed Helmholtz sweet spot.

## 3. The relativistic Boris pusher (Phase 4)

`BorisPusher.hpp` carries each `TestCharge`'s state as proper velocity
`u = gamma v` and does one Boris step per substep: half electric kick,
magnetic rotation (`t = (q/m)(dt/2) B / gamma`, `s = 2t/(1+|t|^2)`), half
electric kick, drift. The field at the particle is `[background] B/E` plus
the on-the-fly 3D Biot-Savart sum (coil decks) or the interpolated in-plane
2D field (wire decks).

The Boris algorithm is exactly phase- and energy-preserving in a static
magnetic field, and the numbers show it (`--boris-selftest`):

- **cyclotron**: orbit radius `gamma v_perp / (|q/m| B0)` to 3e-7, period
  `2 pi gamma / (|q/m| B0)` exact, `d(gamma)/gamma = 6e-16` over 20000 steps
  -- machine precision;
- **E x B drift**: the guiding centre of both a positive and a negative
  charge drifts at `E x B / B^2` to 1e-5, independent of sign.

`magnetic_bottle.toml` (two like-current loops, a mirror trap): the charge
bounces between the throats, `KE` is exactly flat, and the adiabatic
invariant `mu = v_perp^2 / B`, sampled at the mid-plane crossings where the
gyration is cleanest, holds bounce-to-bounce to 3.5% with no secular drift
-- it breathes ~10% within a bounce near the steep-gradient turning points,
which is the expected finite-`r_g/L` behaviour of an *adiabatic* (not exact)
invariant.

## 4. The interactive view

`--interactive` opens `fw::SimApp`: a fullscreen-triangle fragment shader
samples the chosen scalar field from an SSBO (magma for the unsigned `|B|`,
a zero-centred blue-white-red diverging map for the signed `Bx / By / Bz /
A_z`), auto-scaled to what is on screen. A B field-line overlay integrates
RK4 streamlines from a coarse seed grid, stopping each one at a current
source and on loop closure so closed field lines are not drawn many times
over. The arrow keys move an "active" wire (2D path) or coil centre (3D
path) and re-solve -- multigrid warm-started from the current `A_z`, ~15 ms
at 256^2 -- so the field and its lines update live. Test-particle trails are
drawn as slice-projected line strips.

Since the OpenGL window cannot be inspected directly in this environment,
`--render-check <png>` renders one frame into an offscreen FBO (1100x850,
bigger than `GLContext::CreateHidden`'s 64x64 default) and writes a PNG --
the same trick `08_compressible_fluid`'s notes describe. For a particle deck
it first advances the sim so there is a trail to see.

## 5. What further extension would need

- **A GPU multigrid** for interactive grids past ~512^2: the smoother,
  residual, restrict and prolong are all local stencils and map directly to
  compute shaders (the `05_tdse_gpu` FFT-Poisson path is a worked example of
  the SSBO/dispatch/barrier machinery). The CPU version here is the
  validated reference it would cross-check against.
- **Interior conductors** (a charged/current-carrying plate held at fixed
  `A_z` inside the domain, not just on the edge): `SolvePoisson` already
  takes an arbitrary `fixedMask`; the multigrid path would need the mask
  restricted down the hierarchy, or it can fall back to SOR as it does for
  Neumann.
- **Cursor-driven dragging** instead of arrow keys: `fw::ViewInput` carries
  only motion deltas, not an absolute cursor position or button state, so
  true click-and-drag of a wire needs a small addition to that framework
  struct (which `10_fdtd` and `11_retarded_fields` would also use).
