# 03_nbody_gravity_2d: how it works

The 2D companion to `02_nbody_gravity`. Same `nbody_core` library
(`ngrav`, `template<int D>` with `D = 2`), same house model, same
`--*-selftest` conventions. **This doc only covers what is genuinely
2D-specific** -- for the solver internals, the integrator, softening, the
mutual FMM's momentum argument, and the GPU pipeline, see
[`../../02_nbody_gravity/docs/SIMULATION.md`](../../02_nbody_gravity/docs/SIMULATION.md).

## What differs from 3D

**Green's function.** 2D gravity uses the true 2D fundamental solution:
force `~ 1/r`, potential `+ln r` -- not a projected 3D law. Circular speed
is separation-independent, and a `1/r` force is not one of Bertrand's
closed-orbit laws, so **2D orbits precess**. `ngrav::PlummerAccel<2>` /
`PlummerPairPotential<2>` handle this via `if constexpr`. The spline
softening's 2D potential outer branch (`1 <= q < 2`) has no elementary
closed form once the log term enters -- it is built once from a
precomputed Simpson-quadrature table (`Spline2DPotentialTable`, 4096
samples). Only the *diagnostic* potential uses that table; the force the
integrator sees is the exact closed-form `SplineChi2D`.

**Angular momentum** is a scalar -- the out-of-plane component
`r_x v_y - r_y v_x` -- not a 3-vector. `AngMom<2> = double`.

**Tree.** Quadtree (4 children) via the same `AdaptiveTree<D>` template
(`kChildren<2> == 4`).

**High-accuracy FMM.** The in-project `ComplexFmm` (complex-Laurent series,
order p ~ 10) is already a proper order-p dual-tree FMM -- 2D's `ln r`
kernel makes each degree one complex power and M2L O(p) per coefficient.
It is **not retired**; it doubles as both the production O(N) and the
accuracy-reference 2D solver. (Making it *mutual* -- Phase 6c -- is the one
piece of the plan still outstanding; see `../N_Body_Gravity_2D_Plan.md`.)

**No quadrupole BH term.** `AddQuadrupole` is a no-op for D=2 by design --
2D's higher-accuracy path is the arbitrary-order `ComplexFmm`, not a
Cartesian quadrupole.

**GPU: a quadtree, not an octree (Phase 12).** `ngrav::gpu::
ComputeAccelGpuQuadtree`, adapted from the 3D `GpuBarnesHut` port: 4-child
quadrant test, a 30-bit 2D Morton code (two 15-bit axes), `treeChild[4*
cell]`, 2D bounding box, and the 2D `1/r` softened-gravity force law. One
real porting gotcha, caught by inspection against the 3D source before it
ran: `06`'s 3D `ResolveSlots` reuses `treeChild`'s own per-cell array as a
leaf-particle list, which only works because its `kNcrit` equals its
branching factor (8 octants). The quadtree has only 4 slots per cell, so
`kNcrit` had to become 4 here (a GPU-only detail, not the CPU tree's
`ncrit=8`) or the trick would overrun into the next cell.

## Validation

`--selftest` (`ngrav::CoreSelfTest2D`):

- 2D Kepler (precessing -- asserts energy & |L| drift only): energy drift
  **5.4e-6**, |L| drift **1.4e-14**.
- 2D Lagrange equilateral triangle (`Omega = sqrt(Gm)/s`): energy drift
  **3.7e-13**.
- Barnes-Hut theta=0 vs Direct on a 2D cloud: max `|da|/|a|` = **1.9e-14**.
- 2D spline softening: force == Newtonian for `r >= 2h` to **2.2e-16**;
  spline-softened 2D orbit energy drift **2.4e-6**.

`--dt-selftest` exercises the dimension-generic `ngrav::Integrator<D>` via
the 3D eccentric-orbit case (2D's `1/r` force law has no closed-form
vis-viva orbit to validate against directly).

`--gpu-selftest` (`ngrav::gpu::ComputeAccelGpuQuadtree` vs CPU 2D
Barnes-Hut):

- theta=0 max rel err **1.3e-5** (gate <=1e-4).
- theta=0.5 max rel err **1.2e-2** (N=2000) / **7.1e-2** (N=1e5), gate
  <=0.2 -- consistent with 02's own 3D numbers.
- **Speed gate NOT met** on this dev box's integrated GPU: at N=1e5 GPU
  quadtree took 192ms vs the CPU 2D grouped walk's 36.5ms -- **0.19x**,
  against a >=5x gate. Same root cause as the 3D GPU BH (a divergent,
  barrier-heavy shared-stack walk on integrated silicon), re-confirmed
  independently rather than assumed to transfer. Per-stage profiling puts
  167ms of the 192ms in the forces kernel itself.

## Studies (Studies/nbody_gravity_2d/)

Two genuinely-2D questions, currently **executable scaffolds** (each needs
a per-frame particle-position reader wired into its `analyze.py`):

- `two_d_vs_three_d_gravity` -- how the `ln r` potential reshapes a cold
  collapse relative to 3D's `1/r` (no `1/r^2` focusing; a uniform disk
  rotates closer to solid-body).
- `disk_bar_instability` -- the `m=2` bar-mode growth rate vs rotational
  support `f`, mutual FMM vs Barnes-Hut (a momentum-violating far field
  spuriously heats the disk and suppresses the instability, so BH should
  under-report the growth rate).
