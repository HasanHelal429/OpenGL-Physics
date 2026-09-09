# 02_nbody_gravity: how it works

Collisionless self-gravitating N-body dynamics in 3D. Four force solvers
(Direct, Barnes-Hut, a momentum-conserving falcON-style mutual dual-tree
FMM, and an arbitrary-order spherical-harmonic FMM), a KDK-leapfrog
integrator with fixed / adaptive-global / power-of-2-block timesteps,
compact-support spline softening, GPU direct and GPU Barnes-Hut paths, and
realistic equilibrium initial conditions. The shared numerics live in the
`projects/nbody_core` library (namespace `ngrav`, `template<int D>`), which
`03_nbody_gravity_2d` also consumes.

Units: `G = 1` by default (deck `sim.G` overrides). Scenario ICs are in
their own natural units (Plummer/Hernquist/King spheres use `G = M = a = 1`).

The interactive `NBodyApp` (bare invocation or `--interactive`) is a live
four-solver comparison tool. Everything else is deck-driven and headless:
`--deck f.toml --out dir`, `--deck f.toml --render-check out.png`, the
`--*-selftest` flags, and `--bench <solver>`.

---

## 1. Shared core library + flat SoA tree (Phase 1)

`nbody_core` owns the particle state as a **structure of arrays**
(`SoA<D>`: `x,y,z, vx,vy,vz, m, ax,ay,az` as parallel `double` vectors),
the force solvers, the integrator, scenarios, and the deck/selftest glue.
Dimension is a compile-time `template<int D>` parameter (2 or 3); the
binaries are separate, so `kChildren = 1<<D`, the Green's kernel, and the
multipole backend are all zero-cost specializations.

The tree is `AdaptiveTree<D>` -- a **flat, index-addressed** octree
(quadtree for D=2): every node field is a parallel array (`center`,
`halfSize`, `com`, `mass`, `maxRadius`, `children[kNC*N]`, `parent`,
`firstParticle`, `particleCount`, optional traceless quadrupole `Q..`).
Leaves own a contiguous range `[firstParticle, firstParticle+particleCount)`
into a permutation array -- no `std::vector<int>` per node, no per-leaf heap
allocation. `ncrit = 8` (matches `06_tidal_disruption`, bounds near-field
cost). "Child node index > parent index" holds by construction, so the FMM
L2L pass is one increasing-index sweep and the multipole upsweep one
decreasing-index sweep -- no recursion.

Build flags: `-O3 -march=native -funroll-loops` on all of `nbody_core`
(`-ffast-math` deliberately **not** used -- the energy/|L| conservation
gates need strict IEEE).

**Validation** (`nbody_core_selftest`, `--selftest`):

- Kepler two-body (equal masses, eps=0.02, dt=T/2000, 4 periods, Direct):
  periapsis rel-err **2.1e-5**, apoapsis rel-err **-5.6e-10**, energy drift
  **2.1e-5**, |L| drift **1.4e-14**.
- Lagrange equilateral triangle (5 periods): shape deviation **5.0e-6**
  (matches the Python `N_Body_Gravity` reference's 5.05e-6 to the digit),
  energy drift **2.4e-11**.
- Barnes-Hut at theta=0 vs Direct on a 1000-body Plummer sphere: max
  per-particle `|da|/|a|` = **6.2e-15** (theta=0 forces every pair through
  exact near-field).

## 2. House model: deck-driven headless mode (Phase 2)

`ngrav::DeckSim<D> : fw::Simulation` implements `Configure/Reset/Step/
Snapshot/Info/Render` over a `System<D>`. `Snapshot` writes `pos`/`vel`
frame fields plus scalar diagnostics: `energy`, `energy_drift_pct`,
`L_mag`, `L_drift_pct`, `p_mag` (`|sum m_i v_i|`), `com_x/y/z`, `dt_last`,
`rung_max`. Decks in `decks/*.toml`; scenarios via `scenario.type`
(`kepler`, `plummer`, `cold_collapse`, `hernquist`, `king`, `disk_galaxy`,
`ic_file`, ...). `--render-check` produces a PNG from a hidden GL context.

**Validation** (`--selftest` PASS, plus deck runs):

- Kepler deck (100 frames): energy drift **2.1e-3%**, |L| drift **6e-13%**.
- Cold-collapse deck: a uniform sphere free-falls to a ~65%-median-radius
  crash by t≈1, energy conserved to **0.7%** (Barnes-Hut monopole).
- Headless `manifest.json` / `diagnostics.csv` columns == `Info().diagnostics`;
  `--render-check` PNG has nonzero pixel variance.

## 3. Adaptive global timestep (Phase 3)

`dt = eta * min_i sqrt(eps / |a_i|)`, clamped to `(dt_ceiling/4096,
dt_ceiling]`. The fixed-dt path is kept bit-identical. `--dt-selftest`
calibrates a matching fixed dt at runtime (via the leapfrog's confirmed
order p≈1.99) so the comparison is fair.

**Validation** (`--dt-selftest` PASS):

- Fixed-dt path unchanged (mild e=0.36 orbit: drift 8.4e-6, matching the
  Phase-2 gate).
- On an e=0.9 orbit, adaptive (eta=0.015) reaches drift **2.75e-6** at
  **8.9x fewer force evaluations** than the fixed dt that matches its
  accuracy.
- Deck demo (`eccentric_orbit_{fixed,adaptive}.toml`, e=0.9, same nominal
  dt): fixed drifts 1.42%, adaptive drifts 0.027% (53x tighter) by
  shrinking dt from 8.5e-4 to 1.3e-4 through periapsis.

## 4. Compact-support spline softening (Phase 4)

The Hernquist & Katz (1989) gravitational softening kernel -- the point
mass convolved with the normalized cubic B-spline SPH density kernel, so
the force is **exactly Newtonian for r >= 2h** and a smooth, finite
correction inside. Independently re-derived here via the shell-theorem
enclosed-mass integral (closed form in 3D; a Simpson-quadrature table for
the 2D log-potential outer branch). `softening.kind = "spline"`;
`eps` is `h`.

**Validation** (`--selftest`):

- For `r >= 2h`, force == unsoftened Newtonian to **2.2e-16** (max rel err).
- Central potential `phi(0) = -1.4 G m/h` matches the published value;
  both branches agree at q=1.
- Spline-softened orbit energy drift **1.2e-5**.

## 5. Quadrupole Barnes-Hut + relative MAC (Phase 5)

`AdaptiveTree`'s traceless Cartesian quadrupole (`Qxx,Qyy,Qxy,Qxz,Qyz`;
`Qzz = -Qxx-Qyy`, 3D only) is consumed in the accepted-node branch of the
tree walk -- an independently re-derived correction, cross-checked two ways
(the AdaptiveFmm sign convention, and from scratch via
`phi(x) = -G[M/r + Q_ij x_i x_j / 2r^5]`). The GADGET-2/Springel-2005
relative-acceleration MAC (`solver.mac = "relative"`, `alpha`) is wired
into the walk alongside the geometric opening-angle one.

**Validation** (`--selftest`):

- theta=0.5 on a 4000-body Plummer: max rel force err drops **5.2x**
  (monopole 4.2e-2 -> mono+quad 8.1e-3; gate was >=3x).
- theta=0 exactness vs Direct unaffected (still <=1e-12).
- Relative MAC (alpha bisected to match the geometric MAC's mean error)
  visits **1.33x** fewer node interactions.

## 6. Momentum-conserving mutual dual-tree FMM (Phase 6)

`Solver::Fmm` -- a falcON-style (Dehnen 2000/2002) mutual FMM,
monopole+quadrupole order, reusing `AdaptiveTree`'s quadrupole upsweep.
The retired one-directional `AdaptiveFmm` half-measure is **deleted**.

Single-visit unordered traversal: a self-pair `(t,t)` emits only
`(c_i,c_i)` and `(c_i,c_j) i<j` -- half the old "all ordered child pairs"
work, and each distinct pair gets **one mutual M2L** that updates *both*
nodes' local expansions from one shared computation. Because the pair
interaction energy
`U(r) = -G[M_T M_S/r + M_T(Q_S:nn)/2r^3 + M_S(Q_T:nn)/2r^3]` depends only on
the separation vector `r = c_T - c_S`, `F_T = -dU/dc_T = -F_S` by
construction -- momentum conservation is a property of applying one shared
computation to both sides, not a numerical coincidence.

Three real bugs were caught during validation, each by a dedicated check
rather than passing silently: (1) a backwards source/target displacement
sign (momentum ratio read ~2.0); (2) an inherited "degenerate leaf always
aggregates" rule, unsafe once `ncrit=8` made multi-particle leaves routine
(~180% mean error at theta=0); (3) a missing quadrupole *reaction* term --
`U`'s full r-dependence includes the source's mass sitting in the target's
own quadrupole field, `(M_S/M_T)*QuadOnlyTerm(Q_T,d)` -- without which a
~1e-4 residual grew with theta.

**Validation** (`--fmm-selftest`):

- **Momentum**: `|sum m_i a_i| / sum m_i|a_i|` = **4.2e-17 to 7.9e-17**
  (machine precision) at every theta in {0.2, 0.4, 0.5, 0.6, 0.8} -- vs
  Barnes-Hut's incidental ~2e-5 on the same IC. Gate was <=1e-14.
- **theta=0 exactness**: 2.1e-15 vs Direct.
- **Accuracy**: strongly theta-dependent (a structural property of the
  first-order-local cluster-cluster expansion, not a bug): ~2e-4 mean rel
  err by theta=0.2, machine-epsilon by theta<=0.15; **worse than
  Barnes-Hut at BH's typical theta=0.5** (~5e-2 vs ~5e-4). The permanent
  selftest checks the achievable regime (theta<=0.2, <=1e-3).
- **Violent relaxation** (`Studies/nbody_gravity/momentum_conservation_*`):
  through a full cold-collapse bounce, mutual FMM holds `|P|` at
  **9.6e-17** with the centre of mass exactly stationary, vs Barnes-Hut
  **3.7e-4** and the one-directional spherical FMM **2.4e-4** on the
  identical IC.

## 7. Spherical-harmonic FMM (Phase 7)

`Solver::SphericalFmm` -- arbitrary-order solid-harmonic P2M/M2M/M2L/L2L/L2P
(Dehnen 2014 conventions), the accuracy *reference*, not a speed target.
Phase 7 made the expansion order `p` a runtime parameter (was a
compile-time `kOrder=5`) and switched M2L to accumulate in place. The
analytic L2P gradient and rotation-based O(p^3) M2L were deliberately
descoped (high bug-risk for a flexible-schedule phase).

**Validation** (`--spherical-selftest`): accuracy improves monotonically
with p and then plateaus once the fixed opening angle's own `theta^(p+1)`
truncation dominates -- p=2: 1.4e-3, p=4: 1.4e-4, p>=6: ~1.5e-4 (theta=0.5).
p=5 (the old default) gives 1.6e-4, consistent with its neighbours.

## 8. Grouped BH walk + OpenMP-task FMM traversal (Phase 8)

Barnes-Hut: one tree descent per leaf (per `ncrit`-particle group) instead
of per particle, using a conservative group MAC (each leaf's `com` +
`maxRadius` bound every member) that can only reject something the
per-particle test would accept, never the reverse -- so accuracy is
equal-or-better by construction. Mutual FMM traversal: recursion instead of
an explicit stack, OpenMP-tasked for its top 3 levels with per-thread
expansion buffers reduced afterward.

**Validation** (`--fmm-selftest`, benchmarks; N=1e5, theta=0.5):

- Grouped BH walk **1.3x** faster than the per-particle walk (gate guessed
  >=2x -- not met; accuracy came out *better*, monopole max err 4.2e-2 ->
  3.7e-2, which is why).
- FMM traversal stage **3-4.75x** faster; whole-call speedup **2.1-2.3x**
  (Amdahl: near-field + tree build still serial). Gate 0.7 efficiency not
  met for the whole call -- reported honestly.
- Parallel traversal visits the **identical** M2L/near-pair set as the
  serial oracle (pair counts match exactly); forces agree to **1e-14**
  (floating-point reassociation only); momentum conservation unchanged.

## 9. Parallel near-field + L2P (Phase 9)

L2P over leaves is a direct parallel-for (disjoint particle ranges). The
near-field pair sum uses genuine per-thread partial accel buffers (a
particle appears in many cross-leaf pairs), reduced after the loop, gated
behind `nPairs > 20000`. The hand-written SIMD kernel was descoped
(`-march=native` already auto-vectorizes the scalar kernel).

**Validation** (N=1e5): near-field stage **69ms**, down from ~180ms serial
(~2.6x; gate wanted >=3x). Whole-call speedup now **2.07-2.39x**. Forces
agree to 3-4e-14 vs serial; momentum unaffected (4.2-4.3e-17).

## 10. GPU direct sum (Phase 10)

`ngrav::gpu::ComputeAccelGpuDirect` -- one compute shader, one thread per
target, plain loop over every source (Plummer softening, same formula as
`ComputeAccelDirect`). Shader compiled lazily and cached.

**Validation** (`--gpu-selftest`): GPU fp32 vs CPU fp64 max rel err
**5.5e-6** (N=2000) / **1.0e-5** (N=30000), gate <=1e-4. **16.45x** faster
than the (OpenMP, `-O3 -march=native`) CPU direct sum at N=3e4, gate >=10x.
Readback dominates GPU wall time at this N -- the margin only grows larger.

## 11. GPU Barnes-Hut (Phase 11)

Ported from `06_tidal_disruption`'s octree build + grouped shared-stack
gravity walk, stripped of SPH and the TDE central-black-hole term.
`ComputeBoundingBox` (atomic float-flip min/max) -> Morton sort (30-bit,
bitonic) -> level-by-level `ClaimSlots`/`ResolveSlots` (early-exit via
cell-count readback) -> bottom-up mass/COM upsweep -> `Forces()`'s
grouped cooperative tree walk.

**Validation** (`--gpu-selftest`): theta=0 max rel err vs CPU BH **5.6e-6**
(gate <=1e-4); theta=0.5 **8.7e-3** (N=2000) / **5.3e-2** (N=1e5), gate
<=0.2. **Speed gate NOT met on this dev box** (integrated AMD Radeon
860M): at N=1e5 GPU BH took 336ms vs the (Phase-8/9-optimized) CPU grouped
walk's 159ms -- **0.47x**, against a >=5x gate. Per-stage profiling
(explicit `glFinish` between stages, gated behind `stats != nullptr`)
attributes 297ms of 336ms to the gravity kernel itself: the divergent,
barrier-heavy shared-stack walk is a poor fit for an integrated GPU, unlike
Phase 10's embarrassingly-parallel direct sum (16x on the *same* GPU).
Reported honestly; revisit on discrete-GPU hardware.

## 12. Realistic IC generators (Phase 13)

`ngrav/ic/`: **Hernquist** (exact closed-form inverse-CDF positions,
isotropic-Jeans velocities via a quadrature table), **King** (RK4
integration of the dimensionless `W'' + (2/x)W' = -9 g(W)/g(W0)` ODE at
IC-build time -- compact support, finite tidal radius), **DiskBulgeHalo**
(exponential disk + Hernquist bulge + NFW halo; disk stars on near-circular
orbits in the total analytic midplane potential with epicyclic dispersion
+ asymmetric drift). Velocity fields use the isotropic Jeans dispersion,
not each model's exact DF -- documented, not an oversight. `scenario.type =
"ic_file"` loads `06`'s raw 7-float32 format; `tools/make_ic.py` is a
deliberately independent pure-Python reimplementation that writes it.

**Validation** (`CoreIcSelfTest`, in `--selftest`):

- Plummer / Hernquist binned rho(r) median rel err **2.4% / 2.2%** over
  [0.1, 2] r_half (gate <=5%); half-mass radii within 1% of analytic.
- King(W0=6) tidal radius **17.3 r0** (matches the tabulated concentration).
- Virial `2T/|W|` = **0.995-0.998** for all three -- the Jeans dispersion
  is close enough that the scalar virial holds. (The plan's +/-0.02 was
  over-tight for a Jeans field in general; the selftest checks a per-model
  band and says why.)
- Plummer half-mass radius drift over ~6 crossing times of live BH
  integration: **1.1%**.

## 13. Block / power-of-2 rung timesteps (Phase 15)

`System<D>::BlockStep` -- `time.scheme = "block"`. One coarse step of length
`sp.dt` subdivided into `2^rMax` finest substeps; each particle's rung
`k_i` comes from its own `eta*sqrt(eps/|a_i|)` criterion, and a rung-`k`
particle takes a KDK step of `dt/2^k` (opening half-kick at its step start,
one shared global drift every finest substep, closing half-kick + force
refresh for the active set only at its step end). The last finest substep's
active set is all N, so `a(t+dt)` is exact. `ComputeAccel{Direct,BarnesHut}
Targets` compute the force for a target subset. **Direct / Barnes-Hut
only** -- the mutual FMM's symmetric M2L has no target-subset form, so
`block` degrades to `adaptive` for it. `rung_max` diagnostic column.

**Validation** (`--rung-selftest`): 200-body Plummer cloud + one bolted-on
tight fast binary (an explicit timescale separation), integrated to a
fixed t=2 -- fixed global dt: 81k per-particle force evals / 2.3e-3 energy
drift; global-adaptive dt: 2.44M evals / 2.0e-6 drift; **block: 175k evals
(14.0x fewer than adaptive) / 1.9e-6 drift**. Gate wanted >=3x fewer at
matched accuracy. A BH cold-collapse smoke test conserves energy to 2.1e-6.

## 14. Scaling (Studies/nbody_gravity/scaling_and_crossover)

Fitted log-log force-eval-cost exponents on Plummer spheres, N in
[1e3, 6.4e4], theta=0.5: **Direct 2.02** (R^2 0.999), **Barnes-Hut 1.43**
(R^2 0.998), **mutual FMM 0.96** (R^2 0.97), **spherical FMM 1.05**.
Direct is overtaken by Barnes-Hut at **N ≈ 1,900**, by the mutual FMM at
**N ≈ 13,700**, by the huge-prefactor spherical FMM only at **N ≈
350,000**. The mutual FMM's near-linear scaling is real (Phase-8/9
parallelism); its practical value is momentum conservation, not speed at
these N.
