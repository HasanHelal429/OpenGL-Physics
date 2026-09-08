#pragma once

namespace ngrav {

// Core numerics validation with no GL context: the leapfrog integrator, the
// SoA Direct sum, the flat AdaptiveTree, and the Barnes-Hut walk, checked
// against analytic two-body / Lagrange solutions and (BH) against Direct.
// Prints per-check `name value (tol) ok/WRONG` lines. Returns true on PASS.
// Shared by 02/03's `--selftest` and the standalone `nbody_core_selftest`.
bool CoreSelfTest3D();
bool CoreSelfTest2D();

} // namespace ngrav
