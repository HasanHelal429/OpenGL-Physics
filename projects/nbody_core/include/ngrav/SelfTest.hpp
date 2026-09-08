#pragma once

namespace ngrav {

// Core numerics validation with no GL context: the leapfrog integrator, the
// SoA Direct sum, the flat AdaptiveTree, and the Barnes-Hut walk, checked
// against analytic two-body / Lagrange solutions and (BH) against Direct.
// Prints per-check `name value (tol) ok/WRONG` lines. Returns true on PASS.
// Shared by 02/03's `--selftest` and the standalone `nbody_core_selftest`.
bool CoreSelfTest3D();
bool CoreSelfTest2D();

// Adaptive global timestep: a highly-eccentric two-body orbit where a coarse
// fixed dt loses energy at periapsis but adaptive dt = eta*min sqrt(eps/|a|)
// stays bounded, at fewer force evaluations than the fixed dt that matches
// its accuracy. Also checks the fixed-dt path is unchanged.
bool CoreDtSelfTest();

} // namespace ngrav
