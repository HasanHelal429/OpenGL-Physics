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

// Mutual (falcON-style) dual-tree FMM, 3D: momentum conservation (the
// headline result -- machine precision at every theta, vs the retired
// one-directional AdaptiveFmm's ~1e-3..1e-6 and even Barnes-Hut's own
// incidental ~1e-5), theta=0 exactness vs Direct, and the measured O(N)
// scaling exponent.
bool CoreFmmSelfTest();

// Realistic IC generators (P13): each of Plummer / Hernquist / King samples
// its analytic density profile to within a few percent (binned rho(r) vs
// the closed form over [0.1, 2] r_half); the equilibrium models have a
// virial ratio 2T/|W| near 1 at t=0; and a Plummer sphere integrated for
// several crossing times holds its half-mass radius (the one dynamical
// check -- the others are static). Honest thresholds: the velocity fields
// use a Jeans-equation isotropic-dispersion approximation, not each
// model's exact distribution function, so "near 1" is ~10%, not 2%.
bool CoreIcSelfTest();

} // namespace ngrav
