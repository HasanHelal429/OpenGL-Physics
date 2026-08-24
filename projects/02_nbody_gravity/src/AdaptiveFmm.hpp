#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody {

// Fast Multipole Method on an adaptive (Barnes-Hut-style) octree, instead
// of a fixed uniform grid: a dual-tree traversal recurses over *pairs* of
// nodes from the same adaptive tree (one target, one source, both starting
// at the root), splitting whichever side is coarser, until a pair is
// either well-separated -- multipole-to-local (M2L), accumulating a local
// (Taylor-expanded) expansion of the field into the target side, monopole
// + quadrupole order -- or both leaves, a near-field pair handled by exact
// softened summation. Each node's accumulated local expansion is then
// pushed down to its children (L2L) and evaluated at each leaf's member
// particles (L2P). O(N) instead of Barnes-Hut's O(N log N), at a larger
// constant factor and more approximation machinery -- see the accuracy/
// speed comparison surfaced by the app's benchmark panel for when that
// trade actually pays off at this project's particle counts.
//
// Ported from Physics Simulations' Orbital_Dynamics/N_Body_Gravity
// (nbody.py's adaptive_fmm_accel), generalized from 2D to 3D: the
// monopole+quadrupole potential/field/Hessian formulas below were
// re-derived for 3D and checked term-by-term against the Python
// implementation's 2D closed forms (restricting to z=0 reproduces them
// exactly). Also carries forward that implementation's real bugs-turned-
// fixes: near-field pairs recorded only once (smaller particle index
// first) to avoid double-counting, an absolute minimum-separation floor on
// the well-separated test (the opening-angle criterion alone is scale-
// invariant and the unsoftened multipole field is numerically catastrophic
// far inside the softening length), and degenerate (multi-particle) leaves
// treated as a single aggregate point mass for all external interactions
// plus an exact brute-force sum for their own internal ones.
void ComputeAccelAdaptiveFmm(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                              double softening, double theta, std::vector<glm::dvec3>& accelOut);

} // namespace nbody
