#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody {

// Adaptive FMM using arbitrary-order complex spherical-harmonic multipole/
// local expansions (SphericalHarmonics.hpp) instead of AdaptiveFmm.cpp's
// hand-derived Cartesian monopole+quadrupole tensors -- a new, separately
// selectable solver (not a replacement) so it can be A/B tested against
// the existing, working Cartesian FMM before any decision to promote it.
// See this project's spherical-harmonic plan doc for the full rationale.
//
// Shares AdaptiveOctree/OctreeNode (Octree.hpp) for spatial partitioning
// only -- every solver-specific quantity (each node's own multipole
// expansion, its own max-particle-radius, and its accumulated local
// expansion) lives in side arrays here rather than in OctreeNode itself,
// so Barnes-Hut and the Cartesian FMM are untouched by this file.
struct SphericalFmmStats {
    double buildMs = 0.0;
    double multipoleMs = 0.0;
    double seedMs = 0.0;
    double traverseMs = 0.0;
    double l2lMs = 0.0;
    double l2pMs = 0.0;
    double nearFieldMs = 0.0;
    int nodeCount = 0;
    int numTargets = 0;
    size_t nearPairCount = 0;
};

// `order` is the runtime expansion degree p (Phase 7 -- was a compile-time
// kOrder=5 constant); default matches the original fixed value.
void ComputeAccelSphericalFmm(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                               double softening, double theta, std::vector<glm::dvec3>& accelOut,
                               SphericalFmmStats* stats = nullptr, int order = 5);

} // namespace nbody
