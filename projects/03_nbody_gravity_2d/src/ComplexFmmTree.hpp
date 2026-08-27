#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody2d {

// Adaptive FMM using arbitrary-order complex Laurent-series multipole/
// local expansions (ComplexFmm.hpp) -- the 2D analogue of the 3D project's
// SphericalFmm.hpp/.cpp, sharing AdaptiveQuadtree for spatial partitioning
// only (every solver-specific quantity -- each node's own multipole
// expansion and its exact max-particle-radius -- lives in side arrays
// here, never added to QuadNode itself, matching the 3D project's own
// design).
struct FmmStats {
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

void ComputeAccelComplexFmm(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                             double softening, double theta, std::vector<glm::dvec2>& accelOut,
                             FmmStats* stats = nullptr);

} // namespace nbody2d
