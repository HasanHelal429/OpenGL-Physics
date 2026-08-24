#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody {

// Barnes-Hut gravity: builds an adaptive octree over the current particle
// positions (top-down point insertion, one leaf per particle unless a
// depth cap is hit) and walks it once per particle with the standard
// opening-angle criterion (theta = node_size / distance), treating
// well-separated nodes as a single point mass at their center of mass.
// O(N log N) instead of direct summation's O(N^2); accuracy vs. speed is
// tuned by theta. The tree is rebuilt from scratch on every call, matching
// physical reality (positions move every step) at the cost of a fresh
// build each time -- fine at the particle counts this real-time demo
// targets, unlike a production astrophysics code.
void ComputeAccelBarnesHut(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                            double softening, double theta, std::vector<glm::dvec3>& accelOut);

} // namespace nbody
