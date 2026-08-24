#pragma once

#include <array>
#include <glm/glm.hpp>
#include <vector>

namespace nbody {

// One node of an adaptive octree: top-down point insertion, one leaf per
// particle unless a depth cap is hit (a "degenerate" leaf then holds more
// than one). Shared by Barnes-Hut (mass/com only) and the adaptive FMM
// solver (which also needs quadrupole moments and parent links for its
// dual-tree M2L/L2L passes) -- both algorithms want the exact same
// adaptive spatial partition, just walked differently.
struct OctreeNode {
    glm::dvec3 center{0.0}; // geometric box center (used only to pick octants during insertion)
    double halfSize = 0.0;
    double mass = 0.0;
    glm::dvec3 com{0.0};
    std::array<int, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
    int parent = -1;
    bool isLeaf = true;
    std::vector<int> particles; // meaningful only while isLeaf; almost always size 0 or 1

    // Traceless Cartesian quadrupole tensor about `com`: Qxx+Qyy+Qzz == 0,
    // so Qzz is never stored (recover it as -Qxx-Qyy). Zero until
    // ComputeQuadrupoles() is called -- Barnes-Hut never needs it.
    double Qxx = 0.0, Qyy = 0.0, Qxy = 0.0, Qxz = 0.0, Qyz = 0.0;
};

// Flat, index-addressed adaptive octree over the given particle positions.
// Every access goes through m_nodes[idx] rather than a reference cached
// across a call that might grow the vector, so construction stays correct
// even though std::vector can reallocate mid-build. A node's children
// always have a larger index than the node itself (an invariant of the
// insertion order below), which the adaptive FMM's L2L pass relies on to
// push local expansions down the tree with one increasing-index sweep
// instead of an explicit recursion/BFS.
class AdaptiveOctree {
public:
    AdaptiveOctree(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass);

    // Second bottom-up pass computing each node's quadrupole moment from
    // its (already known) mass/com and its children's/members' positions.
    // Only the adaptive FMM solver needs this -- skip it for Barnes-Hut.
    void ComputeQuadrupoles();

    const std::vector<OctreeNode>& Nodes() const { return m_nodes; }
    const std::vector<glm::dvec3>& Positions() const { return m_pos; }
    const std::vector<double>& Masses() const { return m_mass; }

private:
    void BuildRoot();
    void Insert(int nodeIdx, int p, int depth);
    void InsertIntoChild(int nodeIdx, int p, int depth);

    const std::vector<glm::dvec3>& m_pos;
    const std::vector<double>& m_mass;
    std::vector<OctreeNode> m_nodes;
};

// Barnes-Hut gravity: walks the adaptive octree once per particle with the
// standard opening-angle criterion (theta = node_size / distance), treating
// well-separated nodes as a single point mass at their center of mass.
// O(N log N) instead of direct summation's O(N^2); accuracy vs. speed is
// tuned by theta. The tree is rebuilt from scratch on every call, matching
// physical reality (positions move every step) at the cost of a fresh
// build each time -- fine at the particle counts this real-time demo
// targets, unlike a production astrophysics code.
void ComputeAccelBarnesHut(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                            double softening, double theta, std::vector<glm::dvec3>& accelOut);

} // namespace nbody
