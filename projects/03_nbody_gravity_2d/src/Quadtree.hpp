#pragma once

#include <array>
#include <glm/glm.hpp>
#include <vector>

namespace nbody2d {

// One node of an adaptive quadtree: top-down point insertion, one leaf per
// particle unless a depth cap is hit (a "degenerate" leaf then holds more
// than one). The 2D analogue of 02_nbody_gravity/src/Octree.hpp's
// OctreeNode/AdaptiveOctree -- same structure with 4 children instead of
// 8, no Cartesian quadrupole moment (this project's higher-accuracy solver
// carries complex Laurent-series moments instead, computed separately in
// ComplexFmmTree.cpp via a side array, not stored on the node itself).
struct QuadNode {
    glm::dvec2 center{0.0}; // geometric box center (used only to pick quadrants during insertion)
    double halfSize = 0.0;
    double mass = 0.0;
    glm::dvec2 com{0.0};
    std::array<int, 4> children{-1, -1, -1, -1};
    int parent = -1;
    bool isLeaf = true;
    std::vector<int> particles; // meaningful only while isLeaf; almost always size 0 or 1
};

// Flat, index-addressed adaptive quadtree over the given particle
// positions. A node's children always have a larger index than the node
// itself (an invariant of the insertion order below), which the FMM
// solver's bottom-up multipole pass and L2L pass both rely on.
class AdaptiveQuadtree {
public:
    AdaptiveQuadtree(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass);

    const std::vector<QuadNode>& Nodes() const { return m_nodes; }
    const std::vector<glm::dvec2>& Positions() const { return m_pos; }
    const std::vector<double>& Masses() const { return m_mass; }

private:
    void BuildRoot();
    void Insert(int nodeIdx, int p, int depth);
    void InsertIntoChild(int nodeIdx, int p, int depth);

    const std::vector<glm::dvec2>& m_pos;
    const std::vector<double>& m_mass;
    std::vector<QuadNode> m_nodes;
};

struct BarnesHutStats {
    double buildMs = 0.0;
    double walkMs = 0.0;
    int nodeCount = 0;
};

// Barnes-Hut gravity: walks the adaptive quadtree once per particle with
// the standard opening-angle criterion, treating well-separated nodes as
// a single point mass at their center of mass. Monopole-only (no
// quadrupole correction) -- matching the 3D project's own Barnes-Hut path,
// since the complex FMM below carries the higher accuracy instead. Uses
// the 2D force law (a ~ G*m*d/r^2, not 3D's d/r^3 -- see ComplexFmm.hpp's
// header comment on why 2D gravity's force falls off as 1/r).
void ComputeAccelBarnesHut(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                            double softening, double theta, std::vector<glm::dvec2>& accelOut,
                            BarnesHutStats* stats = nullptr);

} // namespace nbody2d
