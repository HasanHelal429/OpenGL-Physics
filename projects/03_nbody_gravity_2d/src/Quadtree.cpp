#include "Quadtree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace nbody2d {

namespace {

// Safety valve, not a target depth -- see Octree.cpp's identical constant
// for the 3D project; only bites for near-duplicate positions.
constexpr int kMaxDepth = 40;

// Leaf bucket size: a leaf holds up to this many particles before
// splitting (instead of splitting as soon as a 2nd particle would land in
// it). Standard FMM/tree-code practice (often called "ncrit") -- a
// single-particle-per-leaf tree makes every leaf-vs-anything interaction
// its own M2L call (cost ~O(p^2), independent of how many particles it
// represents) or its own near-field pair, multiplying both the node count
// and the interaction-list size for no benefit once real particle counts
// get large. Confirmed empirically to be the dominant remaining cost in
// ComplexFmmTree's traversal after fixing the tie-break bug and the
// blind fixed-depth parallel pre-split.
constexpr int kMaxLeafParticles = 16;

int Quadrant(const glm::dvec2& center, const glm::dvec2& p) {
    int q = 0;
    if (p.x >= center.x) q |= 1;
    if (p.y >= center.y) q |= 2;
    return q;
}

glm::dvec2 QuadrantDir(int q) { return glm::dvec2((q & 1) ? 1.0 : -1.0, (q & 2) ? 1.0 : -1.0); }

} // namespace

AdaptiveQuadtree::AdaptiveQuadtree(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass)
    : m_pos(pos), m_mass(mass) {
    m_nodes.reserve(pos.size() * 2 + 64);
    BuildRoot();
    for (int i = 0; i < static_cast<int>(pos.size()); ++i) Insert(0, i, 0);
}

void AdaptiveQuadtree::BuildRoot() {
    glm::dvec2 lo(std::numeric_limits<double>::max());
    glm::dvec2 hi(std::numeric_limits<double>::lowest());
    for (const glm::dvec2& p : m_pos) {
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    if (m_pos.empty()) {
        lo = glm::dvec2(0.0);
        hi = glm::dvec2(0.0);
    }

    QuadNode root;
    root.center = 0.5 * (lo + hi);
    const glm::dvec2 extent = hi - lo;
    double halfSize = 0.5 * std::max(extent.x, extent.y);
    if (!(halfSize > 1e-12)) halfSize = 1.0; // degenerate: single particle or all-coincident positions
    root.halfSize = halfSize * 1.001;        // small pad so boundary points land strictly inside
    m_nodes.push_back(root);
}

void AdaptiveQuadtree::Insert(int nodeIdx, int p, int depth) {
    const double m = m_mass[static_cast<size_t>(p)];
    const size_t idx = static_cast<size_t>(nodeIdx);
    const double newMass = m_nodes[idx].mass + m;
    m_nodes[idx].com = (m_nodes[idx].com * m_nodes[idx].mass + m_pos[static_cast<size_t>(p)] * m) / newMass;
    m_nodes[idx].mass = newMass;

    if (m_nodes[idx].isLeaf) {
        if (m_nodes[idx].particles.size() < static_cast<size_t>(kMaxLeafParticles) || depth >= kMaxDepth) {
            m_nodes[idx].particles.push_back(p);
            return;
        }
        std::vector<int> existing = std::move(m_nodes[idx].particles);
        m_nodes[idx].particles.clear();
        m_nodes[idx].isLeaf = false;
        for (int q : existing) InsertIntoChild(nodeIdx, q, depth);
    }
    InsertIntoChild(nodeIdx, p, depth);
}

void AdaptiveQuadtree::InsertIntoChild(int nodeIdx, int p, int depth) {
    const size_t idx = static_cast<size_t>(nodeIdx);
    const int q = Quadrant(m_nodes[idx].center, m_pos[static_cast<size_t>(p)]);
    int childIdx = m_nodes[idx].children[static_cast<size_t>(q)];
    if (childIdx == -1) {
        QuadNode child;
        child.halfSize = 0.5 * m_nodes[idx].halfSize;
        child.center = m_nodes[idx].center + QuadrantDir(q) * child.halfSize;
        child.parent = nodeIdx;
        m_nodes.push_back(child); // may reallocate -- fine, nothing above holds a stale reference
        childIdx = static_cast<int>(m_nodes.size()) - 1;
        m_nodes[idx].children[static_cast<size_t>(q)] = childIdx;
    }
    Insert(childIdx, p, depth + 1);
}

namespace {

glm::dvec2 WalkBarnesHut(const AdaptiveQuadtree& tree, int nodeIdx, int i, double G, double theta2, double eps2) {
    const QuadNode& node = tree.Nodes()[static_cast<size_t>(nodeIdx)];
    if (node.mass <= 0.0) return glm::dvec2(0.0);
    const std::vector<glm::dvec2>& pos = tree.Positions();
    const std::vector<double>& mass = tree.Masses();

    if (node.isLeaf) {
        glm::dvec2 acc(0.0);
        for (int p : node.particles) {
            if (p == i) continue;
            const glm::dvec2 d = pos[static_cast<size_t>(p)] - pos[static_cast<size_t>(i)];
            const double dist2 = glm::dot(d, d) + eps2;
            acc += G * mass[static_cast<size_t>(p)] * d / dist2; // 2D force: ~1/r, so d/r^2
        }
        return acc;
    }

    const glm::dvec2 d = node.com - pos[static_cast<size_t>(i)];
    const double dist2 = glm::dot(d, d);
    const double size = 2.0 * node.halfSize;
    if (size * size < theta2 * dist2) {
        const double distSoft2 = dist2 + eps2;
        return G * node.mass * d / distSoft2;
    }

    glm::dvec2 acc(0.0);
    for (int c : node.children) {
        if (c != -1) acc += WalkBarnesHut(tree, c, i, G, theta2, eps2);
    }
    return acc;
}

} // namespace

void ComputeAccelBarnesHut(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                            double softening, double theta, std::vector<glm::dvec2>& accelOut,
                            BarnesHutStats* stats) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec2(0.0));
    if (n == 0) return;

    const auto t0 = std::chrono::steady_clock::now();
    const AdaptiveQuadtree tree(pos, mass);
    const auto t1 = std::chrono::steady_clock::now();
    const double eps2 = softening * softening;
    const double theta2 = theta * theta;

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        accelOut[static_cast<size_t>(i)] = WalkBarnesHut(tree, 0, i, G, theta2, eps2);
    }
    const auto t2 = std::chrono::steady_clock::now();

    if (stats) {
        stats->buildMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->walkMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats->nodeCount = static_cast<int>(tree.Nodes().size());
    }
}

} // namespace nbody2d
