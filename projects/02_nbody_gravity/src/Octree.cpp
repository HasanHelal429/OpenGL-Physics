#include "Octree.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace nbody {

namespace {

// Safety valve, not a target depth: an octree over well-spread points
// reaches maybe 15-20 levels even at N in the hundreds of thousands. This
// only bites for near-duplicate positions, where insertion would otherwise
// recurse forever trying to separate two points that are (numerically)
// coincident -- past this depth, particles that still haven't separated
// are grouped into one degenerate leaf.
constexpr int kMaxDepth = 40;

struct Node {
    glm::dvec3 center{0.0};
    double halfSize = 0.0;
    double mass = 0.0;
    glm::dvec3 com{0.0};
    std::array<int, 8> children{-1, -1, -1, -1, -1, -1, -1, -1};
    bool isLeaf = true;
    std::vector<int> particles; // meaningful only while isLeaf; almost always size 0 or 1
};

int Octant(const glm::dvec3& center, const glm::dvec3& p) {
    int oct = 0;
    if (p.x >= center.x) oct |= 1;
    if (p.y >= center.y) oct |= 2;
    if (p.z >= center.z) oct |= 4;
    return oct;
}

glm::dvec3 OctantDir(int oct) {
    return glm::dvec3((oct & 1) ? 1.0 : -1.0, (oct & 2) ? 1.0 : -1.0, (oct & 4) ? 1.0 : -1.0);
}

// Flat, index-addressed octree. Every access goes through m_nodes[idx]
// rather than a Node& cached across a call that might grow the vector, so
// construction stays correct even though std::vector can reallocate
// mid-build (push_back during InsertIntoChild).
class BHTree {
public:
    BHTree(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass) : m_pos(pos), m_mass(mass) {
        m_nodes.reserve(pos.size() * 2 + 64);
        BuildRoot();
        for (int i = 0; i < static_cast<int>(pos.size()); ++i) Insert(0, i, 0);
    }

    glm::dvec3 Accel(int i, double G, double theta, double eps2) const { return Walk(0, i, G, theta * theta, eps2); }

private:
    void BuildRoot() {
        glm::dvec3 lo(std::numeric_limits<double>::max());
        glm::dvec3 hi(std::numeric_limits<double>::lowest());
        for (const glm::dvec3& p : m_pos) {
            lo = glm::min(lo, p);
            hi = glm::max(hi, p);
        }
        if (m_pos.empty()) {
            lo = glm::dvec3(0.0);
            hi = glm::dvec3(0.0);
        }

        Node root;
        root.center = 0.5 * (lo + hi);
        const glm::dvec3 extent = hi - lo;
        double halfSize = 0.5 * std::max({extent.x, extent.y, extent.z});
        if (!(halfSize > 1e-12)) halfSize = 1.0; // degenerate: single particle or all-coincident positions
        root.halfSize = halfSize * 1.001;        // small pad so boundary points land strictly inside
        m_nodes.push_back(root);
    }

    void Insert(int nodeIdx, int p, int depth) {
        const double m = m_mass[static_cast<size_t>(p)];
        const size_t idx = static_cast<size_t>(nodeIdx);
        const double newMass = m_nodes[idx].mass + m;
        m_nodes[idx].com = (m_nodes[idx].com * m_nodes[idx].mass + m_pos[static_cast<size_t>(p)] * m) / newMass;
        m_nodes[idx].mass = newMass;

        if (m_nodes[idx].isLeaf) {
            if (m_nodes[idx].particles.empty() || depth >= kMaxDepth) {
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

    void InsertIntoChild(int nodeIdx, int p, int depth) {
        const size_t idx = static_cast<size_t>(nodeIdx);
        const int oct = Octant(m_nodes[idx].center, m_pos[static_cast<size_t>(p)]);
        int childIdx = m_nodes[idx].children[static_cast<size_t>(oct)];
        if (childIdx == -1) {
            Node child;
            child.halfSize = 0.5 * m_nodes[idx].halfSize;
            child.center = m_nodes[idx].center + OctantDir(oct) * child.halfSize;
            m_nodes.push_back(child); // may reallocate -- fine, nothing above holds a stale Node&
            childIdx = static_cast<int>(m_nodes.size()) - 1;
            m_nodes[idx].children[static_cast<size_t>(oct)] = childIdx;
        }
        Insert(childIdx, p, depth + 1);
    }

    glm::dvec3 Walk(int nodeIdx, int i, double G, double theta2, double eps2) const {
        const Node& node = m_nodes[static_cast<size_t>(nodeIdx)];
        if (node.mass <= 0.0) return glm::dvec3(0.0);

        if (node.isLeaf) {
            glm::dvec3 acc(0.0);
            for (int p : node.particles) {
                if (p == i) continue;
                const glm::dvec3 d = m_pos[static_cast<size_t>(p)] - m_pos[static_cast<size_t>(i)];
                const double dist2 = glm::dot(d, d) + eps2;
                const double invDist = 1.0 / std::sqrt(dist2);
                const double invDist3 = invDist * invDist * invDist;
                acc += G * m_mass[static_cast<size_t>(p)] * invDist3 * d;
            }
            return acc;
        }

        const glm::dvec3 d = node.com - m_pos[static_cast<size_t>(i)];
        const double dist2 = glm::dot(d, d);
        const double size = 2.0 * node.halfSize;
        if (size * size < theta2 * dist2) {
            const double distSoft2 = dist2 + eps2;
            const double invDist = 1.0 / std::sqrt(distSoft2);
            const double invDist3 = invDist * invDist * invDist;
            return G * node.mass * invDist3 * d;
        }

        glm::dvec3 acc(0.0);
        for (int c : node.children) {
            if (c != -1) acc += Walk(c, i, G, theta2, eps2);
        }
        return acc;
    }

    const std::vector<glm::dvec3>& m_pos;
    const std::vector<double>& m_mass;
    std::vector<Node> m_nodes;
};

} // namespace

void ComputeAccelBarnesHut(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                            double softening, double theta, std::vector<glm::dvec3>& accelOut) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec3(0.0));
    if (n == 0) return;

    const BHTree tree(pos, mass);
    const double eps2 = softening * softening;

#pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n; ++i) {
        accelOut[static_cast<size_t>(i)] = tree.Accel(i, G, theta, eps2);
    }
}

} // namespace nbody
