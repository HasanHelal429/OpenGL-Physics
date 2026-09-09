#include "ngrav/Tree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ngrav {

namespace {

template <int D>
int OctantOf(const Vec<D>& center, const Vec<D>& p) {
    int oct = 0;
    if (p.x >= center.x) oct |= 1;
    if (p.y >= center.y) oct |= 2;
    if constexpr (D == 3) {
        if (p.z >= center.z) oct |= 4;
    }
    return oct;
}

template <int D>
Vec<D> OctantDir(int oct) {
    Vec<D> d;
    d.x = (oct & 1) ? 1.0 : -1.0;
    d.y = (oct & 2) ? 1.0 : -1.0;
    if constexpr (D == 3) d.z = (oct & 4) ? 1.0 : -1.0;
    return d;
}

} // namespace

template <int D>
AdaptiveTree<D>::AdaptiveTree(const PosMassView<D>& pts, int ncrit, int maxDepth)
    : m_ncrit(ncrit), m_maxDepth(maxDepth) {
    const int n = static_cast<int>(pts.Count());
    const std::size_t reserve = static_cast<std::size_t>(n) * 2 / std::max(1, ncrit) + 64;
    center.reserve(reserve);
    halfSize.reserve(reserve);
    com.reserve(reserve);
    mass.reserve(reserve);
    children.reserve(reserve);
    childCount.reserve(reserve);
    parent.reserve(reserve);
    firstParticle.reserve(reserve);
    particleCount.reserve(reserve);
    m_buildParticles.reserve(reserve);

    BuildRoot(pts);
    for (int i = 0; i < n; ++i) Insert(pts, 0, i, 0);
    Finalize(pts);
}

template <int D>
void AdaptiveTree<D>::BuildRoot(const PosMassView<D>& pts) {
    Vec<D> lo(std::numeric_limits<double>::max());
    Vec<D> hi(std::numeric_limits<double>::lowest());
    const int n = static_cast<int>(pts.Count());
    for (int i = 0; i < n; ++i) {
        const Vec<D> p = pts.Pos(static_cast<std::size_t>(i));
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
    }
    if (n == 0) {
        lo = Vec<D>(0.0);
        hi = Vec<D>(0.0);
    }

    Vec<D> ext = hi - lo;
    double half = 0.5 * ext.x;
    half = std::max(half, 0.5 * ext.y);
    if constexpr (D == 3) half = std::max(half, 0.5 * ext.z);
    if (!(half > 1e-12)) half = 1.0; // single particle / all-coincident

    std::array<int, kNC> noChildren;
    noChildren.fill(-1);

    center.push_back(0.5 * (lo + hi));
    halfSize.push_back(half * 1.001); // pad so boundary points land strictly inside
    com.push_back(Vec<D>(0.0));
    mass.push_back(0.0);
    children.push_back(noChildren);
    childCount.push_back(0);
    parent.push_back(-1);
    firstParticle.push_back(-1);
    particleCount.push_back(0);
    m_buildParticles.emplace_back();
}

template <int D>
int AdaptiveTree<D>::NewChild(int parentNode, int octant) {
    const std::size_t p = static_cast<std::size_t>(parentNode);
    std::array<int, kNC> noChildren;
    noChildren.fill(-1);

    const int idx = static_cast<int>(mass.size());
    const double childHalf = 0.5 * halfSize[p];
    center.push_back(center[p] + OctantDir<D>(octant) * childHalf);
    halfSize.push_back(childHalf);
    com.push_back(Vec<D>(0.0));
    mass.push_back(0.0);
    children.push_back(noChildren);
    childCount.push_back(0);
    parent.push_back(parentNode);
    firstParticle.push_back(-1);
    particleCount.push_back(0);
    m_buildParticles.emplace_back();

    children[p][static_cast<std::size_t>(octant)] = idx;
    childCount[p] = static_cast<int8_t>(childCount[p] + 1);
    return idx;
}

template <int D>
void AdaptiveTree<D>::Insert(const PosMassView<D>& pts, int node, int p, int depth) {
    const std::size_t nd = static_cast<std::size_t>(node);
    const double mp = pts.m[static_cast<std::size_t>(p)];
    const Vec<D> pp = pts.Pos(static_cast<std::size_t>(p));

    const double newMass = mass[nd] + mp;
    com[nd] = (com[nd] * mass[nd] + pp * mp) / newMass;
    mass[nd] = newMass;

    const bool leaf = (childCount[nd] == 0);
    if (leaf) {
        if (static_cast<int>(m_buildParticles[nd].size()) < m_ncrit || depth >= m_maxDepth) {
            m_buildParticles[nd].push_back(p);
            return;
        }
        // Split: re-insert the bucket's particles into children.
        std::vector<int> existing = std::move(m_buildParticles[nd]);
        m_buildParticles[nd].clear();
        for (int q : existing) InsertIntoChild(pts, node, q, depth);
    }
    InsertIntoChild(pts, node, p, depth);
}

template <int D>
void AdaptiveTree<D>::InsertIntoChild(const PosMassView<D>& pts, int node, int p, int depth) {
    const std::size_t nd = static_cast<std::size_t>(node);
    const int oct = OctantOf<D>(center[nd], pts.Pos(static_cast<std::size_t>(p)));
    int childIdx = children[nd][static_cast<std::size_t>(oct)];
    if (childIdx == -1) childIdx = NewChild(node, oct);
    Insert(pts, childIdx, p, depth + 1);
}

template <int D>
void AdaptiveTree<D>::Finalize(const PosMassView<D>& pts) {
    const int numNodes = static_cast<int>(mass.size());
    order.clear();
    order.reserve(pts.Count());
    maxRadius.assign(static_cast<std::size_t>(numNodes), 0.0);

    // One DFS from the root: emit each leaf's particles contiguously into
    // `order`, recording the [firstParticle, +particleCount) range. Uses an
    // explicit stack (trees can be ~40 deep for pathological inputs).
    std::vector<int> stack;
    stack.push_back(0);
    while (!stack.empty()) {
        const int node = stack.back();
        stack.pop_back();
        const std::size_t nd = static_cast<std::size_t>(node);
        if (childCount[nd] == 0) {
            firstParticle[nd] = static_cast<int>(order.size());
            for (int q : m_buildParticles[nd]) order.push_back(q);
            particleCount[nd] = static_cast<int>(m_buildParticles[nd].size());
            continue;
        }
        // Push children in reverse so they pop in ascending octant order
        // (keeps `order` roughly Morton-sorted, which helps cache locality).
        for (int c = kNC - 1; c >= 0; --c) {
            const int ch = children[nd][static_cast<std::size_t>(c)];
            if (ch != -1) stack.push_back(ch);
        }
    }
    m_buildParticles.clear();
    m_buildParticles.shrink_to_fit();

    // maxRadius bottom-up: decreasing node index is child-before-parent.
    for (int idx = numNodes - 1; idx >= 0; --idx) {
        const std::size_t nd = static_cast<std::size_t>(idx);
        if (mass[nd] <= 0.0) continue;
        if (childCount[nd] == 0) {
            double r = 0.0;
            const int lo = firstParticle[nd];
            const int hi = lo + particleCount[nd];
            for (int k = lo; k < hi; ++k) {
                const Vec<D> d = pts.Pos(static_cast<std::size_t>(order[static_cast<std::size_t>(k)])) - com[nd];
                r = std::max(r, std::sqrt(glm::dot(d, d)));
            }
            maxRadius[nd] = r;
            continue;
        }
        double r = 0.0;
        for (int c = 0; c < kNC; ++c) {
            const int ch = children[nd][static_cast<std::size_t>(c)];
            if (ch == -1) continue;
            const Vec<D> d = com[static_cast<std::size_t>(ch)] - com[nd];
            r = std::max(r, maxRadius[static_cast<std::size_t>(ch)] + std::sqrt(glm::dot(d, d)));
        }
        maxRadius[nd] = r;
    }
}

template <int D>
void AdaptiveTree<D>::ComputeQuadrupoles(const PosMassView<D>& pts) {
    if constexpr (D != 3) {
        return; // 2D uses complex-Laurent moments instead, computed elsewhere
    } else {
        const int numNodes = static_cast<int>(mass.size());
        Qxx.assign(static_cast<std::size_t>(numNodes), 0.0);
        Qyy.assign(static_cast<std::size_t>(numNodes), 0.0);
        Qxy.assign(static_cast<std::size_t>(numNodes), 0.0);
        Qxz.assign(static_cast<std::size_t>(numNodes), 0.0);
        Qyz.assign(static_cast<std::size_t>(numNodes), 0.0);

        for (int idx = numNodes - 1; idx >= 0; --idx) {
            const std::size_t nd = static_cast<std::size_t>(idx);
            if (mass[nd] <= 0.0) continue;

            if (childCount[nd] == 0) {
                const int lo = firstParticle[nd];
                const int hi = lo + particleCount[nd];
                if (hi - lo <= 1) continue; // a single point has zero quadrupole about itself
                for (int k = lo; k < hi; ++k) {
                    const int pi = order[static_cast<std::size_t>(k)];
                    const glm::dvec3 d = pts.Pos(static_cast<std::size_t>(pi)) - com[nd];
                    const double d2 = glm::dot(d, d);
                    const double mp = pts.m[static_cast<std::size_t>(pi)];
                    Qxx[nd] += mp * (3.0 * d.x * d.x - d2);
                    Qyy[nd] += mp * (3.0 * d.y * d.y - d2);
                    Qxy[nd] += mp * 3.0 * d.x * d.y;
                    Qxz[nd] += mp * 3.0 * d.x * d.z;
                    Qyz[nd] += mp * 3.0 * d.y * d.z;
                }
                continue;
            }

            for (int c = 0; c < kNC; ++c) {
                const int ch = children[nd][static_cast<std::size_t>(c)];
                if (ch == -1) continue;
                const std::size_t cs = static_cast<std::size_t>(ch);
                if (mass[cs] <= 0.0) continue;
                const glm::dvec3 d = com[cs] - com[nd];
                const double d2 = glm::dot(d, d);
                Qxx[nd] += Qxx[cs] + mass[cs] * (3.0 * d.x * d.x - d2);
                Qyy[nd] += Qyy[cs] + mass[cs] * (3.0 * d.y * d.y - d2);
                Qxy[nd] += Qxy[cs] + mass[cs] * 3.0 * d.x * d.y;
                Qxz[nd] += Qxz[cs] + mass[cs] * 3.0 * d.x * d.z;
                Qyz[nd] += Qyz[cs] + mass[cs] * 3.0 * d.y * d.z;
            }
        }
    }
}

template class AdaptiveTree<2>;
template class AdaptiveTree<3>;

} // namespace ngrav
