#include "SphericalFmm.hpp"

#include "Octree.hpp"
#include "SphericalHarmonics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

namespace nbody {

namespace {

using sh::Expansion;

// Expansion order: how many degrees (0..p) each multipole/local expansion
// carries. Unlike the Cartesian FMM (where going past quadrupole/octupole
// means hand-deriving a whole new tensor and field formula), this is just a
// bigger loop bound -- see SphericalHarmonics.hpp. Kept a compile-time
// constant for this first pass rather than a runtime slider (see this
// project's plan doc); p=5 is a reasonable balance for the naive O(p^4)
// M2M/M2L/L2L translations this pass uses (no rotation-based speedup yet).
constexpr int kOrder = 5;

// x's children, or {x} itself if x is a leaf -- identical helper to
// AdaptiveFmm.cpp's own ChildrenOrSelf/SplitPair, duplicated rather than
// shared so this solver stays fully independent of the Cartesian one (see
// this file's header comment on why: a new parallel path, not a shared
// refactor, while the two are still being A/B tested).
std::vector<int> ChildrenOrSelf(const std::vector<OctreeNode>& nodes, int x) {
    std::vector<int> out;
    if (nodes[static_cast<size_t>(x)].isLeaf) {
        out.push_back(x);
    } else {
        for (int c : nodes[static_cast<size_t>(x)].children) {
            if (c != -1) out.push_back(c);
        }
    }
    return out;
}

void SplitPair(const std::vector<OctreeNode>& nodes, int a, int b, int depth, std::vector<std::pair<int, int>>& out) {
    if (depth == 0) {
        out.emplace_back(a, b);
        return;
    }
    const std::vector<int> childrenA = ChildrenOrSelf(nodes, a);
    if (a == b) {
        for (int x : childrenA) {
            for (int y : childrenA) SplitPair(nodes, x, y, depth - 1, out);
        }
    } else {
        for (int x : childrenA) SplitPair(nodes, x, b, depth - 1, out);
    }
}

std::vector<std::pair<int, int>> SplitPair(const std::vector<OctreeNode>& nodes, int a, int b, int depth) {
    std::vector<std::pair<int, int>> out;
    SplitPair(nodes, a, b, depth, out);
    return out;
}

// Bottom-up pass (decreasing node index = children-before-parents, same
// invariant ComputeQuadrupoles relies on -- see Octree.cpp) building each
// node's own multipole expansion about its own com, plus `maxRadius`: the
// farthest any of its member particles sits from that com. maxRadius is
// this solver's replacement for the Cartesian FMM's QuadrupoleRatio -- see
// TraverseSubtree's own comment on why a plain, *exact* per-node radius is
// both simpler and more principled than a moment-ratio heuristic once the
// expansion order is a free parameter instead of fixed at quadrupole.
void BuildMultipoles(const AdaptiveOctree& tree, std::vector<Expansion>& multipole, std::vector<double>& maxRadius) {
    const std::vector<OctreeNode>& nodes = tree.Nodes();
    const std::vector<glm::dvec3>& pos = tree.Positions();
    const std::vector<double>& mass = tree.Masses();
    const int numNodes = static_cast<int>(nodes.size());

    multipole.assign(static_cast<size_t>(numNodes), Expansion(kOrder));
    maxRadius.assign(static_cast<size_t>(numNodes), 0.0);

    for (int idx = numNodes - 1; idx >= 0; --idx) {
        const OctreeNode& node = nodes[static_cast<size_t>(idx)];
        if (node.mass <= 0.0) continue;

        if (node.isLeaf) {
            multipole[static_cast<size_t>(idx)] = sh::P2M(node.com, kOrder, pos, mass, node.particles);
            double r = 0.0;
            for (int p : node.particles) r = std::max(r, glm::length(pos[static_cast<size_t>(p)] - node.com));
            maxRadius[static_cast<size_t>(idx)] = r;
            continue;
        }

        Expansion combined(kOrder, node.com);
        double r = 0.0;
        for (int c : node.children) {
            if (c == -1) continue;
            const OctreeNode& child = nodes[static_cast<size_t>(c)];
            if (child.mass <= 0.0) continue;
            combined += sh::M2M(multipole[static_cast<size_t>(c)], node.com);
            r = std::max(r, maxRadius[static_cast<size_t>(c)] + glm::length(child.com - node.com));
        }
        multipole[static_cast<size_t>(idx)] = combined;
        maxRadius[static_cast<size_t>(idx)] = r;
    }
}

// Dual-tree M2L traversal -- structurally identical to AdaptiveFmm.cpp's
// own TraverseSubtree (same stack-based (t,s) walk, same self-pair/cross-
// pair split, same degenerate-leaf and near-field handling), but the
// accept/reject test and the accumulation step are this solver's own: see
// the two comments below for what actually changed and why.
void TraverseSubtree(const std::vector<OctreeNode>& nodes, const std::vector<Expansion>& multipole,
                      const std::vector<double>& maxRadius, double theta2, double minSep2,
                      const std::vector<std::pair<int, int>>& seeds, std::vector<Expansion>& local,
                      std::vector<std::pair<int, int>>& nearPairsOut) {
    std::vector<std::pair<int, int>> stack(seeds);

    while (!stack.empty()) {
        const auto [t, s] = stack.back();
        stack.pop_back();

        const OctreeNode& nodeT = nodes[static_cast<size_t>(t)];
        const OctreeNode& nodeS = nodes[static_cast<size_t>(s)];
        if (nodeT.mass <= 0.0 || nodeS.mass <= 0.0) continue;

        if (t == s) {
            if (nodeT.isLeaf) continue;
            for (int ci : nodeT.children) {
                if (ci == -1) continue;
                for (int cj : nodeT.children) {
                    if (cj == -1) continue;
                    stack.emplace_back(ci, cj);
                }
            }
            continue;
        }

        const glm::dvec3 d = nodeT.com - nodeS.com;
        const double dist2 = glm::dot(d, d);

        const bool degenerateT = nodeT.isLeaf && nodeT.particles.size() > 1;
        const bool degenerateS = nodeS.isLeaf && nodeS.particles.size() > 1;

        // Replaces the Cartesian FMM's QuadrupoleRatio safety net with a
        // tighter, exact-radius version of the same geometric opening-angle
        // test: a degree-p expansion's truncation error for a pair
        // separated by `dist`, where the source's own particles reach out
        // to `a` and the target's to `b`, is bounded by ((a+b)/dist)^(p+1)
        // -- a classical, provable FMM error bound (see this project's plan
        // doc), not a heuristic. Comparing (a+b) against theta*dist is
        // exactly that bound evaluated at tolerance theta^(p+1) (the two
        // are monotonically equivalent, so there's no need to actually
        // raise anything to the p+1 power) -- reusing theta lets this
        // solver share the app's existing opening-angle slider rather than
        // inventing a new tunable. Both radii are included, symmetrically,
        // for the same reason the Cartesian FMM's sizeSum used both node
        // sizes: a target node this large might still have descendants
        // close to the source even when its own com isn't (this is
        // exactly the geometry-vs-com distinction that broke the Task #3
        // near-field-threshold attempt on the Cartesian solver -- see this
        // project's session history -- so both sides are covered here from
        // the start rather than discovered the hard way again).
        //
        // A single-particle leaf has maxRadius exactly 0 (no spread about
        // its own com) -- correct on its own (its own higher moments truly
        // are all zero), but it makes `reach` alone vacuous for leaf-vs-
        // leaf pairs: 0 < theta^2*dist^2 holds for *any* positive dist, so
        // theta stops constraining the single most common pair type in the
        // whole traversal at all. sizeSum (box geometry, never zero even
        // for a one-particle leaf) is the same floor the Cartesian FMM's
        // criterion already relies on for exactly this reason -- required
        // here too, alongside the tighter maxRadius-based bound, rather
        // than replacing it outright.
        const double sizeSum = 2.0 * nodeT.halfSize + 2.0 * nodeS.halfSize;
        const double reach = maxRadius[static_cast<size_t>(t)] + maxRadius[static_cast<size_t>(s)];
        const bool wellSeparated =
            (dist2 > minSep2) && (sizeSum * sizeSum < theta2 * dist2) && (reach * reach < theta2 * dist2);

        if (degenerateT || degenerateS || wellSeparated) {
            // M2L then accumulate -- the direct analogue of
            // AdaptiveFmm.cpp's EvaluateMultipoleField call, but producing
            // a full degree-p local expansion instead of a value+Hessian
            // pair, so nothing is lost when this later gets pushed further
            // down the tree via L2L (unlike the Cartesian path's linear-
            // only local side -- see this project's plan doc).
            local[static_cast<size_t>(t)] += sh::M2L(multipole[static_cast<size_t>(s)], nodeT.com, kOrder);
            continue;
        }

        if (nodeT.isLeaf && nodeS.isLeaf) {
            const int pt = nodeT.particles[0];
            const int ps = nodeS.particles[0];
            if (pt < ps) nearPairsOut.emplace_back(pt, ps);
            continue;
        }

        if (nodeT.isLeaf) {
            for (int cj : nodeS.children) {
                if (cj != -1) stack.emplace_back(t, cj);
            }
        } else if (nodeS.isLeaf) {
            for (int ci : nodeT.children) {
                if (ci != -1) stack.emplace_back(ci, s);
            }
        } else if (nodeT.halfSize >= nodeS.halfSize) {
            for (int ci : nodeT.children) {
                if (ci != -1) stack.emplace_back(ci, s);
            }
        } else {
            for (int cj : nodeS.children) {
                if (cj != -1) stack.emplace_back(t, cj);
            }
        }
    }
}

} // namespace

void ComputeAccelSphericalFmm(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                               double softening, double theta, std::vector<glm::dvec3>& accelOut,
                               SphericalFmmStats* stats) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec3(0.0));
    if (n == 0) return;

    const auto tBuild0 = std::chrono::steady_clock::now();
    AdaptiveOctree tree(pos, mass);
    const auto tBuild1 = std::chrono::steady_clock::now();

    std::vector<Expansion> multipole;
    std::vector<double> maxRadius;
    BuildMultipoles(tree, multipole, maxRadius);
    const auto tMultipole1 = std::chrono::steady_clock::now();

    const std::vector<OctreeNode>& nodes = tree.Nodes();
    const int numNodes = static_cast<int>(nodes.size());

    std::vector<Expansion> local(static_cast<size_t>(numNodes));
    for (int idx = 0; idx < numNodes; ++idx) local[static_cast<size_t>(idx)] = Expansion(kOrder, nodes[static_cast<size_t>(idx)].com);

    const double theta2 = theta * theta;
    // Same physical-softening floor as the Cartesian FMM (AdaptiveFmm.cpp)
    // and for the same reason: the unsoftened M2L/L2P formulas have no
    // softening term at all, so a pair genuinely within a softening length
    // of each other needs the near-field path regardless of how good this
    // solver's truncation-error bound says the multipole approximation is
    // -- that bound only ever speaks to approximation accuracy, not to
    // whether softening the near-field is physically required. Carried
    // over unchanged rather than reinvented (see this project's plan doc).
    const double minSep2 = (5.0 * softening) * (5.0 * softening);

    constexpr int kSplitDepth = 3; // matches AdaptiveFmm.cpp's own choice
    const std::vector<std::pair<int, int>> finalPairs = SplitPair(nodes, 0, 0, kSplitDepth);

    std::vector<int> targetNodes;
    std::vector<int> nodeToTargetIdx(static_cast<size_t>(numNodes), -1);
    auto targetIndexOf = [&](int t) -> int {
        int& idx = nodeToTargetIdx[static_cast<size_t>(t)];
        if (idx == -1) {
            idx = static_cast<int>(targetNodes.size());
            targetNodes.push_back(t);
        }
        return idx;
    };

    std::vector<std::vector<std::pair<int, int>>> seedsPerTarget;
    for (const auto& [t, s] : finalPairs) {
        const int idx = targetIndexOf(t);
        if (idx >= static_cast<int>(seedsPerTarget.size())) seedsPerTarget.resize(static_cast<size_t>(idx) + 1);
        seedsPerTarget[static_cast<size_t>(idx)].emplace_back(t, s);
    }

    const int numThreads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<std::pair<int, int>>> nearPairsPerTarget(static_cast<size_t>(numThreads));

    const int numTargets = static_cast<int>(targetNodes.size());
    const auto tSeed1 = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic, 4)
    for (int ti = 0; ti < numTargets; ++ti) {
        TraverseSubtree(nodes, multipole, maxRadius, theta2, minSep2, seedsPerTarget[static_cast<size_t>(ti)], local,
                         nearPairsPerTarget[static_cast<size_t>(omp_get_thread_num())]);
    }
    const auto tTraverse1 = std::chrono::steady_clock::now();

    // L2L: push each node's accumulated local expansion down to its
    // children (parent-before-child via the same increasing-index sweep
    // AdaptiveFmm.cpp uses), shifting the *entire* degree-p array rather
    // than a linear value+Hessian pair.
    for (int idx = 1; idx < numNodes; ++idx) {
        const OctreeNode& node = nodes[static_cast<size_t>(idx)];
        const int p = node.parent;
        if (p == -1) continue;
        local[static_cast<size_t>(idx)] += sh::L2L(local[static_cast<size_t>(p)], node.com);
    }
    const auto tL2l1 = std::chrono::steady_clock::now();

    // L2P: evaluate each leaf's finalized local expansion at its member
    // particle(s), plus an exact brute-force sum for a degenerate leaf's
    // own mutual interactions -- identical structure to AdaptiveFmm.cpp.
    const double eps2 = softening * softening;
#pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < numNodes; ++idx) {
        const OctreeNode& node = nodes[static_cast<size_t>(idx)];
        if (!node.isLeaf || node.particles.empty()) continue;

        for (int p : node.particles) {
            const sh::PotentialAndAccel field = sh::EvaluateLocal(local[static_cast<size_t>(idx)], pos[static_cast<size_t>(p)], G);
            accelOut[static_cast<size_t>(p)] += field.accel;
        }

        if (node.particles.size() > 1) {
            for (size_t a = 0; a < node.particles.size(); ++a) {
                for (size_t b = a + 1; b < node.particles.size(); ++b) {
                    const int pa = node.particles[a];
                    const int pb = node.particles[b];
                    const glm::dvec3 dab = pos[static_cast<size_t>(pb)] - pos[static_cast<size_t>(pa)];
                    const double dist2 = glm::dot(dab, dab) + eps2;
                    const double invDist = 1.0 / std::sqrt(dist2);
                    const double invDist3 = invDist * invDist * invDist;
                    accelOut[static_cast<size_t>(pa)] += G * mass[static_cast<size_t>(pb)] * invDist3 * dab;
                    accelOut[static_cast<size_t>(pb)] -= G * mass[static_cast<size_t>(pa)] * invDist3 * dab;
                }
            }
        }
    }
    const auto tL2p1 = std::chrono::steady_clock::now();

    // Near-field pairs: exact softened summation -- identical to
    // AdaptiveFmm.cpp (see that file's own comment on why this stays
    // sequential rather than parallelized across buckets).
    for (const std::vector<std::pair<int, int>>& bucket : nearPairsPerTarget) {
        for (const auto& [pi, pj] : bucket) {
            const glm::dvec3 d = pos[static_cast<size_t>(pj)] - pos[static_cast<size_t>(pi)];
            const double dist2 = glm::dot(d, d) + eps2;
            const double invDist = 1.0 / std::sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            accelOut[static_cast<size_t>(pi)] += G * mass[static_cast<size_t>(pj)] * invDist3 * d;
            accelOut[static_cast<size_t>(pj)] -= G * mass[static_cast<size_t>(pi)] * invDist3 * d;
        }
    }
    const auto tNear1 = std::chrono::steady_clock::now();

    if (stats) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        stats->buildMs = ms(tBuild0, tBuild1);
        stats->multipoleMs = ms(tBuild1, tMultipole1);
        stats->seedMs = ms(tMultipole1, tSeed1);
        stats->traverseMs = ms(tSeed1, tTraverse1);
        stats->l2lMs = ms(tTraverse1, tL2l1);
        stats->l2pMs = ms(tL2l1, tL2p1);
        stats->nearFieldMs = ms(tL2p1, tNear1);
        stats->nodeCount = numNodes;
        stats->numTargets = numTargets;
        size_t totalNear = 0;
        for (const auto& bucket : nearPairsPerTarget) totalNear += bucket.size();
        stats->nearPairCount = totalNear;
    }
}

} // namespace nbody
