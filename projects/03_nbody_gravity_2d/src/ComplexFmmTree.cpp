#include "ComplexFmmTree.hpp"

#include "ComplexFmm.hpp"
#include "Quadtree.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <utility>
#include <vector>

namespace nbody2d {

namespace {

using fmm::Expansion;

// Expansion order: degrees 0..p. Deliberately much higher than the 3D
// project's p=5 -- see ComplexFmm.hpp's header comment: 2D's M2L is a
// single sum (O(p) per output coefficient, O(p^2) total per pair), not the
// 3D solver's double sum (O(p^4) total per pair), so a high order costs
// little here. p=10 matches the reference 2D FMM implementation this
// project was compared against.
constexpr int kOrder = 10;

// x's children, or {x} itself if x is a leaf -- identical in spirit to the
// 3D project's ChildrenOrSelf/SplitPair, duplicated (not shared) since the
// two projects' tree/node types differ.
std::vector<int> ChildrenOrSelf(const std::vector<QuadNode>& nodes, int x) {
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

void SplitPair(const std::vector<QuadNode>& nodes, int a, int b, int depth, std::vector<std::pair<int, int>>& out) {
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

std::vector<std::pair<int, int>> SplitPair(const std::vector<QuadNode>& nodes, int a, int b, int depth) {
    std::vector<std::pair<int, int>> out;
    SplitPair(nodes, a, b, depth, out);
    return out;
}

// Bottom-up pass (decreasing node index = children-before-parents, same
// invariant the 3D project's BuildMultipoles relies on) building each
// node's own multipole expansion about its own com, plus `maxRadius`: the
// farthest any of its member particles sits from that com.
void BuildMultipoles(const AdaptiveQuadtree& tree, std::vector<Expansion>& multipole, std::vector<double>& maxRadius) {
    const std::vector<QuadNode>& nodes = tree.Nodes();
    const std::vector<glm::dvec2>& pos = tree.Positions();
    const std::vector<double>& mass = tree.Masses();
    const int numNodes = static_cast<int>(nodes.size());

    multipole.assign(static_cast<size_t>(numNodes), Expansion(kOrder));
    maxRadius.assign(static_cast<size_t>(numNodes), 0.0);

    for (int idx = numNodes - 1; idx >= 0; --idx) {
        const QuadNode& node = nodes[static_cast<size_t>(idx)];
        if (node.mass <= 0.0) continue;

        if (node.isLeaf) {
            multipole[static_cast<size_t>(idx)] = fmm::P2M(fmm::ToComplex(node.com), kOrder, pos, mass, node.particles);
            double r = 0.0;
            for (int p : node.particles) r = std::max(r, glm::length(pos[static_cast<size_t>(p)] - node.com));
            maxRadius[static_cast<size_t>(idx)] = r;
            continue;
        }

        Expansion combined(kOrder, fmm::ToComplex(node.com));
        double r = 0.0;
        for (int c : node.children) {
            if (c == -1) continue;
            const QuadNode& child = nodes[static_cast<size_t>(c)];
            if (child.mass <= 0.0) continue;
            combined += fmm::M2M(multipole[static_cast<size_t>(c)], fmm::ToComplex(node.com));
            r = std::max(r, maxRadius[static_cast<size_t>(c)] + glm::length(child.com - node.com));
        }
        multipole[static_cast<size_t>(idx)] = combined;
        maxRadius[static_cast<size_t>(idx)] = r;
    }
}

// Dual-tree M2L traversal -- same stack-based (t,s) walk shape as the 3D
// project's TraverseSubtree.
void TraverseSubtree(const std::vector<QuadNode>& nodes, const std::vector<Expansion>& multipole,
                      const std::vector<double>& maxRadius, double theta2, double minSep2,
                      const std::vector<std::pair<int, int>>& seeds, std::vector<Expansion>& local,
                      std::vector<std::pair<int, int>>& nearPairsOut) {
    std::vector<std::pair<int, int>> stack(seeds);

    while (!stack.empty()) {
        const auto [t, s] = stack.back();
        stack.pop_back();

        const QuadNode& nodeT = nodes[static_cast<size_t>(t)];
        const QuadNode& nodeS = nodes[static_cast<size_t>(s)];
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

        const glm::dvec2 d = nodeT.com - nodeS.com;
        const double dist2 = glm::dot(d, d);

        const bool degenerateT = nodeT.isLeaf && nodeT.particles.size() > 1;
        const bool degenerateS = nodeS.isLeaf && nodeS.particles.size() > 1;

        // MAC requires *both* the box-geometry opening-angle floor
        // (sizeSum, never zero even for a one-particle leaf) and the
        // tighter exact-radius bound (reach, which *is* exactly zero for a
        // single-particle leaf) -- the 3D project's spherical FMM shipped
        // with only the reach-based bound, which made its criterion
        // vacuously true for leaf-vs-leaf pairs (maxRadius=0 on both
        // sides) regardless of theta or distance, and had to be fixed
        // after the fact. Both are required here from the start.
        const double sizeSum = 2.0 * nodeT.halfSize + 2.0 * nodeS.halfSize;
        const double reach = maxRadius[static_cast<size_t>(t)] + maxRadius[static_cast<size_t>(s)];
        const bool wellSeparated =
            (dist2 > minSep2) && (sizeSum * sizeSum < theta2 * dist2) && (reach * reach < theta2 * dist2);

        if (degenerateT || degenerateS || wellSeparated) {
            local[static_cast<size_t>(t)] += fmm::M2L(multipole[static_cast<size_t>(s)], fmm::ToComplex(nodeT.com), kOrder);
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
        } else if (nodeT.halfSize > nodeS.halfSize || (nodeT.halfSize == nodeS.halfSize && t < s)) {
            // Splitting on a halfSize *tie* must be decided by node identity
            // (t<s), not by which node happens to be labeled "t" -- the
            // dual-tree walk explores both (A,B) and (B,A) as separate
            // stack entries (from the t==s self-pair's own child-pair
            // expansion), and if the tie-break instead always preferred
            // "whichever node is currently first", the two directions can
            // refine at different rates and reach *different* accept/
            // reject decisions for what is mathematically the same pair --
            // confirmed via manual trace to be exactly the source of a
            // real double-counting bug (one direction accepts a coarse
            // M2L pair while the mirror direction keeps splitting past it
            // and finds a near-field sub-pair, adding both).
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

void ComputeAccelComplexFmm(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                             double softening, double theta, std::vector<glm::dvec2>& accelOut, FmmStats* stats) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec2(0.0));
    if (n == 0) return;

    const auto tBuild0 = std::chrono::steady_clock::now();
    AdaptiveQuadtree tree(pos, mass);
    const auto tBuild1 = std::chrono::steady_clock::now();

    std::vector<Expansion> multipole;
    std::vector<double> maxRadius;
    BuildMultipoles(tree, multipole, maxRadius);
    const auto tMultipole1 = std::chrono::steady_clock::now();

    const std::vector<QuadNode>& nodes = tree.Nodes();
    const int numNodes = static_cast<int>(nodes.size());

    std::vector<Expansion> local(static_cast<size_t>(numNodes));
    for (int idx = 0; idx < numNodes; ++idx) local[static_cast<size_t>(idx)] = Expansion(kOrder, fmm::ToComplex(nodes[static_cast<size_t>(idx)].com));

    const double theta2 = theta * theta;
    // Same physical-softening floor as the 3D project's FMM solvers, for
    // the same reason: the unsoftened M2L/L2P formulas have no softening
    // term, so a genuinely close pair still needs the near-field path
    // regardless of what the truncation-error bound says about
    // approximation accuracy.
    //
    // Known accuracy characteristic, found empirically: this solver's
    // accuracy advantage over Barnes-Hut (clear at matched theta for a
    // "normal" softening, e.g. softening a few % of the system's own
    // extent) shrinks and can invert once softening consumes a large
    // fraction of the domain (e.g. RotatingDisk's own 3x-mean-spacing
    // formula on a unit-radius disk) -- confirmed to also affect the 3D
    // project's already-shipped Cartesian FMM in the same scenario shape
    // (Barnes-Hut mean err 1.20e-2 vs. Adaptive FMM's 1.40e-2 there), so
    // this looks like a shared property of the minSep2-gated near-field
    // design itself (forcing deep recursion once minSep2 is a large
    // fraction of the domain) rather than a bug specific to this port.
    // Not chased further -- see this project's plan doc's validation
    // section for the empirical trail.
    const double minSep2 = (5.0 * softening) * (5.0 * softening);

    constexpr int kSplitDepth = 4; // one level deeper than the 3D project's 3, since a quadtree node has only 4 children (vs. 8) -- keeps a comparable target-bucket count (4^4=256 vs 8^3=512)
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
    // the 3D project uses).
    for (int idx = 1; idx < numNodes; ++idx) {
        const QuadNode& node = nodes[static_cast<size_t>(idx)];
        const int p = node.parent;
        if (p == -1) continue;
        local[static_cast<size_t>(idx)] += fmm::L2L(local[static_cast<size_t>(p)], fmm::ToComplex(node.com));
    }
    const auto tL2l1 = std::chrono::steady_clock::now();

    // L2P: evaluate each leaf's finalized local expansion at its member
    // particle(s), plus an exact brute-force sum for a degenerate leaf's
    // own mutual interactions.
    const double eps2 = softening * softening;
#pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < numNodes; ++idx) {
        const QuadNode& node = nodes[static_cast<size_t>(idx)];
        if (!node.isLeaf || node.particles.empty()) continue;

        for (int p : node.particles) {
            const fmm::PotentialAndForce field = fmm::EvaluateLocal(local[static_cast<size_t>(idx)], pos[static_cast<size_t>(p)], G);
            accelOut[static_cast<size_t>(p)] += field.force;
        }

        if (node.particles.size() > 1) {
            for (size_t a = 0; a < node.particles.size(); ++a) {
                for (size_t b = a + 1; b < node.particles.size(); ++b) {
                    const int pa = node.particles[a];
                    const int pb = node.particles[b];
                    const glm::dvec2 dab = pos[static_cast<size_t>(pb)] - pos[static_cast<size_t>(pa)];
                    const double dist2 = glm::dot(dab, dab) + eps2;
                    accelOut[static_cast<size_t>(pa)] += G * mass[static_cast<size_t>(pb)] * dab / dist2;
                    accelOut[static_cast<size_t>(pb)] -= G * mass[static_cast<size_t>(pa)] * dab / dist2;
                }
            }
        }
    }
    const auto tL2p1 = std::chrono::steady_clock::now();

    // Near-field pairs: exact softened summation.
    for (const std::vector<std::pair<int, int>>& bucket : nearPairsPerTarget) {
        for (const auto& [pi, pj] : bucket) {
            const glm::dvec2 d = pos[static_cast<size_t>(pj)] - pos[static_cast<size_t>(pi)];
            const double dist2 = glm::dot(d, d) + eps2;
            accelOut[static_cast<size_t>(pi)] += G * mass[static_cast<size_t>(pj)] * d / dist2;
            accelOut[static_cast<size_t>(pj)] -= G * mass[static_cast<size_t>(pi)] * d / dist2;
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

} // namespace nbody2d
