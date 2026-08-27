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

// Resolves a single (t,s) pair: accepts it via M2L (updating local[t]),
// records a near-field pair, or appends the sub-pairs it splits into to
// `out` (which may safely be the same container the caller is iterating
// or popping from, e.g. `out` == the DFS stack in TraverseSubtree below --
// this function only ever appends). Shared by the parallel per-group DFS
// walk (TraverseSubtree) and the serial BFS ramp-up in
// ComputeAccelComplexFmm, so both apply the exact same accept/split rules.
void ProcessPair(int t, int s, const std::vector<QuadNode>& nodes, const std::vector<Expansion>& multipole,
                  const std::vector<double>& maxRadius, double theta2, double minSep2, std::vector<Expansion>& local,
                  std::vector<std::pair<int, int>>& nearPairsOut, std::vector<std::pair<int, int>>& out) {
    const QuadNode& nodeT = nodes[static_cast<size_t>(t)];
    const QuadNode& nodeS = nodes[static_cast<size_t>(s)];
    if (nodeT.mass <= 0.0 || nodeS.mass <= 0.0) return;

    if (t == s) {
        if (nodeT.isLeaf) return;
        for (int ci : nodeT.children) {
            if (ci == -1) continue;
            for (int cj : nodeT.children) {
                if (cj == -1) continue;
                out.emplace_back(ci, cj);
            }
        }
        return;
    }

    const glm::dvec2 d = nodeT.com - nodeS.com;
    const double dist2 = glm::dot(d, d);

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

    if (wellSeparated) {
        local[static_cast<size_t>(t)] += fmm::M2L(multipole[static_cast<size_t>(s)], fmm::ToComplex(nodeT.com), kOrder);
        return;
    }

    // Neither side can split further (both are leaves, whether because
    // their bucket wasn't yet full or the near-duplicate-position depth
    // cap was hit) and they're not well separated -- exact softened
    // summation over every particle pair between the two leaves (each
    // leaf may hold several, per Quadtree.cpp's bucket size; a leaf-vs-
    // leaf pair is only ever visited from one of its two (t,s)/(s,t)
    // directions -- see the tie-break comment below -- so processing it
    // from the lower-indexed side avoids counting it twice, matching in
    // spirit the single-particle version's `pt<ps` check).
    if (nodeT.isLeaf && nodeS.isLeaf) {
        if (t < s) {
            for (int pt : nodeT.particles) {
                for (int ps : nodeS.particles) nearPairsOut.emplace_back(pt, ps);
            }
        }
        return;
    }

    if (nodeT.isLeaf) {
        for (int cj : nodeS.children) {
            if (cj != -1) out.emplace_back(t, cj);
        }
    } else if (nodeS.isLeaf) {
        for (int ci : nodeT.children) {
            if (ci != -1) out.emplace_back(ci, s);
        }
    } else if (nodeT.halfSize > nodeS.halfSize || (nodeT.halfSize == nodeS.halfSize && t < s)) {
        // Splitting on a halfSize *tie* must be decided by node identity
        // (t<s), not by which node happens to be labeled "t" -- the
        // dual-tree walk explores both (A,B) and (B,A) as separate
        // pairs (from the t==s self-pair's own child-pair expansion),
        // and if the tie-break instead always preferred "whichever node
        // is currently first", the two directions can refine at
        // different rates and reach *different* accept/reject decisions
        // for what is mathematically the same pair -- confirmed via
        // manual trace to be exactly the source of a real double-counting
        // bug (one direction accepts a coarse M2L pair while the mirror
        // direction keeps splitting past it and finds a near-field
        // sub-pair, adding both).
        for (int ci : nodeT.children) {
            if (ci != -1) out.emplace_back(ci, s);
        }
    } else {
        for (int cj : nodeS.children) {
            if (cj != -1) out.emplace_back(t, cj);
        }
    }
}

// Dual-tree M2L traversal: pops (t,s) pairs from `stack` (consumed in
// place) and resolves each via ProcessPair, which may push further
// sub-pairs back onto the same stack -- runs to completion. Used to fully
// resolve one parallel group's share of the leftover pairs from the BFS
// ramp-up below (see ComputeAccelComplexFmm) -- each group only ever
// touches descendants of its own "t", so different groups' local[] writes
// never overlap.
void TraverseSubtree(const std::vector<QuadNode>& nodes, const std::vector<Expansion>& multipole,
                      const std::vector<double>& maxRadius, double theta2, double minSep2,
                      std::vector<std::pair<int, int>>& stack, std::vector<Expansion>& local,
                      std::vector<std::pair<int, int>>& nearPairsOut) {
    while (!stack.empty()) {
        const auto [t, s] = stack.back();
        stack.pop_back();
        ProcessPair(t, s, nodes, multipole, maxRadius, theta2, minSep2, local, nearPairsOut, stack);
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

    const int numThreads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<std::pair<int, int>>> nearPairsPerTarget(static_cast<size_t>(numThreads));

    // Phase 1 (serial): a MAC-first, breadth-first ramp-up from
    // (root,root), accepting pairs at the coarsest level the MAC allows
    // before ever splitting further. This replaces an earlier design that
    // blindly pre-split every pair down to a fixed depth (purely to
    // generate enough independent seeds for the parallel phase below)
    // *before* checking whether the MAC would already accept a much
    // coarser (and far cheaper -- M2L's cost is O(p^2) per call,
    // independent of how many particles either side represents) pair.
    // That forced far more, far smaller M2L calls than necessary and was
    // confirmed empirically to dominate the solver's wall-clock time.
    //
    // Breadth-first (not the depth-first stack TraverseSubtree uses)
    // matters here: this ramp-up stops once the *current level's* pair
    // count reaches a threshold, handing that level to phase 2 as seeds.
    // A depth-first stack's size reflects how many sibling branches are
    // still open, not how much total work remains, so it stays small
    // (bounded by tree depth) even when there's enormous work left to do
    // -- confirmed empirically to never reach any reasonable threshold,
    // running the entire traversal serially. A breadth-first frontier's
    // size actually tracks the width of the remaining problem, growing
    // geometrically with each split, so it reaches the threshold quickly
    // whenever there's enough work to justify parallelizing -- and for
    // small trees, may drain to empty first without ever reaching it,
    // which is correct (nothing left to parallelize).
    std::vector<std::pair<int, int>> frontier{{0, 0}};
    std::vector<std::pair<int, int>> nearPairsSerial;
    const size_t targetSeedCount = static_cast<size_t>(numThreads) * 8;
    while (!frontier.empty() && frontier.size() < targetSeedCount) {
        std::vector<std::pair<int, int>> next;
        for (const auto& [t, s] : frontier) {
            ProcessPair(t, s, nodes, multipole, maxRadius, theta2, minSep2, local, nearPairsSerial, next);
        }
        frontier = std::move(next);
    }

    // Phase 2 (parallel): group the leftover pairs by their "t" side --
    // each group only ever touches descendants of its own "t", so
    // different groups' local[] writes never overlap -- and fully resolve
    // each group's own subtree on its own thread.
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
    for (const auto& [t, s] : frontier) {
        const int idx = targetIndexOf(t);
        if (idx >= static_cast<int>(seedsPerTarget.size())) seedsPerTarget.resize(static_cast<size_t>(idx) + 1);
        seedsPerTarget[static_cast<size_t>(idx)].emplace_back(t, s);
    }

    const int numTargets = static_cast<int>(targetNodes.size());
    const auto tSeed1 = std::chrono::steady_clock::now();

#pragma omp parallel for schedule(dynamic, 4)
    for (int ti = 0; ti < numTargets; ++ti) {
        TraverseSubtree(nodes, multipole, maxRadius, theta2, minSep2, seedsPerTarget[static_cast<size_t>(ti)], local,
                         nearPairsPerTarget[static_cast<size_t>(omp_get_thread_num())]);
    }
    nearPairsPerTarget[0].insert(nearPairsPerTarget[0].end(), nearPairsSerial.begin(), nearPairsSerial.end());
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
    // particle(s), plus an exact brute-force sum for a bucket's own
    // mutual (intra-leaf) interactions -- the M2L/near-field interaction
    // list above only ever covers particles in *other* nodes.
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
