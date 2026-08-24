#include "AdaptiveFmm.hpp"

#include "Octree.hpp"

#include <cmath>
#include <utility>
#include <vector>

namespace nbody {

namespace {

// Symmetric 3x3 tensor: the local expansion's linear (Jacobian-of-
// acceleration) term, and a source node's traceless quadrupole moment
// (Qzz = -Qxx-Qyy is never stored). "H" follows nbody.py's naming: it's
// the Hessian of the *potential* (a = -grad(phi), so d(a_i)/d(x_j) = -H_ij
// -- see the "-H.Apply(d)" shift below).
struct SymMat3 {
    double xx = 0.0, yy = 0.0, zz = 0.0, xy = 0.0, xz = 0.0, yz = 0.0;

    SymMat3& operator+=(const SymMat3& o) {
        xx += o.xx; yy += o.yy; zz += o.zz; xy += o.xy; xz += o.xz; yz += o.yz;
        return *this;
    }

    glm::dvec3 Apply(const glm::dvec3& v) const {
        return glm::dvec3(xx * v.x + xy * v.y + xz * v.z, xy * v.x + yy * v.y + yz * v.z,
                           xz * v.x + yz * v.y + zz * v.z);
    }
};

struct MultipoleField {
    glm::dvec3 accel{0.0};
    SymMat3 hessian;
};

// Monopole+quadrupole field of a source (mass M, traceless quadrupole Q)
// at displacement r = target - source, plus the potential's Hessian at
// that same point (needed to shift this field's *value* down to a nearby
// point later, via a first-order Taylor step -- the L2L/L2P shifts below).
// 3D generalization of nbody.py's `_multipole_field_scalar`; every term
// here was independently derived and then checked against that function's
// exact 2D closed form (setting rz=0, Qxz=Qyz=0 reproduces it termwise).
//
// Written in terms of powers of 1/r^2 (one true division) rather than
// dividing by r3/r5/r7/r9 separately (the original, more directly-
// transcribed-from-the-formulas version did ~19 divisions here) -- this
// function runs once per M2L interaction, i.e. it's the traversal's inner
// loop, and division is far more expensive than multiplication on typical
// hardware. Values are bit-for-bit equivalent modulo floating-point
// reassociation.
MultipoleField EvaluateMultipoleField(double M, const SymMat3& Q, const glm::dvec3& r, double G) {
    const double r2 = glm::dot(r, r);
    const double invR2 = 1.0 / r2;
    const double invR = std::sqrt(invR2); // == 1/|r|, avoids a second division
    const double invR3 = invR2 * invR;
    const double invR5 = invR3 * invR2;
    const double invR7 = invR5 * invR2;
    const double invR9 = invR7 * invR2;

    MultipoleField f;

    // Monopole.
    f.accel = -G * M * invR3 * r;
    f.hessian.xx = G * M * (r2 - 3.0 * r.x * r.x) * invR5;
    f.hessian.yy = G * M * (r2 - 3.0 * r.y * r.y) * invR5;
    f.hessian.zz = G * M * (r2 - 3.0 * r.z * r.z) * invR5;
    f.hessian.xy = -3.0 * G * M * r.x * r.y * invR5;
    f.hessian.xz = -3.0 * G * M * r.x * r.z * invR5;
    f.hessian.yz = -3.0 * G * M * r.y * r.z * invR5;

    // Quadrupole correction.
    const glm::dvec3 Qr = Q.Apply(r);
    const double Qr2 = glm::dot(r, Qr); // Q_ab * r_a * r_b

    f.accel += G * invR5 * Qr - (2.5 * G * Qr2 * invR7) * r;

    f.hessian.xx += -G * Q.xx * invR5 + 10.0 * G * Qr.x * r.x * invR7 + 2.5 * G * Qr2 * invR7 -
                     17.5 * G * Qr2 * r.x * r.x * invR9;
    f.hessian.yy += -G * Q.yy * invR5 + 10.0 * G * Qr.y * r.y * invR7 + 2.5 * G * Qr2 * invR7 -
                     17.5 * G * Qr2 * r.y * r.y * invR9;
    f.hessian.zz += -G * Q.zz * invR5 + 10.0 * G * Qr.z * r.z * invR7 + 2.5 * G * Qr2 * invR7 -
                     17.5 * G * Qr2 * r.z * r.z * invR9;
    f.hessian.xy += -G * Q.xy * invR5 + 5.0 * G * (Qr.x * r.y + Qr.y * r.x) * invR7 - 17.5 * G * Qr2 * r.x * r.y * invR9;
    f.hessian.xz += -G * Q.xz * invR5 + 5.0 * G * (Qr.x * r.z + Qr.z * r.x) * invR7 - 17.5 * G * Qr2 * r.x * r.z * invR9;
    f.hessian.yz += -G * Q.yz * invR5 + 5.0 * G * (Qr.y * r.z + Qr.z * r.y) * invR7 - 17.5 * G * Qr2 * r.y * r.z * invR9;

    return f;
}

struct LocalExpansion {
    glm::dvec3 a0{0.0};
    SymMat3 H;
};

SymMat3 NodeQuadrupole(const OctreeNode& node) {
    SymMat3 q;
    q.xx = node.Qxx;
    q.yy = node.Qyy;
    q.zz = -node.Qxx - node.Qyy;
    q.xy = node.Qxy;
    q.xz = node.Qxz;
    q.yz = node.Qyz;
    return q;
}

// Ratio of the quadrupole term's contribution to the potential against the
// monopole term's, |phi_quad/phi_mono| = |Qr2/(2r^5)| / (M/r) = |Qr2|/(2Mr^4)
// -- an error-controlled complement to the pure opening-angle MAC (Salmon &
// Warren's "adjusted" criteria are the classic version of this idea).
// A fixed geometric size/distance ratio is a poor proxy for how trustworthy
// the *quadrupole-truncated* expansion actually is: it says nothing about
// how the source's mass is actually distributed, which is exactly what the
// quadrupole moment itself measures. A node with a small or symmetric
// quadrupole moment (nearly monopole-like) can safely be treated as well-
// separated much closer than a node with a large one, and this ratio
// quantifies that directly rather than guessing a single fixed distance
// for every source regardless of shape.
//
// This also degrades exactly the right way as r -> 0 (Qr2 ~ Q*r^2 for fixed
// Q, so the ratio ~ Q/(2*M*r^2) blows up) -- i.e. it independently catches
// the same catastrophic-cancellation regime the fixed minSep2 floor exists
// to guard against (see ComputeAccelAdaptiveFmm), just based on the actual
// field magnitude rather than a guessed distance.
double QuadrupoleRatio(double M, const SymMat3& Q, const glm::dvec3& r) {
    const double r2 = glm::dot(r, r);
    const glm::dvec3 Qr = Q.Apply(r);
    const double Qr2 = glm::dot(r, Qr);
    return std::abs(Qr2) / (2.0 * M * r2 * r2);
}

// How large the quadrupole/monopole ratio above is allowed to be before a
// pair falls back to the near-field path regardless of the opening-angle
// test. At the opening-angle boundary itself (node size == theta * dist),
// a maximally asymmetric node's quadrupole moment scales the ratio to
// roughly theta^2/2 (e.g. ~0.125 at theta=0.5) -- this is set well above
// that so typical well-separated pairs are unaffected, catching only
// distributions whose quadrupole moment is unusually large for their size.
constexpr double kQuadrupoleRatioLimit = 0.05;

// Runs the dual-tree M2L traversal starting from `seeds`, writing local
// expansions into `local` and near-field pairs into `nearPairsOut`.
//
// Parallelization note: in this traversal, the *target* index `t` of any
// (t, s) pair reached only ever descends to t's own children (see the
// branch logic below) -- it never jumps to an unrelated node. So a
// traversal seeded with `t` confined to one specific subtree (in practice,
// one of the root's grandchildren -- see ComputeAccelAdaptiveFmm) can only
// ever write to `local[idx]` for `idx` inside that same subtree, for the
// lifetime of this call. Two calls seeded with *different*, disjoint
// subtrees therefore never write the same `local[idx]`, so
// ComputeAccelAdaptiveFmm runs one of these per subtree in parallel, all
// sharing the same `local` vector, with no locking needed. The source
// side `s` still ranges freely across the whole tree (that's what makes
// this M2L rather than a per-subtree-only approximation), so a single call
// here still needs the full `nodes` array, just not exclusive access to
// all of `local`. Each call gets its own `nearPairsOut` purely so multiple
// threads never contend on one vector's growth -- the caller processes
// each bucket's near-field pairs directly rather than merging them.
void TraverseSubtree(const std::vector<OctreeNode>& nodes, double theta2, double minSep2, double G,
                      const std::vector<std::pair<int, int>>& seeds, std::vector<LocalExpansion>& local,
                      std::vector<std::pair<int, int>>& nearPairsOut) {
    std::vector<std::pair<int, int>> stack(seeds);

    while (!stack.empty()) {
        const auto [t, s] = stack.back();
        stack.pop_back();

        const OctreeNode& nodeT = nodes[static_cast<size_t>(t)];
        const OctreeNode& nodeS = nodes[static_cast<size_t>(s)];
        if (nodeT.mass <= 0.0 || nodeS.mass <= 0.0) continue;

        if (t == s) {
            // A self-pair only needs splitting once: (child_i, child_j)
            // for i != j is already handled as an ordinary (t, s) pair by
            // a later iteration, so this just seeds those cross pairs plus
            // each child's own self-pair. Self-pairs recur at every level
            // (not just the root), since this same split can produce new
            // (child, child) self-pairs.
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
        const double sizeSum = 2.0 * nodeT.halfSize + 2.0 * nodeS.halfSize;

        // A degenerate multi-particle leaf has no single particle index to
        // record a near-field pair with -- but it's also smaller than the
        // tree's minimum resolvable size, so treating it as a point mass
        // via M2L for *any* interaction (regardless of the opening-angle
        // test) is an excellent approximation, not a hack.
        const bool degenerateT = nodeT.isLeaf && nodeT.particles.size() > 1;
        const bool degenerateS = nodeS.isLeaf && nodeS.particles.size() > 1;
        // dist2 > minSep2 stays as a cheap, first-pass floor (minSep2 is now
        // small -- see ComputeAccelAdaptiveFmm); the quadrupole-ratio check
        // is what actually decides borderline cases the old, much larger
        // fixed floor used to reject outright regardless of whether the
        // source's mass distribution made that unnecessary.
        const bool wellSeparated = (sizeSum * sizeSum < theta2 * dist2) && (dist2 > minSep2) &&
                                    (QuadrupoleRatio(nodeS.mass, NodeQuadrupole(nodeS), d) < kQuadrupoleRatioLimit);

        if (degenerateT || degenerateS || wellSeparated) {
            const MultipoleField f = EvaluateMultipoleField(nodeS.mass, NodeQuadrupole(nodeS), d, G);
            local[static_cast<size_t>(t)].a0 += f.accel;
            local[static_cast<size_t>(t)].H += f.hessian;
            continue;
        }

        if (nodeT.isLeaf && nodeS.isLeaf) {
            // Dual-tree traversal visits both (A,B) and (B,A) once a self-
            // pair's children are split (needed for M2L, since the two
            // directions feed different local expansions) -- but a near-
            // field pair is symmetric, so only recording it once (smaller
            // particle index first) avoids double-counting the force. This
            // still holds with the traversal split across threads: (A,B)
            // and (B,A) are independent (t,s) pairs that may land in
            // different threads' subtrees, but each is decided purely by
            // its own pt/ps values, so exactly one side records it either
            // way.
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

void ComputeAccelAdaptiveFmm(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                              double softening, double theta, std::vector<glm::dvec3>& accelOut) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec3(0.0));
    if (n == 0) return;

    AdaptiveOctree tree(pos, mass);
    tree.ComputeQuadrupoles();
    const std::vector<OctreeNode>& nodes = tree.Nodes();
    const int numNodes = static_cast<int>(nodes.size());

    std::vector<LocalExpansion> local(static_cast<size_t>(numNodes));

    const double theta2 = theta * theta;
    // The opening-angle test alone is scale-invariant, so a pathologically
    // clustered distribution can recurse the tree down to separations many
    // orders of magnitude below the softening length, where it's still
    // "satisfied" in a relative sense -- but the *unsoftened* multipole
    // field's high-order (1/r^7, 1/r^9) terms are numerically catastrophic
    // there (nbody.py found this via an empirical dist^2 ~ 1e-18 pair
    // producing a ~1e14 spurious acceleration). Below this floor, softening
    // already regularizes the near-field formula anyway, so nothing
    // physical is lost by refusing the multipole shortcut.
    //
    // nbody.py's own fixed floor was 10x softening -- chosen empirically
    // for a pathologically clustered synthetic case many orders of
    // magnitude below softening, not tuned against a realistic scenario.
    // At this project's actual particle counts, that generosity turned out
    // to matter: once a scenario's softening is scaled up relative to the
    // tree's leaf spacing (done deliberately here to avoid close-encounter
    // integration blowup -- see Scenarios.cpp), "10x softening" can be tens
    // of mean interparticle spacings wide, routing millions of ordinary,
    // perfectly safe pairs to near-field for no numerical reason (measured:
    // ~9 million of them at N=25000 in the default disk scenario). Shrunk
    // to 2x softening -- still comfortably (many orders of magnitude) above
    // the regime that actually blew up -- with the quadrupole-ratio check
    // below (see QuadrupoleRatio) as the more precise safety net for
    // whatever this smaller floor alone would let through unsafely: unlike
    // a fixed distance, it adapts to how asymmetric the source's actual
    // mass distribution is, and independently diverges in the same r -> 0
    // regime the floor is guarding against.
    const double minSep2 = (5.0 * softening) * (5.0 * softening);

    // Splitting the traversal by the root's immediate children alone caps
    // parallelism at 8 tasks (an octree root has at most 8 children) --
    // not enough to keep a many-core machine busy, and vulnerable to load
    // imbalance if particles aren't spread evenly across octants. Instead,
    // split two levels down (the root's *grandchildren*, up to 64 tasks):
    // every node "t" reached from a seed only ever descends to t's own
    // children (see TraverseSubtree's comment), so as long as each
    // parallel task's seeds all have `t` confined to one specific
    // grandchild's subtree, tasks still never write the same `local[idx]`.
    //
    // ChildrenOrSelf(x) is x's children, or {x} itself if x is a leaf
    // (can't split further -- x becomes its own, granularity-1 bucket).
    auto childrenOrSelf = [&nodes](int x) {
        std::vector<int> out;
        if (nodes[static_cast<size_t>(x)].isLeaf) {
            out.push_back(x);
        } else {
            for (int c : nodes[static_cast<size_t>(x)].children) {
                if (c != -1) out.push_back(c);
            }
        }
        return out;
    };

    const std::vector<int> level1 = childrenOrSelf(0); // root is never a leaf for N >= 2

    std::vector<std::vector<int>> bucketsOfLevel1(level1.size());
    std::vector<int> nodeToLevel1Idx(static_cast<size_t>(numNodes), -1);
    for (size_t i = 0; i < level1.size(); ++i) {
        bucketsOfLevel1[i] = childrenOrSelf(level1[i]);
        nodeToLevel1Idx[static_cast<size_t>(level1[i])] = static_cast<int>(i);
    }

    std::vector<int> targetNodes;
    std::vector<int> nodeToTargetIdx(static_cast<size_t>(numNodes), -1);
    for (const std::vector<int>& buckets : bucketsOfLevel1) {
        for (int g : buckets) {
            nodeToTargetIdx[static_cast<size_t>(g)] = static_cast<int>(targetNodes.size());
            targetNodes.push_back(g);
        }
    }

    std::vector<std::vector<std::pair<int, int>>> seedsPerTarget(targetNodes.size());
    auto addSeed = [&](int t, int s) { seedsPerTarget[static_cast<size_t>(nodeToTargetIdx[static_cast<size_t>(t)])].emplace_back(t, s); };

    // The root's self-pair split (single-threaded, O(children^2) --
    // trivial) generates every (ci, cj) pair among root's children,
    // exactly as the original single-traversal version's first step did.
    // Each then needs pushing one level finer to reach grandchild
    // granularity:
    //  - a self-pair (ci, ci): split into every (g1, g2) among ci's own
    //    children -- exactly what TraverseSubtree's own self-pair-split
    //    logic would have produced one level later, just precomputed here.
    //  - a cross-pair (ci, cj), ci != cj: only the *target* side needs
    //    splitting for parallelism's sake (the source side can stay at
    //    whatever granularity), so this pairs each of ci's children
    //    against the still-coarse cj. This is a legitimate finer-grained
    //    M2L opportunity, not an approximation of the original algorithm --
    //    dual-tree traversal's result doesn't depend on descent order, and
    //    evaluating the field one level deeper before falling back to an
    //    L2L shift is if anything marginally more accurate.
    for (int ci : level1) {
        const std::vector<int>& bucketsCi = bucketsOfLevel1[static_cast<size_t>(nodeToLevel1Idx[static_cast<size_t>(ci)])];
        for (int cj : level1) {
            if (ci == cj) {
                for (int g1 : bucketsCi) {
                    for (int g2 : bucketsCi) addSeed(g1, g2);
                }
            } else {
                for (int g : bucketsCi) addSeed(g, cj);
            }
        }
    }

    std::vector<std::vector<std::pair<int, int>>> nearPairsPerTarget(targetNodes.size());

    const int numTargets = static_cast<int>(targetNodes.size());
#pragma omp parallel for schedule(dynamic)
    for (int ti = 0; ti < numTargets; ++ti) {
        TraverseSubtree(nodes, theta2, minSep2, G, seedsPerTarget[static_cast<size_t>(ti)], local,
                         nearPairsPerTarget[static_cast<size_t>(ti)]);
    }

    // L2L: push each node's accumulated local expansion down to its
    // children. A single increasing-node-index sweep is already parent-
    // before-child (children always get a larger index than their parent
    // -- an invariant of AdaptiveOctree's insertion order), so this needs
    // no explicit recursion/BFS.
    for (int idx = 1; idx < numNodes; ++idx) {
        const OctreeNode& node = nodes[static_cast<size_t>(idx)];
        const int p = node.parent;
        if (p == -1) continue;
        const glm::dvec3 d = node.com - nodes[static_cast<size_t>(p)].com;
        local[static_cast<size_t>(idx)].a0 += local[static_cast<size_t>(p)].a0 - local[static_cast<size_t>(p)].H.Apply(d);
        local[static_cast<size_t>(idx)].H += local[static_cast<size_t>(p)].H;
    }

    // L2P: evaluate each leaf's finalized local expansion at its member
    // particle(s). A normal leaf's one particle sits exactly at the node's
    // center of mass, so the expansion's value there needs no further
    // shift. A degenerate leaf shares that same value across all its
    // members (they're smaller than the tree's minimum resolvable size,
    // so treating them as coincident for this purpose is fine) plus an
    // exact brute-force sum for their own mutual interactions.
    const double eps2 = softening * softening;
#pragma omp parallel for schedule(dynamic, 64)
    for (int idx = 0; idx < numNodes; ++idx) {
        const OctreeNode& node = nodes[static_cast<size_t>(idx)];
        if (!node.isLeaf || node.particles.empty()) continue;

        const glm::dvec3& a0 = local[static_cast<size_t>(idx)].a0;
        for (int p : node.particles) accelOut[static_cast<size_t>(p)] += a0;

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

    // Near-field pairs: exact softened summation, applied symmetrically.
    // Processed straight out of each target bucket's own vector rather
    // than first merging into one flat list -- avoids a copy that alone
    // cost several ms once the pair count gets large (see below).
    //
    // This can be a genuinely large fraction of the total cost: once a
    // scenario's softening (chosen to avoid close-encounter integration
    // blowup, see Scenarios.cpp) is large relative to the tree's actual
    // leaf spacing at that N, most interactions route through minSep2's
    // near-field fallback rather than the multipole shortcut -- the same
    // "not getting real O(N) leverage" regime nbody.py documented for
    // dense collapse snapshots. Tried parallelizing this loop across
    // buckets too (accelOut writes need to be atomic rather than merely
    // bucket-disjoint, since a pair's particles can belong to a different
    // bucket than the one iterating it -- the source side ranges freely,
    // see TraverseSubtree): measured slower than sequential, not faster --
    // this loop's accesses are effectively random within accelOut/pos/mass
    // (near-field pairs, unlike tree nodes, have no spatial locality by
    // construction), so it's memory-latency-bound already at one thread,
    // and six atomics per pair added synchronization cost without buying
    // back enough real parallelism. Left sequential on that evidence.
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
}

} // namespace nbody
