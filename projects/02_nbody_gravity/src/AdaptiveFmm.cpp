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
MultipoleField EvaluateMultipoleField(double M, const SymMat3& Q, const glm::dvec3& r, double G) {
    const double r2 = glm::dot(r, r);
    const double rmag = std::sqrt(r2);
    const double r3 = r2 * rmag;
    const double r5 = r2 * r3;
    const double r7 = r2 * r5;
    const double r9 = r2 * r7;

    MultipoleField f;

    // Monopole.
    f.accel = -G * M * r / r3;
    f.hessian.xx = G * M * (r2 - 3.0 * r.x * r.x) / r5;
    f.hessian.yy = G * M * (r2 - 3.0 * r.y * r.y) / r5;
    f.hessian.zz = G * M * (r2 - 3.0 * r.z * r.z) / r5;
    f.hessian.xy = -3.0 * G * M * r.x * r.y / r5;
    f.hessian.xz = -3.0 * G * M * r.x * r.z / r5;
    f.hessian.yz = -3.0 * G * M * r.y * r.z / r5;

    // Quadrupole correction.
    const glm::dvec3 Qr = Q.Apply(r);
    const double Qr2 = glm::dot(r, Qr); // Q_ab * r_a * r_b

    f.accel += G * Qr / r5 - 2.5 * G * Qr2 * r / r7;

    f.hessian.xx += -G * Q.xx / r5 + 10.0 * G * Qr.x * r.x / r7 + 2.5 * G * Qr2 / r7 - 17.5 * G * Qr2 * r.x * r.x / r9;
    f.hessian.yy += -G * Q.yy / r5 + 10.0 * G * Qr.y * r.y / r7 + 2.5 * G * Qr2 / r7 - 17.5 * G * Qr2 * r.y * r.y / r9;
    f.hessian.zz += -G * Q.zz / r5 + 10.0 * G * Qr.z * r.z / r7 + 2.5 * G * Qr2 / r7 - 17.5 * G * Qr2 * r.z * r.z / r9;
    f.hessian.xy += -G * Q.xy / r5 + 5.0 * G * (Qr.x * r.y + Qr.y * r.x) / r7 - 17.5 * G * Qr2 * r.x * r.y / r9;
    f.hessian.xz += -G * Q.xz / r5 + 5.0 * G * (Qr.x * r.z + Qr.z * r.x) / r7 - 17.5 * G * Qr2 * r.x * r.z / r9;
    f.hessian.yz += -G * Q.yz / r5 + 5.0 * G * (Qr.y * r.z + Qr.z * r.y) / r7 - 17.5 * G * Qr2 * r.y * r.z / r9;

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
    std::vector<std::pair<int, int>> nearPairs;
    nearPairs.reserve(static_cast<size_t>(n) * 8);

    const double theta2 = theta * theta;
    // The opening-angle test alone is scale-invariant, so a pathologically
    // clustered distribution can recurse the tree down to separations many
    // orders of magnitude below the softening length, where it's still
    // "satisfied" in a relative sense -- but the *unsoftened* multipole
    // field's high-order (1/r^7, 1/r^9) terms are numerically catastrophic
    // there. An absolute floor tied to the softening length -- below which
    // softening already regularizes the near-field formula anyway, so
    // nothing physical is lost by refusing the multipole shortcut -- routes
    // such pairs to the near-field path instead. (nbody.py found this via
    // an empirical dist^2 ~ 1e-18 pair producing a ~1e14 spurious
    // acceleration before this guard existed.)
    const double minSep2 = (10.0 * softening) * (10.0 * softening);

    // Dual-tree traversal: pairs of nodes from the same tree, one target
    // (accumulates a local expansion) one source (contributes a multipole
    // field), splitting whichever side is coarser until well-separated
    // (M2L) or both leaves (near-field pair).
    std::vector<std::pair<int, int>> stack;
    stack.reserve(static_cast<size_t>(numNodes) * 2);
    stack.emplace_back(0, 0);

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
            // each child's own self-pair.
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
        const bool wellSeparated = (sizeSum * sizeSum < theta2 * dist2) && (dist2 > minSep2);

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
            // particle index first) avoids double-counting the force.
            const int pt = nodeT.particles[0];
            const int ps = nodeS.particles[0];
            if (pt < ps) nearPairs.emplace_back(pt, ps);
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
    for (const auto& [pi, pj] : nearPairs) {
        const glm::dvec3 d = pos[static_cast<size_t>(pj)] - pos[static_cast<size_t>(pi)];
        const double dist2 = glm::dot(d, d) + eps2;
        const double invDist = 1.0 / std::sqrt(dist2);
        const double invDist3 = invDist * invDist * invDist;
        accelOut[static_cast<size_t>(pi)] += G * mass[static_cast<size_t>(pj)] * invDist3 * d;
        accelOut[static_cast<size_t>(pj)] -= G * mass[static_cast<size_t>(pi)] * invDist3 * d;
    }
}

} // namespace nbody
