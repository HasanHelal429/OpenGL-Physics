#include "ngrav/MutualFmm.hpp"

#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

// Physical argument for mutuality (why this needs no new derivation beyond
// AddQuadrupole, already validated in Phase 5):
//
// Treat two well-separated clusters T (mass M_T, quadrupole Q_T, center
// c_T) and S (M_S, Q_S, c_S) as frozen rigid multipoles. Their interaction
// energy, to quadrupole order, is
//
//   U(r) = -G[ M_T M_S / r + M_T (Q_S : nn) / (2r^3) + M_S (Q_T : nn) / (2r^3) ]
//
// where r = c_T - c_S, n = r/|r|, and Q:nn = Q_ij n_i n_j. This depends on
// c_T and c_S *only* through r = c_T - c_S (Q_T, Q_S are fixed shape
// parameters, not functions of the centers) -- so by the chain rule,
//   F_T = -dU/dc_T = -U'(r),   F_S = -dU/dc_S = +U'(r) = -F_T
// automatically, for *any* U that depends only on the separation. Momentum
// conservation isn't a special property of this particular U; it's a
// property of the traversal applying ONE shared computation to both sides
// rather than evaluating each side's field independently (which is what
// the old one-directional AdaptiveFmm did, and why it didn't conserve).
//
// Differentiating U(r) termwise reproduces exactly AdaptiveFmm.cpp's
// (Phase-5-reused) monopole force G*M/r^2 and AddQuadrupole's field
// formula -- so the mutual update is just: apply AddQuadrupole's formula
// twice, once to each side, each with *its own* quadrupole moment and the
// displacement pointing at it (d for T-from-S, -d for S-from-T).
namespace ngrav {

namespace {

struct SymMat3 {
    double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
    glm::dvec3 Apply(const glm::dvec3& v) const {
        return {xx * v.x + xy * v.y + xz * v.z, xy * v.x + yy * v.y + yz * v.z, xz * v.x + yz * v.y + zz * v.z};
    }
    SymMat3& operator+=(const SymMat3& o) {
        xx += o.xx;
        yy += o.yy;
        zz += o.zz;
        xy += o.xy;
        xz += o.xz;
        yz += o.yz;
        return *this;
    }
};

SymMat3 NodeQuad(const AdaptiveTree<3>& tree, std::size_t nd) {
    SymMat3 q;
    q.xx = tree.Qxx[nd];
    q.yy = tree.Qyy[nd];
    q.zz = -q.xx - q.yy;
    q.xy = tree.Qxy[nd];
    q.xz = tree.Qxz[nd];
    q.yz = tree.Qyz[nd];
    return q;
}

// Just the quadrupole-shaped term (same formula as AddQuadrupole in
// Solvers.cpp / Phase 5, factored out because it's needed twice below: once
// as "the field T feels from S's moment" and once, mass-ratio-scaled, as
// the reaction term below).
glm::dvec3 QuadOnlyTerm(const SymMat3& Q, const glm::dvec3& d, double G) {
    const double r2 = glm::dot(d, d);
    const double invR2 = 1.0 / r2;
    const double invR = std::sqrt(invR2);
    const double invR5 = invR2 * invR2 * invR;
    const double invR7 = invR5 * invR2;
    const glm::dvec3 Qd = Q.Apply(d);
    const double Qd2 = glm::dot(d, Qd);
    return (-G * invR5) * Qd + (2.5 * G * Qd2 * invR7) * d;
}

// Monopole + quadrupole acceleration contribution at displacement
// d = source_center - field_point (our AccumPair convention: attractive
// acceleration points along +d), from a source of mass M and traceless
// quadrupole Q. Unsoftened -- only ever called on well-separated pairs.
glm::dvec3 FieldAt(double M, const SymMat3& Q, const glm::dvec3& d, double G) {
    const double r2 = glm::dot(d, d);
    const double invR2 = 1.0 / r2;
    const double invR = std::sqrt(invR2);
    const double invR3 = invR2 * invR;
    glm::dvec3 a = (G * M * invR3) * d; // monopole
    a += QuadOnlyTerm(Q, d, G);
    return a;
}

struct LocalExpansion {
    glm::dvec3 a0{0.0};
    SymMat3 H;
};

// Mono+quad field's Hessian at d (needed to L2L-shift a0 to a child center).
// Re-derived to match FieldAt's a = grad(-phi) convention: H_ij = d(a_i)/d(x_j)
// evaluated at the source's *own* location (monopole-only Hessian is exact
// there since it's what L2L needs -- the quadrupole's own Hessian
// contribution is higher order in the shift and dropped, same simplification
// AdaptiveFmm.cpp made).
SymMat3 MonopoleHessian(double M, const glm::dvec3& d, double G) {
    const double r2 = glm::dot(d, d);
    const double invR2 = 1.0 / r2;
    const double invR = std::sqrt(invR2);
    const double invR5 = invR2 * invR2 * invR;
    SymMat3 H;
    H.xx = G * M * (r2 - 3.0 * d.x * d.x) * invR5;
    H.yy = G * M * (r2 - 3.0 * d.y * d.y) * invR5;
    H.zz = G * M * (r2 - 3.0 * d.z * d.z) * invR5;
    H.xy = -3.0 * G * M * d.x * d.y * invR5;
    H.xz = -3.0 * G * M * d.x * d.z * invR5;
    H.yz = -3.0 * G * M * d.y * d.z * invR5;
    return H;
}

// theta^2 * dist2 lower-bounds sizeSum^2; a fixed absolute floor on dist2
// (5*softening) guards the unsoftened field's high-order terms from the
// same catastrophic-cancellation regime the ported nbody.py / AdaptiveFmm
// code documented (see 02's own AdaptiveFmm.cpp comments).
bool WellSeparated(const AdaptiveTree<3>& tree, std::size_t t, std::size_t s, double theta2, double minSep2) {
    const glm::dvec3 d = tree.com[t] - tree.com[s];
    const double dist2 = glm::dot(d, d);
    const double sizeSum = 2.0 * tree.halfSize[t] + 2.0 * tree.halfSize[s];
    return (dist2 > minSep2) && (sizeSum * sizeSum < theta2 * dist2);
}

std::vector<int> ChildrenOrSelf(const AdaptiveTree<3>& tree, int x) {
    std::vector<int> out;
    const std::size_t xd = static_cast<std::size_t>(x);
    if (tree.IsLeaf(x)) {
        out.push_back(x);
    } else {
        for (int c = 0; c < AdaptiveTree<3>::kNC; ++c) {
            const int ch = tree.children[xd][static_cast<std::size_t>(c)];
            if (ch != -1) out.push_back(ch);
        }
    }
    return out;
}

} // namespace

void ComputeAccelMutualFmm(const PosMassView<3>& pts, const StepParams& sp, SoA<3>& out, MutualFmmStats* stats) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    if (n == 0) return;

    const auto t0 = std::chrono::steady_clock::now();
    AdaptiveTree<3> tree(pts);
    tree.ComputeQuadrupoles(pts);
    const auto t1 = std::chrono::steady_clock::now();

    const int numNodes = tree.NumNodes();
    std::vector<LocalExpansion> local(static_cast<std::size_t>(numNodes));

    const double G = sp.G;
    const double theta2 = sp.mac.theta * sp.mac.theta;
    const double minSep2 = (5.0 * sp.soft.eps) * (5.0 * sp.soft.eps);

    std::vector<std::pair<int, int>> nearPairs;
    long m2lCount = 0;

    // Single-visit unordered dual-tree traversal: a self-pair (t,t) only
    // emits (child_i,child_i) [recurse] and (child_i,child_j) i<j [distinct
    // cross-pairs, visited exactly once] -- half the work of the old
    // "emit all ordered child pairs" traversal, and each distinct pair below
    // gets ONE mutual M2L/near-field decision instead of two one-directional
    // ones. t and s are always in disjoint subtrees once split off a
    // self-pair's cross-children (an invariant preserved by always splitting
    // the larger side, since s -- disjoint from all of t's ancestor -- stays
    // disjoint from each of t's own children too).
    std::vector<std::pair<int, int>> stack;
    stack.emplace_back(0, 0);
    while (!stack.empty()) {
        const auto [t, s] = stack.back();
        stack.pop_back();
        const std::size_t td = static_cast<std::size_t>(t), sd = static_cast<std::size_t>(s);
        if (tree.mass[td] <= 0.0 || tree.mass[sd] <= 0.0) continue;

        if (t == s) {
            if (tree.IsLeaf(t)) continue;
            const std::vector<int> children = ChildrenOrSelf(tree, t);
            for (std::size_t i = 0; i < children.size(); ++i) {
                stack.emplace_back(children[i], children[i]); // self-pair, recurse
                for (std::size_t j = i + 1; j < children.size(); ++j)
                    stack.emplace_back(children[i], children[j]); // distinct cross-pair, once
            }
            continue;
        }

        // NOTE: unlike the old one-directional AdaptiveFmm (tuned for a
        // 1-particle-per-leaf tree, where a >1-particle "degenerate" leaf
        // only ever occurred for pathologically near-duplicate positions
        // hitting the depth cap, and was safe to always aggregate), this
        // tree's leaves normally hold up to ncrit=8 particles as routine
        // behavior -- two adjacent 8-particle leaf boxes are NOT
        // automatically far enough apart to treat as point masses. So the
        // MAC applies uniformly regardless of how many particles a leaf
        // holds; a leaf that fails it and can't be split further (it's
        // already a leaf) falls through to the generalized near-field
        // branch below, which loops over each leaf's *actual* particle
        // range rather than assuming exactly one particle per side.
        const bool wellSep = WellSeparated(tree, td, sd, theta2, minSep2);

        if (wellSep) {
            // d = source - target, matching AccumPair's convention (attractive
            // acceleration points along +d): for T's own expansion the source
            // is S, so d_TS = comS - comT.
            const glm::dvec3 d = tree.com[sd] - tree.com[td];
            const SymMat3 Qs = NodeQuad(tree, sd), Qt = NodeQuad(tree, td);
            const double Mt = tree.mass[td], Ms = tree.mass[sd];
            // Momentum conservation to quadrupole order needs TWO terms per
            // side, not one: (i) the standard "cluster's mass responds to
            // the other's monopole+quadrupole field" (FieldAt), and (ii) a
            // reaction term -- T's OWN quadrupole moment couples to the
            // *gradient* of S's field across T's extent, exerting an extra
            // force on T that scales with (M_S/M_T). Dropping (ii) (as a
            // naive port of AddQuadrupole's one-directional formula would)
            // leaves a residual net force of exactly
            //   M_T*QuadOnlyTerm(Q_S,d) - M_S*QuadOnlyTerm(Q_T,d)
            // which is what a standalone momentum check first caught here
            // (residual ~1e-4, growing with theta/M2L usage, vs exact
            // machine-epsilon cancellation at theta=0 where only near-field
            // -- already exactly symmetric -- was exercised). Term (ii)
            // exactly cancels it: re-derived independently from
            // U(r) = -G[M_T M_S/r + M_T(Q_S:nn)/2r^3 + M_S(Q_T:nn)/2r^3]
            // (depends only on r = c_T-c_S, so F_T=-dU/dc_T is manifestly
            // antisymmetric with F_S=-dU/dc_S once *all* of U's
            // r-dependence -- including S's mass sitting in T's own
            // quadrupole field -- is accounted for, not just the "standard"
            // half).
            local[td].a0 += FieldAt(Ms, Qs, d, G) + (Ms / Mt) * QuadOnlyTerm(Qt, d, G);
            local[td].H += MonopoleHessian(Ms, d, G);
            local[sd].a0 += FieldAt(Mt, Qt, -d, G) + (Mt / Ms) * QuadOnlyTerm(Qs, -d, G);
            local[sd].H += MonopoleHessian(Mt, -d, G);
            ++m2lCount;
            continue;
        }

        if (tree.IsLeaf(t) && tree.IsLeaf(s)) {
            // Neither side can split further and they're not well-separated
            // -- exact near-field over every cross-pair between the two
            // leaves' member particles (each leaf may hold up to ncrit).
            const int loT = tree.firstParticle[td], hiT = loT + tree.particleCount[td];
            const int loS = tree.firstParticle[sd], hiS = loS + tree.particleCount[sd];
            for (int a = loT; a < hiT; ++a) {
                const int pt = tree.order[static_cast<std::size_t>(a)];
                for (int b = loS; b < hiS; ++b) {
                    const int ps = tree.order[static_cast<std::size_t>(b)];
                    if (pt < ps)
                        nearPairs.emplace_back(pt, ps);
                    else
                        nearPairs.emplace_back(ps, pt);
                }
            }
            continue;
        }

        if (tree.IsLeaf(t)) {
            for (int c : ChildrenOrSelf(tree, s)) stack.emplace_back(t, c);
        } else if (tree.IsLeaf(s)) {
            for (int c : ChildrenOrSelf(tree, t)) stack.emplace_back(c, s);
        } else if (tree.halfSize[td] >= tree.halfSize[sd]) {
            for (int c : ChildrenOrSelf(tree, t)) stack.emplace_back(c, s);
        } else {
            for (int c : ChildrenOrSelf(tree, s)) stack.emplace_back(t, c);
        }
    }
    const auto t2 = std::chrono::steady_clock::now();

    // L2L: push each node's accumulated local expansion down to its
    // children. Increasing node index is parent-before-child (an
    // AdaptiveTree invariant), so one sweep needs no recursion.
    for (int idx = 1; idx < numNodes; ++idx) {
        const std::size_t nd = static_cast<std::size_t>(idx);
        const int p = tree.parent[nd];
        if (p == -1) continue;
        const std::size_t pd = static_cast<std::size_t>(p);
        const glm::dvec3 d = tree.com[nd] - tree.com[pd];
        local[nd].a0 += local[pd].a0 - local[pd].H.Apply(d);
        local[nd].H += local[pd].H;
    }
    const auto t3 = std::chrono::steady_clock::now();

    // L2P: evaluate each leaf's finalized local expansion at its member
    // particle(s); degenerate multi-particle leaves also get an exact
    // brute-force sum for their own mutual interactions.
    const double eps2 = sp.soft.eps2;
    for (int idx = 0; idx < numNodes; ++idx) {
        const std::size_t nd = static_cast<std::size_t>(idx);
        if (!tree.IsLeaf(idx) || tree.particleCount[nd] == 0) continue;
        const int lo = tree.firstParticle[nd], hi = lo + tree.particleCount[nd];
        for (int k = lo; k < hi; ++k) {
            const int p = tree.order[static_cast<std::size_t>(k)];
            out.ax[static_cast<std::size_t>(p)] += local[nd].a0.x;
            out.ay[static_cast<std::size_t>(p)] += local[nd].a0.y;
            out.az[static_cast<std::size_t>(p)] += local[nd].a0.z;
        }
        if (hi - lo > 1) {
            for (int a = lo; a < hi; ++a) {
                for (int b = a + 1; b < hi; ++b) {
                    const int pa = tree.order[static_cast<std::size_t>(a)];
                    const int pb = tree.order[static_cast<std::size_t>(b)];
                    const double dx = pts.x[static_cast<std::size_t>(pb)] - pts.x[static_cast<std::size_t>(pa)];
                    const double dy = pts.y[static_cast<std::size_t>(pb)] - pts.y[static_cast<std::size_t>(pa)];
                    const double dz = pts.z[static_cast<std::size_t>(pb)] - pts.z[static_cast<std::size_t>(pa)];
                    const double dist2 = dx * dx + dy * dy + dz * dz + eps2;
                    const double invDist = 1.0 / std::sqrt(dist2);
                    const double invDist3 = invDist * invDist * invDist;
                    const double gma = G * pts.m[static_cast<std::size_t>(pa)];
                    const double gmb = G * pts.m[static_cast<std::size_t>(pb)];
                    out.ax[static_cast<std::size_t>(pa)] += gmb * invDist3 * dx;
                    out.ay[static_cast<std::size_t>(pa)] += gmb * invDist3 * dy;
                    out.az[static_cast<std::size_t>(pa)] += gmb * invDist3 * dz;
                    out.ax[static_cast<std::size_t>(pb)] -= gma * invDist3 * dx;
                    out.ay[static_cast<std::size_t>(pb)] -= gma * invDist3 * dy;
                    out.az[static_cast<std::size_t>(pb)] -= gma * invDist3 * dz;
                }
            }
        }
    }

    // Near-field pairs: exact softened summation, Newton's-third-law
    // symmetric (this is where momentum conservation would break if it
    // weren't -- unlike the M2L path, this has always been symmetric, even
    // in the old one-directional AdaptiveFmm).
    for (const auto& [pi, pj] : nearPairs) {
        const double dx = pts.x[static_cast<std::size_t>(pj)] - pts.x[static_cast<std::size_t>(pi)];
        const double dy = pts.y[static_cast<std::size_t>(pj)] - pts.y[static_cast<std::size_t>(pi)];
        const double dz = pts.z[static_cast<std::size_t>(pj)] - pts.z[static_cast<std::size_t>(pi)];
        const double dist2 = dx * dx + dy * dy + dz * dz + eps2;
        const double invDist = 1.0 / std::sqrt(dist2);
        const double invDist3 = invDist * invDist * invDist;
        const double gmi = G * pts.m[static_cast<std::size_t>(pi)];
        const double gmj = G * pts.m[static_cast<std::size_t>(pj)];
        out.ax[static_cast<std::size_t>(pi)] += gmj * invDist3 * dx;
        out.ay[static_cast<std::size_t>(pi)] += gmj * invDist3 * dy;
        out.az[static_cast<std::size_t>(pi)] += gmj * invDist3 * dz;
        out.ax[static_cast<std::size_t>(pj)] -= gmi * invDist3 * dx;
        out.ay[static_cast<std::size_t>(pj)] -= gmi * invDist3 * dy;
        out.az[static_cast<std::size_t>(pj)] -= gmi * invDist3 * dz;
    }
    const auto t4 = std::chrono::steady_clock::now();

    if (stats) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        stats->nodeCount = numNodes;
        stats->m2lPairs = m2lCount;
        stats->nearPairs = static_cast<long>(nearPairs.size());
        stats->buildMs = ms(t0, t1);
        stats->traverseMs = ms(t1, t2);
        stats->l2lMs = ms(t2, t3);
        stats->nearFieldMs = ms(t3, t4);
    }
}

} // namespace ngrav
