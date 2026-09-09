#include "ngrav/Solvers.hpp"

#include "ngrav/Green.hpp"

#include <cmath>
#include <vector>

namespace ngrav {

namespace {

// Plummer-softened acceleration from a source of (G-premultiplied) mass
// `gmj` at displacement (dx,dy,dz), accumulated into (axi,ayi,azi). `gmj` is
// G*m so the per-term product matches the old code's `G * mass * invDist3 * d`
// accumulation exactly (factoring G out of the sum would only reassociate at
// ~1e-16, but the "reproduce old diagnostics" gate is cheapest to hit with a
// term-for-term match). This is the hot path (default softening kind); the
// spline path below is a separate, slightly heavier branch.
template <int D>
inline void AccumPairPlummer(double dx, double dy, double dz, double gmj, double eps2, double& axi, double& ayi,
                             double& azi) {
    if constexpr (D == 3) {
        const double dist2 = dx * dx + dy * dy + dz * dz + eps2;
        const double invDist = 1.0 / std::sqrt(dist2);
        const double invDist3 = invDist * invDist * invDist;
        const double s = gmj * invDist3;
        axi += s * dx;
        ayi += s * dy;
        azi += s * dz;
    } else {
        const double dist2 = dx * dx + dy * dy + eps2;
        const double s = gmj / dist2;
        axi += s * dx;
        ayi += s * dy;
        (void)dz;
        (void)azi;
    }
}

// Compact-support cubic-spline-softened acceleration (Hernquist & Katz 1989):
// exactly Newtonian for r >= 2h, a finite, smooth correction inside. `h` is
// the softening length (Softening::eps); q = r/h.
template <int D>
inline void AccumPairSpline(double dx, double dy, double dz, double gmj, double h, double& axi, double& ayi,
                            double& azi) {
    const double r2 = dx * dx + dy * dy + ((D == 3) ? dz * dz : 0.0);
    const double r = std::sqrt(r2);
    if (r <= 1e-300) return; // coincident points contribute nothing (matches the Plummer limit at eps->0)
    const double q = r / h;
    double s;
    if constexpr (D == 3) {
        const double chi = detail::SplineChi3D(q);
        s = gmj * chi / (r2 * r); // == gmj*chi/r^2, packaged as a d-vector coefficient
    } else {
        const double chi = detail::SplineChi2D(q);
        s = gmj * chi / r2; // a = gmj*chi/r, coefficient on d gives that magnitude along d/r... see below
    }
    axi += s * dx;
    ayi += s * dy;
    if constexpr (D == 3) azi += s * dz;
    (void)dz;
    (void)azi;
}

template <int D>
inline void AccumPair(double dx, double dy, double dz, double gmj, const Softening& soft, double& axi, double& ayi,
                      double& azi) {
    if (soft.kind == SofteningKind::Plummer) {
        AccumPairPlummer<D>(dx, dy, dz, gmj, soft.eps2, axi, ayi, azi);
    } else {
        AccumPairSpline<D>(dx, dy, dz, gmj, soft.eps, axi, ayi, azi);
    }
}

} // namespace

template <int D>
void ComputeAccelDirect(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    const Softening soft = sp.soft;
    const double G = sp.G;

    const double* x = pts.x.data();
    const double* y = pts.y.data();
    const double* z = (D == 3) ? pts.z.data() : nullptr;
    const double* m = pts.m.data();

#pragma omp parallel for schedule(static) if (n > 256)
    for (int i = 0; i < n; ++i) {
        const double xi = x[i];
        const double yi = y[i];
        const double zi = (D == 3) ? z[i] : 0.0;
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = x[j] - xi;
            const double dy = y[j] - yi;
            const double dz = (D == 3) ? (z[j] - zi) : 0.0;
            AccumPair<D>(dx, dy, dz, G * m[j], soft, axi, ayi, azi);
        }
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

template <int D>
void ComputeAccelDirectTargets(const PosMassView<D>& pts, const StepParams& sp, std::span<const int> targets,
                               SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    const int nt = static_cast<int>(targets.size());
    const Softening soft = sp.soft;
    const double G = sp.G;
    const double* x = pts.x.data();
    const double* y = pts.y.data();
    const double* z = (D == 3) ? pts.z.data() : nullptr;
    const double* m = pts.m.data();

#pragma omp parallel for schedule(static) if (nt > 64)
    for (int t = 0; t < nt; ++t) {
        const int i = targets[static_cast<std::size_t>(t)];
        const double xi = x[i], yi = y[i], zi = (D == 3) ? z[i] : 0.0;
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const double dx = x[j] - xi, dy = y[j] - yi, dz = (D == 3) ? (z[j] - zi) : 0.0;
            AccumPair<D>(dx, dy, dz, G * m[j], soft, axi, ayi, azi);
        }
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

namespace {

// Traceless-quadrupole correction to a well-separated node's monopole field,
// evaluated at displacement d = node.com - target (our AccumPair convention:
// attractive acceleration points along +d). Independently re-derived from
// phi(x) = -G[M/r + Q_ij x_i x_j / (2r^5)] with x = target - source = -d
// (standard multipole expansion, e.g. Binney & Tremaine eq. 2.132): with
// Qd = Q.d and Qd2 = d.Qd,
//   a_quad = -G*Qd/r^5 + 2.5*G*Qd2*d/r^7
// 3D only (2D's higher-accuracy path is the complex-Laurent FMM, which
// already carries arbitrary order -- see Scenarios/ComplexFmm).
template <int D>
inline void AddQuadrupole(const AdaptiveTree<D>& tree, std::size_t nd, double dx, double dy, double dz, double dist2,
                         double G, double& axi, double& ayi, double& azi) {
    if constexpr (D == 3) {
        if (!tree.HasQuadrupoles()) return;
        const double Qxx = tree.Qxx[nd], Qyy = tree.Qyy[nd], Qzz = -(Qxx + Qyy);
        const double Qxy = tree.Qxy[nd], Qxz = tree.Qxz[nd], Qyz = tree.Qyz[nd];
        const double Qdx = Qxx * dx + Qxy * dy + Qxz * dz;
        const double Qdy = Qxy * dx + Qyy * dy + Qyz * dz;
        const double Qdz = Qxz * dx + Qyz * dy + Qzz * dz;
        const double Qd2 = dx * Qdx + dy * Qdy + dz * Qdz;
        const double invR2 = 1.0 / dist2;
        const double invR = std::sqrt(invR2);
        const double invR5 = invR2 * invR2 * invR;
        const double invR7 = invR5 * invR2;
        const double c1 = -G * invR5;
        const double c2 = 2.5 * G * Qd2 * invR7;
        axi += c1 * Qdx + c2 * dx;
        ayi += c1 * Qdy + c2 * dy;
        azi += c1 * Qdz + c2 * dz;
    } else {
        (void)tree;
        (void)nd;
        (void)dx;
        (void)dy;
        (void)dz;
        (void)dist2;
        (void)G;
        (void)axi;
        (void)ayi;
        (void)azi;
    }
}

template <int D>
void WalkBH(const AdaptiveTree<D>& tree, const PosMassView<D>& pts, int node, int i, double xi, double yi, double zi,
           double G, double theta2, const Softening& soft, MacKind macKind, double alpha, double aOldI, double& axi,
           double& ayi, double& azi) {
    const std::size_t nd = static_cast<std::size_t>(node);
    if (tree.mass[nd] <= 0.0) return;

    if (tree.childCount[nd] == 0) {
        const int lo = tree.firstParticle[nd];
        const int hi = lo + tree.particleCount[nd];
        for (int k = lo; k < hi; ++k) {
            const int p = tree.order[static_cast<std::size_t>(k)];
            if (p == i) continue;
            const double dx = pts.x[static_cast<std::size_t>(p)] - xi;
            const double dy = pts.y[static_cast<std::size_t>(p)] - yi;
            const double dz = (D == 3) ? (pts.z[static_cast<std::size_t>(p)] - zi) : 0.0;
            AccumPair<D>(dx, dy, dz, G * pts.m[static_cast<std::size_t>(p)], soft, axi, ayi, azi);
        }
        return;
    }

    const Vec<D> comN = tree.com[nd];
    const double dx = comN.x - xi;
    const double dy = comN.y - yi;
    const double dz = (D == 3) ? (Zc(comN) - zi) : 0.0;
    const double dist2 = dx * dx + dy * dy + dz * dz;
    const double size = 2.0 * tree.halfSize[nd];

    bool accept = false;
    if (macKind == MacKind::Relative && aOldI > 0.0) {
        accept = (tree.mass[nd] * size * size) < (alpha * aOldI * dist2 * dist2 / G);
    } else {
        accept = (size * size) < (theta2 * dist2);
    }

    if (accept) {
        AccumPair<D>(dx, dy, dz, G * tree.mass[nd], soft, axi, ayi, azi);
        AddQuadrupole<D>(tree, nd, dx, dy, dz, dist2, G, axi, ayi, azi);
        return;
    }

    for (int c = 0; c < AdaptiveTree<D>::kNC; ++c) {
        const int ch = tree.children[nd][static_cast<std::size_t>(c)];
        if (ch != -1)
            WalkBH<D>(tree, pts, ch, i, xi, yi, zi, G, theta2, soft, macKind, alpha, aOldI, axi, ayi, azi);
    }
}

// Phase 8: grouped/cell walk for the (default, no-per-particle-state)
// Geometric MAC -- one tree descent per *leaf* (ncrit particles share it)
// instead of one descent per particle, using a conservative group MAC so
// accuracy can only be equal-or-better than the per-particle walk it
// replaces (never worse): a leaf's `com`/`maxRadius` bound every member's
// possible offset from the leaf's own center, so testing the *worst-case*
// distance (dist(groupCenter,nodeCom) - groupRadius, by the triangle
// inequality a lower bound on every member's true distance to that node)
// against `theta` accepts a node only when EVERY member would also accept
// it individually -- the group test can reject something the exact
// per-particle test would accept (forcing an extra split there), but can
// never accept something the per-particle test would reject.
//
// The Relative/acceleration MAC (Springel-2005-style) needs each particle's
// own previous-step |a|, which has no shared group value -- it keeps using
// the per-particle `WalkBH` below rather than a second, more speculative
// group-MAC formulation for it (a deliberate Phase-8 scope decision, same
// spirit as Phase 7's descopes: the Geometric MAC is the common case and
// the one this phase's gate targets).
template <int D>
void CollectGroupInteractions(const AdaptiveTree<D>& tree, int node, int group, const Vec<D>& groupCenter,
                              double groupRadius, double theta2, std::vector<int>& farNodes,
                              std::vector<int>& nearLeaves) {
    const std::size_t nd = static_cast<std::size_t>(node);
    if (node == group) return; // the group's own members are handled separately (exact, once)
    if (tree.mass[nd] <= 0.0) return;

    if (tree.IsLeaf(node)) {
        nearLeaves.push_back(node); // matches WalkBH: leaves are always exact, never a MAC monopole
        return;
    }

    const Vec<D> comN = tree.com[nd];
    const double dx = comN.x - groupCenter.x;
    const double dy = comN.y - groupCenter.y;
    const double dz = (D == 3) ? (Zc(comN) - Zc(groupCenter)) : 0.0;
    const double dist2 = dx * dx + dy * dy + dz * dz;
    const double size = 2.0 * tree.halfSize[nd];
    const double worst = std::sqrt(dist2) - groupRadius; // conservative lower bound on any member's true distance

    if (worst > 0.0 && size * size < theta2 * worst * worst) {
        farNodes.push_back(node);
        return;
    }
    for (int c = 0; c < AdaptiveTree<D>::kNC; ++c) {
        const int ch = tree.children[nd][static_cast<std::size_t>(c)];
        if (ch != -1) CollectGroupInteractions<D>(tree, ch, group, groupCenter, groupRadius, theta2, farNodes, nearLeaves);
    }
}

template <int D>
void ComputeAccelBarnesHutGroup(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                                SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    const Softening soft = sp.soft;
    const double G = sp.G;
    const double theta2 = sp.mac.theta * sp.mac.theta;

    std::vector<int> leaves;
    leaves.reserve(static_cast<std::size_t>(tree.NumNodes()) / 4 + 1);
    for (int idx = 0; idx < tree.NumNodes(); ++idx) {
        if (tree.IsLeaf(idx) && tree.particleCount[static_cast<std::size_t>(idx)] > 0) leaves.push_back(idx);
    }
    const int numLeaves = static_cast<int>(leaves.size());

#pragma omp parallel for schedule(dynamic, 32) if (n > 256)
    for (int li = 0; li < numLeaves; ++li) {
        const int g = leaves[static_cast<std::size_t>(li)];
        const std::size_t gd = static_cast<std::size_t>(g);

        // thread_local: capacity is reused call-to-call on the same OS
        // thread instead of a fresh heap alloc per leaf (there are N/ncrit
        // of these, ~1.25e4 at N=1e5) -- cleared, not reallocated, below.
        static thread_local std::vector<int> farNodes, nearLeaves;
        farNodes.clear();
        nearLeaves.clear();
        CollectGroupInteractions<D>(tree, tree.Root(), g, tree.com[gd], tree.maxRadius[gd], theta2, farNodes,
                                    nearLeaves);

        const int lo = tree.firstParticle[gd];
        const int hi = lo + tree.particleCount[gd];
        for (int k = lo; k < hi; ++k) {
            const int i = tree.order[static_cast<std::size_t>(k)];
            const double xi = pts.x[static_cast<std::size_t>(i)];
            const double yi = pts.y[static_cast<std::size_t>(i)];
            const double zi = (D == 3) ? pts.z[static_cast<std::size_t>(i)] : 0.0;
            double axi = 0.0, ayi = 0.0, azi = 0.0;

            for (int nd : farNodes) {
                const std::size_t ndd = static_cast<std::size_t>(nd);
                const Vec<D> comN = tree.com[ndd];
                const double dx = comN.x - xi;
                const double dy = comN.y - yi;
                const double dz = (D == 3) ? (Zc(comN) - zi) : 0.0;
                const double dist2 = dx * dx + dy * dy + dz * dz;
                AccumPair<D>(dx, dy, dz, G * tree.mass[ndd], soft, axi, ayi, azi);
                AddQuadrupole<D>(tree, ndd, dx, dy, dz, dist2, G, axi, ayi, azi);
            }
            for (int leaf : nearLeaves) {
                const std::size_t ld = static_cast<std::size_t>(leaf);
                const int lo2 = tree.firstParticle[ld];
                const int hi2 = lo2 + tree.particleCount[ld];
                for (int k2 = lo2; k2 < hi2; ++k2) {
                    const int p = tree.order[static_cast<std::size_t>(k2)];
                    const double dx = pts.x[static_cast<std::size_t>(p)] - xi;
                    const double dy = pts.y[static_cast<std::size_t>(p)] - yi;
                    const double dz = (D == 3) ? (pts.z[static_cast<std::size_t>(p)] - zi) : 0.0;
                    AccumPair<D>(dx, dy, dz, G * pts.m[static_cast<std::size_t>(p)], soft, axi, ayi, azi);
                }
            }
            // The group's own members (excluding self) -- exact, once per pair.
            for (int k2 = lo; k2 < hi; ++k2) {
                if (k2 == k) continue;
                const int p = tree.order[static_cast<std::size_t>(k2)];
                const double dx = pts.x[static_cast<std::size_t>(p)] - xi;
                const double dy = pts.y[static_cast<std::size_t>(p)] - yi;
                const double dz = (D == 3) ? (pts.z[static_cast<std::size_t>(p)] - zi) : 0.0;
                AccumPair<D>(dx, dy, dz, G * pts.m[static_cast<std::size_t>(p)], soft, axi, ayi, azi);
            }

            out.ax[static_cast<std::size_t>(i)] = axi;
            out.ay[static_cast<std::size_t>(i)] = ayi;
            if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
        }
    }
}

} // namespace

template <int D>
void ComputeAccelBarnesHutPerParticle(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                                      std::span<const double> aOld, SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    const Softening soft = sp.soft;
    const double G = sp.G;
    const double theta2 = sp.mac.theta * sp.mac.theta;
    const bool haveOld = (aOld.size() == static_cast<std::size_t>(n));

#pragma omp parallel for schedule(dynamic, 64) if (n > 256)
    for (int i = 0; i < n; ++i) {
        const double xi = pts.x[static_cast<std::size_t>(i)];
        const double yi = pts.y[static_cast<std::size_t>(i)];
        const double zi = (D == 3) ? pts.z[static_cast<std::size_t>(i)] : 0.0;
        const double aOldI = haveOld ? aOld[static_cast<std::size_t>(i)] : 0.0;
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        WalkBH<D>(tree, pts, tree.Root(), i, xi, yi, zi, G, theta2, soft, sp.mac.kind, sp.mac.alpha, aOldI, axi, ayi,
                  azi);
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

template <int D>
void ComputeAccelBarnesHutTargets(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                                  std::span<const int> targets, SoA<D>& out) {
    const int nt = static_cast<int>(targets.size());
    const Softening soft = sp.soft;
    const double G = sp.G;
    const double theta2 = sp.mac.theta * sp.mac.theta;
    // Always the geometric per-particle walk here: the relative MAC needs
    // each particle's previous |a|, which the block integrator doesn't track
    // per rung -- and geometric is the correct default for this path anyway.
    for (int t = 0; t < nt; ++t) {
        const int i = targets[static_cast<std::size_t>(t)];
        const double xi = pts.x[static_cast<std::size_t>(i)];
        const double yi = pts.y[static_cast<std::size_t>(i)];
        const double zi = (D == 3) ? pts.z[static_cast<std::size_t>(i)] : 0.0;
        double axi = 0.0, ayi = 0.0, azi = 0.0;
        WalkBH<D>(tree, pts, tree.Root(), i, xi, yi, zi, G, theta2, soft, MacKind::Geometric, 0.0, 0.0, axi, ayi, azi);
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                           std::span<const double> aOld, SoA<D>& out) {
    if (sp.mac.kind == MacKind::Geometric) {
        ComputeAccelBarnesHutGroup<D>(pts, sp, tree, out);
    } else {
        ComputeAccelBarnesHutPerParticle<D>(pts, sp, tree, aOld, out);
    }
}

template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out) {
    AdaptiveTree<D> tree(pts);
    if constexpr (D == 3) tree.ComputeQuadrupoles(pts);
    ComputeAccelBarnesHut<D>(pts, sp, tree, {}, out);
}

template void ComputeAccelDirect<2>(const PosMassView<2>&, const StepParams&, SoA<2>&);
template void ComputeAccelDirect<3>(const PosMassView<3>&, const StepParams&, SoA<3>&);
template void ComputeAccelBarnesHutPerParticle<2>(const PosMassView<2>&, const StepParams&, const AdaptiveTree<2>&,
                                                  std::span<const double>, SoA<2>&);
template void ComputeAccelBarnesHutPerParticle<3>(const PosMassView<3>&, const StepParams&, const AdaptiveTree<3>&,
                                                  std::span<const double>, SoA<3>&);
template void ComputeAccelBarnesHut<2>(const PosMassView<2>&, const StepParams&, const AdaptiveTree<2>&,
                                       std::span<const double>, SoA<2>&);
template void ComputeAccelBarnesHut<3>(const PosMassView<3>&, const StepParams&, const AdaptiveTree<3>&,
                                       std::span<const double>, SoA<3>&);
template void ComputeAccelBarnesHut<2>(const PosMassView<2>&, const StepParams&, SoA<2>&);
template void ComputeAccelBarnesHut<3>(const PosMassView<3>&, const StepParams&, SoA<3>&);
template void ComputeAccelDirectTargets<2>(const PosMassView<2>&, const StepParams&, std::span<const int>, SoA<2>&);
template void ComputeAccelDirectTargets<3>(const PosMassView<3>&, const StepParams&, std::span<const int>, SoA<3>&);
template void ComputeAccelBarnesHutTargets<2>(const PosMassView<2>&, const StepParams&, const AdaptiveTree<2>&,
                                              std::span<const int>, SoA<2>&);
template void ComputeAccelBarnesHutTargets<3>(const PosMassView<3>&, const StepParams&, const AdaptiveTree<3>&,
                                              std::span<const int>, SoA<3>&);

} // namespace ngrav
