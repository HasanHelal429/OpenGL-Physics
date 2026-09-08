#include "ngrav/Solvers.hpp"

#include "ngrav/Green.hpp"

#include <cmath>
#include <vector>

namespace ngrav {

namespace {

// Softened acceleration contribution from a source of (G-premultiplied) mass
// `gmj` at displacement (dx,dy,dz), accumulated into (axi,ayi,azi). `gmj` is
// G*m so the per-term product matches the old code's `G * mass * invDist3 * d`
// accumulation exactly (factoring G out of the sum would only reassociate at
// ~1e-16, but the "reproduce old diagnostics" gate is cheapest to hit with a
// term-for-term match).
template <int D>
inline void AccumPair(double dx, double dy, double dz, double gmj, double eps2, double& axi, double& ayi, double& azi) {
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

} // namespace

template <int D>
void ComputeAccelDirect(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    const double eps2 = sp.soft.eps2;
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
            AccumPair<D>(dx, dy, dz, G * m[j], eps2, axi, ayi, azi);
        }
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

namespace {

template <int D>
void WalkBH(const AdaptiveTree<D>& tree, const PosMassView<D>& pts, int node, int i, double xi, double yi, double zi,
           double G, double theta2, double eps2, MacKind macKind, double alpha, double aOldI, double& axi, double& ayi,
           double& azi) {
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
            AccumPair<D>(dx, dy, dz, G * pts.m[static_cast<std::size_t>(p)], eps2, axi, ayi, azi);
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
        AccumPair<D>(dx, dy, dz, G * tree.mass[nd], eps2, axi, ayi, azi);
        return;
    }

    for (int c = 0; c < AdaptiveTree<D>::kNC; ++c) {
        const int ch = tree.children[nd][static_cast<std::size_t>(c)];
        if (ch != -1)
            WalkBH<D>(tree, pts, ch, i, xi, yi, zi, G, theta2, eps2, macKind, alpha, aOldI, axi, ayi, azi);
    }
}

} // namespace

template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                           std::span<const double> aOld, SoA<D>& out) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    const double eps2 = sp.soft.eps2;
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
        WalkBH<D>(tree, pts, tree.Root(), i, xi, yi, zi, G, theta2, eps2, sp.mac.kind, sp.mac.alpha, aOldI, axi, ayi,
                  azi);
        out.ax[static_cast<std::size_t>(i)] = axi;
        out.ay[static_cast<std::size_t>(i)] = ayi;
        if constexpr (D == 3) out.az[static_cast<std::size_t>(i)] = azi;
    }
}

template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out) {
    const AdaptiveTree<D> tree(pts);
    ComputeAccelBarnesHut<D>(pts, sp, tree, {}, out);
}

template void ComputeAccelDirect<2>(const PosMassView<2>&, const StepParams&, SoA<2>&);
template void ComputeAccelDirect<3>(const PosMassView<3>&, const StepParams&, SoA<3>&);
template void ComputeAccelBarnesHut<2>(const PosMassView<2>&, const StepParams&, const AdaptiveTree<2>&,
                                       std::span<const double>, SoA<2>&);
template void ComputeAccelBarnesHut<3>(const PosMassView<3>&, const StepParams&, const AdaptiveTree<3>&,
                                       std::span<const double>, SoA<3>&);
template void ComputeAccelBarnesHut<2>(const PosMassView<2>&, const StepParams&, SoA<2>&);
template void ComputeAccelBarnesHut<3>(const PosMassView<3>&, const StepParams&, SoA<3>&);

} // namespace ngrav
