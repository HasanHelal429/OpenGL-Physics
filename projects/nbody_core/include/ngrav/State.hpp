#pragma once

#include "Vec.hpp"

#include <algorithm>
#include <cstddef>
#include <span>
#include <vector>

// Structure-of-arrays particle state: parallel std::vector<double> for every
// component, so the direct sum, the near-field kernel and the tree walks can
// stream contiguous coordinates and the compiler can vectorise the j-loop.
// (The old code claimed "SoA" but actually stored std::vector<glm::dvec3>.)
namespace ngrav {

template <int D>
struct SoA {
    std::vector<double> x, y, z;    // z unused for D == 2 (kept empty)
    std::vector<double> vx, vy, vz; // "
    std::vector<double> m;
    std::vector<double> ax, ay, az; // acceleration scratch, filled by the solvers

    std::size_t Count() const { return x.size(); }

    void Resize(std::size_t n) {
        x.assign(n, 0.0);
        y.assign(n, 0.0);
        vx.assign(n, 0.0);
        vy.assign(n, 0.0);
        m.assign(n, 0.0);
        ax.assign(n, 0.0);
        ay.assign(n, 0.0);
        if constexpr (D == 3) {
            z.assign(n, 0.0);
            vz.assign(n, 0.0);
            az.assign(n, 0.0);
        }
    }

    // Size/zero only the acceleration arrays -- lets a solver write into a
    // SoA that aliases its own input without clobbering positions.
    void ResizeAccel(std::size_t n) {
        ax.assign(n, 0.0);
        ay.assign(n, 0.0);
        if constexpr (D == 3) az.assign(n, 0.0);
    }

    void SetPos(std::size_t i, const Vec<D>& p) {
        x[i] = p.x;
        y[i] = p.y;
        if constexpr (D == 3) z[i] = p.z;
    }
    void SetVel(std::size_t i, const Vec<D>& v) {
        vx[i] = v.x;
        vy[i] = v.y;
        if constexpr (D == 3) vz[i] = v.z;
    }

    Vec<D> Pos(std::size_t i) const {
        if constexpr (D == 3)
            return Vec<3>(x[i], y[i], z[i]);
        else
            return Vec<2>(x[i], y[i]);
    }
    Vec<D> Vel(std::size_t i) const {
        if constexpr (D == 3)
            return Vec<3>(vx[i], vy[i], vz[i]);
        else
            return Vec<2>(vx[i], vy[i]);
    }
    Vec<D> Acc(std::size_t i) const {
        if constexpr (D == 3)
            return Vec<3>(ax[i], ay[i], az[i]);
        else
            return Vec<2>(ax[i], ay[i]);
    }
    void AddAcc(std::size_t i, const Vec<D>& a) {
        ax[i] += a.x;
        ay[i] += a.y;
        if constexpr (D == 3) az[i] += a.z;
    }
    void ZeroAcc() {
        std::fill(ax.begin(), ax.end(), 0.0);
        std::fill(ay.begin(), ay.end(), 0.0);
        if constexpr (D == 3) std::fill(az.begin(), az.end(), 0.0);
    }
};

// A read-only, non-owning view of just the position + mass arrays -- what the
// force solvers and the tree builder actually need.
template <int D>
struct PosMassView {
    std::span<const double> x, y, z, m;
    std::size_t Count() const { return x.size(); }
    Vec<D> Pos(std::size_t i) const {
        if constexpr (D == 3)
            return Vec<3>(x[i], y[i], z[i]);
        else
            return Vec<2>(x[i], y[i]);
    }
};

template <int D>
inline PosMassView<D> ViewOf(const SoA<D>& s) {
    if constexpr (D == 3)
        return PosMassView<3>{s.x, s.y, s.z, s.m};
    else
        return PosMassView<2>{s.x, s.y, {}, s.m};
}

} // namespace ngrav
