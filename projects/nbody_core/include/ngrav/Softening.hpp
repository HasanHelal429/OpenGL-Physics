#pragma once

#include "Vec.hpp"

#include <array>
#include <cmath>

// Softening models.
//
// Plummer: a ~ m r / (r^2+eps^2)^{3/2} (3D) / m r/(r^2+eps^2) (2D). Simple,
// but biased at *every* radius -- the force is never exactly Newtonian, no
// matter how far apart the pair is.
//
// Spline (compact support): the point mass is convolved with the same
// normalized cubic B-spline (Monaghan & Lattanzio 1985) SPH density kernels
// use, giving a force that is *exactly* Newtonian for r >= H = 2h (h the
// smoothing length) and a smooth, finite-at-the-origin correction inside.
// This is the Hernquist & Katz (1989) gravitational softening kernel,
// standard in GADGET/NBODY-family codes. The 3D force+potential below are
// closed-form (independently re-derived here via the shell-theorem enclosed-
// mass integral of the same kernel SphKernelGlsl uses in 06_tidal_disruption,
// and cross-checked: central potential phi(0) = -1.4 G m/h matches the
// published value, and both branches agree at q=1 and reduce to exactly
// -Gm/r / -Gm/r^2 at q=2). The 2D force uses the same enclosed-mass idea
// with the 2D-normalized kernel (closed form); the 2D potential's outer
// (1<=q<2) branch has no elementary closed form once the log term enters,
// so it's built once from a precomputed quadrature table instead.
namespace ngrav {

enum class SofteningKind {
    Plummer,
    Spline,
};

namespace detail {

// chi(q) = M(<r)/m for the 3D cubic-spline kernel, q=r/h, support q in [0,2).
// Force factor: a(r) = G m chi(r/h) / r^2 (exactly Newtonian for q>=2).
inline double SplineChi3D(double q) {
    if (q >= 2.0) return 1.0;
    if (q < 1.0) return q * q * q * (4.0 / 3.0 - 1.2 * q * q + 0.5 * q * q * q);
    const double t = 2.0 - q;
    const double t2 = t * t, t4 = t2 * t2;
    return 1.0 - 4.0 * (t4 / 4.0 - t4 * t / 5.0 + t4 * t2 / 24.0);
}

// phi(q)*h/(G m) for the 3D kernel (see header derivation). Negative;
// -> -1/q for q>=2 (== -Gm/r).
inline double SplinePotential3D(double q) {
    if (q >= 2.0) return -1.0 / q;
    if (q < 1.0) return -(1.4 - (2.0 / 3.0) * q * q + 0.3 * q * q * q * q - 0.1 * q * q * q * q * q);
    const double t = 2.0 - q;
    const double t2 = t * t, t4 = t2 * t2;
    const double F = t4 / 4.0 - t4 * t / 5.0 + t4 * t2 / 24.0;
    return -((1.0 - 4.0 * F) / q + t4 / 2.0 - t4 * t / 5.0);
}

// chi_2D(q) = Sigma(<r)/m for the 2D-normalized cubic spline (sigma_2D =
// 10/(7 pi h^2)). Force factor: a(r) = G m chi_2D(r/h) / r.
inline double SplineChi2D(double q) {
    if (q >= 2.0) return 1.0;
    if (q < 1.0) return q * q * (10.0 / 7.0 - (15.0 / 14.0) * q * q + (3.0 / 7.0) * q * q * q);
    const double t = 2.0 - q;
    const double t4 = t * t * t * t;
    return 1.0 - (5.0 / 14.0) * t4 + (1.0 / 7.0) * t4 * t;
}

// 2D potential has no elementary closed form past q=1 (the enclosed-mass
// fraction's leading "1" term, divided by q, integrates to a log). But
// d(phi)/dq = chi_2D(q)/q is itself perfectly regular at q=0 (chi_2D ~ q^2,
// so the integrand ~ q, not 1/q) -- there is no singularity to cancel, only
// a closed form to give up on. Built once via Simpson quadrature, integrated
// inward from the exact boundary phi(2) = ln(2) (so phi_actual(r) =
// ln(h) + phi(q) matches ln(r) at r=2h, and is finite at r=0 since phi(q) is
// finite there -- the whole physical point of softening).
class Spline2DPotentialTable {
public:
    // Linear interpolation between samples flattens the true derivative into
    // a step function, so a finite-difference check of d(table)/dq shows
    // O(dq) error near small q where chi_2D(q)/q is itself small (relative
    // error inflates as the reference shrinks, in absolute terms it's tiny).
    // This only affects the 2D energy *diagnostic* in spline mode -- the
    // force used by the integrator is the exact closed-form SplineChi2D, not
    // this table. 4096 samples keeps the diagnostic's own error well under
    // the energy-conservation tolerances it's checked against.
    static constexpr int kN = 4096;
    Spline2DPotentialTable() {
        auto g = [](double q) { return (q <= 1e-12) ? 0.0 : SplineChi2D(q) / q; };
        m_val[kN] = std::log(2.0); // phi(2) -- the exact boundary value
        double acc = 0.0;
        const double dq = 2.0 / kN;
        for (int i = kN; i > 0; --i) {
            const double qa = i * dq, qb = (i - 1) * dq, qm = 0.5 * (qa + qb);
            acc += (dq / 6.0) * (g(qa) + 4.0 * g(qm) + g(qb)); // Simpson, per sub-interval
            m_val[i - 1] = m_val[kN] - acc; // phi(q_{i-1}) = phi(2) - integral_{q_{i-1}}^{2} g dq
        }
    }
    // Returns phi(q) (the h-independent part of the potential; caller adds
    // ln(h)) for q in [0,2]; ln(q) for q >= 2 (exact unsoftened match).
    double operator()(double q) const {
        if (q >= 2.0) return std::log(q);
        if (q <= 0.0) return m_val[0];
        const double x = q / 2.0 * kN;
        const int i0 = std::min(kN - 1, static_cast<int>(x));
        const double frac = x - i0;
        return m_val[i0] * (1.0 - frac) + m_val[i0 + 1] * frac;
    }

private:
    std::array<double, kN + 1> m_val{};
};

inline const Spline2DPotentialTable& Spline2DTable() {
    static const Spline2DPotentialTable table;
    return table;
}

} // namespace detail

struct Softening {
    SofteningKind kind = SofteningKind::Plummer;
    double eps = 0.05; // Plummer: the softening length. Spline: h (support radius H = 2h).
    double eps2 = 0.05 * 0.05;

    static Softening Plummer(double e) {
        Softening s;
        s.kind = SofteningKind::Plummer;
        s.eps = e;
        s.eps2 = e * e;
        return s;
    }
    static Softening Spline(double h) {
        Softening s;
        s.kind = SofteningKind::Spline;
        s.eps = h;
        s.eps2 = h * h;
        return s;
    }
};

} // namespace ngrav
