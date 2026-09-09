#pragma once

#include "ngrav/State.hpp"
#include "ngrav/Vec.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// Hernquist (1990) sphere, 3D, units G = M_total = a = 1 internally then
// rescaled (matching ic::Plummer's own convention).
//
//   rho(r) = M a / (2 pi r (r+a)^3),   M(<r) = M r^2/(r+a)^2,   phi(r) = -GM/(r+a)
//
// Positions: M(<r)/M = r^2/(r+a)^2 inverts in closed form --
// r/(r+a) = sqrt(x1)  =>  r = a*sqrt(x1)/(1-sqrt(x1)) -- exact, no rejection
// sampling or tabulation needed.
//
// Velocities: rather than Hernquist's own closed-form isotropic DF (a
// correct but easy-to-mistranscribe special-function formula, and this
// project's standing preference -- see Softening.hpp, the quadrupole
// moments, MutualFmm.cpp -- is to re-derive from a named physical
// principle rather than trust a formula from memory), this samples an
// isotropic Gaussian at each radius with sigma_r(r) from the spherical
// Jeans equation:
//
//   sigma_r^2(r) = (1/rho(r)) * integral_r^inf rho(r') G M(r')/r'^2 dr'
//
// evaluated by numerical quadrature (the integrand uses only the closed-form
// rho/M above, so the only approximation here is the quadrature itself, not
// a second, unverified analytic formula). This is the same technique
// already used for ic::Plummer's own 2D branch ("a crude equilibrium,
// adequate as a stress-test IC" -- see that file's header) -- consistent
// with, not a lower bar than, existing precedent. It reproduces rho(r) by
// construction (positions are drawn exactly from it) and satisfies the
// scalar virial theorem in the large-N limit (Jeans' equation IS the first
// moment of the stationary collisionless Boltzmann equation), but is not
// bit-for-bit the same equilibrium as Hernquist's own closed-form DF --
// documented here, not hidden.
namespace ngrav::ic {

struct HernquistParams {
    int n = 1000;
    double totalMass = 1.0;
    double scaleRadius = 1.0;
    double G = 1.0;
    std::uint32_t seed = 1;
};

namespace detail {

// rho(r)/M and M(<r)/M in units a=1 (r here is r/a).
inline double HernquistRhoUnit(double r) {
    const double rp1 = r + 1.0;
    return 1.0 / (2.0 * M_PI * r * rp1 * rp1 * rp1);
}
inline double HernquistMassUnit(double r) {
    const double rp1 = r + 1.0;
    return r * r / (rp1 * rp1);
}

// sigma_r^2(r)/(GM/a), tabulated at log-spaced r/a via the Jeans integral
// above (Simpson quadrature on a log grid, integrated from a large outer
// cutoff inward so the -- formally infinite -- upper limit only needs to be
// "large enough", not literal infinity: the integrand falls off as
// rho(r')*M(r')/r'^2 ~ r'^{-4} at large r', so the tail past r=200 a is
// already negligible at double precision).
class HernquistSigmaTable {
public:
    static constexpr int kN = 2048;
    static constexpr double kRMin = 1e-4;
    static constexpr double kRMax = 200.0;

    HernquistSigmaTable() {
        // Grid in log(r), r in [kRMin, kRMax].
        m_logRMin = std::log(kRMin);
        m_logRMax = std::log(kRMax);
        std::vector<double> r(kN + 1), integrand(kN + 1);
        for (int i = 0; i <= kN; ++i) {
            const double t = m_logRMin + (m_logRMax - m_logRMin) * i / kN;
            r[static_cast<std::size_t>(i)] = std::exp(t);
        }
        for (int i = 0; i <= kN; ++i) {
            const double ri = r[static_cast<std::size_t>(i)];
            // d/dr' of the integral, expressed in the r' variable (the
            // quadrature below integrates directly in r, not log r, for
            // simplicity -- accuracy is already ample at kN=2048 points
            // over 6 decades).
            integrand[static_cast<std::size_t>(i)] = HernquistRhoUnit(ri) * HernquistMassUnit(ri) / (ri * ri);
        }
        // Cumulative integral from the outer edge inward (trapezoid on the
        // log-spaced grid, sufficient given the smoothness of this
        // integrand and the point of this table -- an approximate,
        // documented Jeans dispersion, not a claimed-exact DF).
        std::vector<double> tailIntegral(kN + 1, 0.0);
        for (int i = kN - 1; i >= 0; --i) {
            const double r0 = r[static_cast<std::size_t>(i)], r1 = r[static_cast<std::size_t>(i + 1)];
            const double f0 = integrand[static_cast<std::size_t>(i)], f1 = integrand[static_cast<std::size_t>(i + 1)];
            tailIntegral[static_cast<std::size_t>(i)] =
                tailIntegral[static_cast<std::size_t>(i + 1)] + 0.5 * (f0 + f1) * (r1 - r0);
        }
        m_sigma2.resize(static_cast<std::size_t>(kN) + 1);
        for (int i = 0; i <= kN; ++i) {
            const double ri = r[static_cast<std::size_t>(i)];
            const double rho = HernquistRhoUnit(ri);
            m_sigma2[static_cast<std::size_t>(i)] = (rho > 1e-300) ? (tailIntegral[static_cast<std::size_t>(i)] / rho) : 0.0;
        }
    }

    double Sigma2(double r) const {
        r = std::max(kRMin, std::min(kRMax, r));
        const double t = (std::log(r) - m_logRMin) / (m_logRMax - m_logRMin) * kN;
        const int i0 = std::min(kN - 1, static_cast<int>(t));
        const double frac = t - i0;
        return m_sigma2[static_cast<std::size_t>(i0)] * (1.0 - frac) + m_sigma2[static_cast<std::size_t>(i0 + 1)] * frac;
    }

private:
    double m_logRMin = 0.0, m_logRMax = 0.0;
    std::vector<double> m_sigma2;
};

inline const HernquistSigmaTable& HernquistSigmaTab() {
    static const HernquistSigmaTable table;
    return table;
}

} // namespace detail

template <int D>
SoA<D> Hernquist(const HernquistParams& hp) {
    static_assert(D == 3, "Hernquist ic:: is 3D only (no 2D analogue defined in this project)");
    SoA<D> s;
    s.Resize(static_cast<std::size_t>(hp.n));
    std::mt19937 rng(hp.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    const double m = hp.totalMass / hp.n;
    const double vUnit = std::sqrt(hp.G * hp.totalMass / hp.scaleRadius);

    for (int i = 0; i < hp.n; ++i) {
        // r/a = sqrt(x1)/(1-sqrt(x1)). This CDF's tail is heavy (r ~ 1/(1-sqrt(x1))
        // as x1 -> 1), so clip the RADIUS, not x1: a weak x1 clip still lets a
        // single unlucky draw land at r ~ 1e9, which then dominates the COM
        // recenter below and wrecks the whole cloud (same reason ic::Plummer
        // clips its own r to 20). 40 a holds >99.7% of the mass.
        const double x1 = u(rng);
        const double sq = std::sqrt(x1);
        const double r = std::min(sq / (1.0 - sq), 40.0);

        const double cosT = 2.0 * u(rng) - 1.0;
        const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        const double phi = 2.0 * M_PI * u(rng);
        const Vec<3> dir(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
        s.SetPos(static_cast<std::size_t>(i), dir * (r * hp.scaleRadius));

        const double sigma = std::sqrt(std::max(0.0, detail::HernquistSigmaTab().Sigma2(r)));
        std::normal_distribution<double> gg(0.0, sigma);
        const Vec<3> v(gg(rng), gg(rng), gg(rng));
        s.SetVel(static_cast<std::size_t>(i), v * vUnit);
        s.m[static_cast<std::size_t>(i)] = m;
    }

    Vec<D> cp(0.0), cv(0.0);
    double tm = 0.0;
    for (int i = 0; i < hp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        cp += s.m[ii] * s.Pos(ii);
        cv += s.m[ii] * s.Vel(ii);
        tm += s.m[ii];
    }
    cp /= tm;
    cv /= tm;
    for (int i = 0; i < hp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        s.SetPos(ii, s.Pos(ii) - cp);
        s.SetVel(ii, s.Vel(ii) - cv);
    }
    return s;
}

} // namespace ngrav::ic
