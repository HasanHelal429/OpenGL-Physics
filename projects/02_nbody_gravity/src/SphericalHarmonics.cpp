#include "SphericalHarmonics.hpp"

#include <cmath>

namespace nbody::sh {

namespace {

// Associated Legendre function P_n^m(cosTheta), n=0..p, m=0..n, via the
// standard stable recurrence (e.g. Numerical Recipes' plgndr) -- but with
// the usual leading minus sign at the P_m^m seed step dropped, i.e.
// *without* Condon-Shortley phase. Upsilon/Theta's own explicit (-1)^m
// factor (see this file's header) is what supplies the phase instead; the
// pairing is this module's own convention choice (not universal across the
// literature, which is exactly why the whole pipeline is validated end to
// end against brute-force summation in SphericalHarmonicsTest.cpp rather
// than trusted from this derivation alone).
struct LegendreTable {
    std::vector<double> p; // triangular layout, CoeffIndex(n,m), m=0..n

    LegendreTable(int order, double cosTheta, double sinTheta) : p(static_cast<size_t>(CoeffCount(order)), 0.0) {
        p[static_cast<size_t>(CoeffIndex(0, 0))] = 1.0;
        for (int m = 1; m <= order; ++m) {
            p[static_cast<size_t>(CoeffIndex(m, m))] =
                (2.0 * m - 1.0) * sinTheta * p[static_cast<size_t>(CoeffIndex(m - 1, m - 1))];
        }
        for (int m = 0; m < order; ++m) {
            p[static_cast<size_t>(CoeffIndex(m + 1, m))] = (2.0 * m + 1.0) * cosTheta * p[static_cast<size_t>(CoeffIndex(m, m))];
        }
        for (int m = 0; m <= order; ++m) {
            for (int n = m + 2; n <= order; ++n) {
                const double pnm1 = p[static_cast<size_t>(CoeffIndex(n - 1, m))];
                const double pnm2 = p[static_cast<size_t>(CoeffIndex(n - 2, m))];
                p[static_cast<size_t>(CoeffIndex(n, m))] = ((2.0 * n - 1.0) * cosTheta * pnm1 - (n + m - 1.0) * pnm2) / (n - m);
            }
        }
    }

    double operator()(int n, int m) const { return p[static_cast<size_t>(CoeffIndex(n, m))]; }
};

// Factorial table up to 2p (needed for (n+m)!/(n-m)! with n,m<=p) -- double
// is exact for integers this small (factorials up to ~170! before double
// overflow; this project's expansion orders are nowhere near that).
std::vector<double> FactorialTable(int maxN) {
    std::vector<double> f(static_cast<size_t>(maxN) + 1);
    f[0] = 1.0;
    for (int i = 1; i <= maxN; ++i) f[static_cast<size_t>(i)] = f[static_cast<size_t>(i - 1)] * i;
    return f;
}

// Shared setup for EvalUpsilon/EvalTheta: r's magnitude/direction, the
// Legendre table, and the e^{im*phi} table -- everything both solid
// harmonics need except the radial power and the (n+/-m)! normalization,
// which differ between them.
struct SphericalBasis {
    double r = 0.0;
    LegendreTable legendre;
    std::vector<Complex> cis; // e^{i*m*phi}, m=0..p

    SphericalBasis(const glm::dvec3& v, int p, double rMagnitude, double cosTheta, double sinTheta, double phi)
        : r(rMagnitude), legendre(p, cosTheta, sinTheta), cis(static_cast<size_t>(p) + 1) {
        (void)v;
        const Complex step{std::cos(phi), std::sin(phi)};
        cis[0] = Complex{1.0, 0.0};
        for (int m = 1; m <= p; ++m) cis[static_cast<size_t>(m)] = cis[static_cast<size_t>(m - 1)] * step;
    }
};

SphericalBasis MakeBasis(const glm::dvec3& r, int p) {
    const double rMag = glm::length(r);
    const double cosTheta = (rMag > 1e-300) ? (r.z / rMag) : 0.0;
    const double sinTheta = (rMag > 1e-300) ? (std::sqrt(r.x * r.x + r.y * r.y) / rMag) : 0.0;
    const double phi = std::atan2(r.y, r.x);
    return SphericalBasis(r, p, rMag, cosTheta, sinTheta, phi);
}

} // namespace

Expansion EvalUpsilon(const glm::dvec3& r, int p) {
    Expansion out(p);
    // A source exactly at the expansion center (the common single-particle
    // leaf case, where the leaf's center of mass *is* that one particle)
    // has r=0: monopole is 1 (mass carries its own weight in P2M), every
    // higher moment is exactly 0 (r^n factor) -- computed directly rather
    // than through theta/phi, which are undefined at r=0.
    if (glm::length(r) <= 1e-300) {
        out.raw(0, 0) = Complex{1.0, 0.0};
        return out;
    }

    const SphericalBasis b = MakeBasis(r, p);
    const std::vector<double> fact = FactorialTable(2 * p);

    std::vector<double> rPow(static_cast<size_t>(p) + 1);
    rPow[0] = 1.0;
    for (int n = 1; n <= p; ++n) rPow[static_cast<size_t>(n)] = rPow[static_cast<size_t>(n - 1)] * b.r;

    for (int n = 0; n <= p; ++n) {
        for (int m = 0; m <= n; ++m) {
            const double coeff = rPow[static_cast<size_t>(n)] / fact[static_cast<size_t>(n + m)];
            const double sign = (m & 1) ? -1.0 : 1.0;
            out.raw(n, m) = sign * coeff * b.legendre(n, m) * b.cis[static_cast<size_t>(m)];
        }
    }
    return out;
}

Expansion EvalTheta(const glm::dvec3& r, int p) {
    Expansion out(p);
    const SphericalBasis b = MakeBasis(r, p);
    const std::vector<double> fact = FactorialTable(2 * p);

    std::vector<double> rInvPow(static_cast<size_t>(p) + 1); // 1/r^{n+1}
    rInvPow[0] = 1.0 / b.r;
    for (int n = 1; n <= p; ++n) rInvPow[static_cast<size_t>(n)] = rInvPow[static_cast<size_t>(n - 1)] / b.r;

    for (int n = 0; n <= p; ++n) {
        for (int m = 0; m <= n; ++m) {
            const double coeff = fact[static_cast<size_t>(n - m)] * rInvPow[static_cast<size_t>(n)];
            const double sign = (m & 1) ? -1.0 : 1.0;
            out.raw(n, m) = sign * coeff * b.legendre(n, m) * b.cis[static_cast<size_t>(m)];
        }
    }
    return out;
}

Expansion P2M(const glm::dvec3& center, int p, const std::vector<glm::dvec3>& pos, const std::vector<double>& mass,
              const std::vector<int>& indices) {
    Expansion out(p, center);
    for (int idx : indices) {
        const Expansion u = EvalUpsilon(pos[static_cast<size_t>(idx)] - center, p);
        const double m = mass[static_cast<size_t>(idx)];
        for (int n = 0; n <= p; ++n) {
            for (int mm = 0; mm <= n; ++mm) out.raw(n, mm) += m * u.raw(n, mm);
        }
    }
    return out;
}

Expansion M2M(const Expansion& m, const glm::dvec3& newCenter) {
    const int p = m.order;
    const Expansion u = EvalUpsilon(m.center - newCenter, p);
    Expansion out(p, newCenter);
    for (int n = 0; n <= p; ++n) {
        for (int mOut = 0; mOut <= n; ++mOut) {
            Complex sum{0.0, 0.0};
            for (int k = 0; k <= n; ++k) {
                for (int l = -k; l <= k; ++l) sum += u.at(k, l) * m.at(n - k, mOut - l);
            }
            out.raw(n, mOut) = sum;
        }
    }
    return out;
}

Expansion M2L(const Expansion& m, const glm::dvec3& targetCenter, int p) {
    const Expansion th = EvalTheta(targetCenter - m.center, p);
    Expansion out(p, targetCenter);
    for (int n = 0; n <= p; ++n) {
        for (int mOut = 0; mOut <= n; ++mOut) {
            Complex sum{0.0, 0.0};
            const int kMax = p - n;
            for (int k = 0; k <= kMax; ++k) {
                for (int l = -k; l <= k; ++l) sum += std::conj(m.at(k, l)) * th.at(n + k, mOut + l);
            }
            out.raw(n, mOut) = sum;
        }
    }
    return out;
}

Expansion L2L(const Expansion& l, const glm::dvec3& newCenter) {
    const int p = l.order;
    const Expansion u = EvalUpsilon(l.center - newCenter, p);
    Expansion out(p, newCenter);
    for (int n = 0; n <= p; ++n) {
        for (int mOut = 0; mOut <= n; ++mOut) {
            Complex sum{0.0, 0.0};
            const int kMax = p - n;
            for (int k = 0; k <= kMax; ++k) {
                for (int j = -k; j <= k; ++j) sum += std::conj(u.at(k, j)) * l.at(n + k, mOut + j);
            }
            out.raw(n, mOut) = sum;
        }
    }
    return out;
}

namespace {

// Raw (G-free, positive-for-positive-mass) potential Psi = sum Upsilon* .
// L, eq. 3a -- the physical, attractive gravitational potential and its
// sign relative to this are established once, empirically, in
// SphericalHarmonicsTest.cpp rather than assumed here (this expression's
// literal sign is a documented-ambiguous point in the source derivation --
// see this file's header).
double RawPotential(const Expansion& l, const glm::dvec3& point) {
    const Expansion u = EvalUpsilon(l.center - point, l.order);
    Complex sum{0.0, 0.0};
    for (int n = 0; n <= l.order; ++n) {
        for (int m = -n; m <= n; ++m) sum += std::conj(u.at(n, m)) * l.at(n, m);
    }
    return sum.real();
}

} // namespace

PotentialAndAccel EvaluateLocal(const Expansion& l, const glm::dvec3& point, double G) {
    PotentialAndAccel out;
    const double psi = RawPotential(l, point);
    out.potential = -G * psi;

    // Central difference of the already-validated potential evaluator,
    // rather than a fourth independently-derived analytic gradient formula
    // (Upsilon's own degree/order-shifting derivative recurrence) -- a
    // deliberate simplicity-over-speed trade for this first pass; see
    // SphericalFmm's own notes for revisiting this if L2P profiles hot.
    constexpr double kH = 1e-4;
    const double gx = (RawPotential(l, point + glm::dvec3(kH, 0, 0)) - RawPotential(l, point - glm::dvec3(kH, 0, 0))) / (2.0 * kH);
    const double gy = (RawPotential(l, point + glm::dvec3(0, kH, 0)) - RawPotential(l, point - glm::dvec3(0, kH, 0))) / (2.0 * kH);
    const double gz = (RawPotential(l, point + glm::dvec3(0, 0, kH)) - RawPotential(l, point - glm::dvec3(0, 0, kH))) / (2.0 * kH);
    out.accel = G * glm::dvec3(gx, gy, gz);
    return out;
}

} // namespace nbody::sh
