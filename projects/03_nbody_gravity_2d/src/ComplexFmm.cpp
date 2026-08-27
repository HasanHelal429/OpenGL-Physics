#include "ComplexFmm.hpp"

#include <cmath>

namespace nbody2d::fmm {

namespace {

// Pascal's-triangle binomial table up to row `maxN`, needed for M2L's
// C(j+k-1, k-1) with j,k up to p (so rows up to ~2p-1).
class BinomialTable {
public:
    void EnsureAtLeast(int n) {
        if (n <= m_maxN) return;
        m_maxN = n;
        m_table.assign(static_cast<size_t>((n + 1) * (n + 2) / 2), 0.0);
        for (int row = 0; row <= n; ++row) {
            At(row, 0) = 1.0;
            At(row, row) = 1.0;
            for (int k = 1; k < row; ++k) At(row, k) = At(row - 1, k - 1) + At(row - 1, k);
        }
    }

    double operator()(int n, int k) const { return m_table[static_cast<size_t>(n * (n + 1) / 2 + k)]; }

private:
    double& At(int n, int k) { return m_table[static_cast<size_t>(n * (n + 1) / 2 + k)]; }

    int m_maxN = -1;
    std::vector<double> m_table;
};

BinomialTable& Binom() {
    static BinomialTable table;
    return table;
}

} // namespace

Expansion P2M(const Complex& center, int p, const std::vector<glm::dvec2>& pos, const std::vector<double>& mass,
              const std::vector<int>& indices) {
    Expansion out(p, center);
    double m0 = 0.0;
    for (int idx : indices) {
        const double m = mass[static_cast<size_t>(idx)];
        m0 += m;
        const Complex d = ToComplex(pos[static_cast<size_t>(idx)]) - center;
        Complex dPow = d;
        for (int j = 1; j <= p; ++j) {
            out.coeff[static_cast<size_t>(j)] -= m * dPow;
            dPow *= d;
        }
    }
    out.coeff[0] = Complex(m0, 0.0);
    for (int j = 1; j <= p; ++j) out.coeff[static_cast<size_t>(j)] /= static_cast<double>(j);
    return out;
}

Expansion M2M(const Expansion& m, const Complex& newCenter) {
    const int p = m.order;
    Binom().EnsureAtLeast(2 * p);
    const Complex disp = m.center - newCenter;

    std::vector<Complex> dPow(static_cast<size_t>(p) + 1);
    dPow[0] = Complex(1.0, 0.0);
    for (int i = 1; i <= p; ++i) dPow[static_cast<size_t>(i)] = dPow[static_cast<size_t>(i - 1)] * disp;

    Expansion out(p, newCenter);
    out.coeff[0] = m.coeff[0];
    for (int r = 1; r <= p; ++r) {
        Complex val = -m.coeff[0] * dPow[static_cast<size_t>(r)] / static_cast<double>(r);
        for (int k = 1; k <= r; ++k) {
            val += m.coeff[static_cast<size_t>(k)] * dPow[static_cast<size_t>(r - k)] * Binom()(r - 1, k - 1);
        }
        out.coeff[static_cast<size_t>(r)] = val;
    }
    return out;
}

Expansion M2L(const Expansion& m, const Complex& targetCenter, int p) {
    Binom().EnsureAtLeast(2 * p);
    const Complex disp = m.center - targetCenter;
    const Complex dispInv = Complex(1.0, 0.0) / disp;

    std::vector<Complex> dInvPow(static_cast<size_t>(p) + 1);
    dInvPow[0] = Complex(1.0, 0.0);
    for (int i = 1; i <= p; ++i) dInvPow[static_cast<size_t>(i)] = dInvPow[static_cast<size_t>(i - 1)] * dispInv;

    Expansion out(p, targetCenter);

    Complex l0 = m.coeff[0] * std::log(-disp);
    for (int k = 1; k <= p; ++k) {
        const double sign = (k & 1) ? -1.0 : 1.0;
        l0 += sign * m.coeff[static_cast<size_t>(k)] * dInvPow[static_cast<size_t>(k)];
    }
    out.coeff[0] = l0;

    for (int j = 1; j <= p; ++j) {
        Complex lj = -m.coeff[0] / static_cast<double>(j);
        for (int k = 1; k <= p; ++k) {
            const double sign = (k & 1) ? -1.0 : 1.0;
            const double binom = Binom()(j + k - 1, k - 1);
            lj += sign * m.coeff[static_cast<size_t>(k)] * dInvPow[static_cast<size_t>(k)] * binom;
        }
        lj *= dInvPow[static_cast<size_t>(j)];
        out.coeff[static_cast<size_t>(j)] = lj;
    }
    return out;
}

// In-place "Taylor shift": given a truncated Taylor series (a plain
// polynomial in (z-oldCenter)), re-expressed around a new center via
// repeated synthetic division by `disp` -- the standard O(p^2) algorithm
// for shifting a polynomial's expansion point (see e.g. any "Horner/Taylor
// shift" reference), not a combinatorial sum like M2M. `disp` is
// oldCenter - newCenter.
Expansion L2L(const Expansion& l, const Complex& newCenter) {
    const int p = l.order;
    const Complex disp = l.center - newCenter;

    Expansion out(p, newCenter);
    out.coeff = l.coeff;
    for (int j = 0; j < p; ++j) {
        for (int k = p - j - 1; k < p; ++k) {
            out.coeff[static_cast<size_t>(k)] -= disp * out.coeff[static_cast<size_t>(k + 1)];
        }
    }
    return out;
}

PotentialAndForce EvaluateLocal(const Expansion& l, const glm::dvec2& point, double G) {
    const int p = l.order;
    const Complex d = ToComplex(point) - l.center;

    Complex potentialSum = l.coeff[static_cast<size_t>(p)];
    for (int k = p - 1; k >= 0; --k) potentialSum = potentialSum * d + l.coeff[static_cast<size_t>(k)];

    // d/dz of the same Horner polynomial (skips the k=0 term, which has
    // zero derivative) -- see this file's header comment for the
    // Cauchy-Riemann identity packaging this into a real force below.
    Complex deriv = static_cast<double>(p) * l.coeff[static_cast<size_t>(p)];
    for (int k = p - 1; k >= 1; --k) deriv = deriv * d + static_cast<double>(k) * l.coeff[static_cast<size_t>(k)];

    PotentialAndForce out;
    out.potential = -G * potentialSum.real();
    out.force = G * glm::dvec2(-deriv.real(), deriv.imag());
    return out;
}

} // namespace nbody2d::fmm
