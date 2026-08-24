#include "Potentials.hpp"
#include "Quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace hf {

namespace {
constexpr double kPi = std::numbers::pi;
} // namespace

std::vector<double> HartreePotential(const std::vector<double>& r, const std::vector<double>& rho) {
    const size_t n = r.size();
    std::vector<double> qIntegrand(n), sIntegrand(n);
    for (size_t i = 0; i < n; ++i) {
        qIntegrand[i] = 4.0 * kPi * r[i] * r[i] * rho[i];
        sIntegrand[i] = 4.0 * kPi * r[i] * rho[i];
    }
    const std::vector<double> Q = CumulativeTrapezoid(qIntegrand, r);
    const std::vector<double> S = CumulativeTrapezoid(sIntegrand, r);
    const double sTotal = S.back();

    std::vector<double> vH(n);
    for (size_t i = 0; i < n; ++i) {
        vH[i] = Q[i] / r[i] + (sTotal - S[i]);
    }
    return vH;
}

std::vector<double> SlaterExchangePotential(const std::vector<double>& rho, double alpha) {
    std::vector<double> vX(rho.size());
    for (size_t i = 0; i < rho.size(); ++i) {
        vX[i] = -3.0 * alpha * std::cbrt(3.0 * rho[i] / (8.0 * kPi));
    }
    return vX;
}

std::pair<std::vector<double>, std::vector<double>> Pz81Correlation(const std::vector<double>& rho) {
    const size_t n = rho.size();
    std::vector<double> epsC(n), vC(n);

    constexpr double A = 0.0311, B = -0.0480, C = 0.0020, D = -0.0116;
    constexpr double gamma = -0.1423, beta1 = 1.0529, beta2 = 0.3334;

    for (size_t i = 0; i < n; ++i) {
        // Far out on the radial grid the density underflows to exact 0.0;
        // clip to a floor far below any physically meaningful density so
        // rs = (3/(4*pi*rho))**(1/3) stays finite (physically rho=0 means
        // eps_c, V_c -> 0, the true limit of both branches as rs -> infinity).
        const double rhoClamped = std::max(rho[i], 1e-300);
        const double rs = std::cbrt(3.0 / (4.0 * kPi * rhoClamped));

        if (rs < 1.0) {
            const double lnRs = std::log(rs);
            epsC[i] = A * lnRs + B + C * rs * lnRs + D * rs;
            vC[i] = A * lnRs + (B - A / 3.0) + (2.0 / 3.0) * C * rs * lnRs + ((2.0 * D - C) / 3.0) * rs;
        } else {
            const double sqrtRs = std::sqrt(rs);
            const double denom = 1.0 + beta1 * sqrtRs + beta2 * rs;
            epsC[i] = gamma / denom;
            vC[i] = epsC[i] * (1.0 + (7.0 / 6.0) * beta1 * sqrtRs + (4.0 / 3.0) * beta2 * rs) / denom;
        }
    }
    return {epsC, vC};
}

std::pair<std::vector<double>, std::vector<double>> LdaXcPotential(const std::vector<double>& rho) {
    std::vector<double> vX = SlaterExchangePotential(rho, kAlphaLda);
    auto [epsC, vC] = Pz81Correlation(rho);
    std::vector<double> vXc(rho.size());
    for (size_t i = 0; i < rho.size(); ++i) {
        vXc[i] = vX[i] + vC[i];
    }
    return {vXc, epsC};
}

} // namespace hf
