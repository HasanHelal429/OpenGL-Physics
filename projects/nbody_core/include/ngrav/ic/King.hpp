#pragma once

#include "ngrav/State.hpp"
#include "ngrav/Vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

// King (1966) lowered-isothermal-sphere initial conditions, 3D.
//
// The King DF is f(Erel) = rho1/(2 pi sigma^2)^1.5 * [exp(Erel/sigma^2) - 1]
// for Erel > 0 (bound), 0 otherwise, where Erel = Psi(r) - v^2/2 is the
// relative energy and Psi = -phi > 0 the relative potential. Integrating
// this DF over velocity space at fixed Psi (independently re-derived here
// rather than quoted from memory -- substitute u = v/(sqrt(2) sigma),
// W = Psi/sigma^2, and reduce the resulting integral of u^2 e^{-u^2} via
// integration by parts to std::erf) gives the density-potential relation
//
//   rho(W)/rho0 = g(W)/g(W0),   g(W) := e^W erf(sqrt(W)) - sqrt(4W/pi)(1 + 2W/3),   W = Psi/sigma^2 >= 0
//
// (this matches the standard King-model density formula in, e.g., Binney &
// Tremaine eq. 4.114 -- the independent re-derivation is the check that
// it's being transcribed correctly here, not a claim of discovering
// something new). Poisson's equation with this source, in the standard
// dimensionless form x = r/r0 (r0 the King core radius), reads
//
//   d^2W/dx^2 + (2/x) dW/dx = -9 g(W)/g(W0),   W(0) = W0,   W'(0) = 0
//
// solved here by fixed-step RK4 from a small x, seeded by the regular
// series solution near the origin (also independently re-derived: assume
// W = W0 + a x^2 for small x, substitute, match the ODE's leading order to
// get a = -3/2) out to the dimensionless tidal radius x_t where W first
// reaches 0 -- the point where the density (and DF) vanish, by
// construction (g(W) -> 0 as W -> 0+, so the profile has compact support,
// unlike Plummer/Hernquist's infinite tails).
//
// Positions are drawn from the resulting M(<r) (built by numerically
// integrating rho(r) 4 pi r^2 dr along the ODE's own solution grid and
// normalizing to totalMass -- the overall density normalization rho0
// cancels in this ratio, so it's never needed explicitly). Velocities use
// the same Jeans-equation isotropic-dispersion approximation as
// ic::Hernquist (see that file's header for why): sigma_r^2(r) is built
// from the same rho-shape/mass tables, so rho0 cancels there too.
namespace ngrav::ic {

struct KingParams {
    int n = 1000;
    double totalMass = 1.0;
    double coreRadius = 1.0; // r0
    double w0 = 6.0;         // central dimensionless potential (typical range ~1-12)
    double G = 1.0;
    std::uint32_t seed = 1;
};

namespace detail {

inline double KingG(double W) {
    if (W <= 0.0) return 0.0;
    const double sq = std::sqrt(W);
    return std::exp(W) * std::erf(sq) - std::sqrt(4.0 * W / M_PI) * (1.0 + 2.0 * W / 3.0);
}

// Solves the dimensionless King ODE and tabulates everything ic::King needs
// in physical units (given a core radius and total mass): M(<r)/totalMass
// (for inverse-CDF position sampling) and sigma_r^2(r) (for Jeans velocity
// sampling), both built once per King() call (a few thousand RK4 steps,
// negligible next to actually sampling N particles for realistic N).
struct KingProfile {
    double xt = 0.0;               // dimensionless tidal radius
    std::vector<double> r;         // physical radius, r[0]=0 .. r.back()=xt*r0
    std::vector<double> cumMassShape; // integral_0^r g(W(r'/r0))/g0 * r'^2 dr' (unnormalized shape)
    std::vector<double> rhoShape;  // g(W(r/r0))/g0, in [0,1]
    std::vector<double> sigma2;    // sigma_r^2(r), physical (uses G, totalMass)

    KingProfile(double w0, double r0, double G, double totalMass) {
        const double g0 = KingG(w0);
        const double h = 2e-3;
        const double xEps = 1e-4;

        std::vector<double> x{0.0, xEps};
        std::vector<double> W{w0, w0 - 1.5 * xEps * xEps};
        std::vector<double> Wp{0.0, -3.0 * xEps};

        auto deriv = [&](double xx, double WW, double WWp, double& dW, double& dWp) {
            dW = WWp;
            dWp = -9.0 * KingG(WW) / g0 - ((xx > 1e-12) ? (2.0 / xx) * WWp : 0.0);
        };

        double xc = xEps, Wc = W.back(), Wpc = Wp.back();
        while (Wc > 0.0 && x.size() < 200000) {
            double k1W, k1Wp, k2W, k2Wp, k3W, k3Wp, k4W, k4Wp;
            deriv(xc, Wc, Wpc, k1W, k1Wp);
            deriv(xc + h / 2, Wc + h / 2 * k1W, Wpc + h / 2 * k1Wp, k2W, k2Wp);
            deriv(xc + h / 2, Wc + h / 2 * k2W, Wpc + h / 2 * k2Wp, k3W, k3Wp);
            deriv(xc + h, Wc + h * k3W, Wpc + h * k3Wp, k4W, k4Wp);
            Wc = Wc + (h / 6.0) * (k1W + 2 * k2W + 2 * k3W + k4W);
            Wpc = Wpc + (h / 6.0) * (k1Wp + 2 * k2Wp + 2 * k3Wp + k4Wp);
            xc += h;
            x.push_back(xc);
            W.push_back(std::max(Wc, 0.0));
            Wp.push_back(Wpc);
        }
        xt = xc;

        const std::size_t np = x.size();
        r.resize(np);
        rhoShape.resize(np);
        for (std::size_t i = 0; i < np; ++i) {
            r[i] = x[i] * r0;
            rhoShape[i] = KingG(W[i]) / g0;
        }
        // Cumulative mass shape: integral_0^r rhoShape(r') r'^2 dr' (trapezoid).
        cumMassShape.assign(np, 0.0);
        for (std::size_t i = 1; i < np; ++i) {
            const double f0 = rhoShape[i - 1] * r[i - 1] * r[i - 1];
            const double f1 = rhoShape[i] * r[i] * r[i];
            cumMassShape[i] = cumMassShape[i - 1] + 0.5 * (f0 + f1) * (r[i] - r[i - 1]);
        }
        const double totalShape = cumMassShape.back();

        // Physical M(<r), then the Jeans tail integral for sigma_r^2(r)
        // (rho0 cancels in both -- see this file's header).
        std::vector<double> massPhys(np);
        for (std::size_t i = 0; i < np; ++i) massPhys[i] = totalMass * cumMassShape[i] / totalShape;

        std::vector<double> tail(np, 0.0);
        for (std::size_t i = np - 1; i-- > 0;) {
            const double r0i = r[i], r1i = r[i + 1];
            const double f0 = (r0i > 1e-300) ? rhoShape[i] * G * massPhys[i] / (r0i * r0i) : 0.0;
            const double f1 = rhoShape[i + 1] * G * massPhys[i + 1] / (r1i * r1i);
            tail[i] = tail[i + 1] + 0.5 * (f0 + f1) * (r1i - r0i);
        }
        sigma2.resize(np);
        for (std::size_t i = 0; i < np; ++i) {
            sigma2[i] = (rhoShape[i] > 1e-12) ? (tail[i] / rhoShape[i]) : 0.0;
        }
    }

    // Sample r via inverse-CDF (binary search on the monotonic cumMassShape
    // table) given uniform x1 in [0,1).
    double SampleRadius(double x1) const {
        const double target = x1 * cumMassShape.back();
        const auto it = std::lower_bound(cumMassShape.begin(), cumMassShape.end(), target);
        std::size_t i1 = static_cast<std::size_t>(it - cumMassShape.begin());
        i1 = std::clamp(i1, std::size_t{1}, cumMassShape.size() - 1);
        const std::size_t i0 = i1 - 1;
        const double c0 = cumMassShape[i0], c1 = cumMassShape[i1];
        const double frac = (c1 > c0) ? (target - c0) / (c1 - c0) : 0.0;
        return r[i0] * (1.0 - frac) + r[i1] * frac;
    }

    double Sigma2At(double rr) const {
        const auto it = std::lower_bound(r.begin(), r.end(), rr);
        std::size_t i1 = static_cast<std::size_t>(it - r.begin());
        i1 = std::clamp(i1, std::size_t{1}, r.size() - 1);
        const std::size_t i0 = i1 - 1;
        const double r0v = r[i0], r1v = r[i1];
        const double frac = (r1v > r0v) ? (rr - r0v) / (r1v - r0v) : 0.0;
        return sigma2[i0] * (1.0 - frac) + sigma2[i1] * frac;
    }
};

} // namespace detail

template <int D>
SoA<D> King(const KingParams& kp) {
    static_assert(D == 3, "King ic:: is 3D only");
    const detail::KingProfile profile(kp.w0, kp.coreRadius, kp.G, kp.totalMass);

    SoA<D> s;
    s.Resize(static_cast<std::size_t>(kp.n));
    std::mt19937 rng(kp.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const double m = kp.totalMass / kp.n;

    for (int i = 0; i < kp.n; ++i) {
        const double r = profile.SampleRadius(u(rng));

        const double cosT = 2.0 * u(rng) - 1.0;
        const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
        const double phi = 2.0 * M_PI * u(rng);
        const Vec<3> dir(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
        s.SetPos(static_cast<std::size_t>(i), dir * r);

        const double sigma = std::sqrt(std::max(0.0, profile.Sigma2At(r)));
        std::normal_distribution<double> gg(0.0, sigma);
        const Vec<3> v(gg(rng), gg(rng), gg(rng));
        s.SetVel(static_cast<std::size_t>(i), v);
        s.m[static_cast<std::size_t>(i)] = m;
    }

    Vec<D> cp(0.0), cv(0.0);
    double tm = 0.0;
    for (int i = 0; i < kp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        cp += s.m[ii] * s.Pos(ii);
        cv += s.m[ii] * s.Vel(ii);
        tm += s.m[ii];
    }
    cp /= tm;
    cv /= tm;
    for (int i = 0; i < kp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        s.SetPos(ii, s.Pos(ii) - cp);
        s.SetVel(ii, s.Vel(ii) - cv);
    }
    return s;
}

} // namespace ngrav::ic
