#pragma once

#include "ngrav/State.hpp"
#include "ngrav/Vec.hpp"

#include <cmath>
#include <cstdint>
#include <random>

// Plummer-sphere initial conditions in units G = M_total = a = 1 (a = scale
// radius). 3D uses the standard isotropic distribution function (Aarseth,
// Henon & Wielen 1974); the half-mass radius is ~1.305 a, the crossing time
// ~ 2*pi. The 2D variant draws from the 2D Plummer surface density
// Sigma ~ (1 + r^2)^-2 with an isotropic velocity dispersion from the 2D
// Jeans equation -- a crude equilibrium, adequate as a stress-test IC.
namespace ngrav::ic {

struct PlummerParams {
    int n = 1000;
    double totalMass = 1.0;
    double scaleRadius = 1.0;
    double G = 1.0;
    std::uint32_t seed = 1;
};

template <int D>
SoA<D> Plummer(const PlummerParams& pp) {
    SoA<D> s;
    s.Resize(static_cast<std::size_t>(pp.n));
    std::mt19937 rng(pp.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);

    const double m = pp.totalMass / pp.n;
    // Velocity scale so that G = M = a = 1 internally, then rescaled.
    const double vUnit = std::sqrt(pp.G * pp.totalMass / pp.scaleRadius);

    for (int i = 0; i < pp.n; ++i) {
        double r;
        if constexpr (D == 3) {
            const double x1 = u(rng);
            r = 1.0 / std::sqrt(std::pow(x1, -2.0 / 3.0) - 1.0);
        } else {
            const double x1 = u(rng);
            r = std::sqrt(x1 / (1.0 - x1)); // M(<r) = r^2/(1+r^2)
        }
        r = std::min(r, 20.0); // clip the far tail

        // Random direction.
        Vec<D> dir;
        if constexpr (D == 3) {
            const double cosT = 2.0 * u(rng) - 1.0;
            const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
            const double phi = 2.0 * M_PI * u(rng);
            dir = Vec<3>(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
        } else {
            const double phi = 2.0 * M_PI * u(rng);
            dir = Vec<2>(std::cos(phi), std::sin(phi));
        }
        s.SetPos(static_cast<std::size_t>(i), dir * (r * pp.scaleRadius));

        // Speed.
        double speed;
        if constexpr (D == 3) {
            // Rejection-sample q from g(q) = q^2 (1-q^2)^{7/2}, q in [0,1].
            double q, g;
            do {
                q = u(rng);
                g = u(rng) * 0.1;
            } while (g > q * q * std::pow(1.0 - q * q, 3.5));
            const double vEsc = std::sqrt(2.0) * std::pow(1.0 + r * r, -0.25);
            speed = q * vEsc;
        } else {
            // 2D Jeans: sigma^2(r) ~ 1 / (6 (1 + r^2)); draw |v| from a 2D
            // Maxwellian with that dispersion (isotropic).
            const double sigma = std::sqrt(1.0 / (6.0 * (1.0 + r * r)));
            std::normal_distribution<double> gg(0.0, sigma);
            const double vx = gg(rng), vy = gg(rng);
            speed = std::sqrt(vx * vx + vy * vy);
        }

        Vec<D> vdir;
        if constexpr (D == 3) {
            const double cosT = 2.0 * u(rng) - 1.0;
            const double sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
            const double phi = 2.0 * M_PI * u(rng);
            vdir = Vec<3>(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
        } else {
            const double phi = 2.0 * M_PI * u(rng);
            vdir = Vec<2>(std::cos(phi), std::sin(phi));
        }
        s.SetVel(static_cast<std::size_t>(i), vdir * (speed * vUnit));
        s.m[static_cast<std::size_t>(i)] = m;
    }

    // Recenter to zero COM position and momentum (removes sampling drift).
    Vec<D> cp(0.0), cv(0.0);
    double tm = 0.0;
    for (int i = 0; i < pp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        cp += s.m[ii] * s.Pos(ii);
        cv += s.m[ii] * s.Vel(ii);
        tm += s.m[ii];
    }
    cp /= tm;
    cv /= tm;
    for (int i = 0; i < pp.n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        s.SetPos(ii, s.Pos(ii) - cp);
        s.SetVel(ii, s.Vel(ii) - cv);
    }
    return s;
}

} // namespace ngrav::ic
