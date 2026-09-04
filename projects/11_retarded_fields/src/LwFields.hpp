#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <functional>
#include <vector>

namespace lw {

inline constexpr double kPi = 3.14159265358979323846;

// CPU Lienard-Wiechert field of a point charge on an arbitrary path -- a
// line-for-line port of retarded_fields.py's _retarded_time_jit and
// _lw_fields_jit_core, used as the GPU cross-check and by --selftest.
//
// `path` returns (r, v, a) at proper argument tau (given as three callables
// here so both analytic and buffer-backed paths plug in).

struct PathSample {
    glm::dvec3 r{0.0}, v{0.0}, a{0.0};
};
using PathFn = std::function<PathSample(double)>;

// Solve c*(t - t_ret) = |x - r(t_ret)| for t_ret <= t. `guess` NaN = none.
inline double RetardedTime(const PathFn& path, const glm::dvec3& x, double t,
                           double c, double guess) {
    double window = 1.0;
    double tLo = t - window;
    auto g = [&](double tr) {
        return c * (t - tr) - glm::length(x - path(tr).r);
    };
    for (int guard = 0; g(tLo) <= 0.0 && guard < 80; ++guard) {
        window *= 2.0;
        tLo = t - window;
    }
    double tHi = t;
    double tRet = (guess == guess && tLo < guess && guess < tHi)
                      ? guess : 0.5 * (tLo + tHi);
    for (int it = 0; it < 60; ++it) {
        const PathSample s = path(tRet);
        const glm::dvec3 Rv = x - s.r;
        const double R = glm::length(Rv);
        const double gv = c * (t - tRet) - R;
        if (gv > 0.0) tLo = tRet;
        else if (gv < 0.0) tHi = tRet;
        else return tRet;
        const glm::dvec3 n = Rv / R;
        const double gp = -c + glm::dot(n, s.v);
        const double step = gp != 0.0 ? gv / gp : 0.0;
        const double tN = tRet - step;
        if (tLo < tN && tN < tHi) {
            if (std::abs(step) < 1e-13 * std::max(1.0, std::abs(tRet))) return tN;
            tRet = tN;
        } else {
            tRet = 0.5 * (tLo + tHi);
        }
    }
    return tRet;
}

struct LwResult {
    glm::dvec3 E{0.0}, B{0.0};
    double tRet = 0.0;
};

inline LwResult LwFields(const PathFn& path, const glm::dvec3& x, double t,
                         double q, double c, double eps0, double guess) {
    const double tRet = RetardedTime(path, x, t, c, guess);
    const PathSample s = path(tRet);
    const glm::dvec3 Rv = x - s.r;
    const double R = glm::length(Rv);
    const glm::dvec3 n = Rv / R;
    const glm::dvec3 beta = s.v / c;
    const glm::dvec3 betaDot = s.a / c;

    const double kappa = 1.0 - glm::dot(n, beta);
    const double denom = kappa * kappa * kappa;
    const double b2 = glm::dot(beta, beta);

    const glm::dvec3 near = (n - beta) * (1.0 - b2) / (R * R * denom);
    const glm::dvec3 far =
        glm::cross(n, glm::cross(n - beta, betaDot)) / (c * R * denom);

    LwResult out;
    out.E = (q / (4.0 * kPi * eps0)) * (near + far);
    out.B = glm::cross(n, out.E) / c;
    out.tRet = tRet;
    return out;
}

} // namespace lw
