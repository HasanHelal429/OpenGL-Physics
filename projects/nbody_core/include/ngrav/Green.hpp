#pragma once

#include "Vec.hpp"

#include <cmath>

// The gravitational Green's function, dimension-specialised.
//
// 3D uses the ordinary inverse-square force (potential -1/r). 2D uses the
// genuine 2D fundamental solution: force ~ 1/r (potential +ln r), NOT a
// projected 3D law -- this is why 2D circular speed is separation-independent
// and 2D orbits precess (see 03_nbody_gravity_2d's scenario comments).
//
// All functions are G-free: the caller multiplies the returned acceleration
// (or the pairwise energy term) by G. `eps2` is the squared Plummer softening
// length; compact-support (spline) softening is layered on top in
// Softening.hpp and does not touch these.
namespace ngrav {

// Acceleration on a body at the origin due to mass `m` at displacement
// `r = source - target`. Matches the existing per-pair inner loops in both
// projects term for term.
template <int D>
inline Vec<D> PlummerAccel(const Vec<D>& r, double m, double eps2) {
    const double r2 = glm::dot(r, r) + eps2;
    if constexpr (D == 3) {
        const double invDist = 1.0 / std::sqrt(r2);
        const double invDist3 = invDist * invDist * invDist;
        return (m * invDist3) * r;
    } else {
        return (m / r2) * r;
    }
}

// Pairwise potential energy per unit (m_i * m_j), G-free:
//   3D: -1 / sqrt(r^2 + eps2)
//   2D: +0.5 * ln(r^2 + eps2)   ( = ln sqrt(r^2 + eps2) )
template <int D>
inline double PlummerPairPotential(const Vec<D>& r, double eps2) {
    const double r2 = glm::dot(r, r) + eps2;
    if constexpr (D == 3) {
        return -1.0 / std::sqrt(r2);
    } else {
        return 0.5 * std::log(r2);
    }
}

// The far-field multipole kernels (M2L / Taylor) are unsoftened. This is the
// pure Green's function and its first derivative magnitude scale, used by the
// error-controlled MAC and by CartesianTaylor's D^n tensor recursion.
//   3D: phi = -1/r,  |grad phi| ~ 1/r^2
//   2D: phi = +ln r, |grad phi| ~ 1/r
template <int D>
inline double GreenPotential(double r) {
    if constexpr (D == 3) {
        return -1.0 / r;
    } else {
        return std::log(r);
    }
}

} // namespace ngrav
