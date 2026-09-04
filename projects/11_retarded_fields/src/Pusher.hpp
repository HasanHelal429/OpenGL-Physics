#pragma once

#include <glm/glm.hpp>

#include <cmath>

namespace lw {

// Relativistic Boris push (Boris 1970), state carried as proper velocity
// u = gamma v so it stays well-behaved to ultra-relativistic speeds. Shared
// in spirit with 09_magnetostatics' BorisPusher; kept local to avoid a
// cross-project include. Half electric kick / magnetic rotation / half
// electric kick / drift -- 2nd order, phase- and energy-preserving in a
// static B.
struct Mover {
    glm::dvec3 x{0.0};
    glm::dvec3 u{0.0};          // gamma * v
    double qm = 1.0;            // q / m

    double gamma(double c) const {
        return std::sqrt(1.0 + glm::dot(u, u) / (c * c));
    }
    glm::dvec3 v(double c) const { return u / gamma(c); }
    // Relativistic KE per unit mass, (gamma - 1) c^2.
    double keSpecific(double c) const { return (gamma(c) - 1.0) * c * c; }
};

inline void BorisPush(Mover& p, const glm::dvec3& E, const glm::dvec3& B,
                      double dt, double c) {
    const double h = 0.5 * dt * p.qm;
    const glm::dvec3 uMinus = p.u + h * E;
    const double gMinus = std::sqrt(1.0 + glm::dot(uMinus, uMinus) / (c * c));
    const glm::dvec3 t = (h / gMinus) * B;
    const glm::dvec3 s = 2.0 * t / (1.0 + glm::dot(t, t));
    const glm::dvec3 uPrime = uMinus + glm::cross(uMinus, t);
    const glm::dvec3 uPlus = uMinus + glm::cross(uPrime, s);
    p.u = uPlus + h * E;
    const double g = std::sqrt(1.0 + glm::dot(p.u, p.u) / (c * c));
    p.x += dt * p.u / g;
}

// Proper acceleration a = dv/dt from the Lorentz force, for the trajectory
// buffer (the LW formula wants coordinate acceleration). With f = q(E + v x B)
// the specific force per unit mass, and gamma from the current u:
//   dv/dt = (1/gamma) [ f - (f . v) v / c^2 ]
inline glm::dvec3 CoordAccel(const Mover& p, const glm::dvec3& E,
                             const glm::dvec3& B, double c) {
    const double g = p.gamma(c);
    const glm::dvec3 vel = p.u / g;
    const glm::dvec3 f = p.qm * (E + glm::cross(vel, B));
    return (f - glm::dot(f, vel) * vel / (c * c)) / g;
}

} // namespace lw
