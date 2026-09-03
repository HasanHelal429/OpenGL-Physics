#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <deque>
#include <vector>

namespace mag {

// A relativistic test charge, state carried as proper velocity u = gamma * v
// so the push stays well-behaved up to ultra-relativistic speeds.
struct TestCharge {
    glm::dvec3 x{0.0};
    glm::dvec3 u{0.0};        // gamma * v
    double qm = 1.0;          // q / m
    glm::vec3 color{1.0f, 0.85f, 0.2f};
    std::deque<glm::vec3> trail;   // recent positions, for the view
    std::size_t trailCap = 16000;

    double gamma(double c) const {
        return std::sqrt(1.0 + glm::dot(u, u) / (c * c));
    }
    glm::dvec3 velocity(double c) const { return u / gamma(c); }
    // Relativistic kinetic energy per unit mass, (gamma - 1) c^2.
    double keSpecific(double c) const { return (gamma(c) - 1.0) * c * c; }
};

// One Boris step (Boris 1970): half electric kick, magnetic rotation, half
// electric kick, then drift. 2nd-order, phase- and energy-preserving in a
// static magnetic field. `E`, `B` are sampled at the current position.
inline void BorisPush(TestCharge& p, const glm::dvec3& E, const glm::dvec3& B,
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

} // namespace mag
