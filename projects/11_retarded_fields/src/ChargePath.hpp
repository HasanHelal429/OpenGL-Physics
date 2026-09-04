#pragma once

#include <glm/glm.hpp>

#include <cmath>
#include <string>

namespace lw {

// An analytically prescribed path r(t) for a point charge, with the exact
// v(t) and a(t) the Lienard-Wiechert formula needs. Kept in parity with
// Physics Simulations' retarded_fields.py example paths. All in 3D (z = 0 for
// planar motion, but the radiated B still points out of the plane).
struct ChargePath {
    enum Kind { Static, Uniform, Circular, LinearOscillator, Figure8 };

    Kind kind = Circular;
    glm::dvec3 center{0.0};
    glm::dvec3 v0{0.0};        // Uniform: constant velocity
    double radius = 1.0;       // Circular; also A_x for Figure8
    double amplitude = 1.0;    // LinearOscillator amplitude; also A_y for Figure8
    double omega = 1.0;
    double phase = 0.0;
    glm::dvec3 axis{1.0, 0.0, 0.0};   // LinearOscillator direction

    static Kind ParseKind(const std::string& s) {
        if (s == "static") return Static;
        if (s == "uniform") return Uniform;
        if (s == "linear_oscillator") return LinearOscillator;
        if (s == "figure8") return Figure8;
        return Circular;
    }

    glm::dvec3 r(double t) const {
        const double p = omega * t + phase;
        switch (kind) {
        case Static:          return center;
        case Uniform:         return center + v0 * t;
        case LinearOscillator: return center + axis * (amplitude * std::sin(p));
        case Figure8:         return center + glm::dvec3(radius * std::sin(p),
                                                        amplitude * std::sin(2.0 * p), 0.0);
        default:              return center + glm::dvec3(radius * std::cos(p),
                                                        radius * std::sin(p), 0.0);
        }
    }
    glm::dvec3 v(double t) const {
        const double p = omega * t + phase, w = omega;
        switch (kind) {
        case Static:          return glm::dvec3(0.0);
        case Uniform:         return v0;
        case LinearOscillator: return axis * (amplitude * w * std::cos(p));
        case Figure8:         return glm::dvec3(radius * w * std::cos(p),
                                               2.0 * amplitude * w * std::cos(2.0 * p), 0.0);
        default:              return glm::dvec3(-radius * w * std::sin(p),
                                                radius * w * std::cos(p), 0.0);
        }
    }
    glm::dvec3 a(double t) const {
        const double p = omega * t + phase, w2 = omega * omega;
        switch (kind) {
        case Static:
        case Uniform:         return glm::dvec3(0.0);
        case LinearOscillator: return axis * (-amplitude * w2 * std::sin(p));
        case Figure8:         return glm::dvec3(-radius * w2 * std::sin(p),
                                               -4.0 * amplitude * w2 * std::sin(2.0 * p), 0.0);
        default:              return glm::dvec3(-radius * w2 * std::cos(p),
                                                -radius * w2 * std::sin(p), 0.0);
        }
    }
};

} // namespace lw
