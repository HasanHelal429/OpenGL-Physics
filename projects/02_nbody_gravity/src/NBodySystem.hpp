#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody {

enum class SolverType { Direct, BarnesHut };

// Softened-gravity N-body system: SoA position/velocity/mass state plus a
// kick-drift-kick (leapfrog) symplectic integrator. Force evaluation is
// swappable per step (Direct O(N^2) or Barnes-Hut O(N log N), see
// Octree.hpp) so the two can be compared live rather than only offline.
class NBodySystem {
public:
    void SetParticles(std::vector<glm::dvec3> positions, std::vector<glm::dvec3> velocities,
                       std::vector<double> masses);

    // Must be called once after SetParticles (and again if G/softening/
    // solver change while paused) so the first leapfrog half-kick has a
    // valid acceleration to use.
    void PrimeAccelerations(double G, double softening, SolverType solver, double theta);

    void Step(double dt, double G, double softening, SolverType solver, double theta);

    // O(N^2) diagnostics -- not cheap at large N; call sparingly (see
    // NBodyApp's throttled diagnostics tick).
    double TotalEnergy(double G, double softening) const;
    glm::dvec3 AngularMomentum() const; // about the system's center of mass
    glm::dvec3 CenterOfMass() const;

    size_t Count() const { return m_pos.size(); }
    const std::vector<glm::dvec3>& Positions() const { return m_pos; }
    const std::vector<glm::dvec3>& Velocities() const { return m_vel; }
    const std::vector<double>& Masses() const { return m_mass; }

private:
    void RecomputeAccel(double G, double softening, SolverType solver, double theta);

    std::vector<glm::dvec3> m_pos, m_vel, m_accel;
    std::vector<double> m_mass;
};

} // namespace nbody
