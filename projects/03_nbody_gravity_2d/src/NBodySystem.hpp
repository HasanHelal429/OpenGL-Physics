#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "ngrav/System.hpp"

namespace nbody2d {

enum class SolverType { Direct, BarnesHut, ComplexFmm };

// Free-function accel dispatch, unchanged signature -- used by the benchmark
// panel. Direct / Barnes-Hut route through nbody_core (ngrav); ComplexFmm
// still uses this project's own quadtree code.
void ComputeAccel(SolverType solver, const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec2>& accelOut);

// Wire this project's still-in-project ComplexFmm solver onto an
// ngrav::System<2> as an aux adapter.
void RegisterFmmAdaptersOn(ngrav::System<2>& sys);

// Softened-gravity 2D N-body system. Thin wrapper over ngrav::System<2>: it
// owns the SoA state and the leapfrog integrator; NBodySystem keeps an AoS
// mirror so the render / diagnostics call sites that expect
// std::vector<glm::dvec2> stay untouched.
class NBodySystem {
public:
    NBodySystem();

    void SetParticles(std::vector<glm::dvec2> positions, std::vector<glm::dvec2> velocities,
                       std::vector<double> masses);

    void PrimeAccelerations(double G, double softening, SolverType solver, double theta);
    // `adaptive` replaces the fixed `dt` with eta*min sqrt(softening/|a|),
    // clamped to at most `dt`. Returns the dt actually taken.
    double Step(double dt, double G, double softening, SolverType solver, double theta, bool adaptive = false,
                double eta = 0.03);

    double TotalEnergy(double G, double softening) const;
    double AngularMomentum() const; // 2D: the scalar z-component
    glm::dvec2 CenterOfMass() const;

    size_t Count() const { return m_pos.size(); }
    const std::vector<glm::dvec2>& Positions() const { return m_pos; }
    const std::vector<glm::dvec2>& Velocities() const { return m_vel; }
    const std::vector<double>& Masses() const { return m_mass; }

private:
    void RegisterFmmAdapters();
    void SyncMirrorFromCore();
    ngrav::StepParams MakeParams(double G, double softening, SolverType solver, double theta) const;

    ngrav::System<2> m_core;
    std::vector<glm::dvec2> m_pos, m_vel; // AoS mirror
    std::vector<double> m_mass;
};

} // namespace nbody2d
