#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "ngrav/System.hpp"

namespace nbody {

// SolverType is kept as this project's public vocabulary (the ImGui radio
// buttons, the benchmark, the scaling sweep). It maps 1:1 onto ngrav::Solver
// inside NBodySystem.
enum class SolverType { Direct, BarnesHut, AdaptiveFmm, SphericalFmm };

// Free-function accel dispatch, unchanged signature -- still used by the
// benchmark panel and ScalingSweepWorker to run any solver against an
// arbitrary AoS snapshot. Direct / Barnes-Hut route through nbody_core
// (ngrav); the two FMM variants still use this project's own tree code.
void ComputeAccel(SolverType solver, const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec3>& accelOut);

// Wire this project's still-in-project FMM solvers (AdaptiveFmm / SphericalFmm)
// onto an ngrav::System<3> as aux adapters. Used by NBodySystem and by the
// deck-driven NBodySim path in main.cpp.
void RegisterFmmAdaptersOn(ngrav::System<3>& sys);

// Softened-gravity N-body system. Thin wrapper over ngrav::System<3>: it owns
// the SoA state and the kick-drift-kick integrator; NBodySystem keeps an AoS
// position/velocity mirror so the render / diagnostics call sites that expect
// std::vector<glm::dvec3> stay untouched. Force evaluation is swappable per
// step (Direct O(N^2), Barnes-Hut O(N log N), or FMM).
class NBodySystem {
public:
    NBodySystem();

    void SetParticles(std::vector<glm::dvec3> positions, std::vector<glm::dvec3> velocities,
                       std::vector<double> masses);

    // Prime a(t) for the first leapfrog half-kick; call again if
    // G/softening/solver/theta change while paused.
    void PrimeAccelerations(double G, double softening, SolverType solver, double theta);

    void Step(double dt, double G, double softening, SolverType solver, double theta);

    double TotalEnergy(double G, double softening) const;
    glm::dvec3 AngularMomentum() const; // about the center of mass
    glm::dvec3 CenterOfMass() const;

    size_t Count() const { return m_pos.size(); }
    const std::vector<glm::dvec3>& Positions() const { return m_pos; }
    const std::vector<glm::dvec3>& Velocities() const { return m_vel; }
    const std::vector<double>& Masses() const { return m_mass; }

private:
    void RegisterFmmAdapters();
    void SyncMirrorFromCore();
    ngrav::StepParams MakeParams(double G, double softening, SolverType solver, double theta) const;

    ngrav::System<3> m_core;
    std::vector<glm::dvec3> m_pos, m_vel; // AoS mirror, synced after Step/SetParticles
    std::vector<double> m_mass;
};

} // namespace nbody
