#pragma once

#include "framework/Camera.hpp"
#include "framework/ComputeShader.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace tde {

// One SPH particle's persistent CPU-side state, matching the 7-float32
// (x,y,z,mass,vx,vy,vz) record written by tools/make_star_ic.py.
struct ParticleRecord {
    glm::vec3 pos;
    float mass;
    glm::vec3 vel;
};

// Newtonian self-gravitating SPH star (Tier 1 of the tidal-disruption
// project -- see README.md): equal-mass SPH particles feel mutual softened
// self-gravity plus a polytropic pressure/artificial-viscosity force,
// integrated by kick-drift-kick leapfrog, all on the GPU (see kernels.hpp).
// Initial conditions are a Lane-Emden polytrope Monte-Carlo-sampled by
// tools/make_star_ic.py; this phase has no black hole yet -- the deliverable
// is a star that sits in hydrostatic equilibrium instead of collapsing or
// flying apart, checked via the diagnostics this Snapshot()s.
class TdeSim : public fw::Simulation {
public:
    TdeSim() = default;
    ~TdeSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;

private:
    void CreateBuffers();
    void UploadInitial();
    void ComputeDensityAndForces();  // one density pass + one forces pass, with barriers
    void ReadBack();                 // GPU buffers -> m_posMass/m_vel/m_rhoPress mirrors

    int m_n = 0;
    double m_dt = 1e-3;
    int m_substepsPerFrame = 1;

    // Physics parameters (see README.md's deck format for units/derivation).
    double m_G = 1.0;
    double m_softening = 0.05;
    double m_h = 0.1;         // fixed SPH smoothing length
    double m_K = 1.0;         // polytropic EOS constant, P = K rho^Gamma
    double m_gamma = 5.0 / 3.0;
    double m_viscAlpha = 1.0;
    double m_viscBeta = 2.0;

    std::string m_title = "Tidal Disruption -- SPH star";
    std::vector<std::string> m_diagNames;

    // GPU state (see kernels.hpp for the buffer layout).
    GLuint m_posMass = 0;
    GLuint m_vel = 0;
    GLuint m_acc = 0;
    GLuint m_rhoPress = 0;

    fw::ComputeShader m_density;
    fw::ComputeShader m_forces;
    fw::ComputeShader m_kick;
    fw::ComputeShader m_drift;

    // CPU mirrors: m_initial is the reset baseline (never mutated after
    // Configure); the other three are refreshed by ReadBack() (for Snapshot
    // and interactive Render).
    std::vector<ParticleRecord> m_initial;
    std::vector<glm::vec4> m_posMassCpu;
    std::vector<glm::vec4> m_velCpu;
    std::vector<glm::vec4> m_rhoPressCpu;

    // Interactive view.
    fw::Camera m_camera;
    fw::ParticleCloud m_particles;
};

} // namespace tde
