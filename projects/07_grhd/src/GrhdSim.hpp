#pragma once

#include "framework/ComputeShader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace grhd {

// 1D special-relativistic hydrodynamics on flat spacetime (Minkowski
// metric, c=G=1) -- Tier 2 Phase 0 of the tidal-disruption project (see
// 06_tidal_disruption/README.md). Valencia conservative formulation,
// Gamma-law EOS, HLLE Riemann solver, RK2 (Heun) method-of-lines time
// integration, all on the GPU (see kernels.hpp). No curved metric and no
// self-gravity yet -- the deliverable of this phase is a fluid solver
// whose conservative-to-primitive recovery and Riemann solver are actually
// validated (Newtonian limit, conservation, the classic relativistic
// shock tube) before Phase 1 puts a Schwarzschild/Kerr background under it.
class GrhdSim : public fw::Simulation {
public:
    GrhdSim() = default;
    ~GrhdSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    void CreateBuffers();
    void UploadInitial();
    void ReadBack();
    void RunOneRk2Step(); // advances m_cur -> the other slot, m_dt fixed

    int m_n = 0;
    double m_dx = 0.0;
    double m_length = 1.0;
    double m_gamma = 5.0 / 3.0;
    double m_cfl = 0.4;
    double m_dt = 0.0; // = cfl*dx (c=1 is an always-safe SR signal-speed bound)
    int m_consToPrimIters = 25;
    int m_substepsPerFrame = 1;

    // Riemann-problem initial condition: two constant primitive states
    // separated at x0 (fraction of domain length).
    double m_rhoL = 1.0, m_vL = 0.0, m_pL = 1.0;
    double m_rhoR = 0.125, m_vR = 0.0, m_pR = 0.1;
    double m_x0 = 0.5;

    std::string m_title = "GRHD -- 1D SRHD shock tube (flat spacetime)";

    // GPU state (see kernels.hpp for the buffer layout/roles).
    GLuint m_cons[2] = {0, 0};
    GLuint m_stage1 = 0;
    GLuint m_stage2 = 0;
    GLuint m_prim = 0;
    GLuint m_flux = 0;
    int m_cur = 0;

    fw::ComputeShader m_consToPrim;
    fw::ComputeShader m_fluxes;
    fw::ComputeShader m_eulerStep;
    fw::ComputeShader m_combine;

    // CPU mirrors, refreshed by ReadBack() for Snapshot().
    std::vector<glm::vec4> m_initialCons;
    std::vector<glm::vec4> m_consCpu;
    std::vector<glm::vec4> m_primCpu;
};

} // namespace grhd
