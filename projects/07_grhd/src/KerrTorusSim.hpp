#pragma once

#include "framework/ComputeShader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace grhd {

// Full 2D (r,theta) GRHD on a fixed Kerr background -- Tier 2 Phase 2b
// part 2 (see 07_grhd/README.md and kernels_kerr2d.hpp). Validation
// target: initialize the grid to the Fishbone-Moncrief equilibrium torus
// (tools/fishbone_moncrief.py, tools/make_fm_torus_ic.py) and confirm it
// holds steady -- the dynamical half of Phase 2b, whose analytic half
// (the torus solution itself, and the acceleration/source-term physics
// this reuses) was validated in Phase 2b part 1.
class KerrTorusSim : public fw::Simulation {
public:
    KerrTorusSim() = default;
    ~KerrTorusSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    void CreateBuffers();
    void UploadInitial();
    void ReadBack();
    void RunOneRk2Step();

    int m_nr = 0, m_nth = 0;
    double m_rMin = 4.0, m_rMax = 30.0;
    double m_thetaMin = 0.5, m_thetaMax = 0.0; // thetaMax derived: pi - thetaMin
    double m_dr = 0.0, m_dth = 0.0;
    double m_M = 1.0, m_a = 0.0;
    double m_gamma = 4.0 / 3.0;
    double m_cfl = 0.15;
    double m_dt = 0.0;
    int m_consToPrimIters = 40;
    double m_rhoFloor = 1e-8;
    double m_pFloor = 1e-11;
    double m_entropyFloor = 1e-5;
    int m_substepsPerFrame = 1;

    std::string m_icFile;
    std::string m_title = "GRHD -- Kerr Fishbone-Moncrief torus";

    // Conserved: D,Sr,Stheta,tau in vec4 buffers; L separately (5th
    // quantity doesn't fit in a vec4 with the other four -- see
    // kernels_kerr2d.hpp's header comment).
    GLuint m_cons[2] = {0, 0};
    GLuint m_consL[2] = {0, 0};
    GLuint m_stage1 = 0, m_stage2 = 0;
    GLuint m_stage1L = 0, m_stage2L = 0;
    GLuint m_primMain = 0, m_primP = 0;
    GLuint m_fluxR = 0, m_fluxRL = 0;
    GLuint m_fluxTh = 0, m_fluxThL = 0;
    int m_cur = 0;

    fw::ComputeShader m_consToPrim;
    fw::ComputeShader m_fluxesR;
    fw::ComputeShader m_fluxesTheta;
    fw::ComputeShader m_eulerStep;
    fw::ComputeShader m_combine;

    std::vector<glm::vec4> m_initialCons;
    std::vector<float> m_initialConsL;
    std::vector<glm::vec4> m_consCpu;
    std::vector<float> m_consLCpu;
    std::vector<glm::vec4> m_primMainCpu;
    std::vector<float> m_primPCpu;
};

} // namespace grhd
