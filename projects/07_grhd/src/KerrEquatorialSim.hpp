#pragma once

#include "framework/ComputeShader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace grhd {

// GRHD confined to the Kerr equatorial plane -- Tier 2 Phase 2a of the
// tidal-disruption project (see 06_tidal_disruption/README.md's tier
// table and 07_grhd/README.md). An exact invariant submanifold of Kerr
// (reflection symmetry keeps v^theta=0 fluid exactly in the plane), so
// this stays "1D in r" like Phase 0/1, now with an added angular-momentum
// degree of freedom from frame dragging (see kernels_kerr.hpp for the
// physics: conserved variables D/Sr/L/tau, the one genuine geometric
// source term on Sr, and why the HLLE bound here is a conservative
// coordinate-photon-speed estimate rather than the tight flux-Jacobian
// eigenvalues).
//
// Validation target: initialize a ring on an exact circular geodesic
// orbit (tools/kerr_orbits.py, itself derived from the effective-
// potential double-root condition rather than a recalled closed form) and
// confirm it stays circular (v_r stays ~0) over many orbital periods.
class KerrEquatorialSim : public fw::Simulation {
public:
    KerrEquatorialSim() = default;
    ~KerrEquatorialSim() override;

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

    int m_n = 0;
    double m_rMin = 6.0;
    double m_rMax = 20.0;
    double m_dr = 0.0;
    double m_M = 1.0;
    double m_a = 0.0; // spin, |a| < M
    double m_gamma = 4.0 / 3.0;
    double m_cfl = 0.2;
    double m_dt = 0.0;
    int m_consToPrimIters = 40;
    int m_substepsPerFrame = 1;

    std::string m_icFile;
    std::string m_title = "GRHD -- Kerr equatorial circular orbit";

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

    std::vector<glm::vec4> m_initialCons;
    std::vector<glm::vec4> m_consCpu;
    std::vector<glm::vec4> m_primCpu;
};

} // namespace grhd
