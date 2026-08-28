#pragma once

#include "framework/ComputeShader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace grhd {

// 1D general-relativistic hydrodynamics on a FIXED Schwarzschild
// background -- Tier 2 Phase 1 of the tidal-disruption project (see
// 06_tidal_disruption/README.md's tier table and 07_grhd/README.md).
// Radial-only flow, spherical symmetry, Schwarzschild coordinates. Same
// Valencia/HLLE/RK2 structure as Phase 0's flat-spacetime GrhdSim (see
// kernels_schwarzschild.hpp for the curved-spacetime physics: conserved
// variables, flux, and the one genuine geometric source term).
//
// Validation target: the analytic relativistic Bondi accretion solution
// (tools/bondi_analytic.py) -- initialize the grid to it and confirm the
// solver holds it steady, the same "settle then verify it's a genuine
// equilibrium" methodology 06_tidal_disruption used for its SPH star.
class SchwarzschildSim : public fw::Simulation {
public:
    SchwarzschildSim() = default;
    ~SchwarzschildSim() override;

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
    double m_rMin = 3.0;
    double m_rMax = 40.0;
    double m_dr = 0.0;
    double m_M = 1.0;
    double m_gamma = 5.0 / 3.0;
    double m_cfl = 0.2;
    double m_dt = 0.0; // = cfl*dr; coordinate signal speeds are bounded by f<1<c=1, so still a safe bound
    int m_consToPrimIters = 30;
    int m_substepsPerFrame = 1;

    std::string m_icFile;
    glm::vec3 m_ghostHi{0.0f}; // rho,v,P: fixed exterior (Dirichlet) state at r_max --
                                // the outer edge is subsonic, so a zero-gradient outflow
                                // boundary is ill-posed there (see UploadInitial).
    std::string m_title = "GRHD -- Schwarzschild radial accretion (Bondi flow)";

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
