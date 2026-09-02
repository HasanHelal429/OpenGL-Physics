#pragma once

#include "Euler2D.hpp"

#include "framework/Simulation.hpp"

#include <string>

namespace cf {

// fw::Simulation wrapper for the 2D Taylor-Green vortex: a doubly-periodic
// array of counter-rotating vortices with an exact analytic viscous-decay
// solution (incompressible NS) -- u,v decay as exp(-nu*k^2*t), kinetic
// energy (quadratic in velocity) as exp(-2*nu*k^2*t). Unlike Poiseuille
// (Phase 4's steady-state check), this validates the *transient* viscous
// decay rate, and needs no walls at all -- both directions are periodic.
// See tools/plot_taylor_green.py for the comparison this validates against.
class CompressibleSimTaylorGreen : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    Euler2D m_solver;
    double m_u0 = 0.0, m_k = 0.0, m_rho0 = 1.0, m_p0 = 1.0, m_gamma = 1.4;
    double m_dt = 0.0;
    double m_cfl = 0.4;
    int m_substepsPerFrame = 20;
    std::string m_title = "Compressible Navier-Stokes -- 2D Taylor-Green vortex decay";
};

} // namespace cf
