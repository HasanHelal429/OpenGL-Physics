#pragma once

#include "Euler2D.hpp"

#include "framework/Simulation.hpp"

#include <string>

namespace cf {

// fw::Simulation wrapper for plane Poiseuille flow: a channel of height H,
// periodic in the streamwise (x) direction, no-slip walls at y=0 and y=H,
// started from rest and driven by a constant body force, evolved with
// Euler2D's viscous (Newtonian shear stress + Fourier conduction) terms
// until it approaches the analytic steady parabolic profile
// u(y) = (f/(2*mu)) * y*(H-y) -- see tools/plot_poiseuille.py for the
// comparison this validates against.
class CompressibleSimChannel : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    Euler2D m_solver;
    Prim2D m_rest;
    double m_dt = 0.0;
    double m_cfl = 0.4;
    int m_substepsPerFrame = 50;
    std::string m_title = "Compressible Navier-Stokes -- plane Poiseuille flow";
};

} // namespace cf
