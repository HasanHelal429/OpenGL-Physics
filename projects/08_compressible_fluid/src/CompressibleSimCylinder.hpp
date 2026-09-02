#pragma once

#include "Euler2D.hpp"

#include "framework/Simulation.hpp"

#include <string>

namespace cf {

// fw::Simulation wrapper for flow past a circular cylinder in a channel --
// the classic vortex-shedding validation, mirroring the geometry/BCs of
// ../../../../Physics Simulations/Fluid Mechanics/MAC_Grid_Solver's
// Cylinder_Vortex_Shedding.ipynb (D=1, Re=100, U_inf=1, blockage=0.125,
// inflow left / outflow right / free-slip top+bottom, a circular obstacle
// via Euler2D::SetObstacleMask): impulsively started uniform inflow,
// vortex shedding develops downstream of the cylinder after a transient.
// A velocity probe a few diameters downstream (see tools/plot_strouhal.py)
// measures the shedding frequency via FFT, giving the Strouhal number
// St=f*D/U_inf to compare against Roshko's correlation and literature.
class CompressibleSimCylinder : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    Euler2D m_solver;
    Prim2D m_inflow;
    double m_diameter = 1.0;
    double m_cylX = 0.0, m_cylY = 0.0, m_perturbAmplitude = 0.0;
    int m_probeI = 0, m_probeJ = 0;
    double m_dt = 0.0;
    double m_cfl = 0.3;
    int m_substepsPerFrame = 20;
    std::string m_title = "Compressible Navier-Stokes -- cylinder vortex shedding";
};

} // namespace cf
