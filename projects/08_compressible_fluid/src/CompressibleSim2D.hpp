#pragma once

#include "Euler2D.hpp"

#include "framework/Simulation.hpp"

#include <string>

namespace cf {

// fw::Simulation wrapper around Euler2D for a 2D four-quadrant Riemann
// problem (e.g. Kurganov & Tadmor 2002's "Configuration 3", the standard
// genuinely-2D wave-interaction test -- no exact solution, but a
// well-documented reference density pattern to compare against visually,
// see tools/plot_riemann2d.py). Writes rho/u/v/p field frames plus
// mass/energy diagnostics for tools/*.py.
class CompressibleSim2D : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    Euler2D m_solver;
    Prim2D m_ne, m_nw, m_sw, m_se;
    double m_x0 = 0.5, m_y0 = 0.5;
    double m_cfl = 0.4;
    double m_dt = 0.0;
    int m_substepsPerFrame = 4;
    std::string m_title = "Compressible Euler -- 2D Riemann problem";
};

} // namespace cf
