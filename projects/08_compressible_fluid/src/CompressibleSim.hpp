#pragma once

#include "Euler1D.hpp"

#include "framework/Simulation.hpp"

#include <string>

namespace cf {

// fw::Simulation wrapper around Euler1D: one input deck (grid, gas gamma,
// left/right Riemann state, CFL number) fully specifies a reproducible
// batch run, driven by fw::RunHeadless -- writes rho/v/p field frames plus
// mass/momentum/energy diagnostics for tools/plot_shocktube.py.
class CompressibleSim : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    Euler1D m_solver;
    Prim m_left, m_right;
    double m_x0 = 0.5;
    double m_cfl = 0.4;
    double m_dt = 0.0;
    int m_substepsPerFrame = 8;
    std::string m_title = "Compressible Euler -- 1D shock tube";
};

} // namespace cf
