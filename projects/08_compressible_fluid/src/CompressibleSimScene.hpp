#pragma once

#include "Euler2D.hpp"

#include "framework/Simulation.hpp"

#include <functional>
#include <string>
#include <vector>

namespace cf {

// A single, generic 2D fw::Simulation driven entirely by deck content --
// grid, physics, boundary conditions (per side, with an optional inflow
// profile), an obstacle list, an initial-condition type (+ optional
// perturbation), named probes, and a tracer toggle. Replaces
// CompressibleSim2D/CompressibleSimChannel/CompressibleSimTaylorGreen/
// CompressibleSimCylinder: every scenario those four covered is now a deck
// under this one schema (see decks/riemann2d_config3.toml,
// decks/poiseuille_channel.toml, decks/taylor_green.toml,
// decks/cylinder_re100.toml), not a new C++ class. A genuinely new SHAPE of
// initial condition, inflow profile, or obstacle still needs new C++ (a new
// named case in CompressibleSimScene.cpp's parsing) -- but every
// combination of the existing pieces (any grid + any per-side BC + any
// obstacle list + any existing IC type + optional tracer) is pure data.
//
// The 1D shock tube (CompressibleSim, Euler1D) is deliberately NOT folded
// into this schema -- boundary "sides" (plural), 2D obstacles, and probes
// are 2D-specific concepts with no natural 1D analog, and forcing them
// into a shared schema would only obscure both.
class CompressibleSimScene : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

private:
    // Precomputed once in Configure() from the deck's physical (x,y) probe
    // coordinates -- (i,j) never changes since the grid is fixed after Init().
    struct Probe {
        std::string name;
        int i = 0, j = 0;
    };

    Euler2D m_solver;
    // Captures the initial-condition type + any perturbation, fully baked
    // (all deck-read parameters closed over by value) at Configure() time,
    // so Reset() can restore the initial state without re-reading the deck
    // -- the same convention every Simulation in this project follows (see
    // e.g. MDSim.hpp's comment on Reset() not re-parsing the deck).
    std::function<Prim2D(double x, double y)> m_icFn;
    std::vector<Probe> m_probes;
    bool m_writeTracer = false;
    double m_cfl = 0.4;
    int m_substepsPerFrame = 20;
    double m_dt = 0.0;
    std::string m_title = "Compressible Navier-Stokes -- generic 2D scene";
};

} // namespace cf
