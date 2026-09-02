#include "CompressibleSimChannel.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <algorithm>
#include <cmath>

namespace cf {

void CompressibleSimChannel::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    const int nx = deck.GetInt("grid.nx", 8);
    const int ny = deck.GetInt("grid.ny", 50);
    const double lx = deck.GetDouble("grid.lx", 0.1); // streamwise: nothing varies here, keep it small/cheap
    const double h = deck.GetDouble("grid.height", 1.0);
    const double gamma = deck.GetDouble("physics.gamma", 1.4);
    const double mu = deck.GetDouble("physics.mu", 0.05);
    const double conductivity = deck.GetDouble("physics.conductivity", 0.0);

    m_rest = Prim2D{deck.GetDouble("channel.rho0", 1.0), 0.0, 0.0, deck.GetDouble("channel.p0", 1.0)};
    const double bodyForceX = deck.GetDouble("channel.body_force_x", 0.05);

    m_solver.Init(nx, ny, 0.0, lx, 0.0, h, gamma);
    m_solver.SetBoundaryConditions(WallBC::Periodic, WallBC::NoSlipReflective);
    m_solver.SetViscosity(mu, conductivity);
    m_solver.SetBodyForceX(bodyForceX);
    Reset();

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 50);
    // Fixed dt from the rest state's sound speed: this flow stays subsonic
    // and laminar by construction (mu/body_force_x are chosen so u_max <<
    // sound speed -- see the deck's comment), so the sound speed barely
    // changes as the flow spins up from rest, unlike Sod/Config3 where the
    // "bounded by initial data" argument is about genuine shock/rarefaction
    // speeds. Euler2D::Step separately sub-cycles the viscous diffusion
    // update against its own (typically tighter) stability limit.
    m_dt = m_cfl / (m_solver.MaxWaveSpeedX() / m_solver.Dx() + m_solver.MaxWaveSpeedY() / m_solver.Dy());
}

void CompressibleSimChannel::Reset() {
    const Prim2D rest = m_rest;
    m_solver.SetInitialCondition([=](double, double) { return rest; });
}

void CompressibleSimChannel::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSimChannel::Snapshot(fw::OutputWriter& writer) {
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    std::vector<float> rho(static_cast<size_t>(nx * ny)), u(static_cast<size_t>(nx * ny)),
        v(static_cast<size_t>(nx * ny)), p(static_cast<size_t>(nx * ny));
    double uMax = 0.0;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Prim2D pr = m_solver.PrimAt(i, j);
            const size_t idx = static_cast<size_t>(j * nx + i);
            rho[idx] = static_cast<float>(pr.rho);
            u[idx] = static_cast<float>(pr.u);
            v[idx] = static_cast<float>(pr.v);
            p[idx] = static_cast<float>(pr.p);
            uMax = std::max(uMax, pr.u);
        }
    }
    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("u", u.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteScalar("mass_total", m_solver.TotalMass());
    writer.WriteScalar("energy_total", m_solver.TotalEnergy());
    writer.WriteScalar("u_max", uMax);
}

fw::SimInfo CompressibleSimChannel::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Nx();
    info.gridNy = m_solver.Ny();
    info.lx = m_solver.Nx() * m_solver.Dx();
    info.ly = m_solver.Ny() * m_solver.Dy();
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "u", "v", "p"};
    info.diagnostics = {"mass_total", "energy_total", "u_max"};
    return info;
}

} // namespace cf
