#include "CompressibleSimTaylorGreen.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>

namespace cf {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

void CompressibleSimTaylorGreen::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    const int n = deck.GetInt("grid.n", 64);
    const double length = deck.GetDouble("grid.length", 1.0);
    m_gamma = deck.GetDouble("physics.gamma", 1.4);
    m_rho0 = deck.GetDouble("physics.rho0", 1.0);
    m_p0 = deck.GetDouble("physics.p0", 1.0);
    const double mu = deck.GetDouble("physics.mu", 0.01);
    const double conductivity = deck.GetDouble("physics.conductivity", 0.0);

    m_u0 = deck.GetDouble("taylor_green.u0", 0.1);
    const int waveNumberMultiplier = deck.GetInt("taylor_green.wavenumber_multiplier", 1);
    m_k = 2.0 * kPi * waveNumberMultiplier / length;

    m_solver.Init(n, n, 0.0, length, 0.0, length, m_gamma);
    m_solver.SetBoundaryConditions(WallBC::Periodic, WallBC::Periodic, WallBC::Periodic, WallBC::Periodic);
    m_solver.SetViscosity(mu, conductivity);
    Reset();

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 20);
    // Fixed dt from the rest-frame sound speed, same reasoning as
    // CompressibleSimChannel: u0 is chosen well below sound speed (low
    // Mach), so the sound speed barely changes as the vortex decays.
    m_dt = m_cfl / (m_solver.MaxWaveSpeedX() / m_solver.Dx() + m_solver.MaxWaveSpeedY() / m_solver.Dy());
}

void CompressibleSimTaylorGreen::Reset() {
    const double u0 = m_u0, k = m_k, rho0 = m_rho0, p0 = m_p0;
    // Classic 2D Taylor-Green vortex (e.g. Taylor & Green 1937's original
    // decaying-turbulence solution restricted to a single mode): density
    // held uniform at t=0 (the compressible correction to it is a smaller,
    // dynamically-generated effect at this Mach number, not part of the
    // standard IC), pressure carries the leading-order compressible
    // adjustment consistent with the velocity field via the incompressible
    // momentum balance at t=0.
    m_solver.SetInitialCondition([=](double x, double y) {
        const double u = u0 * std::cos(k * x) * std::sin(k * y);
        const double v = -u0 * std::sin(k * x) * std::cos(k * y);
        const double p = p0 - 0.25 * rho0 * u0 * u0 * (std::cos(2.0 * k * x) + std::cos(2.0 * k * y));
        return Prim2D{rho0, u, v, p};
    });
}

void CompressibleSimTaylorGreen::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSimTaylorGreen::Snapshot(fw::OutputWriter& writer) {
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    std::vector<float> rho(static_cast<size_t>(nx * ny)), u(static_cast<size_t>(nx * ny)),
        v(static_cast<size_t>(nx * ny)), p(static_cast<size_t>(nx * ny));
    double kineticEnergy = 0.0;
    const double cellArea = m_solver.Dx() * m_solver.Dy();
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Prim2D pr = m_solver.PrimAt(i, j);
            const size_t idx = static_cast<size_t>(j * nx + i);
            rho[idx] = static_cast<float>(pr.rho);
            u[idx] = static_cast<float>(pr.u);
            v[idx] = static_cast<float>(pr.v);
            p[idx] = static_cast<float>(pr.p);
            kineticEnergy += 0.5 * pr.rho * (pr.u * pr.u + pr.v * pr.v) * cellArea;
        }
    }
    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("u", u.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteScalar("mass_total", m_solver.TotalMass());
    writer.WriteScalar("energy_total", m_solver.TotalEnergy());
    writer.WriteScalar("kinetic_energy", kineticEnergy);
}

fw::SimInfo CompressibleSimTaylorGreen::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Nx();
    info.gridNy = m_solver.Ny();
    info.lx = m_solver.Nx() * m_solver.Dx();
    info.ly = m_solver.Ny() * m_solver.Dy();
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "u", "v", "p"};
    info.diagnostics = {"mass_total", "energy_total", "kinetic_energy"};
    return info;
}

} // namespace cf
