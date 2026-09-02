#include "CompressibleSim2D.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

namespace cf {

namespace {
Prim2D ReadQuadrant(const fw::Deck& deck, const char* key) {
    const std::vector<double> v = deck.GetDoubleArray(key);
    return Prim2D{v[0], v[1], v[2], v[3]};
}
} // namespace

void CompressibleSim2D::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    const int nx = deck.GetInt("grid.nx", 200);
    const int ny = deck.GetInt("grid.ny", 200);
    const double xMin = deck.GetDouble("grid.x_min", 0.0);
    const double xMax = deck.GetDouble("grid.x_max", 1.0);
    const double yMin = deck.GetDouble("grid.y_min", 0.0);
    const double yMax = deck.GetDouble("grid.y_max", 1.0);
    const double gamma = deck.GetDouble("physics.gamma", 1.4);

    m_x0 = deck.GetDouble("riemann2d.x0", 0.5);
    m_y0 = deck.GetDouble("riemann2d.y0", 0.5);
    m_ne = ReadQuadrant(deck, "riemann2d.ne");
    m_nw = ReadQuadrant(deck, "riemann2d.nw");
    m_sw = ReadQuadrant(deck, "riemann2d.sw");
    m_se = ReadQuadrant(deck, "riemann2d.se");

    m_solver.Init(nx, ny, xMin, xMax, yMin, yMax, gamma);
    Reset();

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 4);
    // Fixed dt from the initial condition's max wave speed in each direction
    // (see CompressibleSim.cpp's comment on the 1D case) -- the standard
    // multi-D CFL restriction for a dimensionally-split scheme, combining
    // both sweep directions' stability limits.
    m_dt = m_cfl / (m_solver.MaxWaveSpeedX() / m_solver.Dx() + m_solver.MaxWaveSpeedY() / m_solver.Dy());
}

void CompressibleSim2D::Reset() {
    const double x0 = m_x0, y0 = m_y0;
    const Prim2D ne = m_ne, nw = m_nw, sw = m_sw, se = m_se;
    m_solver.SetInitialCondition([=](double x, double y) {
        if (y >= y0) return x >= x0 ? ne : nw;
        return x >= x0 ? se : sw;
    });
}

void CompressibleSim2D::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSim2D::Snapshot(fw::OutputWriter& writer) {
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    std::vector<float> rho(static_cast<size_t>(nx * ny)), u(static_cast<size_t>(nx * ny)),
        v(static_cast<size_t>(nx * ny)), p(static_cast<size_t>(nx * ny));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Prim2D pr = m_solver.PrimAt(i, j);
            const size_t idx = static_cast<size_t>(j * nx + i);
            rho[idx] = static_cast<float>(pr.rho);
            u[idx] = static_cast<float>(pr.u);
            v[idx] = static_cast<float>(pr.v);
            p[idx] = static_cast<float>(pr.p);
        }
    }
    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("u", u.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteScalar("mass_total", m_solver.TotalMass());
    writer.WriteScalar("energy_total", m_solver.TotalEnergy());
}

fw::SimInfo CompressibleSim2D::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Nx();
    info.gridNy = m_solver.Ny();
    info.lx = m_solver.Nx() * m_solver.Dx();
    info.ly = m_solver.Ny() * m_solver.Dy();
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "u", "v", "p"};
    info.diagnostics = {"mass_total", "energy_total"};
    return info;
}

} // namespace cf
