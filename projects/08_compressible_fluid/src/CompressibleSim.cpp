#include "CompressibleSim.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

namespace cf {

void CompressibleSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    const int n = deck.GetInt("grid.n", 400);
    const double length = deck.GetDouble("grid.length", 1.0);
    const double gamma = deck.GetDouble("physics.gamma", 1.4);

    m_left = Prim{deck.GetDouble("riemann.rho_l", 1.0), deck.GetDouble("riemann.u_l", 0.0),
                  deck.GetDouble("riemann.p_l", 1.0)};
    m_right = Prim{deck.GetDouble("riemann.rho_r", 0.125), deck.GetDouble("riemann.u_r", 0.0),
                   deck.GetDouble("riemann.p_r", 0.1)};
    m_x0 = deck.GetDouble("riemann.x0", 0.5) * length;

    m_solver.Init(n, 0.0, length, gamma);
    m_solver.SetRiemannIC(m_x0, m_left, m_right);

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 8);
    // Fixed dt from the INITIAL max wave speed, not recomputed every step:
    // for a single self-similar Riemann problem, the exact solution's
    // characteristic speeds (shock, contact, rarefaction fan edges) are all
    // bounded by the initial left/right states' own |u|+-c (shocks are
    // compressive -- no new speed extremum is created beyond the data).
    // This is the same simplification 07_grhd makes for its shock tube
    // (there, the bound is simply c=1); a genuinely time-varying flow
    // (Phase 3 onward) will need dt recomputed every substep instead.
    m_dt = m_cfl * m_solver.Dx() / m_solver.MaxWaveSpeed();
}

void CompressibleSim::Reset() {
    m_solver.SetRiemannIC(m_x0, m_left, m_right);
}

void CompressibleSim::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSim::Snapshot(fw::OutputWriter& writer) {
    const int n = m_solver.Count();
    std::vector<float> rho(static_cast<size_t>(n)), v(static_cast<size_t>(n)), p(static_cast<size_t>(n));
    double massTotal = 0.0, momentumTotal = 0.0, energyTotal = 0.0;
    for (int i = 0; i < n; ++i) {
        const Prim pr = m_solver.PrimAt(i);
        rho[static_cast<size_t>(i)] = static_cast<float>(pr.rho);
        v[static_cast<size_t>(i)] = static_cast<float>(pr.u);
        p[static_cast<size_t>(i)] = static_cast<float>(pr.p);
    }
    massTotal = m_solver.TotalMass();
    momentumTotal = m_solver.TotalMomentum();
    energyTotal = m_solver.TotalEnergy();

    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, 1, n);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, 1, n);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, 1, n);
    writer.WriteScalar("mass_total", massTotal);
    writer.WriteScalar("momentum_total", momentumTotal);
    writer.WriteScalar("energy_total", energyTotal);
}

fw::SimInfo CompressibleSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Count();
    info.gridNy = 1;
    info.lx = m_solver.Count() * m_solver.Dx();
    info.ly = 0.0;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "v", "p"};
    info.diagnostics = {"mass_total", "momentum_total", "energy_total"};
    return info;
}

} // namespace cf
