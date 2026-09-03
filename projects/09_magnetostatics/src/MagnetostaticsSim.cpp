#include "MagnetostaticsSim.hpp"

#include "Sources.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace mag {

void MagnetostaticsSim::Configure(const fw::Deck& deck) {
    m_grid.nx = deck.GetInt("grid.nx", 256);
    m_grid.ny = deck.GetInt("grid.ny", m_grid.nx);
    m_grid.lx = deck.GetDouble("grid.lx", 4.0);
    m_grid.ly = deck.GetDouble("grid.ly", m_grid.lx);
    m_mu0 = deck.GetDouble("physics.mu0", 1.0);
    m_title = deck.GetString("title", "2D Magnetostatics");

    m_solveOpt.tol = deck.GetDouble("solver.tol", 1e-9);
    m_solveOpt.maxIterations = deck.GetInt("solver.max_iterations", 200000);
    m_solveOpt.omega = deck.GetDouble("solver.omega", -1.0);
    m_mgOpt.tol = deck.GetDouble("solver.tol", 1e-9);
    m_mgOpt.maxCycles = deck.GetInt("solver.max_cycles", 100);

    m_method = deck.GetString("solver.method", "multigrid");

    const std::string bc = deck.GetString("solver.boundary", "dirichlet");
    m_solveOpt.edgeBC = (bc == "neumann") ? BC::Neumann : BC::Dirichlet;
    // The multigrid path handles only the all-edges-Dirichlet boundary.
    if (m_solveOpt.edgeBC == BC::Neumann) m_method = "rbgs";

    m_Jz = BuildCurrent(m_grid, deck);
    if (m_solveOpt.edgeBC == BC::Neumann) {
        m_fixedMask.assign(m_grid.count(), 0);
        m_fixedValues.assign(m_grid.count(), 0.0);
    } else {
        MakeEdgeDirichlet(m_grid, m_fixedMask, m_fixedValues);
    }
    SolveField();
}

void MagnetostaticsSim::SolveField() {
    std::vector<double> rhs(m_grid.count(), 0.0);
    for (int k = 0; k < m_grid.count(); ++k) rhs[k] = -m_mu0 * m_Jz[k];

    m_Az.assign(m_grid.count(), 0.0);
    if (m_method == "multigrid") {
        const MultigridResult r = SolvePoissonMultigrid(
            m_grid, rhs, m_fixedValues, m_Az, m_mgOpt);
        m_lastSolve.iterations = r.cycles;
        m_lastSolve.residual = r.residual;
        m_lastSolve.converged = r.converged;
        std::printf("[magnetostatics] multigrid: %d V-cycles (%d levels), "
                    "residual %.3e%s\n",
                    r.cycles, r.levels, r.residual,
                    r.converged ? "" : "  (NOT converged)");
    } else {
        m_lastSolve = SolvePoisson(m_grid, rhs, m_fixedMask, m_fixedValues,
                                   m_Az, m_solveOpt);
        std::printf("[magnetostatics] rbgs: %d iterations, residual %.3e%s\n",
                    m_lastSolve.iterations, m_lastSolve.residual,
                    m_lastSolve.converged ? "" : "  (NOT converged)");
    }
    CurlZ(m_grid, m_Az, m_Bx, m_By);
}

void MagnetostaticsSim::Reset() {
    SolveField();
}

void MagnetostaticsSim::Step(int /*substeps*/) {
    // Static field, no test particles yet -- nothing to advance in Phase 1.
}

void MagnetostaticsSim::Snapshot(fw::OutputWriter& writer) {
    writer.WriteField("A_z", m_Az.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
    writer.WriteField("Bx", m_Bx.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
    writer.WriteField("By", m_By.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
    writer.WriteField("Jz", m_Jz.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);

    double bmax = 0.0;
    for (int k = 0; k < m_grid.count(); ++k) {
        const double b2 = m_Bx[k] * m_Bx[k] + m_By[k] * m_By[k];
        if (b2 > bmax) bmax = b2;
    }
    writer.WriteScalar("B_max", std::sqrt(bmax));
    writer.WriteScalar("solve_iterations", m_lastSolve.iterations);
    writer.WriteScalar("solve_residual", m_lastSolve.residual);
}

fw::SimInfo MagnetostaticsSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_grid.nx;
    info.gridNy = m_grid.ny;
    info.lx = m_grid.lx;
    info.ly = m_grid.ly;
    info.dt = 0.0;
    info.substepsPerFrame = 1;
    info.frameFields = {"A_z", "Bx", "By", "Jz"};
    info.diagnostics = {"B_max", "solve_iterations", "solve_residual"};
    return info;
}

} // namespace mag
