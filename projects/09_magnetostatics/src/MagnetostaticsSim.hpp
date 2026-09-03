#pragma once

#include "FieldSolver.hpp"
#include "Grid.hpp"
#include "Multigrid.hpp"

#include "framework/Simulation.hpp"

#include <string>
#include <vector>

namespace fw { class Deck; }

namespace mag {

// 2D magnetostatics on the Simulation/Deck/headless model.
//
// Phase 1: solve  laplacian(A_z) = -mu0 * J_z  by iterative relaxation
// (FieldSolver), then B = curl(A_z zhat). The problem is static, so there is
// no time evolution -- Configure() does the solve, Step() is a no-op, and a
// headless run writes a single frame (A_z, Bx, By, Jz).
//
// Later phases add the GPU multigrid solver + interactive view (Phase 2),
// 3D Biot-Savart coils (Phase 3), and a Boris test-particle pusher (Phase 4).
class MagnetostaticsSim : public fw::Simulation {
public:
    MagnetostaticsSim() = default;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    // Read-only access for --selftest and unit checks.
    const Grid& GridInfo() const { return m_grid; }
    const std::vector<double>& Az() const { return m_Az; }
    const std::vector<double>& Bx() const { return m_Bx; }
    const std::vector<double>& By() const { return m_By; }

private:
    void SolveField();

    Grid m_grid;
    double m_mu0 = 1.0;
    std::string m_title = "2D Magnetostatics";
    std::string m_method = "multigrid";   // "multigrid" | "rbgs"
    SolveOptions m_solveOpt;
    MultigridOptions m_mgOpt;

    std::vector<double> m_Jz;
    std::vector<double> m_Az;
    std::vector<double> m_Bx, m_By;
    std::vector<unsigned char> m_fixedMask;
    std::vector<double> m_fixedValues;

    SolveResult m_lastSolve;
};

} // namespace mag
