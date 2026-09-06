#pragma once

#include "ScfSolver.hpp"
#include "framework/Simulation.hpp"

#include <string>
#include <utility>
#include <vector>

namespace hf {

// Deck-driven headless wrapper around the atomic SCF solver, implementing
// fw::Simulation so a batch run produces the same per-iteration frame stream
// (rho, V_eff, orbital energies, diagnostics) that the ImGui HFVisualizerApp
// shows live -- i.e. an SCF-convergence movie, matching the Python solver's
// scf_movie.py.
//
// The SCF is not a time-stepping process, so Configure() runs the whole solve
// (single-digit seconds in Release for most atoms) and buffers every
// ScfSnapshot; Step()/Snapshot() then replay that history one iteration per
// "frame". main.cpp drives a custom frame loop of length NumIterations()
// rather than fw::RunHeadless (whose fixed [time].frames count does not fit a
// naturally-terminating solve). No GL context is required.
class HFSim : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    // Number of SCF iterations recorded (= number of frames to write).
    int NumIterations() const { return static_cast<int>(m_snapshots.size()); }

    // Final converged results, for main.cpp to write orbitals_final.npy and
    // print a summary line.
    const ScfResult& Result() const { return m_result; }

    // (n, l) shells in the fixed order the "eps" frame field uses.
    const std::vector<ShellKey>& OrbitalLabels() const { return m_labels; }

private:
    ScfParams m_params;
    std::string m_title;
    std::vector<ScfSnapshot> m_snapshots;
    ScfResult m_result;
    std::vector<ShellKey> m_labels;      // sorted (n, l)
    std::vector<std::string> m_diagNames;
    int m_frame = 0;
    int m_n = 0;                          // radial grid points
};

} // namespace hf
