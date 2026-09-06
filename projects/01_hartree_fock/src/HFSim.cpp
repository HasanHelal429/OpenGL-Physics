#include "HFSim.hpp"

#include "Grid.hpp"
#include "Shells.hpp"
#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace hf {

void HFSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "Atomic SCF");

    m_params = ScfParams{};
    m_params.Z = deck.GetInt("atom.Z", 1);
    if (m_params.Z < 1 || m_params.Z > 103) {
        throw std::runtime_error("HFSim: atom.Z must be in 1..103");
    }

    const std::string method = deck.GetString("atom.method", "xalpha");
    if (method == "lda") {
        m_params.method = Method::Lda;
    } else if (method == "xalpha") {
        m_params.method = Method::Xalpha;
    } else {
        throw std::runtime_error("HFSim: atom.method must be \"xalpha\" or \"lda\"");
    }
    m_params.alpha = deck.GetDouble("atom.alpha", kAlphaSchwarz);

    m_params.maxIter = deck.GetInt("scf.max_iter", 200);
    m_params.mixBeta = deck.GetDouble("scf.mix", 0.3);
    m_params.tolE = deck.GetDouble("scf.tol_e", 1e-6);
    m_params.tolN = deck.GetDouble("scf.tol_n", 1e-5);

    if (deck.Has("grid.r_min") || deck.Has("grid.r_max") || deck.Has("grid.n")) {
        const LogGrid def = DefaultGrid(m_params.Z);
        const double rMin = deck.GetDouble("grid.r_min", 1e-5 / m_params.Z);
        const double rMax = deck.GetDouble("grid.r_max", 150.0);
        const int n = deck.GetInt("grid.n", 4000);
        (void)def;
        m_params.grid = MakeLogGrid(rMin, rMax, n);
    }

    m_diagNames = {"e_total", "de", "dn"};
    if (deck.Has("output.diagnostics")) {
        const auto names = deck.GetStringArray("output.diagnostics");
        if (!names.empty()) m_diagNames = names;
    }

    // Run the whole SCF, buffering every iteration's snapshot.
    m_snapshots.clear();
    m_result = RunScf(m_params, [this](const ScfSnapshot& s) { m_snapshots.push_back(s); });

    if (m_snapshots.empty()) throw std::runtime_error("HFSim: SCF produced no iterations");

    m_n = static_cast<int>(m_snapshots.front().r.size());

    // Fixed (n, l) ordering for the "eps" field: sorted by (n, l).
    m_labels.clear();
    for (const auto& [key, eps] : m_snapshots.back().orbitalEnergies) {
        (void)eps;
        m_labels.push_back(key);
    }
    std::sort(m_labels.begin(), m_labels.end());

    m_frame = 0;
}

void HFSim::Reset() { m_frame = 0; }

void HFSim::Step(int substeps) {
    m_frame = std::clamp(m_frame + substeps, 0, NumIterations() - 1);
}

void HFSim::Snapshot(fw::OutputWriter& writer) {
    const ScfSnapshot& s = m_snapshots[static_cast<size_t>(m_frame)];

    writer.WriteField("rho", s.rho.data(), fw::NpyDtype::F8, 1, m_n);
    writer.WriteField("v_eff", s.V.data(), fw::NpyDtype::F8, 1, m_n);

    std::vector<double> eps(m_labels.size(), 0.0);
    for (size_t i = 0; i < m_labels.size(); ++i) {
        auto it = s.orbitalEnergies.find(m_labels[i]);
        eps[i] = (it != s.orbitalEnergies.end()) ? it->second : std::nan("");
    }
    writer.WriteField("eps", eps.data(), fw::NpyDtype::F8, 1, static_cast<int>(eps.size()));

    if (m_frame == 0) {
        writer.WriteField("r", s.r.data(), fw::NpyDtype::F8, 1, m_n);
        std::vector<std::int32_t> nl(2 * m_labels.size());
        for (size_t i = 0; i < m_labels.size(); ++i) {
            nl[i] = m_labels[i].first;                       // n
            nl[m_labels.size() + i] = m_labels[i].second;    // l
        }
        writer.WriteField("eps_nl", nl.data(), fw::NpyDtype::I4, 2, static_cast<int>(m_labels.size()));
    }

    const double de = std::isfinite(s.dE) ? s.dE : std::nan("");
    for (const auto& name : m_diagNames) {
        if (name == "e_total") writer.WriteScalar("e_total", s.eTotal);
        else if (name == "de") writer.WriteScalar("de", de);
        else if (name == "dn") writer.WriteScalar("dn", s.dn);
    }
}

fw::SimInfo HFSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_n;
    info.gridNy = 1;
    info.lx = m_snapshots.empty() ? 0.0 : m_snapshots.front().r.back();
    info.ly = 0.0;
    info.dt = 1.0;                 // one "frame" == one SCF iteration
    info.substepsPerFrame = 1;
    info.frameFields = {"rho", "v_eff", "eps", "eps_nl", "r"};
    info.diagnostics = m_diagNames;
    return info;
}

} // namespace hf
