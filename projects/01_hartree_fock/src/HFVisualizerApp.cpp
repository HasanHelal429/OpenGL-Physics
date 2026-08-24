#include "HFVisualizerApp.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numbers>

namespace hf {

namespace {
constexpr double kPi = std::numbers::pi;
constexpr double kRMinVisual = 0.02;
constexpr double kRMaxVisual = 6.0;

const char* StateLabel(ScfWorker::State state) {
    switch (state) {
        case ScfWorker::State::Idle: return "Idle";
        case ScfWorker::State::Running: return "Running";
        case ScfWorker::State::Converged: return "Converged";
        case ScfWorker::State::NotConverged: return "Stopped (not converged)";
        case ScfWorker::State::Cancelled: return "Cancelled";
        case ScfWorker::State::Failed: return "Failed";
    }
    return "?";
}

// Filters (r, y) pairs to the visualization window, matching the Python
// scf_movie.py mask -- core shells sit at r~1/Z, valence at r~few Bohr, so a
// linear/unfiltered axis would squash the core peak into invisible width.
void MaskedSeries(const std::vector<double>& r, const std::vector<double>& y, std::vector<double>& xOut,
                   std::vector<double>& yOut) {
    xOut.clear();
    yOut.clear();
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] > kRMinVisual && r[i] < kRMaxVisual) {
            xOut.push_back(r[i]);
            yOut.push_back(y[i]);
        }
    }
}

} // namespace

HFVisualizerApp::Config HFVisualizerApp::MakeConfig() {
    Config config;
    config.title = "Atomic SCF Solver (Hartree + Slater/X-alpha / LDA)";
    config.width = 1400;
    config.height = 900;
    return config;
}

HFVisualizerApp::HFVisualizerApp() : fw::Application(MakeConfig()) {}

void HFVisualizerApp::OnStart() {
    ImPlot::CreateContext();
}

void HFVisualizerApp::OnShutdown() {
    ImPlot::DestroyContext();
}

void HFVisualizerApp::StartRun() {
    ScfParams params;
    params.Z = m_Z;
    params.method = m_method;
    params.alpha = static_cast<double>(m_alpha);
    m_energyHistory.clear();
    m_lastRenderedIteration = 0;
    m_worker.Start(params);
}

void HFVisualizerApp::OnImGui() {
    ImGui::Begin("Atomic SCF Solver", nullptr, ImGuiWindowFlags_NoCollapse);
    DrawControls();
    ImGui::Separator();
    DrawPlots();
    ImGui::End();
}

void HFVisualizerApp::DrawControls() {
    ImGui::SliderInt("Atomic number Z", &m_Z, 1, 103, "%d");
    ImGui::SameLine();
    ImGui::Text("(%s)", ElementSymbol(m_Z));

    int methodIdx = (m_method == Method::Xalpha) ? 0 : 1;
    if (ImGui::RadioButton("X-alpha (Slater, tunable)", methodIdx == 0)) m_method = Method::Xalpha;
    ImGui::SameLine();
    if (ImGui::RadioButton("LDA (Kohn-Sham, PZ81)", methodIdx == 1)) m_method = Method::Lda;

    ImGui::BeginDisabled(m_method == Method::Lda);
    ImGui::SliderFloat("alpha", &m_alpha, 0.5f, 1.2f, "%.3f");
    ImGui::EndDisabled();

    const ScfWorker::State state = m_worker.GetState();
    const bool running = (state == ScfWorker::State::Running);

    ImGui::BeginDisabled(running);
    if (ImGui::Button("Run SCF")) StartRun();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::BeginDisabled(!running);
    if (ImGui::Button("Cancel")) m_worker.RequestCancel();
    ImGui::EndDisabled();

    ImGui::SameLine();
    ImGui::Text("Status: %s", StateLabel(state));

    if (const ScfResult* result = m_worker.Result()) {
        ImGui::Text("Configuration: %s", FormatConfiguration(result->config).c_str());
        ImGui::Text("Iterations: %d   E_total = %.6f Ha", result->iterations, result->eTotal);
    } else if (auto snapshot = m_worker.LatestSnapshot()) {
        ImGui::Text("Iteration %d   E_total = %.6f Ha   dE = %.2e   dn = %.2e", snapshot->iteration, snapshot->eTotal,
                    snapshot->dE, snapshot->dn);
    } else {
        ImGui::TextDisabled("No run yet.");
    }
}

void HFVisualizerApp::DrawPlots() {
    auto snapshot = m_worker.LatestSnapshot();
    if (snapshot && snapshot->iteration > m_lastRenderedIteration) {
        m_energyHistory.push_back(snapshot->eTotal);
        m_lastRenderedIteration = snapshot->iteration;
    }

    const ImVec2 plotSize(-1.0f, 280.0f);
    const float columnWidth = ImGui::GetContentRegionAvail().x * 0.5f;

    ImGui::Columns(2, "hf_plot_columns", false);
    ImGui::SetColumnWidth(0, columnWidth);

    // Panel 1: electron density 4*pi*r^2*rho(r).
    if (ImPlot::BeginPlot("Electron density", plotSize)) {
        ImPlot::SetupAxis(ImAxis_X1, "r (Bohr)");
        ImPlot::SetupAxis(ImAxis_Y1, "4*pi*r^2*rho(r)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisLimits(ImAxis_X1, kRMinVisual, kRMaxVisual, ImPlotCond_Once);
        if (snapshot) {
            std::vector<double> xs, density(snapshot->r.size());
            for (size_t i = 0; i < snapshot->r.size(); ++i) {
                density[i] = 4.0 * kPi * snapshot->r[i] * snapshot->r[i] * snapshot->rho[i];
            }
            std::vector<double> xMasked, yMasked;
            MaskedSeries(snapshot->r, density, xMasked, yMasked);
            if (!xMasked.empty()) ImPlot::PlotLine("density", xMasked.data(), yMasked.data(), static_cast<int>(xMasked.size()));
        }
        ImPlot::EndPlot();
    }

    ImGui::NextColumn();

    // Panel 2: effective potential r*V_eff(r), with a -Z bare-nucleus reference.
    if (ImPlot::BeginPlot("Effective potential", plotSize)) {
        ImPlot::SetupAxis(ImAxis_X1, "r (Bohr)");
        ImPlot::SetupAxis(ImAxis_Y1, "r * V_eff(r)");
        ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
        ImPlot::SetupAxisLimits(ImAxis_X1, kRMinVisual, kRMaxVisual, ImPlotCond_Once);
        if (snapshot) {
            std::vector<double> rV(snapshot->r.size());
            for (size_t i = 0; i < snapshot->r.size(); ++i) rV[i] = snapshot->r[i] * snapshot->V[i];
            std::vector<double> xMasked, yMasked;
            MaskedSeries(snapshot->r, rV, xMasked, yMasked);
            if (!xMasked.empty()) ImPlot::PlotLine("r*V_eff", xMasked.data(), yMasked.data(), static_cast<int>(xMasked.size()));

            const double refX[2] = {kRMinVisual, kRMaxVisual};
            const double refY[2] = {static_cast<double>(-m_Z), static_cast<double>(-m_Z)};
            ImPlot::PlotLine("-Z (bare nucleus)", refX, refY, 2);
        }
        ImPlot::EndPlot();
    }

    ImGui::NextColumn();

    // Panel 3: occupied orbital energy levels, grouped by angular momentum l.
    if (ImPlot::BeginPlot("Orbital energy levels", plotSize)) {
        ImPlot::SetupAxis(ImAxis_X1, "shell (l)");
        ImPlot::SetupAxis(ImAxis_Y1, "orbital energy (Ha, symlog)");
        ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_SymLog);
        static const double ticks[4] = {0, 1, 2, 3};
        static const char* tickLabels[4] = {"s", "p", "d", "f"};
        ImPlot::SetupAxisTicks(ImAxis_X1, ticks, 4, tickLabels);
        ImPlot::SetupAxisLimits(ImAxis_X1, -0.5, 3.5, ImPlotCond_Once);
        if (snapshot) {
            std::vector<double> xs[4], ys[4];
            for (const auto& [nl, eps] : snapshot->orbitalEnergies) {
                const int l = nl.second;
                if (l >= 0 && l < 4) {
                    xs[l].push_back(static_cast<double>(l));
                    ys[l].push_back(eps);
                }
            }
            static const char* names[4] = {"s levels", "p levels", "d levels", "f levels"};
            for (int l = 0; l < 4; ++l) {
                if (!xs[l].empty()) {
                    ImPlotSpec spec;
                    spec.Marker = ImPlotMarker_Square;
                    spec.MarkerSize = 8.0f;
                    ImPlot::PlotScatter(names[l], xs[l].data(), ys[l].data(), static_cast<int>(xs[l].size()), spec);
                }
            }
        }
        ImPlot::EndPlot();
    }

    ImGui::NextColumn();

    // Panel 4: total-energy convergence trace.
    if (ImPlot::BeginPlot("Total energy convergence", plotSize)) {
        ImPlot::SetupAxis(ImAxis_X1, "SCF iteration");
        ImPlot::SetupAxis(ImAxis_Y1, "E_total (Ha)");
        if (!m_energyHistory.empty()) {
            std::vector<double> iters(m_energyHistory.size());
            for (size_t i = 0; i < iters.size(); ++i) iters[i] = static_cast<double>(i + 1);
            ImPlot::PlotLine("E_total", iters.data(), m_energyHistory.data(), static_cast<int>(iters.size()));

            ImPlotSpec markerSpec;
            markerSpec.Marker = ImPlotMarker_Circle;
            markerSpec.MarkerSize = 6.0f;
            const double lastIter = iters.back();
            const double lastEnergy = m_energyHistory.back();
            ImPlot::PlotScatter("current", &lastIter, &lastEnergy, 1, markerSpec);
        }
        ImPlot::EndPlot();
    }

    ImGui::Columns(1);
}

} // namespace hf
