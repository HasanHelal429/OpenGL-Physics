#include "HFVisualizerApp.hpp"

#include "RadialFieldSampler.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace hf {

namespace {

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

// Consolas ships with Windows by default; loaded by absolute path rather
// than a vendored asset (this whole toolchain is already Windows/MinGW-
// specific -- see cpp_toolchain_setup notes). Swap for a vendored .ttf
// under assets/fonts/ later if portability off this machine matters.
const char* kFontPath = "C:\\Windows\\Fonts\\consola.ttf";

} // namespace

HFVisualizerApp::Config HFVisualizerApp::MakeConfig() {
    Config config;
    config.title = "Atomic SCF Solver (Hartree + Slater/X-alpha / LDA)";
    config.width = 1500;
    config.height = 950;
    return config;
}

HFVisualizerApp::HFVisualizerApp()
    : fw::Application(MakeConfig()),
      m_font(fw::Font::FromFile(kFontPath, 22.0f)),
      m_textRenderer(),
      m_densityView(),
      m_spectrumView(m_font, m_textRenderer),
      m_convergenceChart(m_font, m_textRenderer) {}

HFVisualizerApp::ViewportRects HFVisualizerApp::ComputeLayout() const {
    ViewportRects vp{};
    const int chartsHeight = static_cast<int>(static_cast<float>(m_height) * 0.35f);
    const int threeDHeight = m_height - chartsHeight;

    vp.threeDX = 0;
    vp.threeDY = chartsHeight;
    vp.threeDW = m_width;
    vp.threeDH = threeDHeight;

    const int halfWidth = m_width / 2;
    vp.spectrumX = 0;
    vp.spectrumY = 0;
    vp.spectrumW = halfWidth;
    vp.spectrumH = chartsHeight;

    vp.convergenceX = halfWidth;
    vp.convergenceY = 0;
    vp.convergenceW = m_width - halfWidth;
    vp.convergenceH = chartsHeight;

    return vp;
}

void HFVisualizerApp::StartRun() {
    ScfParams params;
    params.Z = m_Z;
    params.method = m_method;
    params.alpha = static_cast<double>(m_alpha);

    m_lastRenderedIteration = 0;
    m_isoThreshold = -1.0;
    m_spectrumView.Reset();
    m_convergenceChart.Reset();
    m_worker.Start(params);
}

void HFVisualizerApp::OnRender() {
    const ViewportRects vp = ComputeLayout();

    if (auto snapshot = m_worker.LatestSnapshot()) {
        if (snapshot->iteration > m_lastRenderedIteration) {
            m_lastRenderedIteration = snapshot->iteration;

            if (m_isoThreshold < 0.0) {
                m_isoThreshold = SuggestIsosurfaceThreshold(snapshot->r, snapshot->rho);
            }
            m_densityView.UpdateField(snapshot->r, snapshot->rho, m_isoThreshold);
            m_spectrumView.UpdateEnergies(snapshot->orbitalEnergies);
            m_convergenceChart.AddPoint(snapshot->iteration, snapshot->eTotal);
        }
    }

    m_densityView.Render(vp.threeDX, vp.threeDY, vp.threeDW, vp.threeDH);

    m_spectrumView.SetRegion(vp.spectrumX, vp.spectrumY, vp.spectrumW, vp.spectrumH);
    m_spectrumView.Render();

    m_convergenceChart.SetRegion(vp.convergenceX, vp.convergenceY, vp.convergenceW, vp.convergenceH);
    m_convergenceChart.Render();

    glViewport(0, 0, m_width, m_height);
}

void HFVisualizerApp::OnImGui() {
    ImGui::Begin("Controls", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    DrawControls();
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

    ImGui::Separator();

    int modeIdx = (m_densityView.Mode() == DensityViewMode::ParticleCloud) ? 0 : 1;
    if (ImGui::RadioButton("Particle cloud", modeIdx == 0)) m_densityView.SetMode(DensityViewMode::ParticleCloud);
    ImGui::SameLine();
    if (ImGui::RadioButton("Isosurface", modeIdx == 1)) m_densityView.SetMode(DensityViewMode::Isosurface);

    ImGui::BeginDisabled(m_densityView.Mode() != DensityViewMode::Isosurface);
    float logThreshold = std::log10(std::max(m_isoThreshold, 1e-12));
    if (ImGui::SliderFloat("iso threshold (log10 rho)", &logThreshold, -8.0f, 2.0f, "%.2f")) {
        m_isoThreshold = std::pow(10.0, static_cast<double>(logThreshold));
    }
    ImGui::EndDisabled();

    ImGui::Separator();

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

void HFVisualizerApp::OnMouseButton(int button, int action, int mods) {
    (void)mods;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        if (ImGui::GetIO().WantCaptureMouse) return;
        double x = 0.0, y = 0.0;
        glfwGetCursorPos(m_window, &x, &y);
        const ViewportRects vp = ComputeLayout();
        // GLFW cursor coords are window-space, top-left origin, y-down;
        // vp.threeDH is exactly the height of that same region measured
        // from the window's top, so this comparison lines up directly.
        if (y < static_cast<double>(vp.threeDH)) {
            m_orbiting = true;
            m_lastMouseX = x;
            m_lastMouseY = y;
        }
    } else if (action == GLFW_RELEASE) {
        m_orbiting = false;
    }
}

void HFVisualizerApp::OnMouseMove(double x, double y) {
    if (m_orbiting) {
        const float dx = static_cast<float>(x - m_lastMouseX);
        const float dy = static_cast<float>(y - m_lastMouseY);
        m_densityView.GetCamera().Orbit(dx, dy);
    }
    m_lastMouseX = x;
    m_lastMouseY = y;
}

void HFVisualizerApp::OnScroll(double xoffset, double yoffset) {
    (void)xoffset;
    if (!ImGui::GetIO().WantCaptureMouse) {
        m_densityView.GetCamera().Zoom(static_cast<float>(yoffset));
    }
}

} // namespace hf
