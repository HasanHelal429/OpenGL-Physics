#include "NBodyApp.hpp"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace nbody {

namespace {

// Consolas ships with Windows by default; loaded by absolute path rather
// than a vendored asset, matching 01_hartree_fock's approach on this
// Windows/MinGW-specific toolchain.
const char* kFontPath = "C:\\Windows\\Fonts\\consola.ttf";

// How often (in fixed-update ticks) to run the O(N^2) energy/L diagnostics
// while tracking is enabled -- keeps the conservation chart live without
// paying the full O(N^2) cost every tick at large N.
constexpr int kDiagnosticsEveryNTicks = 6;

constexpr ScenarioType kScenarios[] = {ScenarioType::TwoBodyKepler, ScenarioType::LagrangeTriangle,
                                        ScenarioType::Cluster, ScenarioType::RotatingDisk};

} // namespace

NBodyApp::Config NBodyApp::MakeConfig() {
    Config config;
    config.title = "N-Body Gravity (Direct / Barnes-Hut)";
    config.width = 1500;
    config.height = 950;
    config.fixedTimestep = 1.0 / 60.0;
    return config;
}

NBodyApp::NBodyApp()
    : fw::Application(MakeConfig()),
      m_font(fw::Font::FromFile(kFontPath, 20.0f)),
      m_textRenderer(),
      m_driftChart(m_font, m_textRenderer) {
    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = 4.0f;
    m_camera.pitch = 20.0f;
    m_camera.minDistance = 0.3f;
    m_camera.maxDistance = 100.0f;
}

void NBodyApp::OnStart() { LoadScenario(); }

void NBodyApp::LoadScenario() {
    ScenarioResult result = BuildScenario(m_scenarioType, m_scenarioParams);

    m_G = result.G;
    m_softening = result.softening;
    m_dt = result.suggestedDt;
    m_theta = result.suggestedTheta;

    m_system.SetParticles(std::move(result.pos), std::move(result.vel), std::move(result.mass));
    m_system.PrimeAccelerations(m_G, m_softening, m_solver, m_theta);

    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = result.cameraDistance;
    m_camera.pitch = (m_scenarioType == ScenarioType::RotatingDisk) ? 55.0f : 20.0f; // look down on disks

    m_simTime = 0.0;
    m_stepCount = 0;
    m_diagnosticsTickCounter = 0;
    m_energy0 = m_system.TotalEnergy(m_G, m_softening);
    m_angularMomentum0 = m_system.AngularMomentum();
    m_lastEnergyDriftPct = 0.0;
    m_lastLDriftPct = 0.0;
    m_driftChart.Reset();
}

void NBodyApp::OnUpdate(double dt) {
    (void)dt;
    m_didFixedUpdateThisFrame = false;
}

void NBodyApp::OnFixedUpdate(double fixedDt) {
    if (!m_running && !m_stepOnce) return;

    // Application::Run()'s fixed-timestep loop decides how many times to
    // call OnFixedUpdate this frame from the *previous* frame's measured
    // debt, before it has any idea these calls are slow -- at large N a
    // single physics step can itself take longer than fixedDt, and a naive
    // wall-clock "skip if too soon" gate doesn't help, because a call that
    // always does real (slow) work always makes enough real time pass to
    // satisfy such a gate; only counting calls actually caps them. So this
    // hard-caps real physics work to once per *rendered* frame (the flag is
    // reset in OnUpdate, which Application::Run() calls exactly once before
    // the catch-up loop) -- any further catch-up iterations this frame
    // become a single bool check, and leftover simulated-time debt simply
    // isn't repaid this frame, rather than the loop hammering through its
    // full nominal catch-up count (and the CPU) before a single frame ever
    // renders. An explicit single-step (paused, "Step" button) always runs
    // regardless.
    if (!m_stepOnce) {
        if (m_didFixedUpdateThisFrame) return;
        m_didFixedUpdateThisFrame = true;
    }

    const int maxSteps = m_stepOnce ? 1 : m_substeps;
    m_stepOnce = false;

    // "Substeps" lets a cheap (small-N) simulation advance faster than one
    // step per rendered frame. At large N a single step can itself exceed
    // fixedDt, and Application::Run()'s fixed-timestep loop has no cap on
    // how many (now-expensive) OnFixedUpdate calls it fires back-to-back to
    // catch up -- multiplying that by a fixed substep count compounds into
    // multi-second per-frame stalls well before rendering ever runs. So
    // this always completes at least one step, then bails out of the
    // substep loop once this call has already spent roughly its fixedDt
    // budget, leaving any remaining substeps for later ticks instead of
    // trying to force them all in now. That keeps each OnFixedUpdate call's
    // cost close to fixedDt regardless of N, which is what keeps the
    // catch-up loop from spiraling.
    const auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    for (; steps < maxSteps; ++steps) {
        m_system.Step(m_dt, m_G, m_softening, m_solver, m_theta);
        m_simTime += m_dt;
        ++m_stepCount;

        const double elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (elapsedMs > fixedDt * 1000.0) {
            ++steps;
            break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    m_lastStepMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(steps);

    // The O(N^2) energy diagnostic runs on a background thread (see
    // DiagnosticsWorker) so it never stalls the physics/render loop, no
    // matter how large N gets. RequestCompute is a cheap O(N) state copy
    // that silently no-ops if the previous computation hasn't finished yet.
    ++m_diagnosticsTickCounter;
    if (m_trackDiagnostics && m_diagnosticsTickCounter >= kDiagnosticsEveryNTicks) {
        m_diagnosticsTickCounter = 0;
        m_diagnosticsWorker.RequestCompute(m_simTime, m_system.Positions(), m_system.Velocities(), m_system.Masses(),
                                            m_G, m_softening);
    }

    DiagnosticsWorker::Result diag;
    if (m_diagnosticsWorker.PollResult(diag)) {
        const double L0mag = glm::length(m_angularMomentum0);
        const double Ldeviation = glm::length(diag.angularMomentum - m_angularMomentum0);

        m_lastEnergyDriftPct = (m_energy0 != 0.0) ? 100.0 * (diag.energy - m_energy0) / std::abs(m_energy0) : 0.0;
        m_lastLDriftPct = (L0mag > 1e-12) ? 100.0 * Ldeviation / L0mag : 100.0 * glm::length(diag.angularMomentum);

        m_driftChart.AddPoint(diag.simTime, m_lastEnergyDriftPct, m_lastLDriftPct);
    }
}

NBodyApp::ViewportRects NBodyApp::ComputeLayout() const {
    ViewportRects vp{};
    const int chartHeight = static_cast<int>(static_cast<float>(m_height) * 0.22f);
    const int viewHeight = m_height - chartHeight;

    vp.viewX = 0;
    vp.viewY = chartHeight;
    vp.viewW = m_width;
    vp.viewH = viewHeight;

    vp.chartX = 0;
    vp.chartY = 0;
    vp.chartW = m_width;
    vp.chartH = chartHeight;
    return vp;
}

void NBodyApp::UpdateParticleInstances() {
    const std::vector<glm::dvec3>& pos = m_system.Positions();
    const std::vector<glm::dvec3>& vel = m_system.Velocities();
    const std::vector<double>& mass = m_system.Masses();
    const size_t n = pos.size();

    double maxSpeed = 1e-9;
    double maxMass = 1e-9;
    for (size_t i = 0; i < n; ++i) {
        maxSpeed = std::max(maxSpeed, glm::length(vel[i]));
        maxMass = std::max(maxMass, mass[i]);
    }

    const float baseSize = static_cast<float>(std::clamp(0.6 / std::sqrt(std::max<double>(1.0, static_cast<double>(n))), 0.006, 0.08));

    m_particleScratch.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float t = static_cast<float>(std::clamp(glm::length(vel[i]) / maxSpeed, 0.0, 1.0));
        glm::vec3 color;
        if (t < 0.5f) {
            color = glm::mix(glm::vec3(0.25f, 0.45f, 1.0f), glm::vec3(1.0f, 1.0f, 1.0f), t * 2.0f);
        } else {
            color = glm::mix(glm::vec3(1.0f, 1.0f, 1.0f), glm::vec3(1.0f, 0.55f, 0.15f), (t - 0.5f) * 2.0f);
        }

        fw::ParticleInstance& inst = m_particleScratch[i];
        inst.position = glm::vec3(pos[i]);
        inst.color = glm::vec4(color, 0.9f);
        inst.size = baseSize * static_cast<float>(std::cbrt(mass[i] / maxMass) * 0.6 + 0.4);
    }
    m_particles.SetParticles(m_particleScratch);
}

void NBodyApp::OnRender() {
    const ViewportRects vp = ComputeLayout();

    UpdateParticleInstances();

    glViewport(vp.viewX, vp.viewY, vp.viewW, vp.viewH);
    glEnable(GL_DEPTH_TEST);
    const float aspect = static_cast<float>(vp.viewW) / static_cast<float>(std::max(vp.viewH, 1));
    const glm::mat4 view = m_camera.ViewMatrix();
    const glm::mat4 projection = m_camera.ProjectionMatrix(aspect);
    m_particles.Draw(view, projection, static_cast<float>(vp.viewH));

    m_driftChart.SetRegion(vp.chartX, vp.chartY, vp.chartW, vp.chartH);
    m_driftChart.Render();

    glViewport(0, 0, m_width, m_height);
}

void NBodyApp::OnImGui() {
    ImGui::Begin("N-Body Gravity Controls", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    DrawControls();
    ImGui::End();
}

void NBodyApp::DrawControls() {
    if (ImGui::BeginCombo("Scenario", ScenarioName(m_scenarioType))) {
        for (ScenarioType s : kScenarios) {
            const bool selected = (s == m_scenarioType);
            if (ImGui::Selectable(ScenarioName(s), selected)) m_scenarioType = s;
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    const bool needsN = (m_scenarioType == ScenarioType::Cluster || m_scenarioType == ScenarioType::RotatingDisk);
    ImGui::BeginDisabled(!needsN);
    ImGui::SliderInt("Particle count", &m_scenarioParams.n, 10, 60000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();

    ImGui::BeginDisabled(m_scenarioType != ScenarioType::RotatingDisk);
    float f = static_cast<float>(m_scenarioParams.diskRotationFraction);
    if (ImGui::SliderFloat("Rotational support f", &f, 0.0f, 1.2f, "%.2f")) {
        m_scenarioParams.diskRotationFraction = static_cast<double>(f);
    }
    ImGui::TextDisabled("f=0 cold collapse, f=1 ~circular orbits");
    ImGui::EndDisabled();

    if (ImGui::Button("Load / Reset Scenario")) LoadScenario();

    ImGui::Separator();

    int solverIdx = (m_solver == SolverType::Direct) ? 0 : 1;
    if (ImGui::RadioButton("Direct O(N^2)", solverIdx == 0)) m_solver = SolverType::Direct;
    ImGui::SameLine();
    if (ImGui::RadioButton("Barnes-Hut O(N logN)", solverIdx == 1)) m_solver = SolverType::BarnesHut;

    ImGui::BeginDisabled(m_solver != SolverType::BarnesHut);
    float theta = static_cast<float>(m_theta);
    if (ImGui::SliderFloat("theta (opening angle)", &theta, 0.1f, 1.5f, "%.2f")) m_theta = static_cast<double>(theta);
    ImGui::EndDisabled();

    ImGui::Separator();

    float g = static_cast<float>(m_G);
    if (ImGui::SliderFloat("G", &g, 0.0f, 5.0f, "%.3f")) m_G = static_cast<double>(g);

    float softening = static_cast<float>(m_softening);
    if (ImGui::SliderFloat("softening", &softening, 0.0001f, 0.5f, "%.4f", ImGuiSliderFlags_Logarithmic))
        m_softening = static_cast<double>(softening);

    float dt = static_cast<float>(m_dt);
    if (ImGui::SliderFloat("dt", &dt, 1e-5f, 0.05f, "%.5f", ImGuiSliderFlags_Logarithmic)) m_dt = static_cast<double>(dt);

    ImGui::SliderInt("Substeps / frame", &m_substeps, 1, 30);

    ImGui::Separator();

    if (ImGui::Button(m_running ? "Pause" : "Play")) m_running = !m_running;
    ImGui::SameLine();
    ImGui::BeginDisabled(m_running);
    if (ImGui::Button("Step")) m_stepOnce = true;
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) LoadScenario();

    ImGui::Checkbox("Track energy/L drift (O(N^2), background thread)", &m_trackDiagnostics);

    ImGui::Separator();
    ImGui::Text("N = %zu   sim t = %.4f   steps = %ld", m_system.Count(), m_simTime, m_stepCount);
    ImGui::Text("Solver step: %.3f ms/step", m_lastStepMs);
    ImGui::Text("Energy drift: %+.3f%%   |L| drift: %.3f%%  %s", m_lastEnergyDriftPct, m_lastLDriftPct,
                m_diagnosticsWorker.Busy() ? "(computing...)" : "");
    ImGui::Text("FPS: %.1f", static_cast<double>(ImGui::GetIO().Framerate));
}

void NBodyApp::OnMouseButton(int button, int action, int mods) {
    (void)mods;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;

    if (action == GLFW_PRESS) {
        if (ImGui::GetIO().WantCaptureMouse) return;
        double x = 0.0, y = 0.0;
        glfwGetCursorPos(m_window, &x, &y);
        const ViewportRects vp = ComputeLayout();
        // GLFW cursor coords are window-space, top-left origin, y-down; the
        // 3D view occupies the top vp.viewH pixels of the window.
        if (y < static_cast<double>(vp.viewH)) {
            m_orbiting = true;
            m_lastMouseX = x;
            m_lastMouseY = y;
        }
    } else if (action == GLFW_RELEASE) {
        m_orbiting = false;
    }
}

void NBodyApp::OnMouseMove(double x, double y) {
    if (m_orbiting) {
        const float dx = static_cast<float>(x - m_lastMouseX);
        const float dy = static_cast<float>(y - m_lastMouseY);
        m_camera.Orbit(dx, dy);
    }
    m_lastMouseX = x;
    m_lastMouseY = y;
}

void NBodyApp::OnScroll(double xoffset, double yoffset) {
    (void)xoffset;
    if (!ImGui::GetIO().WantCaptureMouse) m_camera.Zoom(static_cast<float>(yoffset));
}

} // namespace nbody
