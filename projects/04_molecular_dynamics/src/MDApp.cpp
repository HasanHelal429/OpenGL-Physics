#include "MDApp.hpp"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace md {

namespace {

// Consolas ships with Windows by default; loaded by absolute path rather
// than a vendored asset, matching 01_hartree_fock/02_nbody_gravity's
// approach on this Windows/MinGW-specific toolchain.
const char* kFontPath = "C:\\Windows\\Fonts\\consola.ttf";

constexpr Preset kPresets[] = {Preset::Gas, Preset::Liquid, Preset::Solid, Preset::Custom};
constexpr Thermostat kThermostats[] = {Thermostat::None, Thermostat::Berendsen, Thermostat::VelocityRescale};

const char* ThermostatName(Thermostat t) {
    switch (t) {
        case Thermostat::None: return "None (microcanonical, NVE)";
        case Thermostat::Berendsen: return "Berendsen";
        case Thermostat::VelocityRescale: return "Hard rescale (periodic)";
    }
    return "?";
}

} // namespace

MDApp::Config MDApp::MakeConfig() {
    Config config;
    config.title = "Molecular Dynamics (Lennard-Jones)";
    config.width = 1500;
    config.height = 950;
    config.fixedTimestep = 1.0 / 60.0;
    return config;
}

MDApp::MDApp()
    : fw::Application(MakeConfig()),
      m_font(fw::Font::FromFile(kFontPath, 20.0f)),
      m_textRenderer(),
      m_charts(m_font, m_textRenderer) {}

void MDApp::OnStart() { LoadScenario(); }

void MDApp::LoadScenario() {
    ScenarioResult result = BuildFccLattice(m_targetN, m_density, m_temperature, m_seed);
    m_system.SetParticles(std::move(result.pos), std::move(result.vel), result.boxLength);

    m_params.targetT = m_temperature;
    m_system.PrimeForces(m_params);

    const double L = m_system.BoxLength();
    m_camera.target = glm::vec3(static_cast<float>(L * 0.5));
    m_camera.distance = static_cast<float>(L * 1.4);
    m_camera.pitch = 20.0f;
    m_camera.minDistance = static_cast<float>(std::max(0.1, L * 0.05));
    m_camera.maxDistance = static_cast<float>(L * 20.0);

    m_simTime = 0.0;
    m_stepCount = 0;
    m_energy0 = m_system.TotalEnergy();
    m_lastEnergyDriftPct = 0.0;
    m_charts.Reset();

    m_rdfRMax = std::min(5.0, 0.49 * L);
    m_rdfTickCounter = 0;
    m_charts.SetRdf(m_system.ComputeRDF(m_rdfNBins, m_rdfRMax), m_rdfRMax);
}

void MDApp::OnUpdate(double dt) {
    (void)dt;
    m_didFixedUpdateThisFrame = false;
}

void MDApp::OnFixedUpdate(double fixedDt) {
    if (!m_running && !m_stepOnce) return;

    // Same per-rendered-frame physics cap as 02_nbody_gravity's NBodyApp
    // (see its OnFixedUpdate comment for the full rationale): caps real
    // work to once per rendered frame regardless of how many catch-up
    // calls Application::Run()'s fixed-timestep loop fires, so a slow step
    // at large N can't stall rendering by spiraling through a full backlog
    // of catch-up calls before a frame ever presents.
    if (!m_stepOnce) {
        if (m_didFixedUpdateThisFrame) return;
        m_didFixedUpdateThisFrame = true;
    }

    const int maxSteps = m_stepOnce ? 1 : m_substeps;
    m_stepOnce = false;

    const auto t0 = std::chrono::steady_clock::now();
    int steps = 0;
    for (; steps < maxSteps; ++steps) {
        m_system.Step(m_params);
        m_simTime += m_params.dt;
        ++m_stepCount;

        const double elapsedMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (elapsedMs > fixedDt * 1000.0) {
            ++steps;
            break;
        }
    }
    const auto t1 = std::chrono::steady_clock::now();
    m_lastStepMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / static_cast<double>(steps);

    // Diagnostics are O(1) byproducts of the force loop already inside
    // Step (see MDSystem's class comment) -- cheap enough to sample every
    // fixed-update call, unlike 02_nbody_gravity's O(N^2) energy check.
    // Still only once per call (not per substep) so the chart's own
    // rebuild-and-reupload cost (Chart2D::SetContent redoes the whole
    // series each call) stays bounded regardless of the substep count.
    m_charts.AddSample(m_simTime, m_system.Temperature(), m_system.Pressure());

    ++m_rdfTickCounter;
    if (m_rdfAuto && m_rdfTickCounter >= m_rdfEveryNTicks) m_rdfRequested = true;
    MaybeRefreshRdf();

    const double E = m_system.TotalEnergy();
    m_lastEnergyDriftPct = (m_energy0 != 0.0) ? 100.0 * (E - m_energy0) / std::abs(m_energy0) : 0.0;
}

void MDApp::MaybeRefreshRdf() {
    if (!m_rdfRequested) return;
    m_rdfRequested = false;
    m_rdfTickCounter = 0;
    m_charts.SetRdf(m_system.ComputeRDF(m_rdfNBins, m_rdfRMax), m_rdfRMax);
}

MDApp::ViewportRects MDApp::ComputeLayout() const {
    ViewportRects vp{};
    const int chartHeight = static_cast<int>(static_cast<float>(m_height) * 0.24f);
    const int viewHeight = m_height - chartHeight;

    vp.viewX = 0;
    vp.viewY = chartHeight;
    vp.viewW = m_width;
    vp.viewH = viewHeight;

    const int halfW = m_width / 2;
    vp.tpX = 0;
    vp.tpY = 0;
    vp.tpW = halfW;
    vp.tpH = chartHeight;

    vp.rdfX = halfW;
    vp.rdfY = 0;
    vp.rdfW = m_width - halfW;
    vp.rdfH = chartHeight;
    return vp;
}

void MDApp::UpdateParticleInstances() {
    const std::vector<glm::dvec3>& pos = m_system.Positions();
    const std::vector<glm::dvec3>& vel = m_system.Velocities();
    const size_t n = pos.size();

    double maxSpeed = 1e-9;
    for (const glm::dvec3& v : vel) maxSpeed = std::max(maxSpeed, glm::length(v));

    // Fixed world-space radius (LJ particles have a real physical size,
    // ~1 sigma diameter, unlike N-body point masses) rather than an
    // N-dependent size -- and the box itself grows with N at fixed
    // density, so particles naturally read smaller relative to the box as
    // N grows, without needing to fake that here.
    constexpr float kRadius = 0.42f;

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
        inst.size = kRadius;
    }
    m_particles.SetParticles(m_particleScratch);
}

void MDApp::OnRender() {
    const ViewportRects vp = ComputeLayout();

    UpdateParticleInstances();

    glViewport(vp.viewX, vp.viewY, vp.viewW, vp.viewH);
    glEnable(GL_DEPTH_TEST);
    const float aspect = static_cast<float>(vp.viewW) / static_cast<float>(std::max(vp.viewH, 1));
    const glm::mat4 view = m_camera.ViewMatrix();
    const glm::mat4 projection = m_camera.ProjectionMatrix(aspect);

    m_wireBox.Draw(view, projection, m_system.BoxLength(), glm::vec4(0.55f, 0.65f, 0.75f, 0.6f));
    m_particles.Draw(view, projection, static_cast<float>(vp.viewH));

    m_charts.SetRegions(vp.tpX, vp.tpY, vp.tpW, vp.tpH, vp.rdfX, vp.rdfY, vp.rdfW, vp.rdfH);
    m_charts.Render();

    glViewport(0, 0, m_width, m_height);
}

void MDApp::OnImGui() {
    ImGui::Begin("Molecular Dynamics Controls", nullptr, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_AlwaysAutoResize);
    DrawControls();
    ImGui::End();
}

void MDApp::DrawControls() {
    if (ImGui::BeginCombo("Preset", PresetName(m_preset))) {
        for (Preset p : kPresets) {
            const bool selected = (p == m_preset);
            if (ImGui::Selectable(PresetName(p), selected)) {
                m_preset = p;
                double d = m_density, t = m_temperature;
                if (PresetValues(p, d, t)) {
                    m_density = d;
                    m_temperature = t;
                }
            }
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    ImGui::SliderInt("Target N", &m_targetN, 32, 20000, "%d", ImGuiSliderFlags_Logarithmic);
    ImGui::TextDisabled("Actual N is 4*cells^3 (nearest fit) = %zu", m_system.Count());

    float density = static_cast<float>(m_density);
    if (ImGui::SliderFloat("Density (rho*)", &density, 0.01f, 1.10f, "%.3f", ImGuiSliderFlags_Logarithmic)) {
        m_density = static_cast<double>(density);
        m_preset = Preset::Custom;
    }
    float initT = static_cast<float>(m_temperature);
    if (ImGui::SliderFloat("Initial T*", &initT, 0.05f, 3.0f, "%.3f", ImGuiSliderFlags_Logarithmic)) {
        m_temperature = static_cast<double>(initT);
        m_preset = Preset::Custom;
    }

    if (ImGui::Button("Load / Reset Scenario")) LoadScenario();

    ImGui::Separator();

    if (ImGui::BeginCombo("Thermostat", ThermostatName(m_params.thermostat))) {
        for (Thermostat t : kThermostats) {
            const bool selected = (t == m_params.thermostat);
            if (ImGui::Selectable(ThermostatName(t), selected)) m_params.thermostat = t;
            if (selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
    ImGui::TextDisabled("None = energy-conserving NVE (use to check drift). Live target T below applies while running.");

    float targetT = static_cast<float>(m_params.targetT);
    if (ImGui::SliderFloat("Thermostat target T*", &targetT, 0.05f, 3.0f, "%.3f", ImGuiSliderFlags_Logarithmic))
        m_params.targetT = static_cast<double>(targetT);

    ImGui::BeginDisabled(m_params.thermostat != Thermostat::Berendsen);
    float tau = static_cast<float>(m_params.berendsenTau);
    if (ImGui::SliderFloat("Berendsen tau", &tau, 0.05f, 10.0f, "%.2f", ImGuiSliderFlags_Logarithmic))
        m_params.berendsenTau = static_cast<double>(tau);
    ImGui::EndDisabled();

    ImGui::BeginDisabled(m_params.thermostat != Thermostat::VelocityRescale);
    ImGui::SliderInt("Rescale every N steps", &m_params.rescaleEvery, 1, 200);
    ImGui::EndDisabled();

    ImGui::Separator();

    float cutoff = static_cast<float>(m_params.cutoff);
    if (ImGui::SliderFloat("LJ cutoff (sigma)", &cutoff, 1.5f, 4.0f, "%.2f")) m_params.cutoff = static_cast<double>(cutoff);
    float skin = static_cast<float>(m_params.skin);
    if (ImGui::SliderFloat("Verlet skin (sigma)", &skin, 0.05f, 1.0f, "%.2f")) m_params.skin = static_cast<double>(skin);
    ImGui::SliderInt("Neighbor rebuild every N steps", &m_params.neighborRebuildEvery, 1, 30);

    float dt = static_cast<float>(m_params.dt);
    if (ImGui::SliderFloat("dt", &dt, 0.0002f, 0.01f, "%.4f", ImGuiSliderFlags_Logarithmic)) m_params.dt = static_cast<double>(dt);
    ImGui::SliderInt("Substeps / frame", &m_substeps, 1, 20);

    ImGui::Separator();

    ImGui::Checkbox("Auto-refresh RDF", &m_rdfAuto);
    ImGui::SameLine();
    if (ImGui::Button("Recompute RDF now")) m_rdfRequested = true;
    ImGui::SliderInt("RDF refresh every N ticks", &m_rdfEveryNTicks, 1, 120);

    ImGui::Separator();

    if (ImGui::Button(m_running ? "Pause" : "Play")) m_running = !m_running;
    ImGui::SameLine();
    ImGui::BeginDisabled(m_running);
    if (ImGui::Button("Step")) m_stepOnce = true;
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset")) LoadScenario();

    ImGui::Separator();

    const double L = m_system.BoxLength();
    const double actualDensity = L > 0.0 ? static_cast<double>(m_system.Count()) / (L * L * L) : 0.0;
    ImGui::Text("N = %zu   L = %.3f sigma   rho* = %.4f", m_system.Count(), L, actualDensity);
    ImGui::Text("sim t* = %.4f   steps = %ld   step: %.3f ms", m_simTime, m_stepCount, m_lastStepMs);
    ImGui::Text("T* = %.4f   P* = %.4f", m_system.Temperature(), m_system.Pressure());
    ImGui::Text("KE = %.3f   PE = %.3f   E = %.3f   drift = %+.3f%%", m_system.KineticEnergy(), m_system.PotentialEnergy(),
                m_system.TotalEnergy(), m_lastEnergyDriftPct);
    ImGui::Text("FPS: %.1f", static_cast<double>(ImGui::GetIO().Framerate));
}

void MDApp::OnMouseButton(int button, int action, int mods) {
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

void MDApp::OnMouseMove(double x, double y) {
    if (m_orbiting) {
        const float dx = static_cast<float>(x - m_lastMouseX);
        const float dy = static_cast<float>(y - m_lastMouseY);
        m_camera.Orbit(dx, dy);
    }
    m_lastMouseX = x;
    m_lastMouseY = y;
}

void MDApp::OnScroll(double xoffset, double yoffset) {
    (void)xoffset;
    if (!ImGui::GetIO().WantCaptureMouse) m_camera.Zoom(static_cast<float>(yoffset));
}

} // namespace md
