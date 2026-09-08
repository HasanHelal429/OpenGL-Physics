#pragma once

#include "Camera2D.hpp"
#include "DriftChart.hpp"
#include "NBodySystem.hpp"
#include "Scenarios.hpp"
#include "framework/Application.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Text.hpp"

namespace nbody2d {

// One solver's result from RunBenchmark: average wall-clock cost of one
// force evaluation over several repetitions at the live particle set, plus
// (when a Direct reference was affordable to compute) accuracy relative to
// it.
struct SolverBenchmark {
    bool ran = false;
    double msPerCall = 0.0;
    double meanRelError = -1.0; // -1 = no Direct reference available to compare against
    double maxRelError = -1.0;
};

struct BenchmarkResults {
    int n = 0;
    SolverBenchmark direct; // skipped above a particle-count cap -- see RunBenchmark
    SolverBenchmark barnesHut;
    SolverBenchmark complexFmm;
};

// Real-time 2D N-body gravity: Direct O(N^2), Barnes-Hut O(N log N), and
// the complex-Laurent-series FMM O(N) solvers on a shared leapfrog
// integrator, live-tunable scenario/physics parameters, an orthographic
// particle-cloud view, a live energy/angular-momentum conservation chart,
// and an in-app benchmark comparing the solvers' speed and accuracy. The
// 2D companion to 02_nbody_gravity -- see this project's plan doc for why
// 2D gravity (a genuinely different force law, ~1/r not ~1/r^2) gets its
// own project rather than a mode inside the 3D app.
class NBodyApp : public fw::Application {
public:
    NBodyApp();

protected:
    void OnStart() override;
    void OnUpdate(double dt) override;
    void OnFixedUpdate(double fixedDt) override;
    void OnRender() override;
    void OnImGui() override;
    void OnMouseButton(int button, int action, int mods) override;
    void OnMouseMove(double x, double y) override;
    void OnScroll(double xoffset, double yoffset) override;

private:
    struct ViewportRects {
        int viewX, viewY, viewW, viewH;
        int chartX, chartY, chartW, chartH;
    };
    ViewportRects ComputeLayout() const;

    void LoadScenario();
    void DrawControls();
    void DrawBenchmarkResults();
    void UpdateParticleInstances();
    void RunBenchmark();
    void RunDiagnostics();

    static Config MakeConfig();

    fw::Font m_font;
    fw::TextRenderer m_textRenderer;
    Camera2D m_camera;
    fw::ParticleCloud m_particles;
    DriftChart m_driftChart;

    NBodySystem m_system;
    std::vector<fw::ParticleInstance> m_particleScratch; // reused across frames to avoid a per-frame heap alloc

    ScenarioType m_scenarioType = ScenarioType::RotatingDisk;
    ScenarioParams m_scenarioParams;

    double m_G = 1.0;
    double m_softening = 0.05;
    double m_dt = 1e-3; // fixed step, or the adaptive ceiling when m_adaptiveDt
    double m_theta = 0.5;
    SolverType m_solver = SolverType::BarnesHut;
    int m_substeps = 4; // leapfrog steps per fixed-update tick -- the sim-speed dial

    bool m_adaptiveDt = false;
    double m_eta = 0.03;
    double m_lastDtTaken = 0.0;

    bool m_running = true;
    bool m_stepOnce = false;
    bool m_didFixedUpdateThisFrame = false;
    bool m_trackDiagnostics = true;
    int m_diagnosticsTickCounter = 0;

    double m_simTime = 0.0;
    long m_stepCount = 0;
    double m_energy0 = 0.0;
    double m_angularMomentum0 = 0.0;
    double m_lastEnergyDriftPct = 0.0;
    double m_lastLDriftPct = 0.0;
    double m_lastStepMs = 0.0;

    BenchmarkResults m_benchmark;

    bool m_panning = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};

} // namespace nbody2d
