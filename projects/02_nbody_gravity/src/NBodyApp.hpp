#pragma once

#include "DiagnosticsWorker.hpp"
#include "DriftChart.hpp"
#include "NBodySystem.hpp"
#include "ScalingSweepWorker.hpp"
#include "Scenarios.hpp"
#include "framework/Application.hpp"
#include "framework/Camera.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Text.hpp"

namespace nbody {

// One solver's result from RunBenchmark: average wall-clock cost of one
// force evaluation over several repetitions at the live particle set, plus
// (when a Direct reference was affordable to compute) accuracy relative to
// it -- speed alone doesn't say whether a solver's approximation is even
// reasonable at the current theta/softening.
struct SolverBenchmark {
    bool ran = false;
    double msPerCall = 0.0;
    double meanRelError = -1.0; // -1 = no Direct reference available to compare against
    double maxRelError = -1.0;
};

struct BenchmarkResults {
    int n = 0;
    SolverBenchmark direct;    // skipped above a particle-count cap -- see RunBenchmark
    SolverBenchmark barnesHut;
    SolverBenchmark fmm;
};

// Real-time N-body gravity: direct O(N^2), Barnes-Hut O(N log N), and
// adaptive FMM O(N) solvers on a shared leapfrog integrator, live-tunable
// scenario/physics parameters, an OpenGL particle-cloud view (colored by
// speed), a live energy/angular-momentum conservation chart, and an
// in-app benchmark comparing the solvers' speed and accuracy at the
// current live particle set. The C++/OpenGL take on Physics Simulations'
// Orbital_Dynamics/N_Body_Gravity notebooks, redesigned around real-time
// interactivity (drag sliders, watch it respond) rather than a port of
// the offline-notebook structure.
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
    void DrawScalingResults();
    void UpdateParticleInstances();
    void RunBenchmark();

    static Config MakeConfig();

    fw::Font m_font;
    fw::TextRenderer m_textRenderer;
    fw::Camera m_camera;
    fw::ParticleCloud m_particles;
    DriftChart m_driftChart;

    NBodySystem m_system;
    DiagnosticsWorker m_diagnosticsWorker;
    std::vector<fw::ParticleInstance> m_particleScratch; // reused across frames to avoid a per-frame heap alloc

    ScenarioType m_scenarioType = ScenarioType::RotatingDisk;
    ScenarioParams m_scenarioParams;

    double m_G = 1.0;
    double m_softening = 0.05;
    double m_dt = 1e-3;
    double m_theta = 0.5;
    SolverType m_solver = SolverType::BarnesHut;
    int m_substeps = 4; // leapfrog steps per fixed-update tick -- the sim-speed dial

    bool m_running = true;
    bool m_stepOnce = false;
    bool m_didFixedUpdateThisFrame = false;
    bool m_trackDiagnostics = true;
    int m_diagnosticsTickCounter = 0;

    double m_simTime = 0.0;
    long m_stepCount = 0;
    double m_energy0 = 0.0;
    glm::dvec3 m_angularMomentum0{0.0};
    double m_lastEnergyDriftPct = 0.0;
    double m_lastLDriftPct = 0.0;
    double m_lastStepMs = 0.0;

    BenchmarkResults m_benchmark;

    ScalingSweepWorker m_scalingWorker;
    std::vector<ScalingSweepPoint> m_scalingPoints;
    bool m_scalingStarted = false;

    bool m_orbiting = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};

} // namespace nbody
