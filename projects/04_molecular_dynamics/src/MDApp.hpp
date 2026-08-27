#pragma once

#include "MDCharts.hpp"
#include "MDSystem.hpp"
#include "Scenarios.hpp"
#include "WireBox.hpp"
#include "framework/Application.hpp"
#include "framework/Camera.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Text.hpp"

namespace md {

// Real-time 3D periodic Lennard-Jones molecular dynamics: velocity-Verlet
// integration on a linked-cell/Verlet-list force evaluation (see
// MDSystem), a Berendsen or hard-rescale thermostat, and a live OpenGL
// view (particle cloud colored by speed inside a wireframe periodic box)
// alongside temperature/pressure and radial-distribution-function charts.
// The C++/OpenGL sibling of Physics Simulations' Thermodynamics/Ideal_Gas
// and Van_der_Waals_Gas notebooks, taken to a full 3D periodic system with
// real-time interactive density/temperature/thermostat controls in place
// of those notebooks' separate offline runs.
class MDApp : public fw::Application {
public:
    MDApp();

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
        int tpX, tpY, tpW, tpH;
        int rdfX, rdfY, rdfW, rdfH;
    };
    ViewportRects ComputeLayout() const;

    void LoadScenario();
    void DrawControls();
    void UpdateParticleInstances();
    void MaybeRefreshRdf();

    static Config MakeConfig();

    fw::Font m_font;
    fw::TextRenderer m_textRenderer;
    fw::Camera m_camera;
    fw::ParticleCloud m_particles;
    WireBox m_wireBox;
    MDCharts m_charts;

    MDSystem m_system;
    MDParams m_params;
    std::vector<fw::ParticleInstance> m_particleScratch; // reused across frames to avoid a per-frame heap alloc

    Preset m_preset = Preset::Liquid;
    double m_density = 0.70;
    double m_temperature = 1.00; // initial scenario temperature (thermostat target is m_params.targetT, live-editable)
    int m_targetN = 500;
    unsigned m_seed = 1;

    bool m_running = true;
    bool m_stepOnce = false;
    bool m_didFixedUpdateThisFrame = false;
    int m_substeps = 1;

    double m_simTime = 0.0;
    long m_stepCount = 0;
    double m_energy0 = 0.0;
    double m_lastEnergyDriftPct = 0.0;
    double m_lastStepMs = 0.0;

    bool m_rdfAuto = true;
    int m_rdfEveryNTicks = 20;
    int m_rdfTickCounter = 0;
    int m_rdfNBins = 100;
    double m_rdfRMax = 5.0;
    bool m_rdfRequested = false;

    bool m_orbiting = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};

} // namespace md
