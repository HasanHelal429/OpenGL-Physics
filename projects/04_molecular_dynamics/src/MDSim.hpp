#pragma once

#include "MDSystem.hpp"
#include "WireBox.hpp"

#include "framework/Camera.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Simulation.hpp"
#include "framework/Text.hpp"

#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace md {

// fw::Simulation wrapper around MDSystem + Scenarios::BuildFccLattice: one
// input deck (particle count/density/temperature/thermostat/barostat/time)
// fully specifies a reproducible run, driven by either fw::RunHeadless
// (batch: writes pos/vel/diagnostics for tools/ to consume) or fw::SimApp
// (interactive: orbit camera + free-floating HUD). This is the "05 onward"
// deck-driven path -- the older ImGui + live-chart path (MDApp/MDCharts) is
// unchanged and lives alongside it as a separate debug build; see main.cpp.
class MDSim : public fw::Simulation {
public:
    MDSim() = default;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;

private:
    MDSystem m_system;
    MDParams m_params;

    // Reset() baseline -- the Configure()-time initial condition. The deck
    // itself is not re-parsed by Reset() (matches TdeSim/GrhdSim convention).
    std::vector<glm::dvec3> m_initialPos, m_initialVel;
    double m_initialL = 1.0;

    std::string m_title = "Molecular Dynamics -- Lennard-Jones fluid";
    double m_dt = 0.002;
    int m_substepsPerFrame = 20;
    std::vector<std::string> m_diagNames;

    // Interactive view. Deferred construction to Configure(): these
    // constructors call OpenGL functions immediately (shader compile, VAO/
    // VBO), and main.cpp's interactive path constructs MDSim before
    // fw::SimApp creates the window and loads GL function pointers (same
    // ordering constraint as TdeSim -- see its header comment).
    fw::Camera m_camera;
    std::unique_ptr<fw::ParticleCloud> m_particles;
    std::unique_ptr<WireBox> m_box;
    fw::Font m_font;
    std::unique_ptr<fw::TextRenderer> m_text;
};

} // namespace md
