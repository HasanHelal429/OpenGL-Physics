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

// One leg of a piecewise-linear thermostat.target_t(t) schedule -- see
// MDSim::UpdateRampTargetT. Deck: [[ramp.segments]] { t0, t1, t_start, t_end }.
struct RampSegment {
    double t0 = 0.0, t1 = 0.0;
    double tStart = 0.0, tEnd = 0.0;
};

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
    void UpdateRampTargetT();
    // `prevPos` is MDSystem::Positions() (wrapped) from just before the
    // physics step just taken; folds the frame-to-frame minimum-image
    // displacement into m_unwrappedPos. NOTE: only exact for a fixed box --
    // an active barostat's uniform position rescale isn't a pure
    // translation, so it isn't correctly unwrapped by this (not used
    // together with the ramp/Lindemann feature in this project's decks).
    void UpdateUnwrappedPositions(const std::vector<glm::dvec3>& prevPos);
    // rms displacement from the initial FCC lattice site / nearest-neighbor
    // distance -- see MDSim.cpp's comment for the melting-detection idea.
    double LindemannParameter() const;

    MDSystem m_system;
    MDParams m_params;

    // Reset() baseline -- the Configure()-time initial condition. The deck
    // itself is not re-parsed by Reset() (matches TdeSim/GrhdSim convention).
    std::vector<glm::dvec3> m_initialPos, m_initialVel;
    std::vector<int> m_initialSpecies; // empty unless system.species_b_fraction > 0
    double m_initialL = 1.0;
    bool m_isMixture = false;

    // Slab-in-vacuum coexistence mode (system.slab_fraction > 0, mutually
    // exclusive with the mixture mode above -- see MDSim.cpp's Configure).
    // m_slabZ1 is the ORIGINAL boundary [0, slabZ1) the condensed phase
    // occupied at Configure() time -- fixed, not recomputed later, so
    // vapor_fraction always means "outside where the slab started",
    // regardless of how far the interface has since moved.
    bool m_isSlab = false;
    double m_slabZ1 = 0.0;

    std::string m_title = "Molecular Dynamics -- Lennard-Jones fluid";
    double m_dt = 0.002;
    int m_substepsPerFrame = 20;
    std::vector<std::string> m_diagNames;

    // Optional thermostat.target_t(t) schedule (decks/melting_ramp.toml) --
    // empty if the deck has no [ramp] section, in which case target_t stays
    // fixed at whatever Configure() set from thermostat.target_t.
    std::vector<RampSegment> m_ramp;
    double m_simTime = 0.0; // this sim's own clock, advanced dt per substep in Step()

    // Lindemann-parameter tracking: m_unwrappedPos reconstructs each
    // particle's true (non-periodic-wrapped) trajectory incrementally every
    // substep (exact as long as the box is fixed, unlike a post-hoc
    // reconstruction from sparsely-saved frames); m_initialPos doubles as
    // the FCC reference lattice site (the IC starts exactly on the
    // lattice). m_nnDistance is the FCC nearest-neighbor spacing at the
    // deck's initial density, independent of how the box later evolves.
    std::vector<glm::dvec3> m_unwrappedPos;
    double m_nnDistance = 1.0;

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
