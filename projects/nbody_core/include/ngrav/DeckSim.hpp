#pragma once

#include "Scenarios.hpp"
#include "Solvers.hpp"
#include "System.hpp"
#include "Vec.hpp"

#include "framework/Camera.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Simulation.hpp"

#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace ngrav {

// fw::Simulation over ngrav::System<D>: one implementation drives both the
// headless batch runner and (via fw::SimApp) a plain interactive view. The
// rich ImGui NBodyApp stays a separate host per project. `registerAux` lets
// a project wire its still-in-project FMM solvers (P1) before Configure.
template <int D>
class DeckSim : public fw::Simulation {
public:
    using AuxRegistrar = std::function<void(System<D>&)>;
    explicit DeckSim(AuxRegistrar registerAux = {}) : m_registerAux(std::move(registerAux)) {}

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;
    void Render(int fbWidth, int fbHeight) override;

    const System<D>& Sys() const { return m_sys; }

private:
    void LoadScenario();

    AuxRegistrar m_registerAux;
    System<D> m_sys;
    StepParams m_sp;
    ScenarioType m_scenarioType = ScenarioType::PlummerSphere;
    ScenarioParams m_scenarioParams;
    int m_substepsPerFrame = 4;
    double m_time = 0.0;
    long m_step = 0;
    double m_lastDt = 0.0;
    double m_energy0 = 0.0;
    double m_lmag0 = 0.0;
    std::string m_title = "N-Body Gravity";

    // Deck-vs-scenario override tracking (a scenario carries tuned defaults).
    bool m_deckHasG = false, m_deckHasEps = false, m_deckHasDt = false;
    double m_deckG = 1.0, m_deckEps = 0.05, m_deckDt = 1e-3;
    SofteningKind m_deckSoftKind = SofteningKind::Plummer;

    // Built lazily on the first Render() -- its shader compile needs a
    // current GL context, which the headless batch path deliberately never
    // creates (see fw::RunHeadless). Interactive / --render-check paths
    // call Render() with a live context, so the emplace there is safe.
    std::optional<fw::ParticleCloud> m_cloud;
    fw::Camera m_camera;
    std::vector<fw::ParticleInstance> m_scratch;
    double m_cameraScale = 4.0;
    bool m_haveGL = false;
};

} // namespace ngrav
