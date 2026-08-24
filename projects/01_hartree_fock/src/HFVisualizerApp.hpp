#pragma once

#include "ScfWorker.hpp"
#include "framework/Application.hpp"

namespace hf {

// Live visualization of the atomic SCF loop: controls to pick an element and
// method, and four panels (density, effective potential, orbital energy
// levels, total-energy convergence) that update every SCF iteration while
// ScfWorker runs the solve on a background thread. Mirrors the Python
// project's post-hoc scf_movie.py 4-panel layout, but live.
class HFVisualizerApp : public fw::Application {
public:
    HFVisualizerApp();

protected:
    void OnStart() override;
    void OnImGui() override;
    void OnShutdown() override;

private:
    void DrawControls();
    void DrawPlots();
    void StartRun();

    static Config MakeConfig();

    ScfWorker m_worker;

    int m_Z = 18; // Argon by default, matches the Python reference animation
    Method m_method = Method::Xalpha;
    float m_alpha = static_cast<float>(kAlphaSchwarz);

    std::vector<double> m_energyHistory;
    int m_lastRenderedIteration = 0;
};

} // namespace hf
