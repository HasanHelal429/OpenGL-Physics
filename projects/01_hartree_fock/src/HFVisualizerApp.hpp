#pragma once

#include "ConvergenceChart.hpp"
#include "DensityView3D.hpp"
#include "ScfWorker.hpp"
#include "SpectrumView.hpp"
#include "framework/Application.hpp"
#include "framework/Text.hpp"

namespace hf {

// Live visualization of the atomic SCF loop, entirely custom-rendered
// OpenGL (no ImPlot): a 3D view of the density (particle cloud or
// isosurface, toggleable) on top, a traditional line-spectrum plot of
// orbital energies and an E_total convergence chart on the bottom, plus an
// ImGui control panel (Z/method/alpha, view mode, Run/Cancel, status).
class HFVisualizerApp : public fw::Application {
public:
    HFVisualizerApp();

protected:
    void OnRender() override;
    void OnImGui() override;
    void OnMouseButton(int button, int action, int mods) override;
    void OnMouseMove(double x, double y) override;
    void OnScroll(double xoffset, double yoffset) override;

private:
    struct ViewportRects {
        int threeDX, threeDY, threeDW, threeDH;
        int spectrumX, spectrumY, spectrumW, spectrumH;
        int convergenceX, convergenceY, convergenceW, convergenceH;
    };
    ViewportRects ComputeLayout() const;

    void DrawControls();
    void StartRun();

    static Config MakeConfig();

    // Declaration order matters: m_font/m_textRenderer must exist before
    // m_spectrumView/m_convergenceChart (constructed from references to
    // them in the member-initializer list).
    fw::Font m_font;
    fw::TextRenderer m_textRenderer;
    DensityView3D m_densityView;
    SpectrumView m_spectrumView;
    ConvergenceChart m_convergenceChart;
    ScfWorker m_worker;

    int m_Z = 18; // Argon by default, matches the Python reference animation
    Method m_method = Method::Xalpha;
    float m_alpha = static_cast<float>(kAlphaSchwarz);
    double m_isoThreshold = -1.0; // sentinel: auto-suggested from the first snapshot of a run

    int m_lastRenderedIteration = 0;

    bool m_orbiting = false;
    double m_lastMouseX = 0.0;
    double m_lastMouseY = 0.0;
};

} // namespace hf
