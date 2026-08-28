#pragma once

#include "framework/Application.hpp"
#include "framework/Hud.hpp"
#include "framework/Text.hpp"

#include <memory>
#include <string>

namespace fw {

class Simulation;
class Deck;
class OutputWriter;

// Interactive host for a fw::Simulation: a window with a free-floating HUD
// (play/pause/step/reset/speed/record) and mouse-driven view control, no ImGui.
// The sim advances once per rendered frame (its dt is a numerical parameter,
// decoupled from wall-clock -- the base fixed-timestep loop is not used).
//
// Construct with an unconfigured sim; SimApp calls Configure() in OnStart once
// the GL context exists.
class SimApp : public Application {
public:
    SimApp(Simulation& sim, const Deck& deck, std::string title);
    ~SimApp() override;

protected:
    void OnStart() override;
    void OnUpdate(double dt) override;
    void OnRender() override;
    void OnKey(int key, int scancode, int action, int mods) override;
    void OnMouseButton(int button, int action, int mods) override;
    void OnMouseMove(double x, double y) override;
    void OnScroll(double xoffset, double yoffset) override;

private:
    static Config MakeConfig(const std::string& title);
    void ToggleRecording();

    Simulation& m_sim;
    const Deck& m_deck;
    std::string m_title;

    Hud m_hud;
    HudState m_hudState;
    Font m_font;
    TextRenderer m_text;

    std::unique_ptr<OutputWriter> m_recorder;
    double m_simTime = 0.0;
    long m_step = 0;

    bool m_dragging = false;
    bool m_screenshotRequested = false;
    double m_lastX = 0.0;
    double m_lastY = 0.0;
};

} // namespace fw
