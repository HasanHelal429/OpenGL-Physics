#pragma once

#include "framework/Shader.hpp"

#include <glad/glad.h>

#include <array>
#include <string>

namespace fw {

class Font;
class TextRenderer;

// Shared control state between the HUD and its host (SimApp). Fields ending in
// "Requested"/"Toggled" are edge events the host consumes (and clears) each frame.
struct HudState {
    bool playing = true;
    bool stepRequested = false;
    bool resetRequested = false;
    int speed = 1;              // sim substeps per rendered frame; cycles 1,2,4,8,16
    bool recording = false;
    bool recordToggled = false;
    bool showHelp = false;
};

// A small cluster of free-floating controls pinned to the top-right of the
// framebuffer, drawn as plain OpenGL geometry (no ImGui): play/pause, step,
// reset, speed-cycle, record. Each control is an icon on its own translucent
// tile -- deliberately not one docked panel.
//
// Host wiring: call Draw() once per frame inside the render pass; feed cursor
// moves to SetCursor() and clicks to HandleClick(); forward key presses to
// HandleKey().
class Hud {
public:
    Hud();
    ~Hud();
    Hud(const Hud&) = delete;
    Hud& operator=(const Hud&) = delete;

    void Draw(HudState& s, int fbW, int fbH, Font* font, TextRenderer* text, const std::string& title);

    void SetCursor(double x, double y) { m_cursorX = x; m_cursorY = y; }
    // Returns true if the click landed on a control (and mutated `s`).
    bool HandleClick(HudState& s, int fbW, int fbH);
    // GLFW key + action (GLFW_PRESS/REPEAT). Returns true if handled.
    bool HandleKey(HudState& s, int key, int action);

    static int NextSpeed(int speed);

private:
    enum Btn { kPlay, kStep, kReset, kSpeed, kRecord, kBtnCount };
    struct Rect { float x, y, w, h; };

    std::array<Rect, kBtnCount> Layout(int fbW, int fbH) const;
    void FillRect(const Rect& r, const float rgba[4]);
    void DrawTris(const float* verts, int count, const float rgba[4]);

    Shader m_shader;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    double m_cursorX = 0.0;
    double m_cursorY = 0.0;
};

} // namespace fw
