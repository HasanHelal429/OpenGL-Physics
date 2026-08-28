#include "framework/Hud.hpp"

#include "framework/Text.hpp"

#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <cmath>
#include <vector>

namespace fw {

namespace {

constexpr float kTile = 34.0f;
constexpr float kGap = 8.0f;
constexpr float kMargin = 16.0f;

const char* kVert = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
uniform mat4 uProj;
void main() { gl_Position = uProj * vec4(aPos, 0.0, 1.0); }
)";

const char* kFrag = R"(
#version 460 core
out vec4 FragColor;
uniform vec4 uColor;
void main() { FragColor = uColor; }
)";

void AddQuad(std::vector<float>& v, float x0, float y0, float x1, float y1) {
    v.insert(v.end(), {x0, y0, x1, y0, x1, y1, x0, y0, x1, y1, x0, y1});
}

void AddTri(std::vector<float>& v, float ax, float ay, float bx, float by, float cx, float cy) {
    v.insert(v.end(), {ax, ay, bx, by, cx, cy});
}

// Filled disc as a triangle fan flattened to a list.
void AddDisc(std::vector<float>& v, float cx, float cy, float r, int seg = 22) {
    for (int i = 0; i < seg; ++i) {
        float a0 = (float(i) / seg) * 6.2831853f;
        float a1 = (float(i + 1) / seg) * 6.2831853f;
        AddTri(v, cx, cy, cx + r * std::cos(a0), cy + r * std::sin(a0), cx + r * std::cos(a1),
               cy + r * std::sin(a1));
    }
}

// Ring arc (a curved-arrow body) from angle a0..a1, as a triangle-list strip.
void AddArc(std::vector<float>& v, float cx, float cy, float rIn, float rOut, float a0, float a1,
           int seg = 20) {
    for (int i = 0; i < seg; ++i) {
        float t0 = a0 + (a1 - a0) * (float(i) / seg);
        float t1 = a0 + (a1 - a0) * (float(i + 1) / seg);
        float c0 = std::cos(t0), s0 = std::sin(t0), c1 = std::cos(t1), s1 = std::sin(t1);
        AddTri(v, cx + rIn * c0, cy + rIn * s0, cx + rOut * c0, cy + rOut * s0, cx + rOut * c1,
               cy + rOut * s1);
        AddTri(v, cx + rIn * c0, cy + rIn * s0, cx + rOut * c1, cy + rOut * s1, cx + rIn * c1,
               cy + rIn * s1);
    }
}

} // namespace

int Hud::NextSpeed(int speed) {
    int n = speed * 2;
    return n > 16 ? 1 : n;
}

Hud::Hud() {
    m_shader = Shader::FromSource(kVert, kFrag);
    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

Hud::~Hud() {
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

std::array<Hud::Rect, Hud::kBtnCount> Hud::Layout(int fbW, int fbH) const {
    (void)fbH;
    const float total = static_cast<float>(int(kBtnCount)) * kTile + static_cast<float>(int(kBtnCount) - 1) * kGap;
    float x = static_cast<float>(fbW) - kMargin - total;
    const float y = kMargin;
    std::array<Rect, kBtnCount> r{};
    for (int i = 0; i < kBtnCount; ++i) {
        r[i] = Rect{x, y, kTile, kTile};
        x += kTile + kGap;
    }
    return r;
}

void Hud::DrawTris(const float* verts, int count, const float rgba[4]) {
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(count * 2 * sizeof(float)), verts,
                 GL_DYNAMIC_DRAW);
    m_shader.SetVec4("uColor", glm::vec4(rgba[0], rgba[1], rgba[2], rgba[3]));
    glDrawArrays(GL_TRIANGLES, 0, count);
    glBindVertexArray(0);
}

void Hud::FillRect(const Rect& r, const float rgba[4]) {
    std::vector<float> v;
    AddQuad(v, r.x, r.y, r.x + r.w, r.y + r.h);
    DrawTris(v.data(), 6, rgba);
}

void Hud::Draw(HudState& s, int fbW, int fbH, Font* font, TextRenderer* text, const std::string& title) {
    const auto rects = Layout(fbW, fbH);

    GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
    GLboolean blendWas = glIsEnabled(GL_BLEND);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    m_shader.Use();
    const glm::mat4 proj = glm::ortho(0.0f, static_cast<float>(fbW), static_cast<float>(fbH), 0.0f);
    m_shader.SetMat4("uProj", proj);

    const float tileBg[4] = {0.10f, 0.11f, 0.13f, 0.62f};
    const float tileHot[4] = {0.16f, 0.18f, 0.22f, 0.78f};
    const float ink[4] = {0.90f, 0.92f, 0.95f, 0.95f};
    const float accent[4] = {0.35f, 0.72f, 1.0f, 1.0f};
    const float rec[4] = {0.95f, 0.28f, 0.30f, 1.0f};

    for (int i = 0; i < kBtnCount; ++i) {
        const Rect& r = rects[i];
        const bool hot = m_cursorX >= r.x && m_cursorX <= r.x + r.w && m_cursorY >= r.y &&
                         m_cursorY <= r.y + r.h;
        FillRect(r, hot ? tileHot : tileBg);

        const float cx = r.x + r.w * 0.5f;
        const float cy = r.y + r.h * 0.5f;
        std::vector<float> g;

        switch (i) {
            case kPlay:
                if (s.playing) {  // show pause
                    AddQuad(g, cx - 7, cy - 8, cx - 2, cy + 8);
                    AddQuad(g, cx + 2, cy - 8, cx + 7, cy + 8);
                } else {  // show play
                    AddTri(g, cx - 6, cy - 8, cx - 6, cy + 8, cx + 8, cy);
                }
                DrawTris(g.data(), static_cast<int>(g.size() / 2), s.playing ? ink : accent);
                break;
            case kStep:
                AddTri(g, cx - 8, cy - 7, cx - 8, cy + 7, cx + 2, cy);
                AddQuad(g, cx + 3, cy - 7, cx + 7, cy + 7);
                DrawTris(g.data(), static_cast<int>(g.size() / 2), ink);
                break;
            case kReset:
                AddArc(g, cx, cy, 3.5f, 6.0f, -2.3f, 2.6f);
                AddTri(g, cx + 6.0f, cy - 8.5f, cx + 11.5f, cy - 5.0f, cx + 4.5f, cy - 3.0f);
                DrawTris(g.data(), static_cast<int>(g.size() / 2), ink);
                break;
            case kSpeed:
                // icon drawn as text below; leave a faint chevron as a hint
                AddTri(g, cx - 6, cy - 6, cx - 6, cy + 6, cx + 1, cy);
                AddTri(g, cx + 0, cy - 6, cx + 0, cy + 6, cx + 7, cy);
                DrawTris(g.data(), static_cast<int>(g.size() / 2), ink);
                break;
            case kRecord:
                AddDisc(g, cx, cy, 7.0f);
                DrawTris(g.data(), static_cast<int>(g.size() / 2), s.recording ? rec : ink);
                break;
        }
    }

    if (font && text) {
        text->SetViewport(fbW, fbH);
        const std::string spd = "x" + std::to_string(s.speed);
        const float w = TextRenderer::MeasureWidth(*font, spd, 0.62f);
        const Rect& sr = rects[kSpeed];
        text->Draw(*font, spd, {sr.x + sr.w * 0.5f - w * 0.5f, sr.y + sr.h + 12.0f},
                   {0.85f, 0.88f, 0.92f, 0.95f}, 0.62f);
        if (!title.empty()) {
            text->Draw(*font, title, {kMargin, kMargin + 14.0f}, {0.82f, 0.85f, 0.9f, 0.9f}, 0.7f);
        }
        if (s.showHelp) {
            const char* lines[] = {"space  play/pause", ".      step one frame", "r      reset",
                                   "s      cycle speed", "F10    record", "h      toggle help"};
            float ty = kMargin + 40.0f;
            for (const char* ln : lines) {
                text->Draw(*font, ln, {kMargin, ty}, {0.75f, 0.78f, 0.82f, 0.9f}, 0.6f);
                ty += 18.0f;
            }
        }
    }

    if (depthWas) glEnable(GL_DEPTH_TEST);
    if (!blendWas) glDisable(GL_BLEND);
}

bool Hud::HandleClick(HudState& s, int fbW, int fbH) {
    const auto rects = Layout(fbW, fbH);
    for (int i = 0; i < kBtnCount; ++i) {
        const Rect& r = rects[i];
        if (m_cursorX < r.x || m_cursorX > r.x + r.w || m_cursorY < r.y || m_cursorY > r.y + r.h)
            continue;
        switch (i) {
            case kPlay: s.playing = !s.playing; break;
            case kStep: s.stepRequested = true; s.playing = false; break;
            case kReset: s.resetRequested = true; break;
            case kSpeed: s.speed = NextSpeed(s.speed); break;
            case kRecord: s.recording = !s.recording; s.recordToggled = true; break;
        }
        return true;
    }
    return false;
}

bool Hud::HandleKey(HudState& s, int key, int action) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return false;
    switch (key) {
        case GLFW_KEY_SPACE: s.playing = !s.playing; return true;
        case GLFW_KEY_PERIOD:
        case GLFW_KEY_RIGHT: s.stepRequested = true; s.playing = false; return true;
        case GLFW_KEY_R: s.resetRequested = true; return true;
        case GLFW_KEY_S: s.speed = NextSpeed(s.speed); return true;
        case GLFW_KEY_F10: s.recording = !s.recording; s.recordToggled = true; return true;
        case GLFW_KEY_H: s.showHelp = !s.showHelp; return true;
    }
    return false;
}

} // namespace fw
