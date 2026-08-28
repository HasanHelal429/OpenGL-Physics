#include "framework/SimApp.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"
#include "framework/Simulation.hpp"

#include <GLFW/glfw3.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <vector>

namespace fw {

namespace {
// Consolas is present on every Windows install; matches the other projects.
// Swap for a vendored .ttf if this ever needs to run off this machine.
const char* kFontPath = "C:\\Windows\\Fonts\\consola.ttf";
} // namespace

Application::Config SimApp::MakeConfig(const std::string& title) {
    Application::Config c;
    c.width = 1280;
    c.height = 800;
    c.title = title.empty() ? "OpenGL Physics" : title;
    c.enableImGui = false;
    c.vsync = true;
    c.glMajor = 4;
    c.glMinor = 6;
    return c;
}

SimApp::SimApp(Simulation& sim, const Deck& deck, std::string title)
    : Application(MakeConfig(title)), m_sim(sim), m_deck(deck), m_title(std::move(title)) {}

SimApp::~SimApp() {
    if (m_recorder) m_recorder->Finish();
}

void SimApp::OnStart() {
    m_font = Font::FromFile(kFontPath, 20.0f);
    m_sim.Configure(m_deck);
    const SimInfo info = m_sim.Info();
    m_hudState.speed = info.substepsPerFrame > 0 ? info.substepsPerFrame : 1;
}

void SimApp::ToggleRecording() {
    if (m_hudState.recording) {
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const long secs = std::chrono::duration_cast<std::chrono::seconds>(now).count();
        char dir[64];
        std::snprintf(dir, sizeof(dir), "recordings/rec_%ld", secs);
        m_recorder = std::make_unique<OutputWriter>(dir, m_sim.Info(), m_deck);
        std::printf("[rec] -> %s\n", dir);
    } else if (m_recorder) {
        m_recorder->Finish();
        std::printf("[rec] wrote %d frames\n", m_recorder->FramesWritten());
        m_recorder.reset();
    }
}

void SimApp::OnUpdate(double dt) {
    (void)dt;
    if (m_hudState.resetRequested) {
        m_hudState.resetRequested = false;
        m_sim.Reset();
        m_simTime = 0.0;
        m_step = 0;
    }

    const int n = m_hudState.speed < 1 ? 1 : m_hudState.speed;
    const bool advance = m_hudState.playing || m_hudState.stepRequested;
    m_hudState.stepRequested = false;
    if (advance) {
        m_sim.Step(n);
        m_step += n;
        m_simTime += n * m_sim.Info().dt;
    }

    if (m_hudState.recordToggled) {
        m_hudState.recordToggled = false;
        ToggleRecording();
    }
    if (m_recorder) {
        m_recorder->BeginFrame(m_simTime, m_step);
        m_sim.Snapshot(*m_recorder);
        m_recorder->EndFrame();
    }
}

void SimApp::OnRender() {
    m_sim.Render(m_width, m_height);
    m_hud.Draw(m_hudState, m_width, m_height, &m_font, &m_text, m_title);

    if (m_screenshotRequested) {
        m_screenshotRequested = false;
        const int w = m_width, h = m_height;
        std::vector<unsigned char> px(static_cast<size_t>(w) * h * 3);
        glReadBuffer(GL_BACK);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
        std::vector<unsigned char> flip(px.size());
        for (int y = 0; y < h; ++y) {
            std::memcpy(&flip[static_cast<size_t>(y) * w * 3],
                        &px[static_cast<size_t>(h - 1 - y) * w * 3], static_cast<size_t>(w) * 3);
        }
        std::filesystem::create_directories("screenshots");
        const auto now = std::chrono::system_clock::now().time_since_epoch();
        const long secs = std::chrono::duration_cast<std::chrono::seconds>(now).count();
        char path[64];
        std::snprintf(path, sizeof(path), "screenshots/shot_%ld.png", secs);
        stbi_write_png(path, w, h, 3, flip.data(), w * 3);
        std::printf("[shot] %s\n", path);
    }
}

void SimApp::OnKey(int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS && key == GLFW_KEY_F12) {
        m_screenshotRequested = true;
        return;
    }
    if (m_hud.HandleKey(m_hudState, key, action)) return;
    if (action == GLFW_PRESS || action == GLFW_REPEAT) m_sim.OnKey(key, action);
    Application::OnKey(key, scancode, action, mods);
}

void SimApp::OnMouseButton(int button, int action, int mods) {
    (void)mods;
    if (button != GLFW_MOUSE_BUTTON_LEFT) return;
    if (action == GLFW_PRESS) {
        if (!m_hud.HandleClick(m_hudState, m_width, m_height)) m_dragging = true;
    } else if (action == GLFW_RELEASE) {
        m_dragging = false;
    }
}

void SimApp::OnMouseMove(double x, double y) {
    const double dx = x - m_lastX;
    const double dy = y - m_lastY;
    m_lastX = x;
    m_lastY = y;
    m_hud.SetCursor(x, y);
    if (m_dragging) {
        ViewInput vi;
        vi.dx = dx;
        vi.dy = dy;
        vi.dragging = true;
        vi.fbWidth = m_width;
        vi.fbHeight = m_height;
        m_sim.OnViewInput(vi);
    }
}

void SimApp::OnScroll(double xoffset, double yoffset) {
    (void)xoffset;
    ViewInput vi;
    vi.scrollDelta = yoffset;
    vi.fbWidth = m_width;
    vi.fbHeight = m_height;
    m_sim.OnViewInput(vi);
}

} // namespace fw
