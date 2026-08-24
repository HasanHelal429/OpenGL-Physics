#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <string>

struct GLFWwindow;

namespace fw {

// Base class for a simulation window: owns the GLFW/OpenGL context, runs the
// main loop with a fixed-timestep physics update decoupled from rendering,
// and dispatches input to virtual hooks. Derive from this per project and
// override the On* hooks.
class Application {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        std::string title = "OpenGL Physics";
        bool vsync = true;
        bool enableImGui = true;
        int glMajor = 4;
        int glMinor = 6;
        double fixedTimestep = 1.0 / 120.0; // physics step, seconds
    };

    explicit Application(const Config& config);
    virtual ~Application();

    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    // Starts the main loop; blocks until the window is closed.
    void Run();

protected:
    // Called once after the GL context and ImGui are ready.
    virtual void OnStart() {}
    // Called at a fixed rate (config.fixedTimestep) for physics integration.
    virtual void OnFixedUpdate(double fixedDt) { (void)fixedDt; }
    // Called once per rendered frame before OnRender, variable dt.
    virtual void OnUpdate(double dt) { (void)dt; }
    // Called once per rendered frame to issue draw calls.
    virtual void OnRender() {}
    // Called once per rendered frame, inside an ImGui frame, for debug UI.
    virtual void OnImGui() {}
    // Called once before the window/context is destroyed.
    virtual void OnShutdown() {}

    virtual void OnKey(int key, int scancode, int action, int mods);
    virtual void OnMouseMove(double x, double y) { (void)x; (void)y; }
    virtual void OnMouseButton(int button, int action, int mods) { (void)button; (void)action; (void)mods; }
    virtual void OnScroll(double xoffset, double yoffset) { (void)xoffset; (void)yoffset; }
    virtual void OnResize(int width, int height);

    GLFWwindow* m_window = nullptr;
    int m_width;
    int m_height;

private:
    void InitWindow(const Config& config);
    void InitImGui();
    void ShutdownImGui();

    static void FramebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void CursorPosCallback(GLFWwindow* window, double x, double y);
    static void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    bool m_enableImGui;
    double m_fixedTimestep;
};

} // namespace fw
