#pragma once

struct GLFWwindow;

namespace fw {

// An offscreen OpenGL context for headless (no-window) runs: a hidden GLFW
// window with a current 4.6 core context and glad loaded. RAII -- destroying it
// tears down the window and terminates GLFW.
class GLContext {
public:
    static GLContext CreateHidden(int major = 4, int minor = 6);

    GLContext() = default;
    ~GLContext();
    GLContext(GLContext&& other) noexcept;
    GLContext& operator=(GLContext&& other) noexcept;
    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    GLFWwindow* Window() const { return m_window; }

private:
    GLFWwindow* m_window = nullptr;
};

} // namespace fw
