#pragma once

struct GLFWwindow;

namespace fw {

// An offscreen OpenGL context for headless (no-window) runs, with a current
// 4.6 core context and glad loaded. RAII -- destroying it tears the context
// down.
//
// Two backends:
//   Glfw  a hidden GLFW window. Needs a display server (X11/Wayland), so it
//         works on a desktop but not on a batch node with no $DISPLAY.
//   Egl   EGL_PLATFORM_DEVICE_EXT -- a real GPU context with no display
//         server at all. This is the one that works on an HPC compute node.
//         Only built when CMake found EGL (PHYSGL_HAVE_EGL).
//
// CreateHidden() picks a backend automatically: EGL when it is available and
// no display server is in the environment, GLFW otherwise. Override with
//   PHYSGL_GL_BACKEND=egl|glfw
// Nothing in the framework or the projects uses Window(), so the EGL backend
// is a drop-in for every headless run.
class GLContext {
public:
    enum class Backend { None, Glfw, Egl };

    static GLContext CreateHidden(int major = 4, int minor = 6);

    GLContext() = default;
    ~GLContext();
    GLContext(GLContext&& other) noexcept;
    GLContext& operator=(GLContext&& other) noexcept;
    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    // Non-null only for the GLFW backend.
    GLFWwindow* Window() const { return m_window; }
    Backend Which() const { return m_backend; }
    const char* BackendName() const;

private:
    void Destroy() noexcept;
    void StealFrom(GLContext& other) noexcept;

    Backend m_backend = Backend::None;
    GLFWwindow* m_window = nullptr;

    // Opaque EGLDisplay / EGLContext / EGLSurface -- kept as void* so this
    // header stays free of EGL headers.
    void* m_eglDisplay = nullptr;
    void* m_eglContext = nullptr;
    void* m_eglSurface = nullptr;
};

} // namespace fw
