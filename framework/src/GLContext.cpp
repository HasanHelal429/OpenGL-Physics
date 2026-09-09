#include "framework/GLContext.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef PHYSGL_HAVE_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

namespace fw {
namespace {

[[maybe_unused]] bool EnvSaysEmpty(const char* name) {
    const char* v = std::getenv(name);
    return v == nullptr || *v == '\0';
}

// No X11/Wayland in the environment -> a hidden GLFW window cannot work.
[[maybe_unused]] bool NoDisplayServer() {
    return EnvSaysEmpty("DISPLAY") && EnvSaysEmpty("WAYLAND_DISPLAY");
}

#ifdef PHYSGL_HAVE_EGL
// eglGetProcAddress returns void(*)(void); glad wants void*(const char*).
// EGL_KHR_client_get_all_proc_addresses (which the NVIDIA driver advertises)
// is what makes this legal for core GL entry points, not just extensions.
void* EglLoader(const char* name) {
    return reinterpret_cast<void*>(eglGetProcAddress(name));
}

// Build an EGL context on a physical device with no display server.
// Returns true on success and fills the three handles.
bool TryCreateEgl(int major, int minor, void** outDpy, void** outCtx, void** outSurf,
                  std::string* err) {
    auto queryDevices =
        reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(eglGetProcAddress("eglQueryDevicesEXT"));
    auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!queryDevices || !getPlatformDisplay) {
        *err = "EGL_EXT_platform_device unavailable";
        return false;
    }

    constexpr EGLint kMaxDevices = 16;
    EGLDeviceEXT devices[kMaxDevices];
    EGLint deviceCount = 0;
    if (!queryDevices(kMaxDevices, devices, &deviceCount) || deviceCount == 0) {
        *err = "eglQueryDevicesEXT found no devices";
        return false;
    }

    // With several GPUs visible, honour the usual HPC selector so rank/device
    // pinning still means something.
    EGLint first = 0;
    if (const char* sel = std::getenv("PHYSGL_EGL_DEVICE")) {
        const int want = std::atoi(sel);
        if (want >= 0 && want < deviceCount) first = want;
    }

    for (EGLint n = 0; n < deviceCount; ++n) {
        const EGLint i = (first + n) % deviceCount;

        EGLDisplay dpy = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[i], nullptr);
        if (dpy == EGL_NO_DISPLAY) continue;

        EGLint emaj = 0, emin = 0;
        if (!eglInitialize(dpy, &emaj, &emin)) continue;
        if (!eglBindAPI(EGL_OPENGL_API)) { eglTerminate(dpy); continue; }

        const EGLint configAttribs[] = {EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
                                        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                        EGL_RED_SIZE,        8,
                                        EGL_GREEN_SIZE,      8,
                                        EGL_BLUE_SIZE,       8,
                                        EGL_ALPHA_SIZE,      8,
                                        EGL_DEPTH_SIZE,      24,
                                        EGL_NONE};
        EGLConfig config = nullptr;
        EGLint configCount = 0;
        if (!eglChooseConfig(dpy, configAttribs, &config, 1, &configCount) || configCount == 0) {
            eglTerminate(dpy);
            continue;
        }

        const EGLint contextAttribs[] = {EGL_CONTEXT_MAJOR_VERSION, major,
                                         EGL_CONTEXT_MINOR_VERSION, minor,
                                         EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                         EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                         EGL_NONE};
        EGLContext ctx = eglCreateContext(dpy, config, EGL_NO_CONTEXT, contextAttribs);
        if (ctx == EGL_NO_CONTEXT) { eglTerminate(dpy); continue; }

        // A small pbuffer keeps the context complete; every project renders
        // into its own FBO anyway.
        const EGLint pbufferAttribs[] = {EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE};
        EGLSurface surf = eglCreatePbufferSurface(dpy, config, pbufferAttribs);
        if (surf == EGL_NO_SURFACE || !eglMakeCurrent(dpy, surf, surf, ctx)) {
            if (surf != EGL_NO_SURFACE) eglDestroySurface(dpy, surf);
            eglDestroyContext(dpy, ctx);
            eglTerminate(dpy);
            continue;
        }

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(EglLoader))) {
            eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
            eglDestroySurface(dpy, surf);
            eglDestroyContext(dpy, ctx);
            eglTerminate(dpy);
            *err = "glad load failed on EGL context";
            return false;
        }

        *outDpy = dpy;
        *outCtx = ctx;
        *outSurf = surf;
        return true;
    }

    *err = "no EGL device yielded an OpenGL " + std::to_string(major) + "." +
           std::to_string(minor) + " core context";
    return false;
}
#endif // PHYSGL_HAVE_EGL

// Hidden GLFW window -- the original path. Needs a display server.
bool TryCreateGlfw(int major, int minor, GLFWwindow** outWindow, std::string* err) {
    if (!glfwInit()) {
        *err = "glfwInit failed (no display server?)";
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, major);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(64, 64, "headless", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        *err = "failed to create hidden window/context";
        return false;
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        glfwTerminate();
        *err = "glad load failed on GLFW context";
        return false;
    }

    *outWindow = window;
    return true;
}

} // namespace

GLContext GLContext::CreateHidden(int major, int minor) {
    const char* forced = std::getenv("PHYSGL_GL_BACKEND");
    const bool forceEgl = forced && std::strcmp(forced, "egl") == 0;
    [[maybe_unused]] const bool forceGlfw = forced && std::strcmp(forced, "glfw") == 0;

    GLContext ctx;
#ifdef PHYSGL_HAVE_EGL
    std::string eglErr = "not attempted";
#else
    std::string eglErr = "backend not compiled in";
#endif
    std::string glfwErr = "not attempted";

#ifdef PHYSGL_HAVE_EGL
    // Prefer EGL when asked for it, or when there is no display server to
    // hang a hidden window off.
    if (!forceGlfw && (forceEgl || NoDisplayServer())) {
        if (TryCreateEgl(major, minor, &ctx.m_eglDisplay, &ctx.m_eglContext, &ctx.m_eglSurface,
                         &eglErr)) {
            ctx.m_backend = Backend::Egl;
            return ctx;
        }
    }
#endif

    if (!forceEgl) {
        if (TryCreateGlfw(major, minor, &ctx.m_window, &glfwErr)) {
            ctx.m_backend = Backend::Glfw;
            return ctx;
        }
    }

    throw std::runtime_error("GLContext: no offscreen OpenGL context (EGL: " + eglErr +
                             "; GLFW: " + glfwErr + ")");
}

const char* GLContext::BackendName() const {
    switch (m_backend) {
        case Backend::Egl: return "egl";
        case Backend::Glfw: return "glfw";
        default: return "none";
    }
}

void GLContext::Destroy() noexcept {
    if (m_backend == Backend::Glfw && m_window) {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
#ifdef PHYSGL_HAVE_EGL
    if (m_backend == Backend::Egl && m_eglDisplay) {
        EGLDisplay dpy = static_cast<EGLDisplay>(m_eglDisplay);
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (m_eglSurface) eglDestroySurface(dpy, static_cast<EGLSurface>(m_eglSurface));
        if (m_eglContext) eglDestroyContext(dpy, static_cast<EGLContext>(m_eglContext));
        eglTerminate(dpy);
    }
#endif
    m_backend = Backend::None;
    m_window = nullptr;
    m_eglDisplay = nullptr;
    m_eglContext = nullptr;
    m_eglSurface = nullptr;
}

void GLContext::StealFrom(GLContext& other) noexcept {
    m_backend = other.m_backend;
    m_window = other.m_window;
    m_eglDisplay = other.m_eglDisplay;
    m_eglContext = other.m_eglContext;
    m_eglSurface = other.m_eglSurface;

    other.m_backend = Backend::None;
    other.m_window = nullptr;
    other.m_eglDisplay = nullptr;
    other.m_eglContext = nullptr;
    other.m_eglSurface = nullptr;
}

GLContext::~GLContext() { Destroy(); }

GLContext::GLContext(GLContext&& other) noexcept { StealFrom(other); }

GLContext& GLContext::operator=(GLContext&& other) noexcept {
    if (this != &other) {
        Destroy();
        StealFrom(other);
    }
    return *this;
}

} // namespace fw
