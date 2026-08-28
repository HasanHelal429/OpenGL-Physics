#include "framework/GLContext.hpp"

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <stdexcept>

namespace fw {

GLContext GLContext::CreateHidden(int major, int minor) {
    if (!glfwInit()) {
        throw std::runtime_error("GLContext: glfwInit failed");
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
        throw std::runtime_error("GLContext: failed to create hidden window/context");
    }
    glfwMakeContextCurrent(window);
    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        glfwDestroyWindow(window);
        glfwTerminate();
        throw std::runtime_error("GLContext: glad load failed");
    }

    GLContext ctx;
    ctx.m_window = window;
    return ctx;
}

GLContext::~GLContext() {
    if (m_window) {
        glfwDestroyWindow(m_window);
        glfwTerminate();
    }
}

GLContext::GLContext(GLContext&& other) noexcept : m_window(other.m_window) {
    other.m_window = nullptr;
}

GLContext& GLContext::operator=(GLContext&& other) noexcept {
    if (this != &other) {
        if (m_window) {
            glfwDestroyWindow(m_window);
            glfwTerminate();
        }
        m_window = other.m_window;
        other.m_window = nullptr;
    }
    return *this;
}

} // namespace fw
