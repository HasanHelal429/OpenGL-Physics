#include "framework/Application.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace fw {

Application::Application(const Config& config)
    : m_width(config.width)
    , m_height(config.height)
    , m_enableImGui(config.enableImGui)
    , m_fixedTimestep(config.fixedTimestep) {
    InitWindow(config);
    if (m_enableImGui) {
        InitImGui();
    }
}

Application::~Application() {
    if (m_enableImGui) {
        ShutdownImGui();
    }
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    glfwTerminate();
}

void Application::InitWindow(const Config& config) {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, config.glMajor);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, config.glMinor);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

    m_window = glfwCreateWindow(config.width, config.height, config.title.c_str(), nullptr, nullptr);
    if (!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwMakeContextCurrent(m_window);
    glfwSwapInterval(config.vsync ? 1 : 0);

    if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
        throw std::runtime_error("Failed to initialize glad (OpenGL loader)");
    }

    glfwSetFramebufferSizeCallback(m_window, &Application::FramebufferSizeCallback);
    glfwSetKeyCallback(m_window, &Application::KeyCallback);
    glfwSetCursorPosCallback(m_window, &Application::CursorPosCallback);
    glfwSetMouseButtonCallback(m_window, &Application::MouseButtonCallback);
    glfwSetScrollCallback(m_window, &Application::ScrollCallback);

    glViewport(0, 0, config.width, config.height);
    glEnable(GL_DEPTH_TEST);
}

void Application::InitImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 460 core");
}

void Application::ShutdownImGui() {
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void Application::Run() {
    OnStart();

    double lastTime = glfwGetTime();
    double accumulator = 0.0;

    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();

        const double now = glfwGetTime();
        double frameTime = now - lastTime;
        lastTime = now;
        // Avoid the "spiral of death" if a frame takes too long (e.g. window drag).
        frameTime = std::min(frameTime, 0.25);

        OnUpdate(frameTime);

        accumulator += frameTime;
        while (accumulator >= m_fixedTimestep) {
            OnFixedUpdate(m_fixedTimestep);
            accumulator -= m_fixedTimestep;
        }

        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        OnRender();

        if (m_enableImGui) {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            OnImGui();
            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glfwSwapBuffers(m_window);
    }

    OnShutdown();
}

void Application::OnKey(int key, int scancode, int action, int mods) {
    (void)scancode;
    (void)mods;
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        glfwSetWindowShouldClose(m_window, GLFW_TRUE);
    }
}

void Application::OnResize(int width, int height) {
    m_width = width;
    m_height = height;
    glViewport(0, 0, width, height);
}

void Application::FramebufferSizeCallback(GLFWwindow* window, int width, int height) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) app->OnResize(width, height);
}

void Application::KeyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) app->OnKey(key, scancode, action, mods);
}

void Application::CursorPosCallback(GLFWwindow* window, double x, double y) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) app->OnMouseMove(x, y);
}

void Application::MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) app->OnMouseButton(button, action, mods);
}

void Application::ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    if (app) app->OnScroll(xoffset, yoffset);
}

} // namespace fw
