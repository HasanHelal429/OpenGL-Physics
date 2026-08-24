// Sanity-check project: proves the framework (GLFW window, glad GL loading,
// fixed-timestep loop, Shader, ImGui overlay) builds and runs end to end.
// The triangle's rotation is integrated in OnFixedUpdate to demonstrate the
// physics-step / render-step decoupling that later simulations rely on.
#include "framework/Application.hpp"
#include "framework/Shader.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

namespace {

const char* kVertexShader = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 uTransform;

out vec3 vColor;

void main() {
    gl_Position = uTransform * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
)";

const char* kFragmentShader = R"(
#version 460 core
in vec3 vColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(vColor, 1.0);
}
)";

} // namespace

class HelloTriangleApp : public fw::Application {
public:
    HelloTriangleApp() : fw::Application(MakeConfig()) {}

protected:
    void OnStart() override {
        m_shader = fw::Shader::FromSource(kVertexShader, kFragmentShader);

        // clang-format off
        const float vertices[] = {
            // position       // color
             0.0f,  0.6f,     1.0f, 0.35f, 0.35f,
            -0.6f, -0.5f,     0.35f, 1.0f, 0.35f,
             0.6f, -0.5f,     0.35f, 0.55f, 1.0f,
        };
        // clang-format on

        glGenVertexArrays(1, &m_vao);
        glGenBuffers(1, &m_vbo);

        glBindVertexArray(m_vao);
        glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);

        glBindVertexArray(0);
    }

    void OnFixedUpdate(double fixedDt) override {
        m_angleRadians += static_cast<float>(fixedDt) * m_spinSpeed;
    }

    void OnRender() override {
        glm::mat4 transform = glm::rotate(glm::mat4(1.0f), m_angleRadians, glm::vec3(0.0f, 0.0f, 1.0f));

        m_shader.Use();
        m_shader.SetMat4("uTransform", transform);

        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindVertexArray(0);
    }

    void OnImGui() override {
        ImGui::Begin("Framework Sanity Check");
        ImGui::Text("If you can see this and a spinning triangle,");
        ImGui::Text("GLFW + glad + Shader + ImGui are all wired up.");
        ImGui::SliderFloat("Spin speed (rad/s)", &m_spinSpeed, -10.0f, 10.0f);
        ImGui::Text("Frame time: %.3f ms", 1000.0 / ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void OnShutdown() override {
        glDeleteBuffers(1, &m_vbo);
        glDeleteVertexArrays(1, &m_vao);
    }

private:
    static fw::Application::Config MakeConfig() {
        fw::Application::Config config;
        config.title = "OpenGL Physics - Framework Sanity Check";
        return config;
    }

    fw::Shader m_shader;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    float m_angleRadians = 0.0f;
    float m_spinSpeed = 1.5f;
};

int main() {
    HelloTriangleApp app;
    app.Run();
    return 0;
}
