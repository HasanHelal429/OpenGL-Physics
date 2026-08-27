#include "WireBox.hpp"

namespace md {

namespace {

const char* kVertexShader = R"(
#version 460 core
layout (location = 0) in vec3 aPos;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uBoxLength;

void main() {
    gl_Position = uProjection * uView * vec4(aPos * uBoxLength, 1.0);
}
)";

const char* kFragmentShader = R"(
#version 460 core
uniform vec4 uColor;
out vec4 FragColor;

void main() {
    FragColor = uColor;
}
)";

// Unit cube [0,1]^3 edges as 12 GL_LINES segments (24 vertices); scaled to
// the live box length by uBoxLength in the vertex shader so the VBO never
// needs re-uploading when the box is resized.
constexpr float kUnitCubeEdges[] = {
    0, 0, 0, 1, 0, 0,  1, 0, 0, 1, 1, 0,  1, 1, 0, 0, 1, 0,  0, 1, 0, 0, 0, 0, // bottom face
    0, 0, 1, 1, 0, 1,  1, 0, 1, 1, 1, 1,  1, 1, 1, 0, 1, 1,  0, 1, 1, 0, 0, 1, // top face
    0, 0, 0, 0, 0, 1,  1, 0, 0, 1, 0, 1,  1, 1, 0, 1, 1, 1,  0, 1, 0, 0, 1, 1, // verticals
};

} // namespace

WireBox::WireBox() {
    m_shader = fw::Shader::FromSource(kVertexShader, kFragmentShader);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(kUnitCubeEdges), kUnitCubeEdges, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
}

WireBox::~WireBox() {
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void WireBox::Draw(const glm::mat4& view, const glm::mat4& projection, double boxLength, glm::vec4 color) {
    m_shader.Use();
    m_shader.SetMat4("uView", view);
    m_shader.SetMat4("uProjection", projection);
    m_shader.SetFloat("uBoxLength", static_cast<float>(boxLength));
    m_shader.SetVec4("uColor", color);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_LINES, 0, 24);
    glBindVertexArray(0);
}

} // namespace md
