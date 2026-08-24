#include "framework/ParticleCloud.hpp"

#include <cstddef>

namespace fw {

namespace {

const char* kVertexShader = R"(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in float aSize;

uniform mat4 uView;
uniform mat4 uProjection;
uniform float uViewportHeight;

out vec4 vColor;

void main() {
    vec4 viewPos = uView * vec4(aPos, 1.0);
    gl_Position = uProjection * viewPos;

    float dist = max(-viewPos.z, 0.01);
    // Perspective-correct screen-space size from a world-space radius:
    // projection[1][1] == 1/tan(fovY/2), so this converts world size at
    // `dist` into pixels for the current viewport height.
    gl_PointSize = aSize * uProjection[1][1] * uViewportHeight * 0.5 / dist;

    vColor = aColor;
}
)";

const char* kFragmentShader = R"(
#version 460 core
in vec4 vColor;
out vec4 FragColor;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float dist2 = dot(d, d);
    if (dist2 > 0.25) discard;
    float alpha = smoothstep(0.25, 0.1, dist2);
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
)";

} // namespace

ParticleCloud::ParticleCloud() {
    m_shader = Shader::FromSource(kVertexShader, kFragmentShader);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleInstance),
                           reinterpret_cast<void*>(offsetof(ParticleInstance, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(ParticleInstance),
                           reinterpret_cast<void*>(offsetof(ParticleInstance, color)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleInstance),
                           reinterpret_cast<void*>(offsetof(ParticleInstance, size)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

ParticleCloud::~ParticleCloud() {
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void ParticleCloud::SetParticles(const std::vector<ParticleInstance>& particles) {
    m_count = static_cast<GLsizei>(particles.size());
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(particles.size() * sizeof(ParticleInstance)), particles.data(),
                 GL_DYNAMIC_DRAW);
}

void ParticleCloud::Draw(const glm::mat4& view, const glm::mat4& projection, float viewportHeightPx) {
    if (m_count == 0) return;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // additive: overlapping shells glow rather than occlude
    glDepthMask(GL_FALSE); // additive-blended points shouldn't write depth

    m_shader.Use();
    m_shader.SetMat4("uView", view);
    m_shader.SetMat4("uProjection", projection);
    m_shader.SetFloat("uViewportHeight", viewportHeightPx);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, m_count);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
    glDisable(GL_PROGRAM_POINT_SIZE);
}

} // namespace fw
