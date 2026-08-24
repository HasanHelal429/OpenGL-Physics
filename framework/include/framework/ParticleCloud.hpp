#pragma once

#include "framework/Shader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <vector>

namespace fw {

struct ParticleInstance {
    glm::vec3 position;
    glm::vec4 color; // rgba, alpha used with additive blending
    float size = 0.05f; // world-space radius
};

// GPU point-sprite cloud: one GL_POINTS vertex per particle (the hardware-
// native way to draw large point clouds -- cheaper than instancing a quad
// mesh per point), perspective-attenuated size, soft circular falloff and
// additive blending in the fragment shader for a glow-like look. Re-upload
// only when the particle set actually changes (SetParticles), not per
// frame -- Draw just issues one draw call against the last-uploaded buffer.
class ParticleCloud {
public:
    ParticleCloud();
    ~ParticleCloud();

    ParticleCloud(const ParticleCloud&) = delete;
    ParticleCloud& operator=(const ParticleCloud&) = delete;

    void SetParticles(const std::vector<ParticleInstance>& particles);

    void Draw(const glm::mat4& view, const glm::mat4& projection, float viewportHeightPx);

private:
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    Shader m_shader;
    GLsizei m_count = 0;
};

} // namespace fw
