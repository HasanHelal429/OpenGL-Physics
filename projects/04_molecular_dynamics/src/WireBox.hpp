#pragma once

#include "framework/Shader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

namespace md {

// A single-color GL_LINES cube outline spanning [0,L]^3 -- the periodic
// simulation box boundary, so it's visually obvious when a particle wraps
// around an edge. No project in framework/ needed a plain wireframe
// primitive before this one (ParticleCloud and Mesh are both filled/
// point-sprite geometry), so this stays local rather than becoming
// reusable framework infrastructure for a 12-line-segment cube.
class WireBox {
public:
    WireBox();
    ~WireBox();

    WireBox(const WireBox&) = delete;
    WireBox& operator=(const WireBox&) = delete;

    void Draw(const glm::mat4& view, const glm::mat4& projection, double boxLength, glm::vec4 color);

private:
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    fw::Shader m_shader;
};

} // namespace md
