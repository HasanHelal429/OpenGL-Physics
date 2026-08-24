#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

namespace fw {

struct MeshVertex {
    glm::vec3 position;
    glm::vec3 normal;
};

struct MeshData {
    std::vector<MeshVertex> vertices;
    std::vector<uint32_t> indices;
};

// Unit sphere centered at the origin, built by subdividing an icosahedron.
// subdivisions=0 gives the base 20-triangle icosahedron; each further level
// roughly quadruples the triangle count. 2-3 is plenty for a shaded, mostly
// round-looking sphere at typical viewing distances.
MeshData GenerateIcosphere(int subdivisions);

// GPU-resident mesh (position+normal interleaved, indexed triangles).
class Mesh {
public:
    static Mesh Upload(const MeshData& data);

    Mesh() = default;
    ~Mesh();

    Mesh(const Mesh&) = delete;
    Mesh& operator=(const Mesh&) = delete;
    Mesh(Mesh&& other) noexcept;
    Mesh& operator=(Mesh&& other) noexcept;

    void Draw() const;
    // Assumes per-instance attributes have already been bound to this
    // mesh's VAO by the caller (see ParticleCloud-style instanced draws).
    void DrawInstanced(GLsizei instanceCount) const;

    GLuint VAO() const { return m_vao; }

private:
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLuint m_ebo = 0;
    GLsizei m_indexCount = 0;
};

} // namespace fw
