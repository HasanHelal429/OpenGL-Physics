#include "framework/Mesh.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <utility>

namespace fw {

namespace {

uint32_t MidpointIndex(std::vector<glm::vec3>& positions, std::map<std::pair<uint32_t, uint32_t>, uint32_t>& cache,
                        uint32_t a, uint32_t b) {
    const auto key = a < b ? std::make_pair(a, b) : std::make_pair(b, a);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    const glm::vec3 mid = glm::normalize(positions[a] + positions[b]);
    const uint32_t index = static_cast<uint32_t>(positions.size());
    positions.push_back(mid);
    cache.emplace(key, index);
    return index;
}

} // namespace

MeshData GenerateIcosphere(int subdivisions) {
    const float t = (1.0f + std::sqrt(5.0f)) / 2.0f;

    std::vector<glm::vec3> positions = {
        glm::normalize(glm::vec3(-1, t, 0)),  glm::normalize(glm::vec3(1, t, 0)),  glm::normalize(glm::vec3(-1, -t, 0)),
        glm::normalize(glm::vec3(1, -t, 0)),  glm::normalize(glm::vec3(0, -1, t)), glm::normalize(glm::vec3(0, 1, t)),
        glm::normalize(glm::vec3(0, -1, -t)), glm::normalize(glm::vec3(0, 1, -t)), glm::normalize(glm::vec3(t, 0, -1)),
        glm::normalize(glm::vec3(t, 0, 1)),   glm::normalize(glm::vec3(-t, 0, -1)), glm::normalize(glm::vec3(-t, 0, 1)),
    };

    std::vector<std::array<uint32_t, 3>> faces = {
        {0, 11, 5}, {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9},  {5, 11, 4}, {11, 10, 2},
        {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},  {4, 9, 5},
        {2, 4, 11}, {6, 2, 10}, {8, 6, 7},  {9, 8, 1},
    };

    for (int level = 0; level < subdivisions; ++level) {
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpointCache;
        std::vector<std::array<uint32_t, 3>> nextFaces;
        nextFaces.reserve(faces.size() * 4);

        for (const auto& face : faces) {
            const uint32_t a = face[0], b = face[1], c = face[2];
            const uint32_t ab = MidpointIndex(positions, midpointCache, a, b);
            const uint32_t bc = MidpointIndex(positions, midpointCache, b, c);
            const uint32_t ca = MidpointIndex(positions, midpointCache, c, a);

            nextFaces.push_back({a, ab, ca});
            nextFaces.push_back({b, bc, ab});
            nextFaces.push_back({c, ca, bc});
            nextFaces.push_back({ab, bc, ca});
        }
        faces = std::move(nextFaces);
    }

    MeshData mesh;
    mesh.vertices.reserve(positions.size());
    for (const glm::vec3& p : positions) {
        mesh.vertices.push_back({p, p}); // unit sphere: normal == position
    }
    mesh.indices.reserve(faces.size() * 3);
    for (const auto& face : faces) {
        mesh.indices.push_back(face[0]);
        mesh.indices.push_back(face[1]);
        mesh.indices.push_back(face[2]);
    }
    return mesh;
}

Mesh Mesh::Upload(const MeshData& data) {
    Mesh mesh;
    mesh.m_indexCount = static_cast<GLsizei>(data.indices.size());

    glGenVertexArrays(1, &mesh.m_vao);
    glGenBuffers(1, &mesh.m_vbo);
    glGenBuffers(1, &mesh.m_ebo);

    glBindVertexArray(mesh.m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, mesh.m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.vertices.size() * sizeof(MeshVertex)), data.vertices.data(),
                 GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mesh.m_ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.indices.size() * sizeof(uint32_t)),
                 data.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(MeshVertex), reinterpret_cast<void*>(offsetof(MeshVertex, normal)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    return mesh;
}

Mesh::~Mesh() {
    if (m_ebo) glDeleteBuffers(1, &m_ebo);
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

Mesh::Mesh(Mesh&& other) noexcept
    : m_vao(other.m_vao), m_vbo(other.m_vbo), m_ebo(other.m_ebo), m_indexCount(other.m_indexCount) {
    other.m_vao = other.m_vbo = other.m_ebo = 0;
    other.m_indexCount = 0;
}

Mesh& Mesh::operator=(Mesh&& other) noexcept {
    if (this != &other) {
        if (m_ebo) glDeleteBuffers(1, &m_ebo);
        if (m_vbo) glDeleteBuffers(1, &m_vbo);
        if (m_vao) glDeleteVertexArrays(1, &m_vao);
        m_vao = other.m_vao;
        m_vbo = other.m_vbo;
        m_ebo = other.m_ebo;
        m_indexCount = other.m_indexCount;
        other.m_vao = other.m_vbo = other.m_ebo = 0;
        other.m_indexCount = 0;
    }
    return *this;
}

void Mesh::Draw() const {
    glBindVertexArray(m_vao);
    glDrawElements(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr);
    glBindVertexArray(0);
}

void Mesh::DrawInstanced(GLsizei instanceCount) const {
    glBindVertexArray(m_vao);
    glDrawElementsInstanced(GL_TRIANGLES, m_indexCount, GL_UNSIGNED_INT, nullptr, instanceCount);
    glBindVertexArray(0);
}

} // namespace fw
