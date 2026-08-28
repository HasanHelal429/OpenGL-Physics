#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <unordered_map>

namespace fw {

// A GL_COMPUTE_SHADER program with cached uniform lookups. Companion to Shader
// (which is vertex+fragment only).
class ComputeShader {
public:
    static ComputeShader FromSource(const std::string& src);

    ComputeShader() = default;
    ~ComputeShader();

    ComputeShader(const ComputeShader&) = delete;
    ComputeShader& operator=(const ComputeShader&) = delete;
    ComputeShader(ComputeShader&& other) noexcept;
    ComputeShader& operator=(ComputeShader&& other) noexcept;

    void Use() const;
    GLuint Id() const { return m_program; }

    // Dispatch `groups*` workgroups. Call Use() first.
    void Dispatch(GLuint groupsX, GLuint groupsY = 1, GLuint groupsZ = 1) const;
    // Convenience: bind an SSBO to a std430 binding point.
    static void BindBuffer(GLuint binding, GLuint buffer);
    // glMemoryBarrier wrapper; default covers SSBO reads/writes between passes.
    static void Barrier(GLbitfield barriers = GL_SHADER_STORAGE_BARRIER_BIT);

    void SetInt(const std::string& name, int value);
    void SetUInt(const std::string& name, unsigned value);
    void SetFloat(const std::string& name, float value);
    void SetVec2(const std::string& name, const glm::vec2& value);
    void SetVec4(const std::string& name, const glm::vec4& value);
    void SetIVec2(const std::string& name, const glm::ivec2& value);
    void SetIntArray(const std::string& name, const int* values, int count);
    void SetVec4Array(const std::string& name, const glm::vec4* values, int count);

private:
    explicit ComputeShader(GLuint program) : m_program(program) {}
    GLint Location(const std::string& name);

    GLuint m_program = 0;
    std::unordered_map<std::string, GLint> m_uniformCache;
};

} // namespace fw
