#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <filesystem>
#include <string>
#include <unordered_map>

namespace fw {

// Compiles and links a vertex+fragment GLSL program, with cached uniform
// location lookups and glm-typed setters.
class Shader {
public:
    // Loads and compiles shader source from files on disk.
    static Shader FromFiles(const std::filesystem::path& vertexPath,
                             const std::filesystem::path& fragmentPath);
    // Compiles shader source already in memory.
    static Shader FromSource(const std::string& vertexSrc, const std::string& fragmentSrc);

    Shader() = default;
    ~Shader();

    Shader(const Shader&) = delete;
    Shader& operator=(const Shader&) = delete;
    Shader(Shader&& other) noexcept;
    Shader& operator=(Shader&& other) noexcept;

    void Use() const;
    GLuint Id() const { return m_program; }

    void SetBool(const std::string& name, bool value);
    void SetInt(const std::string& name, int value);
    void SetFloat(const std::string& name, float value);
    void SetVec2(const std::string& name, const glm::vec2& value);
    void SetVec3(const std::string& name, const glm::vec3& value);
    void SetVec4(const std::string& name, const glm::vec4& value);
    void SetMat3(const std::string& name, const glm::mat3& value);
    void SetMat4(const std::string& name, const glm::mat4& value);

private:
    explicit Shader(GLuint program) : m_program(program) {}

    GLint Location(const std::string& name);
    static GLuint CompileStage(GLenum stage, const std::string& src);
    static GLuint LinkProgram(GLuint vertex, GLuint fragment);

    GLuint m_program = 0;
    std::unordered_map<std::string, GLint> m_uniformCache;
};

} // namespace fw
