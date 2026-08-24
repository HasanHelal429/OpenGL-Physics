#include "framework/Shader.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fw {

namespace {

std::string ReadFile(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace

GLuint Shader::CompileStage(GLenum stage, const std::string& src) {
    GLuint id = glCreateShader(stage);
    const char* csrc = src.c_str();
    glShaderSource(id, 1, &csrc, nullptr);
    glCompileShader(id);

    GLint success = GL_FALSE;
    glGetShaderiv(id, GL_COMPILE_STATUS, &success);
    if (!success) {
        GLint logLen = 0;
        glGetShaderiv(id, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen > 0 ? logLen : 1);
        glGetShaderInfoLog(id, logLen, nullptr, log.data());
        glDeleteShader(id);
        const char* stageName = (stage == GL_VERTEX_SHADER) ? "vertex" : "fragment";
        throw std::runtime_error(std::string("Shader compile error (") + stageName + "): " + log.data());
    }
    return id;
}

GLuint Shader::LinkProgram(GLuint vertex, GLuint fragment) {
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);

    GLint success = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        GLint logLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &logLen);
        std::vector<char> log(logLen > 0 ? logLen : 1);
        glGetProgramInfoLog(program, logLen, nullptr, log.data());
        glDeleteProgram(program);
        throw std::runtime_error(std::string("Shader link error: ") + log.data());
    }
    return program;
}

Shader Shader::FromFiles(const std::filesystem::path& vertexPath, const std::filesystem::path& fragmentPath) {
    return FromSource(ReadFile(vertexPath), ReadFile(fragmentPath));
}

Shader Shader::FromSource(const std::string& vertexSrc, const std::string& fragmentSrc) {
    GLuint vertex = CompileStage(GL_VERTEX_SHADER, vertexSrc);
    GLuint fragment = CompileStage(GL_FRAGMENT_SHADER, fragmentSrc);
    GLuint program = LinkProgram(vertex, fragment);
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    return Shader(program);
}

Shader::~Shader() {
    if (m_program) {
        glDeleteProgram(m_program);
    }
}

Shader::Shader(Shader&& other) noexcept
    : m_program(other.m_program)
    , m_uniformCache(std::move(other.m_uniformCache)) {
    other.m_program = 0;
}

Shader& Shader::operator=(Shader&& other) noexcept {
    if (this != &other) {
        if (m_program) {
            glDeleteProgram(m_program);
        }
        m_program = other.m_program;
        m_uniformCache = std::move(other.m_uniformCache);
        other.m_program = 0;
    }
    return *this;
}

void Shader::Use() const {
    glUseProgram(m_program);
}

GLint Shader::Location(const std::string& name) {
    auto it = m_uniformCache.find(name);
    if (it != m_uniformCache.end()) {
        return it->second;
    }
    GLint loc = glGetUniformLocation(m_program, name.c_str());
    m_uniformCache.emplace(name, loc);
    return loc;
}

void Shader::SetBool(const std::string& name, bool value) { glUniform1i(Location(name), static_cast<int>(value)); }
void Shader::SetInt(const std::string& name, int value) { glUniform1i(Location(name), value); }
void Shader::SetFloat(const std::string& name, float value) { glUniform1f(Location(name), value); }
void Shader::SetVec2(const std::string& name, const glm::vec2& value) { glUniform2fv(Location(name), 1, glm::value_ptr(value)); }
void Shader::SetVec3(const std::string& name, const glm::vec3& value) { glUniform3fv(Location(name), 1, glm::value_ptr(value)); }
void Shader::SetVec4(const std::string& name, const glm::vec4& value) { glUniform4fv(Location(name), 1, glm::value_ptr(value)); }
void Shader::SetMat3(const std::string& name, const glm::mat3& value) { glUniformMatrix3fv(Location(name), 1, GL_FALSE, glm::value_ptr(value)); }
void Shader::SetMat4(const std::string& name, const glm::mat4& value) { glUniformMatrix4fv(Location(name), 1, GL_FALSE, glm::value_ptr(value)); }

} // namespace fw
