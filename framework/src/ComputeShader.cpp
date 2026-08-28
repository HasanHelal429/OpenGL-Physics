#include "framework/ComputeShader.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <stdexcept>
#include <vector>

namespace fw {

ComputeShader ComputeShader::FromSource(const std::string& src) {
    GLuint shader = glCreateShader(GL_COMPUTE_SHADER);
    const char* csrc = src.c_str();
    glShaderSource(shader, 1, &csrc, nullptr);
    glCompileShader(shader);

    GLint ok = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 0 ? len : 1);
        glGetShaderInfoLog(shader, len, nullptr, log.data());
        glDeleteShader(shader);
        throw std::runtime_error(std::string("Compute shader compile error: ") + log.data());
    }

    GLuint program = glCreateProgram();
    glAttachShader(program, shader);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 0 ? len : 1);
        glGetProgramInfoLog(program, len, nullptr, log.data());
        glDeleteProgram(program);
        glDeleteShader(shader);
        throw std::runtime_error(std::string("Compute shader link error: ") + log.data());
    }
    glDeleteShader(shader);
    return ComputeShader(program);
}

ComputeShader::~ComputeShader() {
    if (m_program) glDeleteProgram(m_program);
}

ComputeShader::ComputeShader(ComputeShader&& other) noexcept
    : m_program(other.m_program), m_uniformCache(std::move(other.m_uniformCache)) {
    other.m_program = 0;
}

ComputeShader& ComputeShader::operator=(ComputeShader&& other) noexcept {
    if (this != &other) {
        if (m_program) glDeleteProgram(m_program);
        m_program = other.m_program;
        m_uniformCache = std::move(other.m_uniformCache);
        other.m_program = 0;
    }
    return *this;
}

void ComputeShader::Use() const { glUseProgram(m_program); }

void ComputeShader::Dispatch(GLuint gx, GLuint gy, GLuint gz) const {
    glDispatchCompute(gx, gy, gz);
}

void ComputeShader::BindBuffer(GLuint binding, GLuint buffer) {
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buffer);
}

void ComputeShader::Barrier(GLbitfield barriers) {
    glMemoryBarrier(barriers);
}

GLint ComputeShader::Location(const std::string& name) {
    auto it = m_uniformCache.find(name);
    if (it != m_uniformCache.end()) return it->second;
    GLint loc = glGetUniformLocation(m_program, name.c_str());
    m_uniformCache.emplace(name, loc);
    return loc;
}

void ComputeShader::SetInt(const std::string& name, int value) { glUniform1i(Location(name), value); }
void ComputeShader::SetUInt(const std::string& name, unsigned value) { glUniform1ui(Location(name), value); }
void ComputeShader::SetFloat(const std::string& name, float value) { glUniform1f(Location(name), value); }
void ComputeShader::SetVec2(const std::string& name, const glm::vec2& v) { glUniform2fv(Location(name), 1, glm::value_ptr(v)); }
void ComputeShader::SetVec4(const std::string& name, const glm::vec4& v) { glUniform4fv(Location(name), 1, glm::value_ptr(v)); }
void ComputeShader::SetIVec2(const std::string& name, const glm::ivec2& v) { glUniform2iv(Location(name), 1, glm::value_ptr(v)); }
void ComputeShader::SetIntArray(const std::string& name, const int* values, int count) { glUniform1iv(Location(name), count, values); }
void ComputeShader::SetVec4Array(const std::string& name, const glm::vec4* values, int count) {
    glUniform4fv(Location(name), count, glm::value_ptr(values[0]));
}

} // namespace fw
