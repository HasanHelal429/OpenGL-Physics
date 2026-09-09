#include "ngrav/gpu/GpuDirect.hpp"

#include "framework/ComputeShader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <chrono>
#include <vector>

namespace ngrav::gpu {

namespace {

// One thread per target particle, plain loop over every source -- the
// naive O(N^2) sum, matching ComputeAccelDirect's own formula term-for-term
// (Plummer softening: a = -G*m*r/(r^2+eps^2)^1.5) so the two are directly
// comparable in --gpu-selftest without a second, differently-derived
// reference. i==j gives rij=(0,0,0), invR3 finite (eps^2 alone in the
// denominator) and contributes exactly zero acceleration -- no special
// case needed, matching 06_tidal_disruption's own convention.
constexpr const char* kDirectComputeSrc = R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) writeonly buffer Accel { vec4 accel[]; };

uniform int uN;
uniform float uG;
uniform float uSoftening2;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    vec3 a = vec3(0.0);
    for (uint j = 0u; j < uint(uN); ++j) {
        vec3 rij = ri - posMass[j].xyz;
        float r2 = dot(rij, rij);
        float invR3 = pow(r2 + uSoftening2, -1.5);
        a -= uG * posMass[j].w * invR3 * rij;
    }
    accel[i] = vec4(a, 0.0);
}
)";

// Lazily compiled on first use -- compiling a GL_COMPUTE_SHADER needs an
// active GL context, which this library (a plain CPU static lib otherwise)
// has no guarantee of at static-init time; the caller creates the context
// (fw::GLContext::CreateHidden or the interactive window) before ever
// calling ComputeAccelGpuDirect. Cached across calls within one context's
// lifetime -- recompiling the same program every frame would be wasteful.
fw::ComputeShader& DirectShader() {
    static fw::ComputeShader shader = fw::ComputeShader::FromSource(kDirectComputeSrc);
    return shader;
}

} // namespace

void ComputeAccelGpuDirect(const PosMassView<3>& pts, double G, double eps2, SoA<3>& out, GpuDirectStats* stats) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    if (n == 0) return;

    const auto t0 = std::chrono::steady_clock::now();

    std::vector<glm::vec4> posMass(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        posMass[ii] = glm::vec4(static_cast<float>(pts.x[ii]), static_cast<float>(pts.y[ii]),
                                static_cast<float>(pts.z[ii]), static_cast<float>(pts.m[ii]));
    }

    GLuint bufPosMass = 0, bufAccel = 0;
    glCreateBuffers(1, &bufPosMass);
    glCreateBuffers(1, &bufAccel);
    glNamedBufferData(bufPosMass, static_cast<GLsizeiptr>(posMass.size() * sizeof(glm::vec4)), posMass.data(),
                      GL_STREAM_DRAW);
    glNamedBufferData(bufAccel, static_cast<GLsizeiptr>(posMass.size() * sizeof(glm::vec4)), nullptr, GL_STREAM_READ);

    const auto t1 = std::chrono::steady_clock::now();

    fw::ComputeShader& shader = DirectShader();
    shader.Use();
    shader.SetInt("uN", n);
    shader.SetFloat("uG", static_cast<float>(G));
    shader.SetFloat("uSoftening2", static_cast<float>(eps2));
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(1, bufAccel);
    const GLuint groups = static_cast<GLuint>((n + 255) / 256);
    shader.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const auto t2 = std::chrono::steady_clock::now();

    std::vector<glm::vec4> accel(static_cast<std::size_t>(n));
    glGetNamedBufferSubData(bufAccel, 0, static_cast<GLsizeiptr>(accel.size() * sizeof(glm::vec4)), accel.data());

    glDeleteBuffers(1, &bufPosMass);
    glDeleteBuffers(1, &bufAccel);

    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        out.ax[ii] = accel[ii].x;
        out.ay[ii] = accel[ii].y;
        out.az[ii] = accel[ii].z;
    }

    const auto t3 = std::chrono::steady_clock::now();
    if (stats) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        stats->uploadMs = ms(t0, t1);
        stats->dispatchMs = ms(t1, t2);
        stats->readbackMs = ms(t2, t3);
    }
}

} // namespace ngrav::gpu
