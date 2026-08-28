#pragma once

#include <string>

// GLSL compute-shader source for the self-gravitating SPH star: smoothed
// particle hydrodynamics (density -> pressure -> pressure/viscosity force)
// plus brute-force softened self-gravity, integrated by kick-drift-kick
// leapfrog. Kept as string builders rather than files, matching 05_tdse_gpu's
// inline-shader convention.
//
// Buffer layout (std430 SSBOs, one array element per particle):
//   binding 0  PosMass   vec4(x, y, z, mass)
//   binding 1  Vel       vec4(vx, vy, vz, _)
//   binding 2  Acc       vec4(ax, ay, az, _)          -- gravity + SPH, summed in one pass
//   binding 3  RhoPress  vec4(rho, pressure, soundspeed, _)
//
// All kernels are one-thread-per-particle with a uN bounds check (particle
// count need not be a multiple of the workgroup size, unlike the FFT's
// power-of-two rows). Self-terms (i==j) need no special-casing: the cubic
// spline's gradient is exactly zero at r=0, and gravity's own softened
// numerator (ri-rj) is exactly zero when i==j, so both loops can safely
// include j==i.

namespace tde::kernels {

inline constexpr int kWorkgroupSize = 256;

// Cubic spline kernel (Monaghan & Lattanzio 1985), 3D normalization 1/pi.
// W(q) sigma/h^3 * f(q), dW/dr = sigma/h^4 * f'(q), q = r/h.
inline const char* SphKernelGlsl = R"(
float kernelW(float r, float h) {
    float q = r / h;
    float sigma = 1.0 / (3.14159265358979 * h * h * h);
    if (q < 1.0) return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    if (q < 2.0) { float t = 2.0 - q; return sigma * 0.25 * t * t * t; }
    return 0.0;
}
// dW/dr (signed; note this is <= 0 everywhere since W is non-increasing in r).
float kernelDwDr(float r, float h) {
    float q = r / h;
    float sigma = 1.0 / (3.14159265358979 * h * h * h);
    if (q < 1.0) return (sigma / h) * (-3.0 * q + 2.25 * q * q);
    if (q < 2.0) { float t = 2.0 - q; return (sigma / h) * (-0.75 * t * t); }
    return 0.0;
}
)";

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) buffer Vel { vec4 vel[]; };
layout(std430, binding = 2) buffer Acc { vec4 acc[]; };
layout(std430, binding = 3) buffer RhoPress { vec4 rhoPress[]; };

uniform int uN;
)";
}

// Pass 1: density (kernel-weighted mass sum) + EOS pressure + sound speed,
// all rho-local so they can be written in the same pass. Polytropic EOS
// P = K*rho^Gamma matches the Lane-Emden profile the initial conditions were
// sampled from (tools/make_star_ic.py), so a correctly-sampled star starts
// (approximately) in hydrostatic equilibrium.
inline std::string Density() {
    return CommonHeader() + SphKernelGlsl + R"(
uniform float uH;
uniform float uK;
uniform float uGamma;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    float rho = 0.0;
    for (uint j = 0u; j < uint(uN); ++j) {
        float r = length(ri - posMass[j].xyz);
        rho += posMass[j].w * kernelW(r, uH);
    }
    float pressure = uK * pow(rho, uGamma);
    float csound = sqrt(uGamma * pressure / rho);
    rhoPress[i] = vec4(rho, pressure, csound, 0.0);
}
)";
}

// Pass 2: total acceleration = self-gravity (softened pairwise) + SPH
// pressure gradient + Monaghan artificial viscosity. Fused into one O(N^2)
// loop since both terms need the same pairwise iteration.
inline std::string Forces() {
    return CommonHeader() + SphKernelGlsl + R"(
uniform float uH;
uniform float uG;
uniform float uSoftening2;   // softening length squared
uniform float uViscAlpha;
uniform float uViscBeta;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    vec3 vi = vel[i].xyz;
    float rhoi = rhoPress[i].x;
    float Pi = rhoPress[i].y;
    float ci = rhoPress[i].z;

    vec3 a = vec3(0.0);
    for (uint j = 0u; j < uint(uN); ++j) {
        vec3 rij = ri - posMass[j].xyz;
        float r2 = dot(rij, rij);
        float mj = posMass[j].w;

        // self-gravity (Plummer-softened)
        float invR3 = pow(r2 + uSoftening2, -1.5);
        a -= uG * mj * invR3 * rij;

        // SPH pressure gradient + artificial viscosity
        float r = sqrt(r2);
        float dWdr = kernelDwDr(r, uH);
        if (dWdr != 0.0) {
            vec3 gradW = (r > 0.0) ? (dWdr / r) * rij : vec3(0.0);
            float rhoj = rhoPress[j].x;
            float Pj = rhoPress[j].y;

            float piVisc = 0.0;
            vec3 vij = vi - vel[j].xyz;
            float vijDotRij = dot(vij, rij);
            if (vijDotRij < 0.0) {
                float cj = rhoPress[j].z;
                float mu = uH * vijDotRij / (r2 + 0.01 * uH * uH);
                float cbar = 0.5 * (ci + cj);
                float rhobar = 0.5 * (rhoi + rhoj);
                piVisc = (-uViscAlpha * cbar * mu + uViscBeta * mu * mu) / rhobar;
            }

            a -= mj * (Pi / (rhoi * rhoi) + Pj / (rhoj * rhoj) + piVisc) * gradW;
        }
    }
    acc[i] = vec4(a, 0.0);
}
)";
}

// Leapfrog half-kick: vel += halfDt * acc.
inline std::string Kick() {
    return CommonHeader() + R"(
uniform float uHalfDt;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vel[i].xyz += uHalfDt * acc[i].xyz;
}
)";
}

// Leapfrog drift: pos += dt * vel. PosMass is read-write here (unlike the
// Density/Forces passes), so this kernel declares its own binding-0 buffer.
inline std::string Drift() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) readonly buffer Vel { vec4 vel[]; };

uniform int uN;
uniform float uDt;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    posMass[i].xyz += uDt * vel[i].xyz;
}
)";
}

} // namespace tde::kernels
