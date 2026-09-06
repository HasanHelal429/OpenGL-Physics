#pragma once

#include <string>

// Phase-8 GLSL: the Kohn-Sham effective potential (nuclear + FFT-Poisson
// Hartree + ALDA exchange-correlation), imaginary-time relaxation, real-time
// propagation with a density-dependent V, the delta-kick, and a shared-memory
// moment reduction. Layout convention as kernels3d.hpp: vec2 complex,
// index = (z*N + y)*N + x, x contiguous; coordinate x[i] = (i - N/2)*dx.

namespace tddft::ks {

// rho[i] = weight * |psi[i]|^2   (float output; call once per occupied orbital
// with weight = occupation, accumulate=false on the first).
inline std::string Density(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) buffer Rho { float rho[]; };
uniform float uWeight;
uniform int uAccumulate;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    vec2 p = psi[i];
    float v = uWeight * dot(p, p);
    rho[i] = (uAccumulate != 0) ? rho[i] + v : v;
}
)";
}

// Real density -> complex buffer for the Poisson FFT: c[i] = vec2(rho[i], 0).
inline std::string RealToComplex(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer R { float r[]; };
layout(std430, binding = 1) writeonly buffer C { vec2 c[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < TOTAL) c[i] = vec2(r[i], 0.0); }
)";
}

// k-space Poisson multiply: data[i] *= 4*pi / |k|^2   (k = 0 -> 0), in the
// (z,y,x) layout with the standard fftfreq sign convention.
inline std::string PoissonK(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(std430, binding = 0) buffer Data { vec2 data[]; };
uniform float uKScale;   // 2*pi / L
void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    int z = int(gl_GlobalInvocationID.z);
    if (x >= N || y >= N || z >= N) return;
    int fx = (x <= N/2) ? x : x - N;
    int fy = (y <= N/2) ? y : y - N;
    int fz = (z <= N/2) ? z : z - N;
    float k2 = uKScale*uKScale * (float(fx*fx) + float(fy*fy) + float(fz*fz));
    int i = (z*N + y)*N + x;
    const float FOUR_PI = 12.566370614359172;
    data[i] = (k2 > 1e-12) ? data[i] * (FOUR_PI / k2) : vec2(0.0);
}
)";
}

// Assemble V_eff = V_nuc + Re(V_H_complex) + v_xc[rho]  (ALDA: Slater exchange
// at alpha = 2/3  +  Perdew-Zunger 1981 correlation potential). rho is the
// spin-summed density.
inline std::string AssembleVeff(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Vnuc { float vnuc[]; };
layout(std430, binding = 1) readonly buffer VhC  { vec2 vhc[]; };   // Hartree, complex (take .x)
layout(std430, binding = 2) readonly buffer Rho  { float rho[]; };
layout(std430, binding = 3) writeonly buffer Veff { float veff[]; };

const float PI = 3.14159265358979324;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    float r = max(rho[i], 1e-30);

    // Slater exchange, alpha = 2/3:  V_x = -3 alpha (3 rho / 8 pi)^{1/3}
    float vx = -2.0 * pow(3.0 * r / (8.0 * PI), 1.0 / 3.0);

    // PZ81 correlation potential
    float rs = pow(3.0 / (4.0 * PI * r), 1.0 / 3.0);
    float vc;
    if (rs < 1.0) {
        float A = 0.0311, B = -0.0480, C = 0.0020, D = -0.0116;
        float l = log(rs);
        vc = A * l + (B - A / 3.0) + (2.0 / 3.0) * C * rs * l + ((2.0 * D - C) / 3.0) * rs;
    } else {
        float g = -0.1423, b1 = 1.0529, b2 = 0.3334;
        float srs = sqrt(rs);
        float den = 1.0 + b1 * srs + b2 * rs;
        float epsc = g / den;
        vc = epsc * (1.0 + (7.0 / 6.0) * b1 * srs + (4.0 / 3.0) * b2 * rs) / den;
    }
    veff[i] = vnuc[i] + vhc[i].x + vx + vc;
}
)";
}

// Real-time half kick from a real potential: psi[i] *= exp(-i V[i] dt/2).
inline std::string HalfKick(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer V { float v[]; };
uniform float uDt;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    float ph = -0.5 * v[i] * uDt;
    float c = cos(ph), s = sin(ph);
    vec2 p = psi[i];
    psi[i] = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
}
)";
}

// Imaginary-time half step (real, non-unitary): psi[i] *= exp(-V[i] dtau/2).
inline std::string HalfKickImag(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer V { float v[]; };
uniform float uDtau;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    psi[i] *= exp(-0.5 * v[i] * uDtau);
}
)";
}

// Multiply psi-hat by a precomputed real k-space factor (imaginary-time
// kinetic step: exp(-k^2 dtau / 2)).
inline std::string MulRealK(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer K { float k[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < TOTAL) psi[i] *= k[i]; }
)";
}

// Instantaneous dipole/boost kick: psi[i] *= exp(i kappa * coord_axis).
inline std::string Kick(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
#define TOTAL )" + std::to_string((long)n * n * n) + R"(u
#define HALF )" + std::to_string(n / 2) + R"(.0
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
uniform float uKappa;
uniform float uDx;
uniform int uAxis;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= TOTAL) return;
    uint x = g % N;
    uint y = (g / N) % N;
    uint z = g / (N * N);
    uint c = (uAxis == 0) ? x : (uAxis == 1) ? y : z;
    float coord = (float(c) - HALF) * uDx;
    float ph = uKappa * coord;
    float cs = cos(ph), sn = sin(ph);
    vec2 p = psi[g];
    psi[g] = vec2(p.x * cs - p.y * sn, p.x * sn + p.y * cs);
}
)";
}

// Shared-memory partial-sum reduction of  coord_axis(i) * |psi[i]|^2  (uAxis
// in 0..2), or of |psi[i]|^2 when uAxis < 0. One float per workgroup; sum the
// partials on the CPU, then multiply by dx^3.
inline std::string ReduceMoment(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
#define TOTAL )" + std::to_string((long)n * n * n) + R"(u
#define HALF )" + std::to_string(n / 2) + R"(.0
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) writeonly buffer Part { float part[]; };
uniform int uAxis;   // -1 -> just |psi|^2 ; 0/1/2 -> coord * |psi|^2
uniform float uDx;
shared float s[256];
void main() {
    uint g = gl_GlobalInvocationID.x;
    uint l = gl_LocalInvocationID.x;
    float v = 0.0;
    if (g < TOTAL) {
        vec2 p = psi[g];
        float d = dot(p, p);
        if (uAxis < 0) {
            v = d;
        } else {
            uint x = g % N;
            uint y = (g / N) % N;
            uint z = g / (N * N);
            uint c = (uAxis == 0) ? x : (uAxis == 1) ? y : z;
            v = (float(c) - HALF) * uDx * d;
        }
    }
    s[l] = v;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (l < stride) s[l] += s[l + stride];
        barrier();
    }
    if (l == 0u) part[gl_WorkGroupID.x] = s[0];
}
)";
}

// veff[i] += uField * coord_axis(i)   (length-gauge laser term E(t) x).
inline std::string AddLinearField(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
#define TOTAL )" + std::to_string((long)n * n * n) + R"(u
#define HALF )" + std::to_string(n / 2) + R"(.0
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Veff { float veff[]; };
uniform float uField;
uniform float uDx;
uniform int uAxis;
void main() {
    uint g = gl_GlobalInvocationID.x;
    if (g >= TOTAL) return;
    uint x = g % N, y = (g / N) % N, z = g / (N * N);
    uint c = (uAxis == 0) ? x : (uAxis == 1) ? y : z;
    veff[g] += uField * (float(c) - HALF) * uDx;
}
)";
}

// psi[i] *= mask[i]   (real absorbing-boundary window).
inline std::string MaskMul(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer M { float m[]; };
void main() { uint i = gl_GlobalInvocationID.x; if (i < TOTAL) psi[i] *= m[i]; }
)";
}

// psi[i] *= uFactor  (real scalar).
inline std::string Scale(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Psi { vec2 psi[]; };
uniform float uFactor;
void main() { uint i = gl_GlobalInvocationID.x; if (i < TOTAL) psi[i] *= uFactor; }
)";
}

// Shared-memory partial sums of  w[i] * |a[i]|^2  (a complex, w real). Used for
// <V_eff> (w = V_eff) and <T> (w = k^2/2, on the FFT'd buffer). One float per
// workgroup; sum on the CPU.
inline std::string ReduceWeighted(int n) {
    return R"(#version 460 core
#define TOTAL )" + std::to_string((long)n * n * n) + R"(u
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer A { vec2 a[]; };
layout(std430, binding = 1) readonly buffer W { float w[]; };
layout(std430, binding = 2) writeonly buffer Part { float part[]; };
shared float s[256];
void main() {
    uint g = gl_GlobalInvocationID.x;
    uint l = gl_LocalInvocationID.x;
    float v = 0.0;
    if (g < TOTAL) { vec2 p = a[g]; v = w[g] * dot(p, p); }
    s[l] = v;
    barrier();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (l < stride) s[l] += s[l + stride];
        barrier();
    }
    if (l == 0u) part[gl_WorkGroupID.x] = s[0];
}
)";
}

} // namespace tddft::ks
