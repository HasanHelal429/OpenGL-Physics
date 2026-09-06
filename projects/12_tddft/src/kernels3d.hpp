#pragma once

#include <string>

// GLSL compute-shader source for the 3D split-step-Fourier rt-TDDFT
// propagator (Stage 2 of Physics Simulations/Quantum Mechanics/TDDFT). Built
// for a concrete cube size N (a power of two).
//
// Complex data is vec2 (re, im) in std430 SSBOs. The working layout is
// (z, y, x) with x contiguous:  index = (z*N + y)*N + x.
//
// The 3D FFT is done as three passes of "FFT the contiguous axis, then
// cyclically rotate the axes (z,y,x) -> (x,z,y)". Three rotations return the
// buffer to its original layout with all three axes transformed. The 1D FFT
// kernel is lifted verbatim from 05_tdse_gpu (a proven radix-2 Cooley-Tukey);
// here it is dispatched over N*N rows instead of N.

namespace tddft::kernels {

inline int Log2(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

// 1-D radix-2 Cooley-Tukey FFT, one workgroup per row of length N, N/2
// threads. Bit-reversed shared-memory load, log2(N) butterfly stages against
// a precomputed twiddle table, scaled write-back. Verbatim from
// 05_tdse_gpu/kernels.hpp Fft1D -- for the 3D case we simply Dispatch(N*N).
inline std::string Fft1D(int n) {
    const int logn = Log2(n);
    const int half = n / 2;
    return R"(#version 460 core
#define FFT_N )" + std::to_string(n) + R"(u
#define FFT_LOGN )" + std::to_string(logn) + R"(u
layout(local_size_x = )" + std::to_string(half) + R"() in;

layout(std430, binding = 0) buffer Data { vec2 data[]; };
layout(std430, binding = 1) readonly buffer Twiddle { vec2 tw[]; };

uniform int uSign;      // -1 forward, +1 inverse
uniform float uScale;   // 1.0 forward, 1.0/N inverse (per axis)

shared vec2 s[FFT_N];

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

void main() {
    uint row = gl_WorkGroupID.x;
    uint base = row * FFT_N;
    uint t = gl_LocalInvocationID.x;

    for (uint p = 0u; p < 2u; ++p) {
        uint i = t + p * (FFT_N / 2u);
        uint r = bitfieldReverse(i) >> (32u - FFT_LOGN);
        s[i] = data[base + r];
    }
    memoryBarrierShared();
    barrier();

    for (uint stage = 1u; stage <= FFT_LOGN; ++stage) {
        uint hs = 1u << (stage - 1u);
        uint step = 1u << stage;
        uint k = t;
        uint group = k / hs;
        uint pair = k - group * hs;
        uint i = group * step + pair;
        uint j = i + hs;
        uint twIdx = (pair * (FFT_N / step)) & (FFT_N - 1u);
        vec2 w = tw[twIdx];
        if (uSign > 0) w.y = -w.y;
        vec2 a = s[i];
        vec2 b = cmul(s[j], w);
        s[i] = a + b;
        s[j] = a - b;
        memoryBarrierShared();
        barrier();
    }

    for (uint p = 0u; p < 2u; ++p) {
        uint i = t + p * (FFT_N / 2u);
        data[base + i] = s[i] * uScale;
    }
}
)";
}

// Cyclic axis rotation (a, b, c) -> (c, a, b), c the contiguous axis of src,
// b the contiguous axis of dst. Applied after each 1D-FFT pass; three of them
// compose to the identity, so a full forward/inverse 3D FFT is three
// (FFT + rotate) passes.
inline std::string Rotate(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(std430, binding = 0) readonly buffer Src { vec2 src[]; };
layout(std430, binding = 1) writeonly buffer Dst { vec2 dst[]; };

void main() {
    uint c = gl_GlobalInvocationID.x;   // contiguous in src
    uint b = gl_GlobalInvocationID.y;
    uint a = gl_GlobalInvocationID.z;
    if (a >= N || b >= N || c >= N) return;
    dst[(c * N + a) * N + b] = src[(a * N + b) * N + c];
}
)";
}

// Pointwise complex multiply over the whole cube: a[i] <- a[i] * b[i].
// Dispatched flat, ceil(N^3 / 64) groups.
inline std::string CMul(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;

layout(std430, binding = 0) buffer A { vec2 a[]; };
layout(std430, binding = 1) readonly buffer B { vec2 b[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    vec2 p = a[i];
    vec2 q = b[i];
    a[i] = vec2(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x);
}
)";
}

// Accumulate weighted density: rho[i] += w * |psi[i]|^2  (rho is a float
// buffer). Called once per occupied orbital.
inline std::string AccumRho(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) buffer Rho { float rho[]; };
uniform float uWeight;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    vec2 p = psi[i];
    rho[i] += uWeight * dot(p, p);
}
)";
}

// Half-kick multiplier from a real potential: vprop[i] = exp(-i V[i] dt/2).
inline std::string BuildVprop(int n) {
    const long total = (long)n * n * n;
    return R"(#version 460 core
#define TOTAL )" + std::to_string(total) + R"(u
layout(local_size_x = 64) in;
layout(std430, binding = 0) readonly buffer V { float v[]; };
layout(std430, binding = 1) writeonly buffer Vp { vec2 vprop[]; };
uniform float uDt;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= TOTAL) return;
    float ph = -0.5 * v[i] * uDt;
    vprop[i] = vec2(cos(ph), sin(ph));
}
)";
}

} // namespace tddft::kernels
