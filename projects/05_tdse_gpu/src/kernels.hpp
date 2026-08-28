#pragma once

#include <string>

// GLSL compute-shader source for the split-step-Fourier propagator, built for a
// concrete grid size N (a power of two). Kept as string builders rather than
// files to match the framework's inline-shader convention and avoid runtime
// path lookups.
//
// Complex data is vec2 (re, im) in std430 SSBOs, row-major (index = y*N + x).

namespace tdse::kernels {

inline int Log2(int n) {
    int l = 0;
    while ((1 << l) < n) ++l;
    return l;
}

// 1-D radix-2 Cooley-Tukey FFT, one workgroup per row, N/2 threads. Loads the
// row into shared memory (bit-reversed), runs log2(N) butterfly stages against
// a precomputed twiddle table, writes back scaled. uSign: -1 forward, +1
// inverse; uScale: 1 forward, 1/N inverse. This is the kernel worth tuning.
inline std::string Fft1D(int n) {
    const int logn = Log2(n);
    const int half = n / 2;
    std::string s = R"(#version 460 core
#define FFT_N )" + std::to_string(n) + R"(u
#define FFT_LOGN )" + std::to_string(logn) + R"(u
layout(local_size_x = )" + std::to_string(half) + R"() in;

layout(std430, binding = 0) buffer Data { vec2 data[]; };
layout(std430, binding = 1) readonly buffer Twiddle { vec2 tw[]; }; // length N, e^{-i 2pi m / N}

uniform int uSign;      // -1 forward, +1 inverse
uniform float uScale;   // 1.0 forward, 1.0/N inverse

shared vec2 s[FFT_N];

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

void main() {
    uint row = gl_WorkGroupID.x;
    uint base = row * FFT_N;
    uint t = gl_LocalInvocationID.x; // 0 .. N/2-1

    // Bit-reversed load: two elements per thread.
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
        uint k = t;                 // one butterfly per thread
        uint group = k / hs;
        uint pair = k - group * hs;
        uint i = group * step + pair;
        uint j = i + hs;
        uint twIdx = (pair * (FFT_N / step)) & (FFT_N - 1u);
        vec2 w = tw[twIdx];
        if (uSign > 0) w.y = -w.y;  // inverse: conjugate twiddle
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
    return s;
}

// Tiled square transpose (N x N), binding 0 -> binding 1.
inline std::string Transpose(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Src { vec2 src[]; };
layout(std430, binding = 1) writeonly buffer Dst { vec2 dst[]; };

shared vec2 tile[16][17];

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x < N && y < N) tile[gl_LocalInvocationID.y][gl_LocalInvocationID.x] = src[y * N + x];
    memoryBarrierShared();
    barrier();
    uint tx = gl_WorkGroupID.y * 16u + gl_LocalInvocationID.x;
    uint ty = gl_WorkGroupID.x * 16u + gl_LocalInvocationID.y;
    if (tx < N && ty < N) dst[ty * N + tx] = tile[gl_LocalInvocationID.x][gl_LocalInvocationID.y];
}
)";
}

// Reduce max |psi|^2 over the grid into a single uint (float bits; |psi|^2 >= 0
// so the bit order is monotonic). Clear binding 1 to 0 before dispatching.
// Used by the interactive view to auto-scale brightness to the current density.
inline std::string ReduceMax(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) buffer Stat { uint sMaxBits; };

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= N || y >= N) return;
    vec2 p = psi[y * N + x];
    atomicMax(sMaxBits, floatBitsToUint(dot(p, p)));
}
)";
}

// Probability current j = Im(conj(psi) grad psi)  (hbar = m = 1), central
// differences with periodic wrap. Reads psi (binding 0), writes j as vec2
// (binding 1).
inline std::string Current(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) writeonly buffer Cur { vec2 j[]; };

uniform float uDx;
uniform float uDy;

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= N || y >= N) return;
    int xp = (x + 1) % N, xm = (x + N - 1) % N;
    int yp = (y + 1) % N, ym = (y + N - 1) % N;
    vec2 p  = psi[y * N + x];
    vec2 gx = (psi[y * N + xp] - psi[y * N + xm]) / (2.0 * uDx);
    vec2 gy = (psi[yp * N + x] - psi[ym * N + x]) / (2.0 * uDy);
    // Im(conj(p) * g) = p.x*g.y - p.y*g.x
    j[y * N + x] = vec2(p.x * gx.y - p.y * gx.x, p.x * gy.y - p.y * gy.x);
}
)";
}

// Mean-field (Hartree) helpers.
// Density: write rho = |psi|^2 into a complex buffer (imag = 0) for the FFT.
inline std::string MeanFieldRho(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) writeonly buffer Rho { vec2 rho[]; };
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= N || y >= N) return;
    vec2 p = psi[y * N + x];
    rho[y * N + x] = vec2(dot(p, p), 0.0);
}
)";
}
// Poisson multiply in k-space: V_hat(k) = coupling * rho_hat(k) / k^2 (k=0 -> 0).
// Data is rho_hat in the transposed [kxIdx*N + kyIdx] layout.
inline std::string PoissonMul(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Data { vec2 data[]; };
uniform float uCoupling;
uniform float uKxScale;   // 2*pi / lx
uniform float uKyScale;   // 2*pi / ly
void main() {
    int a = int(gl_GlobalInvocationID.y);   // kx index
    int b = int(gl_GlobalInvocationID.x);   // ky index
    if (a >= N || b >= N) return;
    int fa = (a <= N / 2) ? a : a - N;
    int fb = (b <= N / 2) ? b : b - N;
    float kx = uKxScale * float(fa), ky = uKyScale * float(fb);
    float k2 = kx * kx + ky * ky;
    int i = a * N + b;
    data[i] = k2 > 1e-12 ? data[i] * (uCoupling / k2) : vec2(0.0);
}
)";
}
// Copy the real part of a complex buffer into a float buffer.
inline std::string ExtractReal(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Src { vec2 src[]; };
layout(std430, binding = 1) writeonly buffer Dst { float dst[]; };
void main() {
    int x = int(gl_GlobalInvocationID.x), y = int(gl_GlobalInvocationID.y);
    if (x >= N || y >= N) return;
    dst[y * N + x] = src[y * N + x].x;
}
)";
}

// Fourier-shift phase for one shear of the 3-shear rotation. Data is in
// half-transformed layout [row][col=freq bin]; this multiplies element (r, c)
// by exp(-i * k(c) * uAmount * coord(r)), which shifts row r by -uAmount*coord(r)
// along the transformed axis. coord(r) = uCoordOrigin + (r+0.5)*uCoordStep,
// k(c) = uKScale * (c <= N/2 ? c : c - N).
inline std::string ShearPhase(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Data { vec2 data[]; };

uniform float uAmount;
uniform float uCoordOrigin;
uniform float uCoordStep;
uniform float uKScale;

void main() {
    int r = int(gl_GlobalInvocationID.y);
    int c = int(gl_GlobalInvocationID.x);
    if (r >= N || c >= N) return;
    float coord = uCoordOrigin + (float(r) + 0.5) * uCoordStep;
    int fi = (c <= N / 2) ? c : c - N;
    float k = uKScale * float(fi);
    float ph = -k * uAmount * coord;
    float cs = cos(ph), sn = sin(ph);
    vec2 z = data[r * N + c];
    data[r * N + c] = vec2(z.x * cs - z.y * sn, z.x * sn + z.y * cs);
}
)";
}

// Build the half-kick multiplier Vprop = exp(-i (V0 + sum drives(x,t)) dt/2) *
// exp(-W dt/2) from the static external potential (binding 0), the absorbing
// rate W (binding 1), and up to 4 time-dependent drive terms. Dispatched once
// per substep when the deck has [[drive]] terms.
inline std::string BuildVprop(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer V0 { float v0[]; };
layout(std430, binding = 1) readonly buffer Cap { float wcap[]; };
layout(std430, binding = 2) writeonly buffer Vp { vec2 vprop[]; };
layout(std430, binding = 3) readonly buffer VH { float vh[]; };   // Hartree potential (0 if unused)

uniform float uDt;
uniform float uTime;
uniform float uX0;   // world x of the grid origin; x(i) = uX0 + (i+0.5)*uDx
uniform float uDx;
uniform float uY0;
uniform float uDy;
uniform int  uDriveCount;
uniform int  uDriveType[4];  // 0 = tilt, 1 = gate
uniform vec4 uDriveA[4];
uniform vec4 uDriveB[4];

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
    if (i >= N || j >= N) return;
    int k = j * N + i;
    float x = uX0 + (float(i) + 0.5) * uDx;
    float y = uY0 + (float(j) + 0.5) * uDy;

    float V = v0[k] + vh[k];
    for (int d = 0; d < uDriveCount; ++d) {
        if (uDriveType[d] == 0) {           // tilt: E0 sin(w t + phi) env(t) (dx*x + dy*y)
            float E0 = uDriveA[d].x, w = uDriveA[d].y, phi = uDriveA[d].z, ramp = uDriveA[d].w;
            float env = ramp > 0.0 ? clamp(uTime / ramp, 0.0, 1.0) : 1.0;
            V += E0 * sin(w * uTime + phi) * env * (uDriveB[d].x * x + uDriveB[d].y * y);
        } else if (uDriveType[d] == 1) {    // gate: amp env(t) exp(-r^2 / 2 s^2)
            float amp = uDriveA[d].x, s = uDriveA[d].y, tOn = uDriveA[d].z, tRamp = uDriveA[d].w;
            float dx = x - uDriveB[d].x, dy = y - uDriveB[d].y;
            float env = clamp((uTime - tOn) / max(tRamp, 1e-6), 0.0, 1.0);
            V += amp * env * exp(-(dx * dx + dy * dy) / (2.0 * s * s));
        }
    }

    float ph = -V * uDt * 0.5;
    float damp = exp(-wcap[k] * uDt * 0.5);
    vprop[k] = vec2(damp * cos(ph), damp * sin(ph));
}
)";
}

// Un-transpose + fftshift the diagnostic FFT result for display: reads
// psihat in the transposed, DC-at-origin layout (binding 0, index kxIdx*N+kyIdx)
// and writes it row-major with DC at the grid centre (binding 1, index y*N+x).
inline std::string FftShift(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer Src { vec2 src[]; };  // [kxIdx*N + kyIdx]
layout(std430, binding = 1) writeonly buffer Dst { vec2 dst[]; }; // [y*N + x], centred

void main() {
    int x = int(gl_GlobalInvocationID.x);
    int y = int(gl_GlobalInvocationID.y);
    if (x >= N || y >= N) return;
    int kxIdx = (x + N / 2) % N;
    int kyIdx = (y + N / 2) % N;
    dst[y * N + x] = src[kxIdx * N + kyIdx];
}
)";
}

// Pointwise complex multiply: a[i] <- a[i] * b[i]  (binding 0 *= binding 1).
inline std::string CMul(int n) {
    return R"(#version 460 core
#define N )" + std::to_string(n) + R"(u
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) buffer A { vec2 a[]; };
layout(std430, binding = 1) readonly buffer B { vec2 b[]; };

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= N || y >= N) return;
    uint i = y * N + x;
    vec2 p = a[i];
    vec2 q = b[i];
    a[i] = vec2(p.x * q.x - p.y * q.y, p.x * q.y + p.y * q.x);
}
)";
}

} // namespace tdse::kernels
