#pragma once

#include <string>

// GLSL compute-shader port of Euler1D.{hpp,cpp} -- same algorithm, same
// formulas, transliterated to GLSL for a bit-for-bit-in-spirit (float32 vs
// the CPU's double) cross-check (see main.cpp's SelfTestGpu). 1D
// compressible Euler, ideal Gamma-law gas, HLLC flux, MinMod-limited
// piecewise-linear reconstruction, RK2 (Heun) time integration.
//
// Boundary condition, without any ghost cells in the buffers: every pass
// clamps its neighbor/interface cell index to [0, N-1]. At i=0 this makes
// the "left neighbor" cell 0 itself, forcing MinMod's slope there to
// exactly 0 (one of the two differences it minmods is exactly zero) --
// which is exactly what Euler1D's explicit zero-gradient ghost cells also
// produce (see Euler1D.cpp's ApplyBoundary): a zero reconstruction slope at
// the boundary cell, so the domain-edge flux reduces to the boundary
// cell's own physical flux (HLLC's self-consistency identity, checked by
// SelfTestHllcConsistency). Clamping is therefore not an approximation of
// the CPU's ghost-cell boundary -- it produces the identical result by a
// different (and here, simpler) route.
//
// Buffer layout (std430 SSBOs, N cells unless noted):
//   binding 0  Cons0   vec4(rho, mom, E, _)   -- ping-pong slot A
//   binding 1  Cons1   vec4(rho, mom, E, _)   -- ping-pong slot B
//   binding 2  Stage1  vec4(rho, mom, E, _)   -- RK2 intermediate state U1
//   binding 3  Stage2  vec4(rho, mom, E, _)   -- RK2 intermediate state U2
//   binding 4  Prim    vec4(rho, u, p, _)     -- recovered from whichever
//                                                Cons buffer is "current"
//   binding 5  Flux    vec4(F_rho, F_mom, F_E, _), N+1 entries (interfaces)
//   binding 6  Slope   vec4(d_rho, d_u, d_p, _) -- MinMod slope per cell

namespace cf::kernels {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uN;
uniform float uGamma;
)";
}

// Closed-form primitive<->conserved conversion and the physical/HLLC flux --
// identical algebra to Euler1D.cpp's ToPrim/ToCons/Flux/HllcFlux, with
// (rho,u,p) and (rho,mom,E) packed as vec3 instead of separate structs.
inline const char* HllcGlsl = R"(
vec3 primToCons(vec3 prim, float gamma) {
    float rho = prim.x, u = prim.y, p = prim.z;
    float mom = rho * u;
    float E = p / (gamma - 1.0) + 0.5 * rho * u * u;
    return vec3(rho, mom, E);
}

vec3 fluxOf(vec3 prim, float gamma) {
    vec3 c = primToCons(prim, gamma);
    float u = prim.y, p = prim.z;
    return vec3(c.y, c.y * u + p, u * (c.z + p));
}

vec3 hllcStarFlux(vec3 k, vec3 uk, vec3 fk, float sk, float sStar) {
    float coef = k.x * (sk - k.y) / (sk - sStar);
    vec3 uStar = vec3(coef, coef * sStar,
                       coef * (uk.z / k.x + (sStar - k.y) * (sStar + k.z / (k.x * (sk - k.y)))));
    return fk + sk * (uStar - uk);
}

// left/right: primitive (rho,u,p) states either side of the interface.
vec3 hllcFlux(vec3 left, vec3 right, float gamma) {
    float cL = sqrt(gamma * left.z / left.x);
    float cR = sqrt(gamma * right.z / right.x);
    float sL = min(left.y - cL, right.y - cR);
    float sR = max(left.y + cL, right.y + cR);

    vec3 fL = fluxOf(left, gamma);
    if (sL >= 0.0) return fL;
    vec3 fR = fluxOf(right, gamma);
    if (sR <= 0.0) return fR;

    float sStar = (right.z - left.z + left.x * left.y * (sL - left.y) - right.x * right.y * (sR - right.y)) /
                  (left.x * (sL - left.y) - right.x * (sR - right.y));

    if (sStar >= 0.0) return hllcStarFlux(left, primToCons(left, gamma), fL, sL, sStar);
    return hllcStarFlux(right, primToCons(right, gamma), fR, sR, sStar);
}
)";

// Pass: Cons (rho,mom,E) -> Prim (rho,u,p), per cell. Closed-form -- no
// Newton iteration (unlike 07_grhd's relativistic con2prim).
inline std::string ConsToPrim() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) buffer PrimBuf { vec4 prim[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vec4 c = cons[i];
    float rho = c.x;
    float u = c.y / rho;
    float p = (uGamma - 1.0) * (c.z - 0.5 * c.y * c.y / rho);
    prim[i] = vec4(rho, u, p, 0.0);
}
)";
}

// Pass: MinMod-limited slope per cell, from Prim. See the header comment for
// why clamped neighbor indices reproduce Euler1D's ghost-cell boundary
// exactly.
inline std::string ComputeSlopes() {
    return CommonHeader() + R"(
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 6) buffer SlopeBuf { vec4 slope[]; };

float minmod(float a, float b) {
    if (a * b <= 0.0) return 0.0;
    return abs(a) < abs(b) ? a : b;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    uint im1 = (i == 0u) ? 0u : (i - 1u);
    uint ip1 = (i + 1u >= uint(uN)) ? uint(uN - 1) : (i + 1u);
    vec3 pm = prim[im1].xyz, p0 = prim[i].xyz, pp = prim[ip1].xyz;
    vec3 d;
    d.x = minmod(p0.x - pm.x, pp.x - p0.x);
    d.y = minmod(p0.y - pm.y, pp.y - p0.y);
    d.z = minmod(p0.z - pm.z, pp.z - p0.z);
    slope[i] = vec4(d, 0.0);
}
)";
}

// Pass: HLLC flux at each of the N+1 cell interfaces, from MinMod-extrapolated
// face states.
inline std::string Fluxes() {
    return CommonHeader() + HllcGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 6) readonly buffer SlopeBuf { vec4 slope[]; };
layout(std430, binding = 5) buffer FluxBuf { vec4 flux[]; };

void main() {
    uint i = gl_GlobalInvocationID.x; // interface index, 0..uN
    if (i > uint(uN)) return;
    int iL = clamp(int(i) - 1, 0, uN - 1);
    int iR = clamp(int(i), 0, uN - 1);
    vec3 faceR = prim[iL].xyz + 0.5 * slope[iL].xyz; // right-extrapolated state of cell iL
    vec3 faceL = prim[iR].xyz - 0.5 * slope[iR].xyz; // left-extrapolated state of cell iR
    flux[i] = vec4(hllcFlux(faceR, faceL, uGamma), 0.0);
}
)";
}

// Pass: one explicit Euler flux-divergence update -- identical role to
// 07_grhd's EulerStep (used twice per RK2/Heun step, see kernels.hpp there).
inline std::string EulerStep() {
    return CommonHeader() + R"(
uniform float uDtOverDx;

layout(std430, binding = 0) readonly buffer ConsIn { vec4 consIn[]; };
layout(std430, binding = 5) readonly buffer FluxBuf { vec4 flux[]; };
layout(std430, binding = 1) buffer ConsOut { vec4 consOut[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    consOut[i] = consIn[i] - uDtOverDx * (flux[i + 1u] - flux[i]);
}
)";
}

// Pass: consOut = 0.5*(consA + consB) -- the RK2/Heun combination step.
inline std::string Combine() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsA { vec4 consA[]; };
layout(std430, binding = 1) readonly buffer ConsB { vec4 consB[]; };
layout(std430, binding = 2) buffer ConsOut { vec4 consOut[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    consOut[i] = 0.5 * (consA[i] + consB[i]);
}
)";
}

} // namespace cf::kernels
