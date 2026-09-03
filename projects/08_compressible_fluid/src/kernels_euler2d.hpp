#pragma once

#include <string>

// GLSL compute-shader port of Euler2D's CORE inviscid method only --
// deliberately scoped the same way kernels_euler1d.hpp is scoped relative to
// Euler1D: HLLC flux, MinMod-limited piecewise-linear reconstruction, RK2
// (Heun) time integration, Strang-split into X/Y sweeps -- with NO
// viscosity, no obstacle mask, no tracer field, and no boundary condition
// other than the zero-gradient "outflow" every side defaults to in
// Euler2D.hpp. This is validated only via main.cpp's SelfTestGpu2D (a
// GPU-vs-CPU cross-check against Euler2D::Step on the isentropic-vortex
// scenario, see SelfTestVortexAdvection) -- it is not wired into any
// deck-driven run, exactly like Euler1D's GPU port.
//
// Each of the three Strang sub-steps (X half, Y full, X half) is one
// fractional RK2 step over the WHOLE grid at once, dispatched as a single
// compute pass rather than Euler2D.cpp's per-row/per-column CPU loop --
// mathematically identical, since dimensional splitting already treats
// every row (X sweep) or column (Y sweep) as independent, so doing all of
// them in one dispatch is just data-parallelism over that same
// independence. An `uAxis` uniform (0=X, 1=Y) selects the neighbor
// stride/interface layout each pass uses, so every kernel below has one
// GLSL definition covering both sweep directions instead of two near-
// duplicate ones.
//
// Boundary condition, without ghost cells: exactly kernels_euler1d.hpp's
// clamped-index trick, generalized to 2D -- MinMod's slope at a domain-edge
// cell clamps its "outside" neighbor back to itself, forcing a zero slope
// there, which reproduces Euler2D.cpp's zero-gradient (Outflow) ghost cells
// exactly (see kernels_euler1d.hpp's header comment for the full argument;
// unchanged here since it's a per-line, per-boundary argument that doesn't
// care how many lines are processed at once).
//
// Buffer layout (std430 SSBOs; Cons/Prim/Slope sized nx*ny, row-major
// idx=j*nx+i like Euler2D.hpp's m_u; Flux sized to the LARGER of the two
// interface counts, (nx+1)*ny and nx*(ny+1), and reused for both axes since
// the three sub-steps run strictly sequentially, never concurrently):
//   binding 0  Cons0   vec4(rho, momX, momY, energy)  -- ping-pong slot A
//   binding 1  Cons1   vec4(rho, momX, momY, energy)  -- ping-pong slot B
//   binding 2  Stage1  vec4(...)  -- RK2 intermediate state U1
//   binding 3  Stage2  vec4(...)  -- RK2 intermediate state U2
//   binding 4  Prim    vec4(rho, u, v, p)
//   binding 5  Flux    vec4(F_rho, F_normalMom, F_transverseMom, F_E)
//   binding 6  Slope   vec4(d_rho, d_u, d_v, d_p) -- MinMod slope per cell

namespace cf::kernels2d {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uNx;
uniform int uNy;
uniform int uAxis; // 0 = X sweep, 1 = Y sweep
uniform float uGamma;
)";
}

// Closed-form conversions and the direction-specific HLLC flux -- identical
// algebra to Euler2D.cpp's ToPrim2D/ToCons2D/FluxX2D/FluxY2D/HllcFluxX/
// HllcFluxY, with (rho,u,v,p) and (rho,momX,momY,E) packed as vec4. The
// transverse momentum component is carried through unchanged into whichever
// star state the sampled point falls in (Toro sec. 10.4's passive-variable
// extension) -- same treatment CPU HllcFluxX/Y give it, just without the
// tracer field (out of scope for this port).
inline const char* HllcGlsl = R"(
vec4 primToCons(vec4 prim, float gamma) {
    float rho = prim.x, u = prim.y, v = prim.z, p = prim.w;
    float energy = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
    return vec4(rho, rho * u, rho * v, energy);
}

vec4 fluxXOf(vec4 prim, float gamma) {
    vec4 c = primToCons(prim, gamma);
    float u = prim.y, p = prim.w;
    return vec4(c.y, c.y * u + p, c.z * u, u * (c.w + p));
}

vec4 fluxYOf(vec4 prim, float gamma) {
    vec4 c = primToCons(prim, gamma);
    float v = prim.z, p = prim.w;
    return vec4(c.z, c.y * v, c.z * v + p, v * (c.w + p));
}

// left/right: primitive (rho,u,v,p) states either side of an X-normal
// interface. Mirrors Euler2D.cpp's HllcFluxX line for line.
vec4 hllcFluxX(vec4 left, vec4 right, float gamma) {
    float cL = sqrt(gamma * left.w / left.x);
    float cR = sqrt(gamma * right.w / right.x);
    float sL = min(left.y - cL, right.y - cR);
    float sR = max(left.y + cL, right.y + cR);

    vec4 fL = fluxXOf(left, gamma);
    if (sL >= 0.0) return fL;
    vec4 fR = fluxXOf(right, gamma);
    if (sR <= 0.0) return fR;

    vec4 uL = primToCons(left, gamma);
    vec4 uR = primToCons(right, gamma);
    float sStar = (right.w - left.w + left.x * left.y * (sL - left.y) - right.x * right.y * (sR - right.y)) /
                  (left.x * (sL - left.y) - right.x * (sR - right.y));

    if (sStar >= 0.0) {
        float coef = left.x * (sL - left.y) / (sL - sStar);
        vec4 uStar;
        uStar.x = coef;
        uStar.y = coef * sStar;
        uStar.z = coef * left.z; // transverse v: carried through unchanged
        uStar.w = coef * (uL.w / left.x + (sStar - left.y) * (sStar + left.w / (left.x * (sL - left.y))));
        return fL + sL * (uStar - uL);
    }
    float coef = right.x * (sR - right.y) / (sR - sStar);
    vec4 uStar;
    uStar.x = coef;
    uStar.y = coef * sStar;
    uStar.z = coef * right.z;
    uStar.w = coef * (uR.w / right.x + (sStar - right.y) * (sStar + right.w / (right.x * (sR - right.y))));
    return fR + sR * (uStar - uR);
}

// left/right: primitive (rho,u,v,p) states either side of a Y-normal
// interface. Mirrors Euler2D.cpp's HllcFluxY line for line (normal velocity
// is v=.z here, transverse is u=.y).
vec4 hllcFluxY(vec4 left, vec4 right, float gamma) {
    float cL = sqrt(gamma * left.w / left.x);
    float cR = sqrt(gamma * right.w / right.x);
    float sL = min(left.z - cL, right.z - cR);
    float sR = max(left.z + cL, right.z + cR);

    vec4 fL = fluxYOf(left, gamma);
    if (sL >= 0.0) return fL;
    vec4 fR = fluxYOf(right, gamma);
    if (sR <= 0.0) return fR;

    vec4 uL = primToCons(left, gamma);
    vec4 uR = primToCons(right, gamma);
    float sStar = (right.w - left.w + left.x * left.z * (sL - left.z) - right.x * right.z * (sR - right.z)) /
                  (left.x * (sL - left.z) - right.x * (sR - right.z));

    if (sStar >= 0.0) {
        float coef = left.x * (sL - left.z) / (sL - sStar);
        vec4 uStar;
        uStar.x = coef;
        uStar.z = coef * sStar;
        uStar.y = coef * left.y; // transverse u: carried through unchanged
        uStar.w = coef * (uL.w / left.x + (sStar - left.z) * (sStar + left.w / (left.x * (sL - left.z))));
        return fL + sL * (uStar - uL);
    }
    float coef = right.x * (sR - right.z) / (sR - sStar);
    vec4 uStar;
    uStar.x = coef;
    uStar.z = coef * sStar;
    uStar.y = coef * right.y;
    uStar.w = coef * (uR.w / right.x + (sStar - right.z) * (sStar + right.w / (right.x * (sR - right.z))));
    return fR + sR * (uStar - uR);
}
)";

// Pass: Cons (rho,momX,momY,E) -> Prim (rho,u,v,p), per cell. Axis-
// independent -- one call covers both sweep directions.
inline std::string ConsToPrim() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) buffer PrimBuf { vec4 prim[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNx * uNy)) return;
    vec4 c = cons[idx];
    float rho = c.x;
    float u = c.y / rho;
    float v = c.z / rho;
    float p = (uGamma - 1.0) * (c.w - 0.5 * (c.y * c.y + c.z * c.z) / rho);
    prim[idx] = vec4(rho, u, v, p);
}
)";
}

// Pass: MinMod-limited slope per cell, along whichever direction uAxis
// selects. See the header comment for why clamping the "outside" neighbor
// back to the cell itself reproduces Euler2D.cpp's ghost-cell boundary.
inline std::string ComputeSlopes() {
    return CommonHeader() + R"(
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 6) buffer SlopeBuf { vec4 slope[]; };

float minmod(float a, float b) {
    if (a * b <= 0.0) return 0.0;
    return abs(a) < abs(b) ? a : b;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNx * uNy)) return;
    int i = int(idx) % uNx;
    int j = int(idx) / uNx;
    uint idxM, idxP;
    if (uAxis == 0) {
        int iM = clamp(i - 1, 0, uNx - 1);
        int iP = clamp(i + 1, 0, uNx - 1);
        idxM = uint(j * uNx + iM);
        idxP = uint(j * uNx + iP);
    } else {
        int jM = clamp(j - 1, 0, uNy - 1);
        int jP = clamp(j + 1, 0, uNy - 1);
        idxM = uint(jM * uNx + i);
        idxP = uint(jP * uNx + i);
    }
    vec4 pm = prim[idxM], p0 = prim[idx], pp = prim[idxP];
    vec4 d;
    d.x = minmod(p0.x - pm.x, pp.x - p0.x);
    d.y = minmod(p0.y - pm.y, pp.y - p0.y);
    d.z = minmod(p0.z - pm.z, pp.z - p0.z);
    d.w = minmod(p0.w - pm.w, pp.w - p0.w);
    slope[idx] = d;
}
)";
}

// Pass: HLLC flux at each interface normal to uAxis, from MinMod-
// extrapolated face states. Dispatched over (uNx+1)*uNy interfaces for an
// X sweep, uNx*(uNy+1) for a Y sweep -- caller picks the dispatch size.
inline std::string Fluxes() {
    return CommonHeader() + HllcGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 6) readonly buffer SlopeBuf { vec4 slope[]; };
layout(std430, binding = 5) buffer FluxBuf { vec4 flux[]; };

void main() {
    uint k = gl_GlobalInvocationID.x;
    if (uAxis == 0) {
        int total = (uNx + 1) * uNy;
        if (int(k) >= total) return;
        int ix = int(k) % (uNx + 1);
        int j = int(k) / (uNx + 1);
        int iL = clamp(ix - 1, 0, uNx - 1);
        int iR = clamp(ix, 0, uNx - 1);
        uint cellL = uint(j * uNx + iL);
        uint cellR = uint(j * uNx + iR);
        vec4 faceR = prim[cellL] + 0.5 * slope[cellL]; // right-extrapolated state of cellL
        vec4 faceL = prim[cellR] - 0.5 * slope[cellR]; // left-extrapolated state of cellR
        flux[k] = hllcFluxX(faceR, faceL, uGamma);
    } else {
        int total = uNx * (uNy + 1);
        if (int(k) >= total) return;
        int i = int(k) % uNx;
        int jy = int(k) / uNx;
        int jL = clamp(jy - 1, 0, uNy - 1);
        int jR = clamp(jy, 0, uNy - 1);
        uint cellL = uint(jL * uNx + i);
        uint cellR = uint(jR * uNx + i);
        vec4 faceR = prim[cellL] + 0.5 * slope[cellL];
        vec4 faceL = prim[cellR] - 0.5 * slope[cellR];
        flux[k] = hllcFluxY(faceR, faceL, uGamma);
    }
}
)";
}

// Pass: one explicit Euler flux-divergence update along uAxis, from
// whichever Flux buffer contents the preceding Fluxes() dispatch (over the
// matching interface count) just produced.
inline std::string EulerStep() {
    return CommonHeader() + R"(
uniform float uDtOverD; // dt/dx for an X sweep, dt/dy for a Y sweep

layout(std430, binding = 0) readonly buffer ConsIn { vec4 consIn[]; };
layout(std430, binding = 5) readonly buffer FluxBuf { vec4 flux[]; };
layout(std430, binding = 1) buffer ConsOut { vec4 consOut[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNx * uNy)) return;
    int i = int(idx) % uNx;
    int j = int(idx) / uNx;
    vec4 fPrev, fNext;
    if (uAxis == 0) {
        fPrev = flux[uint(j * (uNx + 1) + i)];
        fNext = flux[uint(j * (uNx + 1) + i + 1)];
    } else {
        fPrev = flux[uint(j * uNx + i)];
        fNext = flux[uint((j + 1) * uNx + i)];
    }
    consOut[idx] = consIn[idx] - uDtOverD * (fNext - fPrev);
}
)";
}

// Pass: consOut = 0.5*(consA + consB) -- the RK2/Heun combination step.
// Axis-independent -- identical to kernels_euler1d.hpp's Combine.
inline std::string Combine() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsA { vec4 consA[]; };
layout(std430, binding = 1) readonly buffer ConsB { vec4 consB[]; };
layout(std430, binding = 2) buffer ConsOut { vec4 consOut[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNx * uNy)) return;
    consOut[idx] = 0.5 * (consA[idx] + consB[idx]);
}
)";
}

} // namespace cf::kernels2d
