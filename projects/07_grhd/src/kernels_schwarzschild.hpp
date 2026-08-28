#pragma once

#include <string>

// GLSL compute-shader source for 1D general-relativistic hydrodynamics on a
// FIXED Schwarzschild background (Tier 2 Phase 1, see 07_grhd/README.md
// and 06_tidal_disruption/README.md's tier table) -- radial-only flow,
// spherical symmetry, Schwarzschild coordinates (t,r,theta,phi):
//
//   ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2,   f = 1 - 2M/r
//
// static (no shift, alpha=sqrt(f) is the only nontrivial lapse), so this
// reuses the same buffer layout and RK2/HLLE machinery as kernels.hpp's
// flat-spacetime solver -- only the per-cell/per-interface physics (the
// conserved-variable definitions, the flux, and a genuine momentum source
// term from spacetime curvature) changes. Derived from first principles
// (∇_mu T^{mu nu}=0 projected onto the r and t directions, using the
// standard Schwarzschild Christoffel symbols) and cross-checked three
// ways before trusting it: (1) the Newtonian limit (M/r->0, v,eps->0)
// reduces exactly to the classical spherical Euler momentum source
// -rho*GM+2*P*r; (2) the energy equation comes out source-free, matching
// the general theorem that a stationary spacetime's timelike Killing
// vector gives an exactly conserved current (Noether/Killing energy
// conservation), not assumed but confirmed by direct calculation; (3) the
// flux, stripped of its r^2 and lapse factors, reduces to exactly Phase
// 0's already-validated flat SRHD flux form. See 07_grhd/README.md's
// Physics section for the full derivation.
//
// Conserved variables (D,S,tau), all "areal" (per unit r, i.e. already
// including the r^2 spherical-shell factor -- same per-cell storage
// convention as kernels.hpp, so no change to buffer layout):
//   D   = r^2 * rho * W / alpha
//   S   = r^2 * rho * h * W^2 * v
//   tau = r^2 * (rho*h*W^2 - P) - D
// with v the LOCAL (orthonormal, static-observer-measured) radial
// velocity, W=1/sqrt(1-v^2), h=1+Gamma*P/((Gamma-1)*rho) exactly as in
// flat SRHD -- the metric only enters through r, f=1-2M/r, alpha=sqrt(f).
//
// Evolution:
//   d_t D   + d_r(f*v*D)                 = 0
//   d_t S   + d_r(f*(v*S + r^2*P))       = -M*rho*h + 2*M*P + 2*f*P*r
//   d_t tau + d_r(f*(S - v*D))           = 0
//
// Buffer layout: identical to kernels.hpp (binding 0/1 Cons0/Cons1,
// binding 2/3 Stage1/Stage2, binding 4 Prim, binding 5 Flux). Grid is
// r in [uRmin, uRmin+uN*uDr), cell i centered at uRmin+(i+0.5)*uDr,
// interface i at uRmin+i*uDr.

namespace grhd::kernels_sch {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uN;
uniform float uGamma;
uniform float uM;
uniform float uRmin;
uniform float uDr;

float metricF(float r) { return 1.0 - 2.0 * uM / r; }
)";
}

inline const char* ConsToPrimSchGlsl = R"(
// Same Newton-Raphson structure as flat SRHD (kernels.hpp), generalized
// with the r^2/alpha factors from the Schwarzschild conserved variables
// above (denom = tau + D + r^2*P reduces to flat SRHD's tau+D+P at r=1,
// alpha=1; rho = D*alpha/(r^2*W) reduces to flat SRHD's D/W the same way).
float pOfConsSch(float D, float S, float tau, float P, float r, float alpha, float Gamma) {
    float r2 = r * r;
    float denom = tau + D + r2 * P;
    float v2 = clamp((S * S) / (denom * denom), 0.0, 0.999999);
    float W = 1.0 / sqrt(1.0 - v2);
    float rho = D * alpha / (r2 * W);
    float h = denom / (r2 * rho * W * W);
    float eps = h - 1.0 - P / rho;
    return (Gamma - 1.0) * rho * eps;
}

vec3 recoverPrimitivesSch(float D, float S, float tau, float pGuess, float r, float alpha, float Gamma, int iters) {
    float P = max(pGuess, 1e-12);
    for (int it = 0; it < iters; ++it) {
        float f0 = pOfConsSch(D, S, tau, P, r, alpha, Gamma) - P;
        float dP = max(1e-6 * abs(P), 1e-12);
        float fPlus = pOfConsSch(D, S, tau, P + dP, r, alpha, Gamma) - (P + dP);
        float fMinus = pOfConsSch(D, S, tau, P - dP, r, alpha, Gamma) - (P - dP);
        float deriv = (fPlus - fMinus) / (2.0 * dP);
        if (abs(deriv) > 1e-12) {
            P = max(P - f0 / deriv, 1e-12);
        }
    }
    float r2 = r * r;
    float denom = tau + D + r2 * P;
    float v2 = clamp((S * S) / (denom * denom), 0.0, 0.999999);
    float W = 1.0 / sqrt(1.0 - v2);
    float rho = D * alpha / (r2 * W);
    float v = S / denom;
    return vec3(rho, v, P);
}
)";

inline std::string ConsToPrim() {
    return CommonHeader() + ConsToPrimSchGlsl + R"(
uniform int uIters;

layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) buffer PrimBuf { vec4 prim[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    float r = uRmin + (float(i) + 0.5) * uDr;
    float alpha = sqrt(metricF(r));
    vec4 U = cons[i];
    vec3 p = recoverPrimitivesSch(U.x, U.y, U.z, prim[i].z, r, alpha, uGamma, uIters);
    prim[i] = vec4(p, 0.0);
}
)";
}

inline std::string Fluxes() {
    return CommonHeader() + R"(
uniform vec4 uGhostHi; // fixed exterior (Dirichlet) primitive state at r_max --
                        // see kernels_schwarzschild.hpp's header comment: the outer
                        // edge is subsonic, so zero-gradient outflow is ill-posed there.

layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 5) buffer FluxBuf { vec4 flux[]; };

// Side flux + HLLE signal speeds at one interface, reconstructing this
// side's (D,S,tau) AT THE INTERFACE's r from its cell-centered primitives
// (rho,v,P don't depend on position for a piecewise-constant
// reconstruction, but the conserved-variable *representation* of that
// state does, through the metric's r-dependence).
void sideFluxSch(vec3 P, float r, float f, float alpha, out vec3 U, out vec3 F, out float lamMinus, out float lamPlus) {
    float rho = P.x, v = P.y, p = P.z;
    float h = 1.0 + uGamma * p / ((uGamma - 1.0) * rho);
    float W = 1.0 / sqrt(clamp(1.0 - v * v, 1e-12, 1.0));
    float r2 = r * r;
    float D = r2 * rho * W / alpha;
    float S = r2 * rho * h * W * W * v;
    float tau = r2 * (rho * h * W * W - p) - D;
    U = vec3(D, S, tau);
    F = vec3(f * v * D, f * (v * S + r2 * p), f * (S - v * D));

    float cs2 = clamp(uGamma * p / (rho * h), 0.0, 0.999999);
    float cs = sqrt(cs2);
    lamMinus = f * (v - cs) / (1.0 - v * cs);
    lamPlus = f * (v + cs) / (1.0 + v * cs);
}

void main() {
    uint i = gl_GlobalInvocationID.x; // interface index, 0..uN
    if (i > uint(uN)) return;
    uint iL = (i == 0u) ? 0u : (i - 1u);
    // Inner edge (i==0): supersonic, plain extrapolation outflow (self) is
    // correct. Outer edge (i==uN): subsonic, so the right state is the
    // fixed exterior ghost, not another self-extrapolation.
    bool outerBoundary = (i == uint(uN));
    uint iR = outerBoundary ? uint(uN - 1) : i;

    float rFace = uRmin + float(i) * uDr;
    float f = metricF(rFace);
    float alpha = sqrt(f);

    vec3 UL, UR, FL, FR;
    float lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    sideFluxSch(prim[iL].xyz, rFace, f, alpha, UL, FL, lamMinusL, lamPlusL);
    vec3 primR = outerBoundary ? uGhostHi.xyz : prim[iR].xyz;
    sideFluxSch(primR, rFace, f, alpha, UR, FR, lamMinusR, lamPlusR);

    float sL = min(0.0, min(lamMinusL, lamMinusR));
    float sR = max(0.0, max(lamPlusL, lamPlusR));

    vec3 F;
    if (sL >= 0.0) {
        F = FL;
    } else if (sR <= 0.0) {
        F = FR;
    } else {
        F = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
    }
    flux[i] = vec4(F, 0.0);
}
)";
}

// Euler predictor step + the momentum-only geometric source term
// (-M*rho*h + 2*M*P + 2*f*P*r, evaluated at the cell's own primitives and
// center -- see kernels_schwarzschild.hpp's header comment for the
// derivation). D and tau have exactly zero source (baryon conservation
// has no source ever; the energy equation's source vanishes by the
// Killing-vector argument), so only the S component picks one up.
inline std::string EulerStep() {
    return CommonHeader() + R"(
uniform float uDt;

layout(std430, binding = 0) readonly buffer ConsIn { vec4 consIn[]; };
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 5) readonly buffer FluxBuf { vec4 flux[]; };
layout(std430, binding = 1) buffer ConsOut { vec4 consOut[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    float r = uRmin + (float(i) + 0.5) * uDr;
    float f = metricF(r);

    vec3 rho_v_P = prim[i].xyz;
    float rho = rho_v_P.x, P = rho_v_P.z;
    float h = 1.0 + uGamma * P / ((uGamma - 1.0) * rho);
    float sourceS = -uM * rho * h + 2.0 * uM * P + 2.0 * f * P * r;

    vec4 divF = (flux[i + 1u] - flux[i]) / uDr;
    vec4 source = vec4(0.0, sourceS, 0.0, 0.0);
    consOut[i] = consIn[i] - uDt * divF + uDt * source;
}
)";
}

// Combine (RK2/Heun) -- identical to flat SRHD's, metric-independent.
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

} // namespace grhd::kernels_sch
