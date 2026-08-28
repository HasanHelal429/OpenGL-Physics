#pragma once

#include <string>

// GLSL compute-shader source for 1D special-relativistic hydrodynamics
// (flat spacetime, Minkowski metric, c=G=1) -- Tier 2 Phase 0 of the
// tidal-disruption project (see 06_tidal_disruption/README.md's tier
// table): before adding a curved Schwarzschild/Kerr background, this
// validates the actual relativistic-fluid machinery (conservative
// variables, primitive recovery, a Riemann solver) that Tier 2 needs, in
// the one regime with simple, trustworthy checks (Newtonian limit,
// conservation, and the classic relativistic shock tube).
//
// Valencia formulation (Marti & Muller, Living Rev. Relativity; Rezzolla &
// Zanotti, "Relativistic Hydrodynamics"), ideal Gamma-law EOS
// P=(Gamma-1)*rho*eps. Primitive variables (rho, v, P); conserved
// variables per unit length:
//   D   = rho*W                          (relativistic mass density)
//   S   = rho*h*W^2*v                    (momentum density)
//   tau = rho*h*W^2 - P - D              (energy density minus rest mass)
// with Lorentz factor W=1/sqrt(1-v^2) and specific enthalpy
// h=1+Gamma*P/((Gamma-1)*rho). Flux (1D, flat):
//   F_D=D*v,  F_S=S*v+P,  F_tau=(tau+P)*v
//
// Buffer layout (std430 SSBOs, N cells unless noted):
//   binding 0  Cons0   vec4(D, S, tau, _)   -- ping-pong slot A
//   binding 1  Cons1   vec4(D, S, tau, _)   -- ping-pong slot B
//   binding 2  Stage1  vec4(D, S, tau, _)   -- RK2 intermediate state U1
//   binding 3  Stage2  vec4(D, S, tau, _)   -- RK2 intermediate state U2
//   binding 4  Prim    vec4(rho, v, P, _)   -- recovered from whichever
//                                              Cons buffer is "current"
//   binding 5  Flux    vec4(F_D, F_S, F_tau, _), N+1 entries (interfaces)
//
// No self-gravity, no curved metric yet -- that is the whole point of this
// phase: get the fluid solver right somewhere it can actually be checked
// against an exact answer before trusting it near a black hole.

namespace grhd::kernels {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uN;
uniform float uGamma;
)";
}

// Reusable primitive-recovery function: given conserved (D,S,tau) and a
// pressure guess, one Newton-Raphson iteration on
//   f(P) = P_new(P) - P = 0
// where P_new(P) is the pressure implied by the EOS once P has been used
// to reconstruct v, W, rho, h from the conserved variables (the standard
// 1D primitive-recovery scheme for a Gamma-law gas -- see e.g. Marti &
// Muller's living review, sec. 4.2, or Rezzolla & Zanotti eq. 3.44-3.49).
// The derivative is taken numerically (central difference): this problem's
// f(P) is smooth and well-behaved for the mildly-relativistic shock tubes
// this phase targets, so a numeric derivative avoids deriving (and risking
// a sign error in) the analytic one.
inline const char* ConsToPrimGlsl = R"(
float pOfCons(float D, float S, float tau, float P, float Gamma) {
    float denom = tau + D + P;
    float v2 = clamp((S * S) / (denom * denom), 0.0, 0.999999);
    float W = 1.0 / sqrt(1.0 - v2);
    float rho = D / W;
    float h = denom / (D * W);
    float eps = h - 1.0 - P / rho;
    return (Gamma - 1.0) * rho * eps;
}

// Returns primitives (rho, v, P) for conserved (D,S,tau), Newton-iterating
// from the supplied initial pressure guess pGuess (warm-started from the
// previous step's converged pressure).
vec3 recoverPrimitives(float D, float S, float tau, float pGuess, float Gamma, int iters) {
    float P = max(pGuess, 1e-10);
    for (int it = 0; it < iters; ++it) {
        float f0 = pOfCons(D, S, tau, P, Gamma) - P;
        float dP = max(1e-6 * abs(P), 1e-10);
        float fPlus = pOfCons(D, S, tau, P + dP, Gamma) - (P + dP);
        float fMinus = pOfCons(D, S, tau, P - dP, Gamma) - (P - dP);
        float deriv = (fPlus - fMinus) / (2.0 * dP);
        if (abs(deriv) > 1e-12) {
            P = max(P - f0 / deriv, 1e-10);
        }
    }
    float denom = tau + D + P;
    float v2 = clamp((S * S) / (denom * denom), 0.0, 0.999999);
    float W = 1.0 / sqrt(1.0 - v2);
    float rho = D / W;
    float v = S / denom;
    return vec3(rho, v, P);
}
)";

// Pass: Cons (D,S,tau) -> Prim (rho,v,P), per cell.
inline std::string ConsToPrim() {
    return CommonHeader() + ConsToPrimGlsl + R"(
uniform int uIters;

layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) buffer PrimBuf { vec4 prim[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vec4 U = cons[i];
    vec3 p = recoverPrimitives(U.x, U.y, U.z, prim[i].z, uGamma, uIters);
    prim[i] = vec4(p, 0.0);
}
)";
}

// Pass: HLLE flux at each of the N+1 cell interfaces, from piecewise-constant
// reconstruction (first-order Godunov: interface i's left/right states are
// just cell i-1 and cell i). Outflow (zero-gradient) boundaries: the two
// domain-edge interfaces use the nearest interior cell on both sides, which
// makes HLLE degenerate to that cell's own physical flux -- exactly the
// zero-gradient condition, no special-casing needed beyond clamping the
// neighbor index.
inline std::string Fluxes() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 5) buffer FluxBuf { vec4 flux[]; };

// Physical flux and the two outward characteristic speeds for one side of
// the interface (relativistic sound speed and relativistic velocity
// addition for the signal speeds -- Marti & Muller eq. 42).
void sideFlux(vec4 U, vec3 P, out vec3 F, out float lamMinus, out float lamPlus) {
    float rho = P.x, v = P.y, p = P.z;
    float h = 1.0 + uGamma * p / ((uGamma - 1.0) * rho);
    float cs2 = clamp(uGamma * p / (rho * h), 0.0, 0.999999);
    float cs = sqrt(cs2);
    F = vec3(U.x * v, U.y * v + p, (U.z + p) * v);
    lamMinus = (v - cs) / (1.0 - v * cs);
    lamPlus = (v + cs) / (1.0 + v * cs);
}

void main() {
    uint i = gl_GlobalInvocationID.x; // interface index, 0..uN
    if (i > uint(uN)) return;
    uint iL = (i == 0u) ? 0u : (i - 1u);
    uint iR = (i == uint(uN)) ? uint(uN - 1) : i;

    vec4 UL = cons[iL], UR = cons[iR];
    vec3 PL = prim[iL].xyz, PR = prim[iR].xyz;

    vec3 FL, FR;
    float lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    sideFlux(UL, PL, FL, lamMinusL, lamPlusL);
    sideFlux(UR, PR, FR, lamMinusR, lamPlusR);

    float sL = min(0.0, min(lamMinusL, lamMinusR));
    float sR = max(0.0, max(lamPlusL, lamPlusR));

    vec3 F;
    if (sL >= 0.0) {
        F = FL;
    } else if (sR <= 0.0) {
        F = FR;
    } else {
        F = (sR * FL - sL * FR + sL * sR * (UR.xyz - UL.xyz)) / (sR - sL);
    }
    flux[i] = vec4(F, 0.0);
}
)";
}

// Pass: one explicit Euler flux-divergence update, U_out = U_in -
// (dt/dx)*(flux[i+1]-flux[i]). Used twice per step (RK2/Heun's method: an
// Euler predictor from U0 to U1, then another Euler step from U1 to U2,
// combined 0.5*(U0+U2) by the Combine pass below) rather than a single
// forward-Euler step, for the standard second-order-in-time accuracy of a
// strong-stability-preserving RK2 method-of-lines integrator.
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

} // namespace grhd::kernels
