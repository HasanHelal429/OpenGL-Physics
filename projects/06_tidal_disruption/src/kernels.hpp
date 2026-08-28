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
//   binding 4  H         float smoothing length, per particle (adaptive)
//
// All kernels are one-thread-per-particle with a uN bounds check (particle
// count need not be a multiple of the workgroup size, unlike the FFT's
// power-of-two rows). Self-terms (i==j) need no special-casing: the cubic
// spline's gradient is exactly zero at r=0, and gravity's own softened
// numerator (ri-rj) is exactly zero when i==j, so both loops can safely
// include j==i.
//
// Adaptive smoothing length. A single fixed h under-resolves the star's
// outer envelope (where particles are sparse) relative to its dense core --
// the free-surface density excess documented in README.md's first
// validation pass. Standard fix: solve h_i per particle from
// h_i = eta*(m_i/rho_i)^(1/3) (Springel & Hernquist 2002's ansatz: h such
// that the kernel's ~(4/3)pi(2h)^3 support volume holds a roughly constant
// effective neighbor count, eta~1.2 -> ~40-60 neighbors in 3D for the cubic
// spline). rho_i(h_i) itself depends on h_i, so Density() iterates a small
// fixed-point loop per particle (warm-started from last step's h_i, which
// changes slowly frame to frame) entirely inside one shader invocation --
// each particle's solve is independent, ideal for one-thread-per-particle.
// A density DEPENDS on h "gather"-style (rho_i uses only h_i) as usual for
// this iteration; but Forces() uses the symmetrized pair length
// h_ij = 0.5*(h_i+h_j) for both the kernel gradient AND artificial
// viscosity, so the same W(r,h_ij) is used for both directions of a pair
// and F_ij = -F_ji exactly (momentum conservation to machine precision, as
// before) despite h_i != h_j in general.

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
layout(std430, binding = 4) buffer HBuf { float hArr[]; };

uniform int uN;
)";
}

// Pass 1: adaptive smoothing length + density (kernel-weighted mass sum,
// "gather" form: rho_i uses only h_i) + EOS pressure + sound speed. h_i is
// solved by a short fixed-point loop, warm-started from last step's value
// (see kernels.hpp's header comment for the ansatz and why Forces() uses a
// different, symmetrized h for the pair interaction). Polytropic EOS
// P = K*rho^Gamma matches the Lane-Emden profile the initial conditions were
// sampled from (tools/make_star_ic.py), so a correctly-sampled star starts
// (approximately) in hydrostatic equilibrium.
inline std::string Density() {
    return CommonHeader() + SphKernelGlsl + R"(
uniform float uK;
uniform float uGamma;
uniform float uEta;
uniform int uHIters;
uniform float uHMin;
uniform float uHMax;

// Excludes j==i: W(0,h) is nonzero (unlike the force kernel's gradient,
// which vanishes at r=0), so a naive self-inclusive sum adds a spurious
// m_i/(pi*h^3) term. That's negligible at a large fixed h, but once h
// adapts down toward the true local spacing this self-term becomes 15-30%
// of the total -- a real, measured bias (bin-averaged density running
// 25-30% high through the star's bulk), not a rounding-level effect. Fixed
// by never letting a particle see itself as its own neighbor.
float gatherDensity(vec3 ri, float h, uint selfIdx) {
    float rho = 0.0;
    for (uint j = 0u; j < uint(uN); ++j) {
        if (j == selfIdx) continue;
        float r = length(ri - posMass[j].xyz);
        rho += posMass[j].w * kernelW(r, h);
    }
    return rho;
}

// A particle in the far-stretched, thinned-out tail of a tidal debris
// stream can end up with literally zero neighbors within its kernel
// support even at h=uHMax (measured: ~0.3% of particles, rho exactly 0.0,
// ~30t after pericenter once the stream has stretched over 100+ length
// units) -- rho=0 then makes csound=sqrt(Gamma*pressure/rho) a literal
// 0/0 = NaN, which poisons every OTHER particle's force the very next
// step (Forces() loops over all N, so one NaN position contaminates the
// whole buffer within one frame). Flooring rho alone is not enough to fix
// this: P/rho^2 ~ rho^(Gamma-2) actually *diverges* as rho->0 for
// Gamma<2 (true here, Gamma=5/3), so a naive floor silently replaces a
// crash with a large spurious pressure kick instead. The physically
// correct treatment is that a particle with no real neighbors should feel
// no SPH pressure force at all (it still feels gravity, which doesn't
// depend on rho) -- so pressure/soundspeed are set to exactly 0 below
// that floor rather than computed from a floored rho, and Forces() must
// skip the P/rho^2 term entirely below the same floor (next function).
const float kRhoFloor = 1e-8;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    float mi = posMass[i].w;
    float h = hArr[i];
    float rho = 0.0;
    for (int it = 0; it < uHIters; ++it) {
        rho = gatherDensity(ri, h, i);
        h = clamp(uEta * pow(mi / rho, 1.0 / 3.0), uHMin, uHMax); // rho==0 -> mi/rho=+inf -> clamps to uHMax, no NaN
    }
    rho = gatherDensity(ri, h, i); // final density consistent with the converged h

    hArr[i] = h;
    float pressure = (rho > kRhoFloor) ? uK * pow(rho, uGamma) : 0.0;
    float csound = (rho > kRhoFloor) ? sqrt(uGamma * pressure / rho) : 0.0;
    rhoPress[i] = vec4(rho, pressure, csound, 0.0);
}
)";
}

// Pass 2: total acceleration = self-gravity (softened pairwise) + SPH
// pressure gradient + Monaghan artificial viscosity. Fused into one O(N^2)
// loop since both terms need the same pairwise iteration. Uses the
// symmetrized pair length h_ij = 0.5*(h_i+h_j) (same value regardless of
// which particle is "i"), so the kernel gradient -- and hence the force --
// is exactly antisymmetric under i<->j despite h_i != h_j in general.
inline std::string Forces() {
    return CommonHeader() + SphKernelGlsl + R"(
uniform float uG;
uniform float uSoftening2;   // softening length squared
uniform float uViscAlpha;
uniform float uViscBeta;

// Black hole: a fixed point mass at the origin (M_BH >> M_star, so its
// recoil from the star's gravity is negligible to leading order in the
// mass ratio -- not integrated). uBhType: 0 off, 1 Newtonian point mass,
// 2 Paczynski-Wiita (a = -G*M/(r-r_s)^2, reproduces the true ISCO at 6M and
// diverges at the Schwarzschild radius r_s -- a pseudo-relativistic
// stand-in for pericenter precession/plunge without a real metric). Both
// forms are floored at uSoftening (already sized for the star's own
// resolution) below r_s so a particle passing through the horizon gets a
// large but finite kick instead of a NaN.
uniform int uBhType;
uniform float uBhMass;
uniform float uBhRs;

// See Density()'s kRhoFloor comment: a particle with no real SPH neighbors
// (rho==0, measured in the far-stretched tail of a tidal debris stream)
// must contribute exactly zero pressure/viscosity force -- P/rho^2 diverges
// as rho->0 for Gamma<2 (true here), so skipping the term below the floor
// is required for correctness, not just a NaN-avoidance patch.
const float kRhoFloor = 1e-8;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    vec3 vi = vel[i].xyz;
    float hi = hArr[i];
    float rhoi = rhoPress[i].x;
    float Pi = rhoPress[i].y;
    float ci = rhoPress[i].z;

    vec3 a = vec3(0.0);

    if (uBhType == 1) {
        float r2 = dot(ri, ri);
        a -= uG * uBhMass * pow(r2 + uSoftening2, -1.5) * ri;
    } else if (uBhType == 2) {
        float r = length(ri);
        float dr = max(r - uBhRs, sqrt(uSoftening2));
        a -= uG * uBhMass / (dr * dr) * (ri / max(r, 1e-8));
    }

    for (uint j = 0u; j < uint(uN); ++j) {
        vec3 rij = ri - posMass[j].xyz;
        float r2 = dot(rij, rij);
        float mj = posMass[j].w;

        // self-gravity (Plummer-softened)
        float invR3 = pow(r2 + uSoftening2, -1.5);
        a -= uG * mj * invR3 * rij;

        // SPH pressure gradient + artificial viscosity, symmetrized h
        float hij = 0.5 * (hi + hArr[j]);
        float r = sqrt(r2);
        float dWdr = kernelDwDr(r, hij);
        if (dWdr != 0.0) {
            vec3 gradW = (r > 0.0) ? (dWdr / r) * rij : vec3(0.0);
            float rhoj = rhoPress[j].x;
            float Pj = rhoPress[j].y;

            float rhobar = 0.5 * (rhoi + rhoj);
            float piVisc = 0.0;
            if (rhobar > kRhoFloor) {
                vec3 vij = vi - vel[j].xyz;
                float vijDotRij = dot(vij, rij);
                if (vijDotRij < 0.0) {
                    float cj = rhoPress[j].z;
                    float mu = hij * vijDotRij / (r2 + 0.01 * hij * hij);
                    float cbar = 0.5 * (ci + cj);
                    piVisc = (-uViscAlpha * cbar * mu + uViscBeta * mu * mu) / rhobar;
                }
            }

            float PiOverRho2 = (rhoi > kRhoFloor) ? Pi / (rhoi * rhoi) : 0.0;
            float PjOverRho2 = (rhoj > kRhoFloor) ? Pj / (rhoj * rhoj) : 0.0;
            a -= mj * (PiOverRho2 + PjOverRho2 + piVisc) * gradW;
        }
    }
    acc[i] = vec4(a, 0.0);
}
)";
}

// Leapfrog half-kick: vel += halfDt * acc, then an optional exponential
// velocity damping (uDampingFactor = exp(-damping*halfDt), so 1.0 = off).
// Used only during relaxation runs (decks/star_relax_*.toml) to let a
// Monte-Carlo-sampled star settle into numerical equilibrium -- artificial
// viscosity alone damps convergent/shock flows, not the smooth global
// radial breathing mode a freshly-sampled star tends to ring at, which
// otherwise oscillates indefinitely without growing or decaying. Off
// (uDampingFactor=1) once the star is actually being disrupted, since real
// tidal dynamics must not be damped.
inline std::string Kick() {
    return CommonHeader() + R"(
uniform float uHalfDt;
uniform float uDampingFactor;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vel[i].xyz = (vel[i].xyz + uHalfDt * acc[i].xyz) * uDampingFactor;
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
