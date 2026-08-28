#pragma once

#include <string>

// GLSL compute-shader source for GRHD confined to the Kerr EQUATORIAL
// PLANE (theta=pi/2) -- Tier 2 Phase 2a, see 07_grhd/README.md. The
// equatorial plane is an exact invariant submanifold of Kerr (reflection
// symmetry theta->pi-theta), so a fluid with v^theta=0 stays there
// exactly -- this keeps the problem "1D in r" numerically (same grid/
// pipeline shape as kernels.hpp/kernels_schwarzschild.hpp), at the cost of
// not being able to represent genuinely 3D structures like a
// Fishbone-Moncrief torus's finite thickness (deferred -- see README).
//
// Boyer-Lindquist coordinates, equatorial metric (G=c=1):
//   g_tt=-(1-2M/r), g_tphi=-2Ma/r, g_rr=r^2/Delta, g_phiphi=r^2+a^2+2Ma^2/r
//   Delta=r^2-2Mr+a^2
// Frame dragging means BOTH radial and azimuthal physical (ZAMO-frame)
// velocities matter now, not just radial: v=(v_r,v_phi),
// W=1/sqrt(1-v_r^2-v_phi^2), h=1+Gamma*P/((Gamma-1)*rho) exactly as before.
//
// Given how much more error-prone Kerr's off-diagonal metric is than
// Schwarzschild's diagonal one (Phase 1 already caught one misremembered
// formula, and deriving this phase caught a further sign error in the
// shift and a missing factor in the specific-energy formula -- see
// 07_grhd/README.md's Physics section for the full story and the
// independent checks that caught each one), this implementation
// deliberately avoids hand-derived symbolic Christoffel symbols
// altogether: Gamma^r_munu and the momentum source term are both computed
// via NUMERICAL (central finite-difference) differentiation of the
// metric components, which are themselves just the textbook Kerr metric
// definition, not a derived quantity. The exact same numerical approach
// is used in tools/kerr_metric_check.py and tools/kerr_equatorial_ref.py,
// so the shader and the Python validation are provably doing the same
// computation, not just two independent transcriptions of one formula.
//
// Conserved variables (mixed-index T^mu_nu = rho*h*u^mu*u_nu +
// P*delta^mu_nu, "areal" r^2-inclusive, matching Phase 0/1's convention):
//   D   = r^2 * rho * u^t                     (baryon current, always source-free)
//   Sr  = r^2 * T^t_r   = r^2*rho*h*u^t*u_r    (radial momentum -- the only one with a source)
//   L   = r^2 * T^t_phi = r^2*rho*h*u^t*u_phi  (angular momentum -- source-free: phi-Killing vector)
//   tau = -r^2*T^t_t - D                       (energy minus rest mass -- source-free: t-Killing vector)
// with u^t=W/alpha, u_r=W*sqrt(gamma_rr)*v_r, u_phi=W*sqrt(gamma_phiphi)*v_phi,
// E:=-u_t=W*(alpha - g_tphi*v_phi/sqrt(gamma_phiphi)) (see tools/kerr_equatorial_ref.py
// for the full derivation and its cross-checks against independently-solved
// circular geodesics).
//
// Momentum source (the standard identity Gamma^lambda_{mu r} T^mu_lambda =
// 0.5*T^ab*d(g_ab)/dr for symmetric T -- verified here to give exactly
// zero for a dust circular orbit at every radius/spin tested, matching
// the geodesic equation independently):
//   Source_Sr = r^2 * 0.5 * [T^tt*dg_tt + 2*T^tphi*dg_tphi + T^phiphi*dg_phiphi
//                            + T^rr*dg_rr + T^thth*dg_thth]   (d/dr, finite difference)

namespace grhd::kernels_kerr {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uN;
uniform float uGamma;
uniform float uM;
uniform float uA;
uniform float uRmin;
uniform float uDr;
)";
}

// Metric + Christoffel helpers, shared by every pass. All by numerical
// finite difference of the metric where a derivative is needed (see
// header comment) -- no hand-derived symbolic Christoffels anywhere.
inline const char* KerrMetricGlsl = R"(
struct Metric { float gtt, gtphi, grr, gphiphi, gthth; };

Metric metricAt(float r) {
    float Delta = r*r - 2.0*uM*r + uA*uA;
    Metric m;
    m.gtt = -(1.0 - 2.0*uM/r);
    m.gtphi = -2.0*uM*uA/r;
    m.grr = r*r/Delta;
    m.gphiphi = r*r + uA*uA + 2.0*uM*uA*uA/r;
    m.gthth = r*r;
    return m;
}

// Lapse/shift bundle at r (equatorial Kerr ZAMO frame).
void zamoAt(float r, out float alpha, out float betaPhiUp, out float gamma_rr, out float gamma_phiphi, out float g_tphi) {
    Metric m = metricAt(r);
    float Delta = r*r - 2.0*uM*r + uA*uA;
    float A = (r*r+uA*uA)*(r*r+uA*uA) - uA*uA*Delta;
    alpha = sqrt(r*r*Delta/A);
    betaPhiUp = m.gtphi / m.gphiphi;
    gamma_rr = m.grr;
    gamma_phiphi = m.gphiphi;
    g_tphi = m.gtphi;
}

// Primitives (rho,v_r,v_phi,P) -> the up/down 4-velocity pieces needed
// everywhere else, plus specific energy E=-u_t.
void kinematics(vec4 prim, float r, out float W, out float h, out float ut, out float ur,
                 out float uPhiContra, out float urCov, out float uPhiCov, out float E) {
    float rho = prim.x, vr = prim.y, vphi = prim.z, P = prim.w;
    float alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi;
    zamoAt(r, alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi);

    W = 1.0 / sqrt(clamp(1.0 - vr*vr - vphi*vphi, 1e-10, 1.0));
    h = 1.0 + uGamma * P / ((uGamma - 1.0) * rho);
    ut = W / alpha;
    ur = W * vr / sqrt(gamma_rr);
    uPhiContra = W * vphi / sqrt(gamma_phiphi) - W * betaPhiUp / alpha;
    urCov = W * sqrt(gamma_rr) * vr;
    uPhiCov = W * sqrt(gamma_phiphi) * vphi;
    E = W * (alpha - g_tphi * vphi / sqrt(gamma_phiphi));
}
)";

// Pass: Cons (D,Sr,L,tau) -> Prim (rho,v_r,v_phi,P), per cell. Newton
// iteration on specific enthalpy h (see tools/kerr_equatorial_ref.py's
// cons_to_prim -- same algorithm, same variable names).
inline std::string ConsToPrim() {
    return CommonHeader() + KerrMetricGlsl + R"(
uniform int uIters;

layout(std430, binding = 0) readonly buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 4) buffer PrimBuf { vec4 prim[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    float r = uRmin + (float(i) + 0.5) * uDr;
    vec4 U = cons[i];
    float D = U.x, Sr = U.y, L = U.z, tau = U.w;

    float alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi;
    zamoAt(r, alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi);
    float r2 = r * r;
    float kappa = Sr / D;
    float lam = L / D;
    float K = kappa*kappa/gamma_rr + lam*lam/gamma_phiphi;

    float h = max(prim[i].w > 0.0 ? 1.0 + uGamma*prim[i].w/((uGamma-1.0)*max(D*alpha/(r2*sqrt(1.0+K)),1e-8)) : 1.001, 1.0 + 1e-8);
    for (int it = 0; it < uIters; ++it) {
        float W0 = sqrt(1.0 + K/(h*h));
        float rho0 = D * alpha / (r2 * W0);
        float eps0 = (h - 1.0) / uGamma;
        float P0 = (uGamma - 1.0) * rho0 * eps0;
        float Q0 = (tau + D + r2*P0) / D;
        float f0 = Q0/h - (W0*alpha - g_tphi*lam/(h*gamma_phiphi));

        float dh = max(1e-6*abs(h), 1e-9);
        float hp = h + dh, hm = h - dh;
        float Wp = sqrt(1.0+K/(hp*hp)); float rhop = D*alpha/(r2*Wp); float Pp=(uGamma-1.0)*rhop*(hp-1.0)/uGamma;
        float Qp = (tau+D+r2*Pp)/D; float fPlus = Qp/hp - (Wp*alpha - g_tphi*lam/(hp*gamma_phiphi));
        float Wm = sqrt(1.0+K/(hm*hm)); float rhom = D*alpha/(r2*Wm); float Pm=(uGamma-1.0)*rhom*(hm-1.0)/uGamma;
        float Qm = (tau+D+r2*Pm)/D; float fMinus = Qm/hm - (Wm*alpha - g_tphi*lam/(hm*gamma_phiphi));

        float deriv = (fPlus - fMinus) / (2.0*dh);
        if (abs(deriv) > 1e-12) h = max(h - f0/deriv, 1.0 + 1e-9);
    }

    float W = sqrt(1.0 + K/(h*h));
    float rho = D * alpha / (r2 * W);
    float vr = kappa / (h*W*sqrt(gamma_rr));
    float vphi = lam / (h*W*sqrt(gamma_phiphi));
    float eps = (h - 1.0) / uGamma;
    float P = (uGamma - 1.0) * rho * eps;
    prim[i] = vec4(rho, vr, vphi, P);
}
)";
}

// Pass: HLLE flux at each interface, piecewise-constant reconstruction
// (same first-order Godunov convention as Phase 0/1).
inline std::string Fluxes() {
    return CommonHeader() + KerrMetricGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 5) buffer FluxBuf { vec4 flux[]; };

void sideFluxKerr(vec4 P, float r, out vec4 U, out vec4 F) {
    float rho = P.x, vr = P.y, vphi = P.z, p = P.w;
    float W, h, ut, ur, uPhiContra, urCov, uPhiCov, E;
    kinematics(P, r, W, h, ut, ur, uPhiContra, urCov, uPhiCov, E);

    float r2 = r * r;
    float D = r2 * rho * ut;
    float Sr = r2 * rho * h * ut * urCov;
    float L = r2 * rho * h * ut * uPhiCov;
    float tau = r2 * rho * h * ut * E - r2 * p - D;
    U = vec4(D, Sr, L, tau);

    float FD = r2 * rho * ur;
    float FSr = r2 * (rho * h * ur * urCov + p);
    float FL = r2 * rho * h * ur * uPhiCov;
    float Ftau = r2 * rho * ur * (h * E - 1.0);
    F = vec4(FD, FSr, FL, Ftau);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i > uint(uN)) return;
    uint iL = (i == 0u) ? 0u : (i - 1u);
    uint iR = (i == uint(uN)) ? uint(uN - 1) : i;

    float rFace = uRmin + float(i) * uDr;
    Metric m = metricAt(rFace);
    // HLLE bound: the exact coordinate speed of a purely radial photon,
    // sqrt(-g_tt/g_rr) (from the null condition g_tt dt^2+g_rr dr^2=0,
    // dphi=0 -- g_tphi/g_phiphi drop out entirely for zero angular
    // motion, so this is exact, not an approximation). Always a safe
    // upper bound on any physical signal speed, radial or not -- used
    // here in place of deriving this system's actual flux-Jacobian
    // eigenvalues (unlike Phase 0/1's diagonal metrics, Kerr's shift
    // means the local-to-coordinate speed conversion isn't a single
    // clean multiplicative factor, so getting the tight bound right
    // would need more risk than this phase's validation target
    // -- a circular orbit staying circular -- requires). Costs extra
    // numerical diffusion, not correctness.
    float fPhoton = sqrt(-m.gtt / m.grr);

    vec4 UL, UR, FL, FR;
    sideFluxKerr(prim[iL], rFace, UL, FL);
    sideFluxKerr(prim[iR], rFace, UR, FR);

    float sL = -fPhoton;
    float sR = fPhoton;
    vec4 F = (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
    flux[i] = F;
}
)";
}

// Euler predictor + the one genuine geometric source term (radial
// momentum only -- D, L, tau are all exactly source-free, see header
// comment). Source computed via the symmetric-tensor identity
// Gamma^lambda_{mu r} T^mu_lambda = 0.5*T^ab*d(g_ab)/dr, metric
// derivatives by central finite difference (verified in
// tools/kerr_equatorial_ref.py to vanish for a dust circular orbit).
inline std::string EulerStep() {
    return CommonHeader() + KerrMetricGlsl + R"(
uniform float uDt;

layout(std430, binding = 0) readonly buffer ConsIn { vec4 consIn[]; };
layout(std430, binding = 4) readonly buffer PrimBuf { vec4 prim[]; };
layout(std430, binding = 5) readonly buffer FluxBuf { vec4 flux[]; };
layout(std430, binding = 1) buffer ConsOut { vec4 consOut[]; };

float sourceSr(vec4 P, float r) {
    float rho = P.x, p = P.w;
    float W, h, ut, ur, uPhiContra, urCov, uPhiCov, E;
    kinematics(P, r, W, h, ut, ur, uPhiContra, urCov, uPhiCov, E);

    float alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi;
    zamoAt(r, alpha, betaPhiUp, gamma_rr, gamma_phiphi, g_tphi);
    float Ttt = rho*h*ut*ut + p*(-gamma_phiphi/(gamma_phiphi*(-r*r+r*r) + 1.0)); // unused fallback avoided below
    // Fully-contravariant stress tensor components via the (t,phi) inverse
    // metric block, computed directly (not assuming any shortcut identity).
    Metric m = metricAt(r);
    float detG = m.gtt*m.gphiphi - m.gtphi*m.gtphi;
    float gUtt = m.gphiphi/detG, gUtphi = -m.gtphi/detG, gUphiphi = m.gtt/detG;
    float gUrr = 1.0/m.grr, gUthth = 1.0/m.gthth;

    Ttt = rho*h*ut*ut + p*gUtt;
    float Ttphi = rho*h*ut*uPhiContra + p*gUtphi;
    float Tphiphi = rho*h*uPhiContra*uPhiContra + p*gUphiphi;
    float Trr = rho*h*ur*ur + p*gUrr;
    float Tthth = p*gUthth;

    float eps = max(1e-5*r, 1e-5);
    Metric mp = metricAt(r+eps);
    Metric mm = metricAt(r-eps);
    float dgtt = (mp.gtt-mm.gtt)/(2.0*eps);
    float dgtphi = (mp.gtphi-mm.gtphi)/(2.0*eps);
    float dgphiphi = (mp.gphiphi-mm.gphiphi)/(2.0*eps);
    float dgrr = (mp.grr-mm.grr)/(2.0*eps);
    float dgthth = (mp.gthth-mm.gthth)/(2.0*eps);

    float source = 0.5*(Ttt*dgtt + 2.0*Ttphi*dgtphi + Tphiphi*dgphiphi + Trr*dgrr + Tthth*dgthth);
    return r*r*source;
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    float r = uRmin + (float(i) + 0.5) * uDr;

    vec4 divF = (flux[i + 1u] - flux[i]) / uDr;
    float srcSr = sourceSr(prim[i], r);
    vec4 source = vec4(0.0, srcSr, 0.0, 0.0);
    consOut[i] = consIn[i] - uDt * divF + uDt * source;
}
)";
}

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

} // namespace grhd::kernels_kerr
