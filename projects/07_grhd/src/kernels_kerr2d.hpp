#pragma once

#include <string>

// GLSL compute-shader source for full 2D (r,theta) GRHD on a fixed Kerr
// background -- Tier 2 Phase 2b part 2, see 07_grhd/README.md. Generalizes
// kernels_kerr.hpp's equatorial-plane solver by adding a genuine theta
// grid direction and a third velocity component v_theta; every formula
// here reduces exactly to kernels_kerr.hpp's at theta=pi/2, v_theta=0
// (checked in the selftest).
//
// Full Kerr metric (Boyer-Lindquist, general theta, G=c=1):
//   Sigma=r^2+a^2*cos^2(theta)  Delta=r^2-2Mr+a^2  A=(r^2+a^2)^2-a^2*Delta*sin^2(theta)
//   g_tt=-(1-2Mr/Sigma)  g_tphi=-2Mar*sin^2(theta)/Sigma
//   g_rr=Sigma/Delta  g_thth=Sigma  g_phiphi=A*sin^2(theta)/Sigma
//
// Conserved variables (mixed-index T^mu_nu, weighted by sqrt(-g) -- the
// FULL 4-metric determinant, sqrt(-g)=Sigma*sin(theta) for Kerr, NOT
// sqrt(gamma)=sqrt(g_rr*g_thth*g_phiphi) (the 3-metric determinant --
// see sqrtNegG()'s comment below for why this distinction is a real,
// previously-made bug, not pedantry). sqrt(-g) reduces to r^2 at the
// equator (Sigma=r^2, sin(pi/2)=1), matching Phase 2a's "r^2" convention
// exactly there:
//   D       = sqrt(-g) * rho * u^t                        (source-free)
//   Sr      = sqrt(-g) * T^t_r     = sqrt(-g)*rho*h*u^t*u_r     (has a source)
//   Stheta  = sqrt(-g) * T^t_theta = sqrt(-g)*rho*h*u^t*u_theta (has a source)
//   L       = sqrt(-g) * T^t_phi   = sqrt(-g)*rho*h*u^t*u_phi   (source-free)
//   tau     = -sqrt(-g)*T^t_t - D                                (source-free)
// with u^t=W/alpha, u_r=W*sqrt(gamma_rr)*v_r, u_theta=W*sqrt(gamma_thth)*v_theta,
// u_phi=W*sqrt(gamma_phiphi)*v_phi, E:=-u_t=W*(alpha-g_tphi*v_phi/sqrt(gamma_phiphi))
// -- all identical in form to kernels_kerr.hpp, just with a third velocity
// component along for the ride (verified in tools/kerr_equatorial_ref.py's
// style of derivation, generalized in tools/fishbone_moncrief.py, whose
// acceleration()/kerr_metric_full() functions this shader mirrors exactly
// for the CPU/GPU cross-check).
//
// Primitive recovery generalizes Phase 2a's Newton-on-h iteration exactly
// (same W=sqrt(1+K/h^2) trick, now K has three terms instead of two):
//   kappa_r=Sr/D  kappa_th=Stheta/D  lam=L/D
//   K = kappa_r^2/gamma_rr + kappa_th^2/gamma_thth + lam^2/gamma_phiphi
//   denom(h) = (tau+D+sqrt(-g)*P)/D = h*E
//
// Source terms (both Sr and Stheta now need one -- neither r nor theta is
// a Killing direction), via the same symmetric-tensor identity validated
// in Phase 2a and again in tools/fishbone_moncrief.py (matches Gammie,
// McKinney & Toth 2003's "HARM" formulation, eq. 8-10, which is where the
// sqrt(-g) convention -- as opposed to sqrt(gamma) -- comes from):
//   Source_Sr     = sqrt(-g)*0.5*[T^tt*dg_tt/dr + 2*T^tphi*dg_tphi/dr
//                                  + T^phiphi*dg_phiphi/dr + T^rr*dg_rr/dr + T^thth*dg_thth/dr]
//   Source_Stheta = (same, d/dtheta instead of d/dr)
// (the T^{r theta} cross term some would expect from the general formula
// is identically absent: g_{r theta}=0 for ALL r,theta in Boyer-Lindquist,
// so its derivative -- the only way T^{r theta} could contribute -- is
// identically zero too, not just small.)
//
// Domain choice avoids genuine polar coordinate-singularity handling
// (reflecting boundary conditions, the cot(theta) terms in a full
// treatment, etc.): the validated Fishbone-Moncrief torus this phase
// targets is already known (tools/fishbone_moncrief.py) to taper to zero
// density well before reaching either pole, so uRthetaMin/uThetaMax are
// chosen with a safety margin and plain outflow is used there, same as
// every other boundary in this project -- a deliberate, documented scope
// limit, not an oversight (see README's Physics section).
//
// Buffer layout (std430 SSBOs, Nr*Ntheta cells, row-major index i*Ntheta+j,
// i=0..Nr-1 radial, j=0..Ntheta-1 polar):
//   binding 0/1   Cons0/Cons1     vec4(D, Sr, Stheta, tau)   -- ping-pong
//   binding 2/3   Stage1/Stage2   vec4(D, Sr, Stheta, tau)
//   binding 6/7   Cons0L/Cons1L   float(L)                    -- ping-pong (L is source-free,
//   binding 8/9   Stage1L/Stage2L float(L)                       kept in its own buffer since a
//                                                                 5th conserved quantity doesn't
//                                                                 fit in a vec4 with the other 4)
//   binding 4     PrimMain        vec4(rho, v_r, v_theta, v_phi)
//   binding 10    PrimP           float(P)
//   binding 5     FluxR           vec4(F_D,F_Sr,F_Stheta,F_tau) at r-interfaces, (Nr+1)*Ntheta
//   binding 11    FluxRL          float(F_L) at r-interfaces
//   binding 12    FluxTh          vec4(F_D,F_Sr,F_Stheta,F_tau) at theta-interfaces, Nr*(Ntheta+1)
//   binding 13    FluxThL         float(F_L) at theta-interfaces

namespace grhd::kernels_kerr2d {

inline constexpr int kWorkgroupSize = 256;

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

uniform int uNr;
uniform int uNth;
uniform float uGamma;
uniform float uM;
uniform float uA;
uniform float uRmin;
uniform float uDr;
uniform float uThetaMin;
uniform float uDth;

int cellIndex(int i, int j) { return i * uNth + j; }
)";
}

inline const char* KerrMetric2DGlsl = R"(
struct Metric { float gtt, gtphi, grr, gthth, gphiphi; };

Metric metricAt(float r, float theta) {
    float s = sin(theta), c = cos(theta);
    float sin2 = s * s, cos2 = c * c;
    float Sigma = r * r + uA * uA * cos2;
    float Delta = r * r - 2.0 * uM * r + uA * uA;
    float A = (r * r + uA * uA) * (r * r + uA * uA) - uA * uA * Delta * sin2;
    Metric m;
    m.gtt = -(1.0 - 2.0 * uM * r / Sigma);
    m.gtphi = -2.0 * uM * uA * r * sin2 / Sigma;
    m.grr = Sigma / Delta;
    m.gthth = Sigma;
    m.gphiphi = A * sin2 / Sigma;
    return m;
}

// sqrt(-g), the full 4-metric determinant -- NOT sqrt(gamma)=sqrt(g_rr*g_thth*g_phiphi)
// (the 3-metric determinant), which an earlier version of this file used by
// mistake. They differ by exactly the lapse (sqrt(-g)=alpha*sqrt(gamma), the
// standard ADM identity) -- invisible in Phase 2a because sqrt(-g) happens to
// equal sqrt(gamma)*alpha=r^2 exactly AT the equator for BOTH quantities'
// r^2-like shorthand there, but they diverge off it. The conservative GRHD
// formulation (Gammie, McKinney & Toth 2003, "HARM", eq. 8-10 -- the source
// used to derive kernels_kerr2d.hpp's momentum source term) uses sqrt(-g)
// throughout; using sqrt(gamma) instead breaks both mass conservation and
// the flux/source balance, and was caught only by a from-scratch divergence-
// theorem residual check against the analytic torus, not by any shape-level
// validation (see 07_grhd/README.md's Physics section). For Kerr,
// sqrt(-g)=Sigma*sin(theta) exactly (a standard, independently-checkable
// result -- verified here against a brute-force 4x4 determinant before use).
float sqrtNegG(float r, float theta) {
    Metric m = metricAt(r, theta);
    return m.gthth * sin(theta); // m.gthth = Sigma
}

void zamoAt(float r, float theta, out float alpha, out float betaPhiUp,
            out float gamma_rr, out float gamma_thth, out float gamma_phiphi, out float g_tphi) {
    Metric m = metricAt(r, theta);
    float Delta = r * r - 2.0 * uM * r + uA * uA;
    float sin2 = sin(theta) * sin(theta);
    float A = (r * r + uA * uA) * (r * r + uA * uA) - uA * uA * Delta * sin2;
    float Sigma = m.gthth;
    alpha = sqrt(Sigma * Delta / A);
    betaPhiUp = m.gtphi / m.gphiphi;
    gamma_rr = m.grr;
    gamma_thth = m.gthth;
    gamma_phiphi = m.gphiphi;
    g_tphi = m.gtphi;
}

// (rho,v_r,v_theta,v_phi,P) -> W,h,u^t,u^r,u^theta,u^phi(contra),
// u_r,u_theta,u_phi (cov), E. Direct generalization of kernels_kerr.hpp's
// kinematics(), one extra velocity component along for the ride.
void kinematics2D(float rho, float vr, float vth, float vphi, float P, float r, float theta,
                   out float W, out float h, out float ut, out float ur, out float uth,
                   out float uPhiContra, out float urCov, out float uthCov, out float uPhiCov, out float E) {
    float alpha, betaPhiUp, gamma_rr, gamma_thth, gamma_phiphi, g_tphi;
    zamoAt(r, theta, alpha, betaPhiUp, gamma_rr, gamma_thth, gamma_phiphi, g_tphi);

    W = 1.0 / sqrt(clamp(1.0 - vr*vr - vth*vth - vphi*vphi, 1e-10, 1.0));
    h = 1.0 + uGamma * P / ((uGamma - 1.0) * rho);
    ut = W / alpha;
    ur = W * vr / sqrt(gamma_rr);
    uth = W * vth / sqrt(gamma_thth);
    uPhiContra = W * vphi / sqrt(gamma_phiphi) - W * betaPhiUp / alpha;
    urCov = W * sqrt(gamma_rr) * vr;
    uthCov = W * sqrt(gamma_thth) * vth;
    uPhiCov = W * sqrt(gamma_phiphi) * vphi;
    E = W * (alpha - g_tphi * vphi / sqrt(gamma_phiphi));
}
)";

// ConsToPrim: Newton iteration on h, identical structure to kernels_kerr.hpp,
// generalized with a 3-term K (see header comment). Also enforces a
// density floor: the region between the horizon and the torus's inner
// edge (r_in) is vacuum by construction (see tools/make_fm_torus_ic.py),
// carried at a tiny floor density/pressure at rest. Left alone, flux
// exchange with the real torus material slowly perturbs these
// near-vacuum cells' conserved variables, and dividing by their tiny D to
// recover a velocity (kappaR=Sr/D etc.) amplifies that perturbation into
// a large, spurious v_r/v_theta -- confirmed to be the actual mechanism
// behind an early version of this run's eventual blowup: the runaway
// growth traced to the innermost radial cells at near-floor density, not
// to the torus material itself (which stayed small and well-behaved, as
// expected). This is the same class of bug as 06_tidal_disruption's
// zero-density SPH fix -- floor cells need to be actively RESET each
// step (not just seeded once at t=0), both in the primitives (for this
// step's flux) and written back to the conserved buffer (so the state
// actually stays corrected next step, not just displayed once).
inline std::string ConsToPrim() {
    return CommonHeader() + KerrMetric2DGlsl + R"(
uniform int uIters;
uniform float uRhoFloor;
uniform float uPFloor;
uniform float uEntropyFloor;

layout(std430, binding = 0) buffer ConsBuf { vec4 cons[]; };
layout(std430, binding = 6) buffer ConsLBuf { float consL[]; };
layout(std430, binding = 4) buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) buffer PrimPBuf { float primP[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNr * uNth)) return;
    int i = int(idx) / uNth, j = int(idx) % uNth;
    float r = uRmin + (float(i) + 0.5) * uDr;
    float theta = uThetaMin + (float(j) + 0.5) * uDth;

    float alpha, betaPhiUp, gamma_rr, gamma_thth, gamma_phiphi, g_tphi;
    zamoAt(r, theta, alpha, betaPhiUp, gamma_rr, gamma_thth, gamma_phiphi, g_tphi);
    float sqrtg = sqrtNegG(r, theta);

    vec4 U = cons[idx];
    float D = U.x, Sr = U.y, Sth = U.z, tau = U.w;
    float L = consL[idx];

    float kappaR = Sr / D, kappaTh = Sth / D, lam = L / D;
    float K = kappaR*kappaR/gamma_rr + kappaTh*kappaTh/gamma_thth + lam*lam/gamma_phiphi;

    // Physicality pre-check: does ANY h>=1 solve the residual equation at
    // all? As h->infinity, f(h) -> -alpha/Gamma < 0 always (rho saturates
    // at D*alpha/sqrtg, so P grows only linearly in h, not fast enough to
    // keep f positive); so a root exists in [1,infinity) only if
    // f(h=1)>=0, i.e.:
    //   (tau+D)/D + g_tphi*lam/gamma_phiphi >= sqrt(1+K)*alpha
    // (derived from the residual formula below evaluated at h=1, P=0;
    // verified against synthetic solvable/unsolvable test cases before
    // use -- see tools/ for the derivation notes). When this fails, NO
    // number of Newton iterations can converge (confirmed empirically:
    // raising cons_to_prim_iters from 40 to 200 did not change the
    // observed blowup at all) -- the conserved state itself is
    // momentarily unphysical.
    //
    // Fixup, NOT a full vacuum reset: a first attempt reset unphysical
    // cells all the way to vacuum-at-rest, matching the vacuum-floor
    // branch below -- but this violation turned out to be common at the
    // torus's own low-density surface (any equilibrium's outer layers are
    // the most marginal, easily perturbed across this boundary by a
    // first-order scheme's diffusion every step), so a full reset there
    // was erasing real torus mass wholesale (confirmed: >97% total mass
    // lost over the run, an unphysical result in itself, not a fix).
    // Standard GRMHD practice for a marginally-unphysical state is
    // instead a MINIMAL correction: rescale the momentum (Sr,Sth,L, i.e.
    // kappaR,kappaTh,lam) down by the smallest factor s in (0,1) that
    // restores physicality, preserving as much of the original state as
    // possible rather than discarding it. Found by bisection (the
    // physical/unphysical boundary as a function of s is monotonic: s=0
    // -- pure D,tau, no momentum -- is always physical if tau,D
    // themselves are sane, and s=1 is the already-known-unphysical
    // original state).
    // A small tolerance here (vs. a strict >=0) turned out to change
    // nothing: rerunning decks/kerr_torus.toml with tol=1e-4 gave
    // essentially identical mass-loss numbers to tol=0 at every
    // checkpoint. That rules out "marginal float-noise-level violations
    // right at the boundary" as the trigger -- the fixups below are
    // firing for genuinely, substantially unphysical states, not
    // borderline ones. The real fix for the mass loss this fixup still
    // leaves (see 07_grhd/README.md's Progress section) is therefore NOT
    // a tolerance tweak; kept at 0 here since a nonzero value
    // demonstrably buys nothing.
    float physMargin = (tau + D) / D + g_tphi * lam / gamma_phiphi - sqrt(1.0 + K) * alpha;
    bool physical = physMargin >= 0.0;
    if (!physical) {
        float sLo = 0.0, sHi = 1.0;
        for (int bi = 0; bi < 24; ++bi) {
            float sMid = 0.5 * (sLo + sHi);
            float Kmid = sMid * sMid * K;
            bool midOk = ((tau + D) / D + sMid * g_tphi * lam / gamma_phiphi) >= sqrt(1.0 + Kmid) * alpha;
            if (midOk) sLo = sMid; else sHi = sMid;
        }
        float sFix = sLo * 0.999; // small safety margin inside the physical region
        kappaR *= sFix; kappaTh *= sFix; lam *= sFix;
        K = kappaR*kappaR/gamma_rr + kappaTh*kappaTh/gamma_thth + lam*lam/gamma_phiphi;
        Sr = kappaR * D; Sth = kappaTh * D; L = lam * D;
        cons[idx] = vec4(D, Sr, Sth, tau);
        consL[idx] = L;
        physical = true; // now solvable -- proceed with the normal Newton solve below
    }

    // Lower bound uses 1e-4 (NOT the mathematically-tighter 1e-9 an earlier
    // version used): float32 near h=1 has ~1.19e-7 ULP spacing, so
    // "1.0+1e-9" silently rounds to exactly 1.0 -- making eps=(h-1)/Gamma
    // and hence P exactly 0.0, not just small. That degenerate exact-zero
    // state was traced (via a from-scratch replay of this exact iteration
    // against saved simulation frames) to precede every observed blowup:
    // once h snaps to exactly 1, a subsequent step's flux/source update
    // can leave no h>=1 solution nearby, and the unguarded Newton step
    // below sent h to a nonsensical, enormous value in a single iteration
    // (observed: P jumping from 0 to 2.36e8 between consecutive frames).
    // Both changes below are real, independent safeguards, not just a
    // bigger epsilon: a float32-representable floor, AND a damped step
    // that caps how far a single Newton iteration can move h (a standard
    // primitive-recovery safeguard -- c.f. HARM's "u2p" routines, which
    // are well known to need exactly this kind of guarding), so a bad
    // local derivative estimate can no longer run away unbounded.
    const float kHFloor = 1.0 + 1e-4;
    float hGuess = primP[idx] > 0.0 ? 1.0 + uGamma*primP[idx]/((uGamma-1.0)*max(D*alpha/(sqrtg*sqrt(1.0+K)),1e-8)) : 1.001;
    float h = max(hGuess, kHFloor);
    if (physical) {
        for (int it = 0; it < uIters; ++it) {
            float dh = max(1e-6*abs(h), 1e-9);
            float hp = h + dh, hm = h - dh;

            float Wc = sqrt(1.0+K/(h*h));
            float rhoC = D*alpha/(sqrtg*Wc);
            float Pc = (uGamma-1.0)*rhoC*(h-1.0)/uGamma;
            float f0 = (tau+D+sqrtg*Pc)/D/h - (Wc*alpha - g_tphi*lam/(h*gamma_phiphi));

            float Wp = sqrt(1.0+K/(hp*hp)); float rhoP = D*alpha/(sqrtg*Wp); float Pp=(uGamma-1.0)*rhoP*(hp-1.0)/uGamma;
            float fPlus = (tau+D+sqrtg*Pp)/D/hp - (Wp*alpha - g_tphi*lam/(hp*gamma_phiphi));
            float Wm = sqrt(1.0+K/(hm*hm)); float rhoM = D*alpha/(sqrtg*Wm); float Pm=(uGamma-1.0)*rhoM*(hm-1.0)/uGamma;
            float fMinus = (tau+D+sqrtg*Pm)/D/hm - (Wm*alpha - g_tphi*lam/(hm*gamma_phiphi));

            float deriv = (fPlus - fMinus) / (2.0*dh);
            if (abs(deriv) > 1e-12) {
                float step = clamp(f0 / deriv, -0.5*h, 0.5*h); // damped: at most a 50% change in h per iteration
                h = max(h - step, kHFloor);
            }
        }
    }

    float W = sqrt(1.0 + K/(h*h));
    float rho = D*alpha/(sqrtg*W);
    float vr = kappaR/(h*W*sqrt(gamma_rr));
    float vth = kappaTh/(h*W*sqrt(gamma_thth));
    float vphi = lam/(h*W*sqrt(gamma_phiphi));
    float eps = (h - 1.0) / uGamma;
    float P = (uGamma - 1.0) * rho * eps;

    if (rho < uRhoFloor) {
        // Reset to vacuum-at-rest (v=0 in the ZAMO frame) and recompute
        // the conserved variables to match -- see this function's header
        // comment for why the floor must be written back to cons/consL,
        // not just to the displayed primitives.
        rho = uRhoFloor;
        vr = 0.0; vth = 0.0; vphi = 0.0;
        P = uPFloor;
        float Wf, hf, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef;
        kinematics2D(rho, vr, vth, vphi, P, r, theta, Wf, hf, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef);
        float Df = sqrtg * rho * utf;
        float Srf = sqrtg * rho * hf * utf * urCovf;   // = 0 (urCovf=0)
        float Sthf = sqrtg * rho * hf * utf * uthCovf; // = 0 (uthCovf=0)
        float Lf = sqrtg * rho * hf * utf * uPhiCovf;  // = 0 (uPhiCovf=0)
        float tauf = sqrtg * rho * hf * utf * Ef - sqrtg * P - Df;
        cons[idx] = vec4(Df, Srf, Sthf, tauf);
        consL[idx] = Lf;
    } else {
        // Entropy/pressure floor: P_floor(rho) = uEntropyFloor * rho^Gamma
        // -- a small FRACTION of what the torus's own polytropic relation
        // (P=K*rho^Gamma) would give at this density, i.e. scaled WITH
        // rho, unlike a fixed additive enthalpy floor (tried first: an
        // h>=1.05 floor avoided the crash entirely but corresponds to a
        // FIXED P offset independent of rho, which is negligible for the
        // torus's dense core but dominates -- and badly distorts the
        // physics -- for any low-density gas; confirmed by the 85% total
        // mass INCREASE it caused, a clear sign of spurious energy
        // injection, not a real fix). This scaled version targets only
        // gas that has gone genuinely colder than the flow's own entropy
        // floor should allow, without perturbing normal-density material
        // at all (P_floor is tiny wherever rho is not tiny). Velocity is
        // kept as recovered (unlike the vacuum-floor branch above) --
        // this isn't declaring the cell vacuum, just refusing to let its
        // entropy collapse toward the numerically fragile P~0 regime that
        // preceded every observed blowup (see the kHFloor comment above).
        float pFloorEntropy = uEntropyFloor * pow(rho, uGamma);
        if (P < pFloorEntropy) {
            P = pFloorEntropy;
            float epsF = P / ((uGamma - 1.0) * rho);
            float hF = 1.0 + uGamma * epsF;
            float Wf = sqrt(1.0 + K/(hF*hF));
            // Recompute rho from the (now slightly different) hF via the
            // same W=sqrt(1+K/h^2) relation, keeping D fixed -- consistent
            // with how rho/h/W all relate to the SAME conserved D above.
            float rhoF = D*alpha/(sqrtg*Wf);
            float vrF = kappaR/(hF*Wf*sqrt(gamma_rr));
            float vthF = kappaTh/(hF*Wf*sqrt(gamma_thth));
            float vphiF = lam/(hF*Wf*sqrt(gamma_phiphi));
            P = (uGamma - 1.0) * rhoF * (hF - 1.0) / uGamma;
            rho = rhoF; vr = vrF; vth = vthF; vphi = vphiF;

            float Wf2, hf2, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef;
            kinematics2D(rho, vr, vth, vphi, P, r, theta, Wf2, hf2, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef);
            float Df = sqrtg * rho * utf;
            float Srf = sqrtg * rho * hf2 * utf * urCovf;
            float Sthf = sqrtg * rho * hf2 * utf * uthCovf;
            float Lf = sqrtg * rho * hf2 * utf * uPhiCovf;
            float tauf = sqrtg * rho * hf2 * utf * Ef - sqrtg * P - Df;
            cons[idx] = vec4(Df, Srf, Sthf, tauf);
            consL[idx] = Lf;
        }
    }

    primMain[idx] = vec4(rho, vr, vth, vphi);
    primP[idx] = P;
}
)";
}

// Side state (U, F_r, F_theta) at a cell, from its primitives -- shared by
// both flux passes.
inline const char* SideStateGlsl = R"(
void sideState(vec4 primM, float P, float r, float theta,
                out vec4 U, out float UL, out vec4 Fr, out float FrL, out vec4 Fth, out float FthL) {
    float rho = primM.x, vr = primM.y, vth = primM.z, vphi = primM.w;
    float W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E;
    kinematics2D(rho, vr, vth, vphi, P, r, theta, W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E);
    float sqrtg = sqrtNegG(r, theta);

    float D = sqrtg * rho * ut;
    float Sr = sqrtg * rho * h * ut * urCov;
    float Sth = sqrtg * rho * h * ut * uthCov;
    float L = sqrtg * rho * h * ut * uPhiCov;
    float tau = sqrtg * rho * h * ut * E - sqrtg * P - D;
    U = vec4(D, Sr, Sth, tau);
    UL = L;

    float FrD = sqrtg * rho * ur;
    float FrSr = sqrtg * (rho * h * ur * urCov + P);
    float FrSth = sqrtg * rho * h * ur * uthCov;
    float FrTau = sqrtg * rho * ur * (h * E - 1.0);
    Fr = vec4(FrD, FrSr, FrSth, FrTau);
    FrL = sqrtg * rho * h * ur * uPhiCov;

    float FthD = sqrtg * rho * uth;
    float FthSr = sqrtg * rho * h * uth * urCov;
    float FthSth = sqrtg * (rho * h * uth * uthCov + P);
    float FthTau = sqrtg * rho * uth * (h * E - 1.0);
    Fth = vec4(FthD, FthSr, FthSth, FthTau);
    FthL = sqrtg * rho * h * uth * uPhiCov;
}
)";

// Pass: HLLE flux at each r-interface (fixed theta cell-center), for all
// (Nr+1)*Ntheta interfaces. Conservative bound: exact radial photon
// coordinate speed sqrt(-g_tt/g_rr) (see header comment; same
// deliberately-conservative choice as kernels_kerr.hpp).
inline std::string FluxesR() {
    return CommonHeader() + KerrMetric2DGlsl + SideStateGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 5) buffer FluxRBuf { vec4 fluxR[]; };
layout(std430, binding = 11) buffer FluxRLBuf { float fluxRL[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    int nInterfaces = (uNr + 1) * uNth;
    if (int(idx) >= nInterfaces) return;
    int i = int(idx) / uNth, j = int(idx) % uNth;
    int iL = (i == 0) ? 0 : (i - 1);
    int iR = (i == uNr) ? (uNr - 1) : i;

    float rFace = uRmin + float(i) * uDr;
    float theta = uThetaMin + (float(j) + 0.5) * uDth;
    Metric m = metricAt(rFace, theta);
    float fPhoton = sqrt(-m.gtt / m.grr);

    int idxL = cellIndex(iL, j), idxR = cellIndex(iR, j);
    vec4 UL, UR, FrL4, FrR4, FthDummyL, FthDummyR;
    float ULl, URl, FrLl, FrRl, FthDummyLl, FthDummyRl;
    sideState(primMain[idxL], primP[idxL], rFace, theta, UL, ULl, FrL4, FrLl, FthDummyL, FthDummyLl);
    sideState(primMain[idxR], primP[idxR], rFace, theta, UR, URl, FrR4, FrRl, FthDummyR, FthDummyRl);

    float sL = -fPhoton, sR = fPhoton;
    fluxR[idx] = (sR * FrL4 - sL * FrR4 + sL * sR * (UR - UL)) / (sR - sL);
    fluxRL[idx] = (sR * FrLl - sL * FrRl + sL * sR * (URl - ULl)) / (sR - sL);
}
)";
}

// Pass: HLLE flux at each theta-interface (fixed r cell-center), for all
// Nr*(Ntheta+1) interfaces. Bound: exact theta-photon coordinate speed
// sqrt(-g_tt/g_thth) (same reasoning as the r-direction).
inline std::string FluxesTheta() {
    return CommonHeader() + KerrMetric2DGlsl + SideStateGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 12) buffer FluxThBuf { vec4 fluxTh[]; };
layout(std430, binding = 13) buffer FluxThLBuf { float fluxThL[]; };

int thCellIndex(int i, int j) { return i * (uNth + 1) + j; }

void main() {
    uint idx = gl_GlobalInvocationID.x;
    int nInterfaces = uNr * (uNth + 1);
    if (int(idx) >= nInterfaces) return;
    int i = int(idx) / (uNth + 1), j = int(idx) % (uNth + 1);
    int jL = (j == 0) ? 0 : (j - 1);
    int jR = (j == uNth) ? (uNth - 1) : j;

    float r = uRmin + (float(i) + 0.5) * uDr;
    float thFace = uThetaMin + float(j) * uDth;
    Metric m = metricAt(r, thFace);
    float fPhoton = sqrt(-m.gtt / m.gthth);

    int idxL = cellIndex(i, jL), idxR = cellIndex(i, jR);
    vec4 UL, UR, FrDummyL, FrDummyR, FthL4, FthR4;
    float ULl, URl, FrDummyLl, FrDummyRl, FthLl, FthRl;
    sideState(primMain[idxL], primP[idxL], r, thFace, UL, ULl, FrDummyL, FrDummyLl, FthL4, FthLl);
    sideState(primMain[idxR], primP[idxR], r, thFace, UR, URl, FrDummyR, FrDummyRl, FthR4, FthRl);

    float sL = -fPhoton, sR = fPhoton;
    int outIdx = i * (uNth + 1) + j;
    fluxTh[outIdx] = (sR * FthL4 - sL * FthR4 + sL * sR * (UR - UL)) / (sR - sL);
    fluxThL[outIdx] = (sR * FthLl - sL * FthRl + sL * sR * (URl - ULl)) / (sR - sL);
}
)";
}

// Euler predictor step: 2D flux divergence (r and theta) plus the two
// geometric source terms (Sr, Stheta -- see header comment; D, L, tau are
// exactly source-free).
inline std::string EulerStep() {
    return CommonHeader() + KerrMetric2DGlsl + R"(
uniform float uDt;

layout(std430, binding = 0) readonly buffer ConsIn { vec4 consIn[]; };
layout(std430, binding = 6) readonly buffer ConsLIn { float consLIn[]; };
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 5) readonly buffer FluxRBuf { vec4 fluxR[]; };
layout(std430, binding = 11) readonly buffer FluxRLBuf { float fluxRL[]; };
layout(std430, binding = 12) readonly buffer FluxThBuf { vec4 fluxTh[]; };
layout(std430, binding = 13) readonly buffer FluxThLBuf { float fluxThL[]; };
layout(std430, binding = 1) buffer ConsOut { vec4 consOut[]; };
layout(std430, binding = 7) buffer ConsLOut { float consLOut[]; };

int rFaceIndex(int i, int j) { return i * uNth + j; }         // (uNr+1)*uNth entries
int thFaceIndex(int i, int j) { return i * (uNth + 1) + j; }  // uNr*(uNth+1) entries

void sourceTerms(vec4 primM, float P, float r, float theta, out float srcSr, out float srcSth) {
    float rho = primM.x, vr = primM.y, vth = primM.z, vphi = primM.w;
    float W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E;
    kinematics2D(rho, vr, vth, vphi, P, r, theta, W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E);

    Metric m = metricAt(r, theta);
    float detG = m.gtt*m.gphiphi - m.gtphi*m.gtphi;
    float gUtt = m.gphiphi/detG, gUtphi = -m.gtphi/detG, gUphiphi = m.gtt/detG;
    float gUrr = 1.0/m.grr, gUthth = 1.0/m.gthth;

    float Ttt = rho*h*ut*ut + P*gUtt;
    float Ttphi = rho*h*ut*uPhiContra + P*gUtphi;
    float Tphiphi = rho*h*uPhiContra*uPhiContra + P*gUphiphi;
    float Trr = rho*h*ur*ur + P*gUrr;
    float Tthth = rho*h*uth*uth + P*gUthth;

    float epsR = max(1e-5*r, 1e-5);
    Metric mrp = metricAt(r+epsR, theta); Metric mrm = metricAt(r-epsR, theta);
    float dgtt_r = (mrp.gtt-mrm.gtt)/(2.0*epsR);
    float dgtphi_r = (mrp.gtphi-mrm.gtphi)/(2.0*epsR);
    float dgphiphi_r = (mrp.gphiphi-mrm.gphiphi)/(2.0*epsR);
    float dgrr_r = (mrp.grr-mrm.grr)/(2.0*epsR);
    float dgthth_r = (mrp.gthth-mrm.gthth)/(2.0*epsR);
    srcSr = sqrtNegG(r,theta) * 0.5*(Ttt*dgtt_r + 2.0*Ttphi*dgtphi_r + Tphiphi*dgphiphi_r + Trr*dgrr_r + Tthth*dgthth_r);

    float epsT = 1e-5;
    Metric mtp = metricAt(r, theta+epsT); Metric mtm = metricAt(r, theta-epsT);
    float dgtt_t = (mtp.gtt-mtm.gtt)/(2.0*epsT);
    float dgtphi_t = (mtp.gtphi-mtm.gtphi)/(2.0*epsT);
    float dgphiphi_t = (mtp.gphiphi-mtm.gphiphi)/(2.0*epsT);
    float dgrr_t = (mtp.grr-mtm.grr)/(2.0*epsT);
    float dgthth_t = (mtp.gthth-mtm.gthth)/(2.0*epsT);
    srcSth = sqrtNegG(r,theta) * 0.5*(Ttt*dgtt_t + 2.0*Ttphi*dgtphi_t + Tphiphi*dgphiphi_t + Trr*dgrr_t + Tthth*dgthth_t);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNr * uNth)) return;
    int i = int(idx) / uNth, j = int(idx) % uNth;
    float r = uRmin + (float(i) + 0.5) * uDr;
    float theta = uThetaMin + (float(j) + 0.5) * uDth;

    vec4 divFr = (fluxR[rFaceIndex(i+1, j)] - fluxR[rFaceIndex(i, j)]) / uDr;
    vec4 divFth = (fluxTh[thFaceIndex(i, j+1)] - fluxTh[thFaceIndex(i, j)]) / uDth;
    float divFrL = (fluxRL[rFaceIndex(i+1, j)] - fluxRL[rFaceIndex(i, j)]) / uDr;
    float divFthL = (fluxThL[thFaceIndex(i, j+1)] - fluxThL[thFaceIndex(i, j)]) / uDth;

    float srcSr, srcSth;
    sourceTerms(primMain[idx], primP[idx], r, theta, srcSr, srcSth);
    vec4 source = vec4(0.0, srcSr, srcSth, 0.0);

    consOut[idx] = consIn[idx] - uDt * (divFr + divFth) + uDt * source;
    consLOut[idx] = consLIn[idx] - uDt * (divFrL + divFthL);
}
)";
}

inline std::string Combine() {
    return CommonHeader() + R"(
layout(std430, binding = 0) readonly buffer ConsA { vec4 consA[]; };
layout(std430, binding = 1) readonly buffer ConsB { vec4 consB[]; };
layout(std430, binding = 2) buffer ConsOut { vec4 consOut[]; };
layout(std430, binding = 6) readonly buffer ConsLA { float consLA[]; };
layout(std430, binding = 7) readonly buffer ConsLB { float consLB[]; };
layout(std430, binding = 8) buffer ConsLOut { float consLOut[]; };

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNr * uNth)) return;
    consOut[idx] = 0.5 * (consA[idx] + consB[idx]);
    consLOut[idx] = 0.5 * (consLA[idx] + consLB[idx]);
}
)";
}

} // namespace grhd::kernels_kerr2d
