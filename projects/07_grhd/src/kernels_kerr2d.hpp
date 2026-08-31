#pragma once

#include <string>

// GLSL compute-shader source for full 2D (r,theta) GRHD on a fixed Kerr
// background -- Tier 2 Phase 2b part 2, see 07_grhd/README.md. Uses
// horizon-penetrating spherical KERR-SCHILD (KS) coordinates, NOT
// Boyer-Lindquist (BL) -- see this file's git history and README's mass-
// loss investigation for why: BL's coordinate singularity at the horizon
// forced an artificial inner radial boundary INSIDE the flow (well
// outside the true horizon), and the standard "outflow" boundary
// condition there could not drain real infalling material fast enough
// once the HLLE wave-speed bound was tightened from a raw photon speed to
// an actual sound-speed estimate (see FluxesR's comment) -- mass piled up
// at that artificial cutoff until the scheme diverged. KS avoids the
// problem at its root: r and theta are UNCHANGED from BL (only t and phi
// are redefined), so the domain can extend to/through the horizon with no
// coordinate singularity and no artificial mid-flow boundary at all.
//
// Every formula below was derived and independently verified in
// tools/kerr_schild_derive.py (metric + ADM 3+1 split, verified against a
// from-scratch BL->KS coordinate-transform substitution) and
// tools/kerr_schild_ref.py (primitive recovery, source terms, IC
// transform, all verified against independent invariants -- round-trip
// prim<->cons, direct 4-velocity normalization, known circular orbits,
// conserved energy/angular momentum under the BL->KS transform) and
// tools/wave_speed_check.py (HLLE wave-speed bound) BEFORE being trusted
// here, per this project's standing methodology: never trust a recalled
// GR formula, or even a *carefully hand-derived* one, without a from-
// scratch numerical cross-check -- this reformulation caught two genuine
// derivation bugs this way (a missing beta_i*u^t term in the covariant
// velocity, and a wrong assumption in the con2prim quadratic), neither of
// which any amount of re-reading the algebra caught; only comparing
// against an independent numerical identity did.
//
// Kerr-Schild metric (spherical form, general theta, G=c=1). r,theta same
// as BL; Sigma=r^2+a^2*cos^2(theta), Delta=r^2-2Mr+a^2 (same definitions
// as BL) but the metric itself picks up TWO new nonzero off-diagonal
// components BL didn't have, g_tr and g_rphi (this is the key structural
// difference: KS's spatial 3-metric is NOT diagonal, unlike BL's):
//   g_tt=-(1-2Mr/Sigma)      g_tr=2Mr/Sigma       g_tphi=-2Mar*sin^2(theta)/Sigma
//   gamma_rr=1+2Mr/Sigma     gamma_rphi=-a*sin^2(theta)*(1+2Mr/Sigma)
//   gamma_thth=Sigma         gamma_phiphi=sin^2(theta)*(Sigma+a^2*sin^2(theta)*(1+2Mr/Sigma))
// ADM 3+1 (verified against the metric above): alpha^2=Sigma/(Sigma+2Mr),
// beta^r=2Mr/(Sigma+2Mr), beta^phi=0 (the famous KS result: despite g_tphi
// and g_rphi both nonzero, the frame-dragging shift itself is PURELY
// radial). beta^theta=0 always (the coordinate transform never touches
// theta -- it stays exactly as decoupled from r,phi as it was in BL).
//
// Primitive velocity convention: v^i is now the GENERAL coordinate-frame
// Valencia velocity (u^i=W(v^i-beta^i/alpha), W built from the FULL
// quadratic form v^2=gamma_ij v^i v^j -- NOT a diagonal sum), not the old
// BL-era "physical/ZAMO-orthonormal" convention (which only worked
// because BL's spatial metric was diagonal). Con2prim's Newton iteration
// still runs on h, but W is no longer read off a simple W=sqrt(1+K/h^2)
// shortcut (that identity implicitly assumed beta_r=0, true for BL, false
// for KS): instead u^t is solved from the quadratic normalization
// equation at each trial h (see ConsToPrim's comment).
//
// Conserved variables (mixed-index T^mu_nu, weighted by sqrt(-g) --
// SAME formula and SAME sqrt(-g)=Sigma*sin(theta) as BL, this identity
// doesn't depend on which of BL/KS coordinates realizes it):
//   D    = sqrt(-g)*rho*u^t                     (source-free: baryon conservation)
//   Sr   = sqrt(-g)*rho*h*u^t*u_r                    (has a source)
//   Sth  = sqrt(-g)*rho*h*u^t*u_theta                (has a source)
//   L    = sqrt(-g)*rho*h*u^t*u_phi              (source-free: phi still Killing)
//   tau  = sqrt(-g)*rho*h*u^t*E - sqrt(-g)*P - D  (source-free: t still Killing)
// with u_r, u_phi now needing the FULL general relation u_i=gamma_ij
// u^j+beta_i*u^t (NOT simply gamma_ij u^j -- the beta_i*u^t term is
// exactly zero for BL, where beta_r=0, but nonzero here).
//
// Source terms: same general, metric-structure-independent identity as
// BL (Gammie, McKinney & Toth 2003 eq. 8-10) -- Source_i=sqrt(-g)*0.5*
// sum_munu T^munu*d(g_munu)/dx^i -- just summed over KS's 7 nonzero
// metric components instead of BL's 5 (the two new ones, g_tr and
// g_rphi, each contribute a 2*T^..*dg_../dx^i term, same factor-of-2
// convention as the existing g_tphi term). D, L, tau all stay EXACTLY
// source-free: t and phi are both still Killing vectors (none of the 7
// metric components depend on t or phi, only r and theta) -- verified
// against a KNOWN circular orbit (transformed BL->KS): Source_Sr came out
// ~1e-10 (finite-difference-epsilon-level, i.e. exactly zero) for that
// state, across three spins and four radii.
//
// HLLE wave-speed bound: r-direction needs a genuine local-orthonormal
// projection (gamma_rphi couples r and phi -- see FluxesR's comment for
// the tetrad construction), and its coordinate-speed conversion needs an
// extra -beta^r term BL never needed (beta^r=0 there). theta is
// unaffected (still orthogonal to both r and phi, beta^theta=0 always) --
// identical in form to the old BL treatment. Both directions keep a
// defensive clamp to the raw photon speed (now ASYMMETRIC for r --
// ingoing/outgoing radial photons have different coordinate speeds in a
// horizon-penetrating slicing, unlike BL's symmetric +-sqrt(-g_tt/g_rr)):
// this clamp is what makes the overall bound safe by construction even in
// the rare regime (verified: only very close to the horizon) where the
// tight sound-speed estimate itself isn't perfectly bounded.
//
// Domain choice avoids genuine polar coordinate-singularity handling
// (reflecting boundary conditions, the cot(theta) terms in a full
// treatment, etc.): the validated Fishbone-Moncrief torus this phase
// targets is already known (tools/fishbone_moncrief.py) to taper to zero
// density well before reaching either pole, so uRthetaMin/uThetaMax are
// chosen with a safety margin and plain outflow is used there, same as
// every other boundary in this project -- a deliberate, documented scope
// limit, not an oversight (see README's Physics section). The RADIAL
// inner boundary, by contrast, can now sit at or inside the horizon
// (r_min chosen well inside r_+ is fine -- KS has no coordinate
// singularity there), which is the entire point of this reformulation.
//
// Buffer layout (std430 SSBOs, Nr*Ntheta cells, row-major index i*Ntheta+j,
// i=0..Nr-1 radial, j=0..Ntheta-1 polar) -- UNCHANGED from the BL version:
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
//   binding 16    SlopeRMain      vec4(d(rho,vr,vth,vphi)/di), MinMod-limited, r-direction
//   binding 17    SlopeRP         float(dP/di), MinMod-limited, r-direction
//   binding 18    SlopeThMain     vec4(d(rho,vr,vth,vphi)/dj), MinMod-limited, theta-direction
//   binding 19    SlopeThP        float(dP/dj), MinMod-limited, theta-direction
//
// Reconstruction (ComputeSlopes, then FluxesR/FluxesTheta): each cell's
// primitives are extrapolated a HALF CELL toward each face using a
// MinMod-limited slope, rather than fed into the Riemann solver at their
// raw cell-center value -- standard piecewise-linear HRSC reconstruction
// (van Leer 1979; original HARM used piecewise-linear+MC from its first
// version, not piecewise-constant). This part is UNCHANGED by the KS
// reformulation -- it operates purely on primitive VALUES, with no
// reference to the metric at all.

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
// Bare lower-index KS metric components -- see this file's header
// comment. Cheap to evaluate at a displaced (r,theta) for the source
// terms' finite differences, without also computing the ADM/inverse
// pieces MetricBundle below adds.
struct Metric { float gtt, gtr, gtphi, grr, grphi, gthth, gphiphi; };

Metric metricAt(float r, float theta) {
    float s = sin(theta), c = cos(theta);
    float sin2 = s * s, cos2 = c * c;
    float Sigma = r * r + uA * uA * cos2;
    float twoMrOverSigma = 2.0 * uM * r / Sigma;
    Metric m;
    m.gtt = -(1.0 - twoMrOverSigma);
    m.gtr = twoMrOverSigma;
    m.gtphi = -uA * sin2 * twoMrOverSigma;
    m.grr = 1.0 + twoMrOverSigma;
    m.grphi = -uA * sin2 * (1.0 + twoMrOverSigma);
    m.gthth = Sigma;
    m.gphiphi = sin2 * (Sigma + uA * uA * sin2 * (1.0 + twoMrOverSigma));
    return m;
}

// sqrt(-g) = Sigma*sin(theta), IDENTICAL formula to BL (this identity
// doesn't depend on which coordinates realize the metric) -- verified in
// tools/kerr_schild_derive.py.
float sqrtNegG(float r, float theta) {
    Metric m = metricAt(r, theta);
    return m.gthth * sin(theta);
}

// Full ADM 3+1 + inverse-spatial-metric bundle -- see this file's header
// comment for the verified closed forms. beta^phi is always exactly 0
// for KS (the famous result) and is not included here (nothing downstream
// needs it as anything but a literal 0.0).
struct MetricBundle {
    Metric g;
    float alpha, betaUpR;
    float gammaUpRr, gammaUpRphi, gammaUpPhiphi, gammaUpThth;
};

MetricBundle metricBundleAt(float r, float theta) {
    Metric m = metricAt(r, theta);
    MetricBundle b;
    b.g = m;
    float Sigma = m.gthth;
    b.alpha = sqrt(Sigma / (Sigma + 2.0 * uM * r));
    b.betaUpR = 2.0 * uM * r / (Sigma + 2.0 * uM * r);
    float det2 = m.grr * m.gphiphi - m.grphi * m.grphi;
    b.gammaUpRr = m.gphiphi / det2;
    b.gammaUpRphi = -m.grphi / det2;
    b.gammaUpPhiphi = m.grr / det2;
    b.gammaUpThth = 1.0 / m.gthth;
    return b;
}

// (rho,v^r,v^th,v^phi,P) -> W,h,u^t,u^r,u^th,u^phi(contra),
// u_r,u_th,u_phi (cov), E. v^i is the GENERAL coordinate-frame Valencia
// velocity (see this file's header comment), NOT the old BL-era
// orthonormal convention.
void kinematics2D(float rho, float vr, float vth, float vphi, float P, float r, float theta,
                   out float W, out float h, out float ut, out float ur, out float uth,
                   out float uPhiContra, out float urCov, out float uthCov, out float uPhiCov, out float E) {
    MetricBundle b = metricBundleAt(r, theta);
    float v2 = b.g.grr * vr * vr + b.g.gthth * vth * vth + b.g.gphiphi * vphi * vphi + 2.0 * b.g.grphi * vr * vphi;
    W = 1.0 / sqrt(clamp(1.0 - v2, 1e-10, 1.0));
    h = 1.0 + uGamma * P / ((uGamma - 1.0) * rho);
    ut = W / b.alpha;
    ur = W * (vr - b.betaUpR / b.alpha);
    uth = W * vth;
    uPhiContra = W * vphi; // beta^phi=0 always for KS

    // u_i = gamma_ij u^j + beta_i*u^t -- the FULL relation (u_i is NOT
    // simply gamma_ij u^j once beta^r!=0: an earlier version of this
    // derivation dropped the beta_i*u^t term entirely, caught only by an
    // independent round-trip check, not by re-reading the algebra -- see
    // this file's header comment).
    float betaR = b.g.grr * b.betaUpR;      // beta_r = gamma_rj beta^j = gamma_rr*beta^r (beta^phi=0)
    float betaPhi = b.g.grphi * b.betaUpR;  // beta_phi = gamma_rphi*beta^r
    urCov = b.g.grr * ur + b.g.grphi * uPhiContra + betaR * ut;
    uPhiCov = b.g.grphi * ur + b.g.gphiphi * uPhiContra + betaPhi * ut;
    uthCov = b.g.gthth * uth; // beta_theta=0, no correction needed

    E = -(b.g.gtt * ut + b.g.gtr * ur + b.g.gtphi * uPhiContra);
}

// HLLE wave-speed bound -- see this file's header comment and
// tools/wave_speed_check.py. Local-orthonormal-frame acoustic
// characteristic speed (Marti & Muller / Anile, frame-independent SR
// physics -- unaffected by the KS reformulation, still exact to 1e-15
// against a from-scratch symbolic flux Jacobian).
float soundSpeed2(float rho, float P, float h) {
    return uGamma * P / (rho * h); // cs^2 = Gamma*P/(rho*h), ideal-gas EOS (Font 2008 eq. 68)
}

void charSpeedsLocal(float vx, float vy, float vz, float cs2, out float lamMinus, out float lamPlus) {
    float v2 = vx*vx + vy*vy + vz*vz;
    float denom = 1.0 - v2 * cs2;
    float disc = max(cs2 * (1.0 - v2) * ((1.0 - v2 * cs2) - vx * vx * (1.0 - cs2)), 0.0);
    float root = sqrt(disc);
    float centerTerm = vx * (1.0 - cs2);
    lamPlus = (centerTerm + root) / denom;
    lamMinus = (centerTerm - root) / denom;
}
)";

// ConsToPrim: Newton iteration on h. UNCHANGED in outer structure from the
// BL version (same physicality pre-check + bisection fixup, same
// float32-safe h-floor + damped-Newton-step safeguards, same vacuum-floor
// and entropy-floor branches), but the per-iteration RESIDUAL now solves
// u^t from the general quadratic normalization equation (see below)
// instead of the old W=sqrt(1+K/h^2) shortcut, which implicitly assumed
// beta_r=0 (an accident of BL's beta^r=0, not a general truth -- KS's
// beta^r!=0 makes it wrong, caught by tools/kerr_schild_ref.py's
// round-trip check failing until this was fixed).
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
// Diagnostic-only: counts how many cells the momentum fixup (below) had to
// correct this call -- see 07_grhd/README.md's Physics section.
layout(std430, binding = 14) buffer FixupCountBuf { uint fixupCount[]; };
// Diagnostic-only: counts how many cells the vacuum density-floor branch
// (below) resets per call.
layout(std430, binding = 15) buffer FloorCountBuf { uint floorCount[]; };

// K := gamma^ij kappa_i kappa_j (kappa_i:=S_i/D=h*u_i), general inverse-
// metric contraction (theta decouples; r,phi form a genuine 2x2
// contraction via the off-diagonal gammaUpRphi term).
float computeK(float kappaR, float kappaTh, float kappaPhi, MetricBundle b) {
    return b.gammaUpRr * kappaR * kappaR + b.gammaUpThth * kappaTh * kappaTh
         + b.gammaUpPhiphi * kappaPhi * kappaPhi + 2.0 * b.gammaUpRphi * kappaR * kappaPhi;
}

// Given a trial h (and the FIXED kappa_i, K from the conserved state),
// recover u^t, u^r, u^th, u^phi. X^i:=gamma^ij kappa_j/h is NOT u^i
// itself: raising the lowered kappa_i=h*u_i=h*(gamma_ij u^j+beta_i*u^t)
// with gamma^ij undoes the gamma_ij u^j part but leaves the beta_i*u^t
// part behind as an extra +beta^i*u^t on X^i, i.e. u^i=X^i-beta^i*u^t.
// Substituting into u^mu u_mu=-1 and collecting as a quadratic in u^t
// (done symbolically, not by hand -- see tools/kerr_schild_ref.py) gives
// the coefficients below; the physical root is the one with u^t>0.
void solveUtAndU(float h, float kappaR, float kappaTh, float kappaPhi, float K, MetricBundle b,
                  out float ut, out float ur, out float uth, out float uphi) {
    float Xr = (b.gammaUpRr * kappaR + b.gammaUpRphi * kappaPhi) / h;
    float Xphi = (b.gammaUpRphi * kappaR + b.gammaUpPhiphi * kappaPhi) / h;
    float Xth = (b.gammaUpThth * kappaTh) / h;

    float Acoef = b.g.gtt - b.betaUpR * b.g.gtr; // beta^phi=0, that term dropped
    float Bcoef = (-b.betaUpR * kappaR
                   + b.gammaUpPhiphi * b.g.gtphi * kappaPhi + b.gammaUpRphi * b.g.gtphi * kappaR
                   + b.gammaUpRphi * b.g.gtr * kappaPhi + b.gammaUpRr * b.g.gtr * kappaR) / h;
    float Ccoef = K / (h * h) + 1.0;
    float disc = max(Bcoef * Bcoef - 4.0 * Acoef * Ccoef, 0.0);
    float root1 = (-Bcoef + sqrt(disc)) / (2.0 * Acoef);
    float root2 = (-Bcoef - sqrt(disc)) / (2.0 * Acoef);
    ut = (root1 > 0.0) ? root1 : root2;

    ur = Xr - b.betaUpR * ut;
    uphi = Xphi; // beta^phi=0
    uth = Xth;
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNr * uNth)) return;
    int i = int(idx) / uNth, j = int(idx) % uNth;
    float r = uRmin + (float(i) + 0.5) * uDr;
    float theta = uThetaMin + (float(j) + 0.5) * uDth;

    MetricBundle b = metricBundleAt(r, theta);
    float sqrtg = sqrtNegG(r, theta);

    vec4 U = cons[idx];
    float D = U.x, Sr = U.y, Sth = U.z, tau = U.w;
    float L = consL[idx];

    float kappaR = Sr / D, kappaTh = Sth / D, kappaPhi = L / D;
    float K = computeK(kappaR, kappaTh, kappaPhi, b);

    // Physicality pre-check, same role as the BL version (does ANY h>=1
    // solve the residual at all): evaluate the residual at h=1 (P=0) via
    // the SAME solveUtAndU machinery used everywhere else, rather than a
    // separately-derived closed-form inequality (removes a second place
    // this logic could drift from the main iteration).
    float ut1, ur1, uth1, uphi1;
    solveUtAndU(1.0, kappaR, kappaTh, kappaPhi, K, b, ut1, ur1, uth1, uphi1);
    float E1 = -(b.g.gtt * ut1 + b.g.gtr * ur1 + b.g.gtphi * uphi1);
    float physMargin = (tau + D) / D - E1;
    bool physical = physMargin >= 0.0;
    if (!physical) {
        // Same bisection fixup as BL: rescale momentum (kappaR,kappaTh,
        // kappaPhi) down by the smallest s in (0,1) restoring physicality.
        float sLo = 0.0, sHi = 1.0;
        for (int bi = 0; bi < 24; ++bi) {
            float sMid = 0.5 * (sLo + sHi);
            float Kmid = sMid * sMid * K;
            float utM, urM, uthM, uphiM;
            solveUtAndU(1.0, sMid * kappaR, sMid * kappaTh, sMid * kappaPhi, Kmid, b, utM, urM, uthM, uphiM);
            float EM = -(b.g.gtt * utM + b.g.gtr * urM + b.g.gtphi * uphiM);
            bool midOk = ((tau + D) / D - EM) >= 0.0;
            if (midOk) sLo = sMid; else sHi = sMid;
        }
        float sFix = sLo * 0.999;
        kappaR *= sFix; kappaTh *= sFix; kappaPhi *= sFix;
        K = computeK(kappaR, kappaTh, kappaPhi, b);
        Sr = kappaR * D; Sth = kappaTh * D; L = kappaPhi * D;
        cons[idx] = vec4(D, Sr, Sth, tau);
        consL[idx] = L;
        atomicAdd(fixupCount[0], 1u);
        physical = true;
    }

    // Same float32-safe h-floor + damped-Newton-step safeguards as BL
    // (see kernels_kerr2d.hpp git history/README for why both are needed,
    // not just a smaller epsilon).
    const float kHFloor = 1.0 + 1e-4;
    float hGuess = primP[idx] > 0.0 ? 1.0 + uGamma*primP[idx]/((uGamma-1.0)*max(D*b.alpha/(sqrtg*sqrt(1.0+K)),1e-8)) : 1.001;
    float h = max(hGuess, kHFloor);
    if (physical) {
        for (int it = 0; it < uIters; ++it) {
            float dh = max(1e-6 * abs(h), 1e-9);
            float hp = h + dh, hm = h - dh;

            float ut0, ur0, uth0, uphi0; solveUtAndU(h, kappaR, kappaTh, kappaPhi, K, b, ut0, ur0, uth0, uphi0);
            float Wc = b.alpha * ut0; float rhoC = D * b.alpha / (sqrtg * Wc); float Pc = (uGamma-1.0)*rhoC*(h-1.0)/uGamma;
            float E0 = -(b.g.gtt*ut0 + b.g.gtr*ur0 + b.g.gtphi*uphi0);
            float f0 = (tau + D + sqrtg * Pc) / D / h - E0;

            float utP, urP, uthP, uphiP; solveUtAndU(hp, kappaR, kappaTh, kappaPhi, K, b, utP, urP, uthP, uphiP);
            float WP = b.alpha * utP; float rhoP = D * b.alpha / (sqrtg * WP); float PP=(uGamma-1.0)*rhoP*(hp-1.0)/uGamma;
            float EP = -(b.g.gtt*utP + b.g.gtr*urP + b.g.gtphi*uphiP);
            float fPlus = (tau + D + sqrtg * PP) / D / hp - EP;

            float utM, urM, uthM, uphiM; solveUtAndU(hm, kappaR, kappaTh, kappaPhi, K, b, utM, urM, uthM, uphiM);
            float WM = b.alpha * utM; float rhoM = D * b.alpha / (sqrtg * WM); float PM=(uGamma-1.0)*rhoM*(hm-1.0)/uGamma;
            float EM2 = -(b.g.gtt*utM + b.g.gtr*urM + b.g.gtphi*uphiM);
            float fMinus = (tau + D + sqrtg * PM) / D / hm - EM2;

            float deriv = (fPlus - fMinus) / (2.0 * dh);
            if (abs(deriv) > 1e-12) {
                float step = clamp(f0 / deriv, -0.5 * h, 0.5 * h);
                h = max(h - step, kHFloor);
            }
        }
    }

    float utF, urF0, uthF0, uphiF0; solveUtAndU(h, kappaR, kappaTh, kappaPhi, K, b, utF, urF0, uthF0, uphiF0);
    float W = b.alpha * utF;
    float rho = D * b.alpha / (sqrtg * W);
    float vr = urF0 / W + b.betaUpR / b.alpha;
    float vth = uthF0 / W;
    float vphi = uphiF0 / W;
    float eps = (h - 1.0) / uGamma;
    float P = (uGamma - 1.0) * rho * eps;

    if (rho < uRhoFloor) {
        // Reset to vacuum-at-rest (v^i=0 in the coordinate frame -- NOT
        // the same physical state as "v_hat=0" would have been under the
        // old BL convention, but the natural KS analogue: a fluid element
        // instantaneously comoving with the ZAMO observer) and recompute
        // the conserved variables to match.
        rho = uRhoFloor;
        vr = 0.0; vth = 0.0; vphi = 0.0;
        P = uPFloor;
        float Wf, hf, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef;
        kinematics2D(rho, vr, vth, vphi, P, r, theta, Wf, hf, utf, urf, uthf, uPhiContraf, urCovf, uthCovf, uPhiCovf, Ef);
        float Df = sqrtg * rho * utf;
        float Srf = sqrtg * rho * hf * utf * urCovf;
        float Sthf = sqrtg * rho * hf * utf * uthCovf;
        float Lf = sqrtg * rho * hf * utf * uPhiCovf;
        float tauf = sqrtg * rho * hf * utf * Ef - sqrtg * P - Df;
        cons[idx] = vec4(Df, Srf, Sthf, tauf);
        consL[idx] = Lf;
        atomicAdd(floorCount[0], 1u);
    } else {
        // Entropy/pressure floor: P_floor(rho) = uEntropyFloor*rho^Gamma
        // (see BL version's comment for why this scaled-with-rho form,
        // not a fixed additive one).
        float pFloorEntropy = uEntropyFloor * pow(rho, uGamma);
        if (P < pFloorEntropy) {
            P = pFloorEntropy;
            float epsF = P / ((uGamma - 1.0) * rho);
            float hF = 1.0 + uGamma * epsF;
            float utF2, urF2, uthF2, uphiF2; solveUtAndU(hF, kappaR, kappaTh, kappaPhi, K, b, utF2, urF2, uthF2, uphiF2);
            float Wf = b.alpha * utF2;
            float rhoF = D * b.alpha / (sqrtg * Wf);
            float vrF = urF2 / Wf + b.betaUpR / b.alpha;
            float vthF = uthF2 / Wf;
            float vphiF = uphiF2 / Wf;
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
// both flux passes. UNCHANGED in structure from BL; kinematics2D already
// carries the corrected u_i formulas.
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

// Pass: per-cell MinMod-limited primitive slopes in both r and theta --
// UNCHANGED by the KS reformulation (pure primitive-value reconstruction,
// no metric dependence at all). See this file's header comment.
inline std::string ComputeSlopes() {
    return CommonHeader() + R"(
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 16) writeonly buffer SlopeRMainBuf { vec4 slopeRMain[]; };
layout(std430, binding = 17) writeonly buffer SlopeRPBuf { float slopeRP[]; };
layout(std430, binding = 18) writeonly buffer SlopeThMainBuf { vec4 slopeThMain[]; };
layout(std430, binding = 19) writeonly buffer SlopeThPBuf { float slopeThP[]; };

float minmod(float a, float b) {
    if (a * b <= 0.0) return 0.0;
    return (a > 0.0) ? min(a, b) : max(a, b);
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= uint(uNr * uNth)) return;
    int i = int(idx) / uNth, j = int(idx) % uNth;
    int iM = max(i - 1, 0), iP = min(i + 1, uNr - 1);
    int jM = max(j - 1, 0), jP = min(j + 1, uNth - 1);

    vec4 cC = primMain[idx];
    vec4 cIm = primMain[cellIndex(iM, j)], cIp = primMain[cellIndex(iP, j)];
    vec4 cJm = primMain[cellIndex(i, jM)], cJp = primMain[cellIndex(i, jP)];
    float pC = primP[idx];
    float pIm = primP[cellIndex(iM, j)], pIp = primP[cellIndex(iP, j)];
    float pJm = primP[cellIndex(i, jM)], pJp = primP[cellIndex(i, jP)];

    slopeRMain[idx] = vec4(minmod(cC.x - cIm.x, cIp.x - cC.x), minmod(cC.y - cIm.y, cIp.y - cC.y),
                           minmod(cC.z - cIm.z, cIp.z - cC.z), minmod(cC.w - cIm.w, cIp.w - cC.w));
    slopeThMain[idx] = vec4(minmod(cC.x - cJm.x, cJp.x - cC.x), minmod(cC.y - cJm.y, cJp.y - cC.y),
                            minmod(cC.z - cJm.z, cJp.z - cC.z), minmod(cC.w - cJm.w, cJp.w - cC.w));
    slopeRP[idx] = minmod(pC - pIm, pIp - pC);
    slopeThP[idx] = minmod(pC - pJm, pJp - pC);
}
)";
}

// Pass: HLLE flux at each r-interface (fixed theta cell-center), for all
// (Nr+1)*Ntheta interfaces. Wave-speed bound: sound-speed-based
// Marti-Muller/Anile characteristic speed, but now needing a genuine
// local-orthonormal PROJECTION (not a simple per-axis scaling): since
// gamma_rphi!=0, r and phi are not orthogonal directions in KS's spatial
// metric, so the "physical" r-velocity component is
//   v_r_hat = sqrt(gamma_rr)*v^r + (gamma_rphi/sqrt(gamma_rr))*v^phi
// (a genuine Gram-Schmidt projection onto the unit vector along partial_r
// -- verified: v_r_hat^2 + [remaining (r,phi)-block magnitude] equals the
// FULL v^2 exactly, i.e. this decomposition is exact, not approximate).
// Marti-Muller's formula only ever uses vx and the total v^2 (never the
// transverse components individually), so the "transverse" speed can be
// any (vy,vz) with the right combined magnitude -- sqrt(v^2-v_r_hat^2) is
// used directly, theta's own contribution folded in with it (theta stays
// orthogonal to both r and phi, so this is exact too).
// Coordinate-speed conversion needs an extra -beta^r term BL never needed
// (beta^r=0 there): a wave at local speed lambda relative to the ZAMO
// observer has coordinate speed alpha*lambda/sqrt(gamma_rr) PLUS the ZAMO
// observer's OWN coordinate drift dr/dt|_ZAMO=-beta^r (from the ADM
// normal-observer 4-velocity n^mu=(1/alpha,-beta^i/alpha) -- once the
// shift is nonzero, "the ZAMO observer" is no longer "an observer at
// fixed r,theta", unlike BL where beta^r=0 made those coincide).
// Defensive clamp: the radial photon speed is now ASYMMETRIC (v_in !=
// v_out -- from g_rr*v^2+2*g_tr*v+g_tt=0, ds^2=0 with dtheta=dphi=0;
// unlike BL's symmetric +-sqrt(-g_tt/g_rr), since g_tr=0 there) -- exactly
// the point of a horizon-penetrating slicing: ingoing and outgoing radial
// photons have different coordinate speeds. Verified in
// tools/wave_speed_check.py (extended for KS): the tight bound stays
// safely inside this photon bound throughout the domain a real simulation
// uses (r > r_+ + 0.5M); only very close to the horizon does it need the
// clamp, which is exactly what the clamp is for.
inline std::string FluxesR() {
    return CommonHeader() + KerrMetric2DGlsl + SideStateGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 16) readonly buffer SlopeRMainBuf { vec4 slopeRMain[]; };
layout(std430, binding = 17) readonly buffer SlopeRPBuf { float slopeRP[]; };
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
    MetricBundle b = metricBundleAt(rFace, theta);
    // Radial photon speeds (asymmetric -- see this function's comment):
    // g_rr*v^2 + 2*g_tr*v + g_tt = 0.
    float photonDisc = max(b.g.gtr * b.g.gtr - b.g.grr * b.g.gtt, 0.0);
    float photonRoot = sqrt(photonDisc);
    float vPhotonOut = (-b.g.gtr + photonRoot) / b.g.grr;
    float vPhotonIn = (-b.g.gtr - photonRoot) / b.g.grr;

    int idxL = cellIndex(iL, j), idxR = cellIndex(iR, j);
    vec4 primML = primMain[idxL] + 0.5 * slopeRMain[idxL];
    vec4 primMR = primMain[idxR] - 0.5 * slopeRMain[idxR];
    primML.x = max(primML.x, 1e-12);
    primMR.x = max(primMR.x, 1e-12);
    float PL = max(primP[idxL] + 0.5 * slopeRP[idxL], 1e-12);
    float PR = max(primP[idxR] - 0.5 * slopeRP[idxR], 1e-12);

    vec4 UL, UR, FrL4, FrR4, FthDummyL, FthDummyR;
    float ULl, URl, FrLl, FrRl, FthDummyLl, FthDummyRl;
    sideState(primML, PL, rFace, theta, UL, ULl, FrL4, FrLl, FthDummyL, FthDummyLl);
    sideState(primMR, PR, rFace, theta, UR, URl, FrR4, FrRl, FthDummyR, FthDummyRl);

    float sqrtGammaRR = sqrt(b.g.grr);
    float hL = 1.0 + uGamma * PL / ((uGamma - 1.0) * primML.x);
    float hR = 1.0 + uGamma * PR / ((uGamma - 1.0) * primMR.x);
    float cs2L = soundSpeed2(primML.x, PL, hL);
    float cs2R = soundSpeed2(primMR.x, PR, hR);

    float v2L = b.g.grr*primML.y*primML.y + b.g.gthth*primML.z*primML.z + b.g.gphiphi*primML.w*primML.w
              + 2.0*b.g.grphi*primML.y*primML.w;
    float vRHatL = sqrtGammaRR*primML.y + (b.g.grphi/sqrtGammaRR)*primML.w;
    float vTransL = sqrt(max(v2L - vRHatL*vRHatL, 0.0));
    float v2R = b.g.grr*primMR.y*primMR.y + b.g.gthth*primMR.z*primMR.z + b.g.gphiphi*primMR.w*primMR.w
              + 2.0*b.g.grphi*primMR.y*primMR.w;
    float vRHatR = sqrtGammaRR*primMR.y + (b.g.grphi/sqrtGammaRR)*primMR.w;
    float vTransR = sqrt(max(v2R - vRHatR*vRHatR, 0.0));

    float lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    charSpeedsLocal(vRHatL, vTransL, 0.0, cs2L, lamMinusL, lamPlusL);
    charSpeedsLocal(vRHatR, vTransR, 0.0, cs2R, lamMinusR, lamPlusR);

    float sL = b.alpha * min(lamMinusL, lamMinusR) / sqrtGammaRR - b.betaUpR;
    float sR = b.alpha * max(lamPlusL, lamPlusR) / sqrtGammaRR - b.betaUpR;
    sL = max(sL, vPhotonIn);
    sR = min(sR, vPhotonOut);

    fluxR[idx] = (sR * FrL4 - sL * FrR4 + sL * sR * (UR - UL)) / (sR - sL);
    fluxRL[idx] = (sR * FrLl - sL * FrRl + sL * sR * (URl - ULl)) / (sR - sL);
}
)";
}

// Pass: HLLE flux at each theta-interface (fixed r cell-center), for all
// Nr*(Ntheta+1) interfaces. Same sound-speed-based bound as FluxesR, but
// simpler: theta stays orthogonal to both r and phi (beta^theta=0 always,
// no coordinate transform ever touches theta), so no tetrad projection or
// shift correction is needed -- identical in form to the old BL
// treatment. Photon bound also unaffected: g_t,theta=0 for BOTH BL and KS
// (the null cone with dr=dphi=0 doesn't see g_tr or g_rphi at all), so
// sqrt(-g_tt/g_thth) stays symmetric and correct as-is.
inline std::string FluxesTheta() {
    return CommonHeader() + KerrMetric2DGlsl + SideStateGlsl + R"(
layout(std430, binding = 4) readonly buffer PrimMainBuf { vec4 primMain[]; };
layout(std430, binding = 10) readonly buffer PrimPBuf { float primP[]; };
layout(std430, binding = 18) readonly buffer SlopeThMainBuf { vec4 slopeThMain[]; };
layout(std430, binding = 19) readonly buffer SlopeThPBuf { float slopeThP[]; };
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
    MetricBundle b = metricBundleAt(r, thFace);
    float fPhoton = sqrt(-b.g.gtt / b.g.gthth);

    int idxL = cellIndex(i, jL), idxR = cellIndex(i, jR);
    vec4 primML = primMain[idxL] + 0.5 * slopeThMain[idxL];
    vec4 primMR = primMain[idxR] - 0.5 * slopeThMain[idxR];
    primML.x = max(primML.x, 1e-12);
    primMR.x = max(primMR.x, 1e-12);
    float PL = max(primP[idxL] + 0.5 * slopeThP[idxL], 1e-12);
    float PR = max(primP[idxR] - 0.5 * slopeThP[idxR], 1e-12);

    vec4 UL, UR, FrDummyL, FrDummyR, FthL4, FthR4;
    float ULl, URl, FrDummyLl, FrDummyRl, FthLl, FthRl;
    sideState(primML, PL, r, thFace, UL, ULl, FrDummyL, FrDummyLl, FthL4, FthLl);
    sideState(primMR, PR, r, thFace, UR, URl, FrDummyR, FrDummyRl, FthR4, FthRl);

    float sqrtGammaThth = sqrt(b.g.gthth);
    float hL = 1.0 + uGamma * PL / ((uGamma - 1.0) * primML.x);
    float hR = 1.0 + uGamma * PR / ((uGamma - 1.0) * primMR.x);
    float cs2L = soundSpeed2(primML.x, PL, hL);
    float cs2R = soundSpeed2(primMR.x, PR, hR);

    float v2L = b.g.grr*primML.y*primML.y + b.g.gthth*primML.z*primML.z + b.g.gphiphi*primML.w*primML.w
              + 2.0*b.g.grphi*primML.y*primML.w;
    float vThHatL = sqrtGammaThth * primML.z;
    float vTransL = sqrt(max(v2L - vThHatL*vThHatL, 0.0));
    float v2R = b.g.grr*primMR.y*primMR.y + b.g.gthth*primMR.z*primMR.z + b.g.gphiphi*primMR.w*primMR.w
              + 2.0*b.g.grphi*primMR.y*primMR.w;
    float vThHatR = sqrtGammaThth * primMR.z;
    float vTransR = sqrt(max(v2R - vThHatR*vThHatR, 0.0));

    float lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    charSpeedsLocal(vThHatL, vTransL, 0.0, cs2L, lamMinusL, lamPlusL);
    charSpeedsLocal(vThHatR, vTransR, 0.0, cs2R, lamMinusR, lamPlusR);

    float sL = b.alpha * min(lamMinusL, lamMinusR) / sqrtGammaThth;
    float sR = b.alpha * max(lamPlusL, lamPlusR) / sqrtGammaThth;
    sL = max(sL, -fPhoton);
    sR = min(sR, fPhoton);

    int outIdx = i * (uNth + 1) + j;
    fluxTh[outIdx] = (sR * FthL4 - sL * FthR4 + sL * sR * (UR - UL)) / (sR - sL);
    fluxThL[outIdx] = (sR * FthLl - sL * FthRl + sL * sR * (URl - ULl)) / (sR - sL);
}
)";
}

// Euler predictor step: 2D flux divergence (r and theta) plus the two
// geometric source terms (Sr, Stheta -- D, L, tau are exactly
// source-free: t and phi both still Killing for KS, see this file's
// header comment). Same general identity as BL, extended to KS's 7
// nonzero metric components (2 more than BL's 5: g_tr, g_rphi).
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

// Full inverse 4-metric from the ADM pieces (standard identities, valid
// for ANY lapse/shift/spatial-metric): g^tt=-1/alpha^2, g^ti=beta^i/
// alpha^2, g^ij=gamma^ij-beta^i beta^j/alpha^2.
void inverse4Metric(MetricBundle b, out float gUtt, out float gUtr, out float gUtphi,
                     out float gUrr, out float gUrphi, out float gUphiphi, out float gUthth) {
    float alpha2 = b.alpha * b.alpha;
    gUtt = -1.0 / alpha2;
    gUtr = b.betaUpR / alpha2;
    gUtphi = 0.0; // beta^phi=0
    gUrr = b.gammaUpRr - b.betaUpR * b.betaUpR / alpha2;
    gUrphi = b.gammaUpRphi; // beta^phi=0, no correction
    gUphiphi = b.gammaUpPhiphi;
    gUthth = b.gammaUpThth;
}

void sourceTerms(vec4 primM, float P, float r, float theta, out float srcSr, out float srcSth) {
    float rho = primM.x, vr = primM.y, vth = primM.z, vphi = primM.w;
    float W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E;
    kinematics2D(rho, vr, vth, vphi, P, r, theta, W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E);

    MetricBundle b = metricBundleAt(r, theta);
    float gUtt, gUtr, gUtphi, gUrr, gUrphi, gUphiphi, gUthth;
    inverse4Metric(b, gUtt, gUtr, gUtphi, gUrr, gUrphi, gUphiphi, gUthth);

    float Ttt = rho*h*ut*ut + P*gUtt;
    float Ttr = rho*h*ut*ur + P*gUtr;
    float Ttphi = rho*h*ut*uPhiContra + P*gUtphi;
    float Trr = rho*h*ur*ur + P*gUrr;
    float Trphi = rho*h*ur*uPhiContra + P*gUrphi;
    float Tphiphi = rho*h*uPhiContra*uPhiContra + P*gUphiphi;
    float Tthth = rho*h*uth*uth + P*gUthth;

    float epsR = max(1e-5*r, 1e-5);
    Metric mrp = metricAt(r+epsR, theta); Metric mrm = metricAt(r-epsR, theta);
    float dgtt_r = (mrp.gtt-mrm.gtt)/(2.0*epsR);
    float dgtr_r = (mrp.gtr-mrm.gtr)/(2.0*epsR);
    float dgtphi_r = (mrp.gtphi-mrm.gtphi)/(2.0*epsR);
    float dgrr_r = (mrp.grr-mrm.grr)/(2.0*epsR);
    float dgrphi_r = (mrp.grphi-mrm.grphi)/(2.0*epsR);
    float dgphiphi_r = (mrp.gphiphi-mrm.gphiphi)/(2.0*epsR);
    float dgthth_r = (mrp.gthth-mrm.gthth)/(2.0*epsR);
    srcSr = sqrtNegG(r,theta) * 0.5*(Ttt*dgtt_r + 2.0*Ttr*dgtr_r + 2.0*Ttphi*dgtphi_r
                                     + Trr*dgrr_r + 2.0*Trphi*dgrphi_r + Tphiphi*dgphiphi_r + Tthth*dgthth_r);

    float epsT = 1e-5;
    Metric mtp = metricAt(r, theta+epsT); Metric mtm = metricAt(r, theta-epsT);
    float dgtt_t = (mtp.gtt-mtm.gtt)/(2.0*epsT);
    float dgtr_t = (mtp.gtr-mtm.gtr)/(2.0*epsT);
    float dgtphi_t = (mtp.gtphi-mtm.gtphi)/(2.0*epsT);
    float dgrr_t = (mtp.grr-mtm.grr)/(2.0*epsT);
    float dgrphi_t = (mtp.grphi-mtm.grphi)/(2.0*epsT);
    float dgphiphi_t = (mtp.gphiphi-mtm.gphiphi)/(2.0*epsT);
    float dgthth_t = (mtp.gthth-mtm.gthth)/(2.0*epsT);
    srcSth = sqrtNegG(r,theta) * 0.5*(Ttt*dgtt_t + 2.0*Ttr*dgtr_t + 2.0*Ttphi*dgtphi_t
                                      + Trr*dgrr_t + 2.0*Trphi*dgrphi_t + Tphiphi*dgphiphi_t + Tthth*dgthth_t);
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
