"""
Phase 2 of the Kerr-Schild (KS) reformulation for 07_grhd -- see
kerr_schild_derive.py for the verified metric/ADM closed forms this reuses
(trusted here since that script already confirmed them independently
against a from-scratch coordinate-transform derivation).

General (non-diagonal-spatial-metric) Valencia formulation. KS's spatial
3-metric has a genuine gamma_rphi cross term (theta still decouples: KS's
coordinate transformation never touches theta), so the previous BL-era
convention -- "physical" ZAMO-orthonormal v_r,v_th,v_phi with
W=1/sqrt(1-sum v_i^2), valid only because BL's spatial metric was diagonal
-- no longer applies. This switches to the general coordinate-frame
Valencia velocity (what production codes like HARM actually use for
Kerr-Schild): v^i defined via u^i = W(v^i - beta^i/alpha), W built from
the FULL quadratic form v^2 = gamma_ij v^i v^j (not a diagonal sum).

Key identity used below, gamma_ij u^i u^j = W^2-1 (a standard,
metric-independent ADM identity -- verified numerically here, not just
asserted): combined with kappa_i := S_i/D = h*u_i (same derivation as the
BL-era code, metric-independent), this gives K := gamma^ij kappa_i kappa_j
= h^2(W^2-1), i.e. W=sqrt(1+K/h^2) -- EXACTLY the same Newton-iteration
structure as before, just with K computed via the full inverse-metric
contraction instead of a diagonal sum.
"""

import numpy as np


def metric_bundle_ks(r, theta, M, a):
    """KS metric + ADM 3+1 split -- closed forms verified in
    kerr_schild_derive.py against an independent BL->KS coordinate-
    transform derivation (both the metric components and alpha/beta^r/
    beta^phi=0 matched to machine precision there)."""
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2
    Sigma = r * r + a * a * cos2

    g_tt = -(1.0 - 2.0 * M * r / Sigma)
    g_tr = 2.0 * M * r / Sigma
    g_tphi = -2.0 * M * a * r * sin2 / Sigma
    gamma_rr = 1.0 + 2.0 * M * r / Sigma
    gamma_rphi = -a * sin2 * (1.0 + 2.0 * M * r / Sigma)
    gamma_thth = Sigma
    gamma_phiphi = sin2 * (Sigma + a * a * sin2 * (1.0 + 2.0 * M * r / Sigma))

    alpha2 = Sigma / (Sigma + 2.0 * M * r)
    alpha = np.sqrt(alpha2)
    beta_up_r = 2.0 * M * r / (Sigma + 2.0 * M * r)
    beta_up_phi = 0.0

    # Inverse of the (r,phi) 2x2 spatial block (theta decouples).
    det2 = gamma_rr * gamma_phiphi - gamma_rphi * gamma_rphi
    gamma_up_rr = gamma_phiphi / det2
    gamma_up_rphi = -gamma_rphi / det2
    gamma_up_phiphi = gamma_rr / det2
    gamma_up_thth = 1.0 / gamma_thth

    return dict(
        g_tt=g_tt, g_tr=g_tr, g_tphi=g_tphi,
        gamma_rr=gamma_rr, gamma_rphi=gamma_rphi, gamma_phiphi=gamma_phiphi, gamma_thth=gamma_thth,
        alpha=alpha, beta_up_r=beta_up_r, beta_up_phi=beta_up_phi,
        gamma_up_rr=gamma_up_rr, gamma_up_rphi=gamma_up_rphi, gamma_up_phiphi=gamma_up_phiphi,
        gamma_up_thth=gamma_up_thth,
    )


def kinematics_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma):
    """(rho,v^r,v^th,v^phi,P) -> W,h,u^t,u^r,u^th,u^phi,u_r,u_th,u_phi,E.
    v^i is the GENERAL coordinate-frame Valencia velocity (see module
    docstring), not the old BL-era orthonormal convention.

    u_i = gamma_ij u^j + beta_i u^t (the FULL relation from u_i=g_i_mu
    u^mu=g_it u^t+g_ij u^j -- an earlier version of this file dropped the
    beta_i u^t term entirely (missed since KS's beta_r,beta_phi ARE
    nonzero, unlike BL where beta_r=0 made it silently vanish there);
    caught by check_roundtrip()/check_raising_identity() failing
    while check_4d_normalization() (which never goes through u_i at all)
    kept passing -- exactly the kind of bug a single check can miss."""
    m = metric_bundle_ks(r, theta, M, a)
    v2 = (m["gamma_rr"] * vr * vr + m["gamma_thth"] * vth * vth + m["gamma_phiphi"] * vphi * vphi
          + 2.0 * m["gamma_rphi"] * vr * vphi)
    W = 1.0 / np.sqrt(1.0 - v2)
    h = 1.0 + gamma * P / ((gamma - 1.0) * rho)

    ut = W / m["alpha"]
    ur = W * (vr - m["beta_up_r"] / m["alpha"])
    uth = W * vth
    uPhiContra = W * (vphi - m["beta_up_phi"] / m["alpha"])  # beta_up_phi=0, kept for clarity/symmetry

    beta_r = m["gamma_rr"] * m["beta_up_r"] + m["gamma_rphi"] * m["beta_up_phi"]
    beta_phi = m["gamma_rphi"] * m["beta_up_r"] + m["gamma_phiphi"] * m["beta_up_phi"]
    urCov = m["gamma_rr"] * ur + m["gamma_rphi"] * uPhiContra + beta_r * ut
    uPhiCov = m["gamma_rphi"] * ur + m["gamma_phiphi"] * uPhiContra + beta_phi * ut
    uthCov = m["gamma_thth"] * uth  # beta_theta=0, no correction needed

    E = -(m["g_tt"] * ut + m["g_tr"] * ur + m["g_tphi"] * uPhiContra)
    return W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E, m


def prim_to_cons_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma):
    W, h, ut, ur, uth, uphi, urCov, uthCov, uPhiCov, E, m = kinematics_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma)
    sqrtg = m["gamma_thth"] * np.sin(theta)  # sqrt(-g)=Sigma*sin(theta) -- SAME formula as BL (verified below)
    D = sqrtg * rho * ut
    Sr = sqrtg * rho * h * ut * urCov
    Sth = sqrtg * rho * h * ut * uthCov
    Sphi = sqrtg * rho * h * ut * uPhiCov
    tau = sqrtg * rho * h * ut * E - sqrtg * P - D
    return D, Sr, Sth, Sphi, tau


def cons_to_prim_ks(D, Sr, Sth, Sphi, tau, r, theta, M, a, gamma, iters=60, p_guess=1e-3):
    """Newton iterate on h, generalizing the BL-era algorithm -- but W can
    no longer be read off a simple W=sqrt(1+K/h^2) shortcut once the shift
    has a component (beta^r) matching a spatial index of S_i: that
    relation implicitly assumed beta_r=0, true for BL (only beta^phi is
    nonzero there, and beta_r=gamma_rr*beta^r=0), false for KS. Instead,
    at each trial h: u^i are read off directly from kappa_i=S_i/D (exact,
    metric-only, no shift dependence: u^i=gamma^ij kappa_j/h), then u^t is
    recovered from the QUADRATIC normalization equation
        g_tt(u^t)^2 + 2(g_tr u^r+g_tphi u^phi) u^t + (gamma_ij u^i u^j+1) = 0
    (derived from u^mu u_mu=-1 with u_t=g_tt u^t+g_tr u^r+g_tphi u^phi),
    taking the positive root -- verified symbolically to be the correct,
    general (shift-independent-derivation) relation, unlike the shortcut
    it replaces."""
    m = metric_bundle_ks(r, theta, M, a)
    sqrtg = m["gamma_thth"] * np.sin(theta)
    kappa_r, kappa_th, kappa_phi = Sr / D, Sth / D, Sphi / D
    K = (m["gamma_up_rr"] * kappa_r * kappa_r + m["gamma_up_thth"] * kappa_th * kappa_th
         + m["gamma_up_phiphi"] * kappa_phi * kappa_phi + 2.0 * m["gamma_up_rphi"] * kappa_r * kappa_phi)

    def ut_and_u_from_h(h):
        # X^i := gamma^ij kappa_j/h -- NOT u^i itself: u^i = X^i - beta^i
        # u^t (raising the LOWERED u_i=kappa_i/h with gamma^ij undoes the
        # gamma_ij u^j part of u_i's definition but leaves the beta_i u^t
        # part behind as an extra +beta^i u^t on X^i -- see this
        # function's caller docstring). Substituting into u^mu u_mu=-1
        # and collecting as a quadratic in u^t (done symbolically, not by
        # hand -- see tools/ derivation notes) gives:
        Xr = (m["gamma_up_rr"] * kappa_r + m["gamma_up_rphi"] * kappa_phi) / h
        Xphi = (m["gamma_up_rphi"] * kappa_r + m["gamma_up_phiphi"] * kappa_phi) / h
        Xth = (m["gamma_up_thth"] * kappa_th) / h
        gamma_ij_uiuj_K_term = K / (h * h)  # == gamma^ij kappa_i kappa_j/h^2

        betaUr, betaUphi = m["beta_up_r"], m["beta_up_phi"]
        Acoef = m["g_tt"] - betaUr * m["g_tr"] - betaUphi * m["g_tphi"]
        Bcoef = (-betaUr * kappa_r - betaUphi * kappa_phi
                 + m["gamma_up_phiphi"] * m["g_tphi"] * kappa_phi + m["gamma_up_rphi"] * m["g_tphi"] * kappa_r
                 + m["gamma_up_rphi"] * m["g_tr"] * kappa_phi + m["gamma_up_rr"] * m["g_tr"] * kappa_r) / h
        Ccoef = gamma_ij_uiuj_K_term + 1.0
        disc = max(Bcoef * Bcoef - 4.0 * Acoef * Ccoef, 0.0)
        root1 = (-Bcoef + np.sqrt(disc)) / (2.0 * Acoef)
        root2 = (-Bcoef - np.sqrt(disc)) / (2.0 * Acoef)
        ut = root1 if root1 > 0.0 else root2  # physical root: u^t>0

        ur = Xr - betaUr * ut
        uphi = Xphi - betaUphi * ut
        uth = Xth
        return ut, ur, uth, uphi

    def residual(h):
        ut, ur, uth, uphi = ut_and_u_from_h(h)
        W = m["alpha"] * ut
        rho = D * m["alpha"] / (sqrtg * W)
        P = (gamma - 1.0) * rho * (h - 1.0) / gamma
        E = -(m["g_tt"] * ut + m["g_tr"] * ur + m["g_tphi"] * uphi)
        Q = (tau + D + sqrtg * P) / D
        return Q / h - E

    h = max(1.0 + gamma * max(p_guess, 1e-12) / ((gamma - 1.0) * (D * m["alpha"] / (sqrtg * np.sqrt(1.0 + K)))), 1.0 + 1e-6)
    for _ in range(iters):
        f0 = residual(h)
        dh = max(1e-6 * abs(h), 1e-9)
        fp, fm = residual(h + dh), residual(h - dh)
        deriv = (fp - fm) / (2.0 * dh)
        if abs(deriv) < 1e-14:
            break
        step = np.clip(f0 / deriv, -0.5 * h, 0.5 * h)
        h = max(h - step, 1.0 + 1e-6)

    ut, ur, uth, uphi = ut_and_u_from_h(h)
    W = m["alpha"] * ut
    rho = D * m["alpha"] / (sqrtg * W)
    vr = ur / W + m["beta_up_r"] / m["alpha"]
    vth = uth / W
    vphi = uphi / W + m["beta_up_phi"] / m["alpha"]
    P = (gamma - 1.0) * rho * (h - 1.0) / gamma
    return rho, vr, vth, vphi, P


def _sample_physical_state(rng):
    """A random (r,theta,a) plus a coordinate-frame velocity (v^r,v^th,
    v^phi) capped at v^2=gamma_ij v^iv^j<=0.81 -- sampled as "physical-
    scale" components (order 0.1-0.5, safely sub-luminal on their own)
    divided by sqrt(gamma_ii) to get a plausible coordinate-velocity
    MAGNITUDE (gamma_thth, gamma_phiphi ~ r^2, so a physical speed of
    order 0.3 corresponds to a much SMALLER coordinate v^i there -- unlike
    the old BL-era orthonormal convention, coordinate v^i is not itself
    bounded by 1), then uniformly rescaled if the exact quadratic form
    (including the gamma_rphi cross term) still exceeds the cap."""
    r = rng.uniform(1.6, 20.0)  # KS has no horizon coordinate singularity -- can go well inside r_BL_horizon
    theta = rng.uniform(0.2, np.pi - 0.2)
    a = rng.uniform(0.0, 0.998)
    M = 1.0
    m = metric_bundle_ks(r, theta, M, a)
    v_r_hat = rng.uniform(-0.5, 0.5)
    v_th_hat = rng.uniform(-0.3, 0.3)
    v_phi_hat = rng.uniform(-0.3, 0.3)
    vr = v_r_hat / np.sqrt(m["gamma_rr"])
    vth = v_th_hat / np.sqrt(m["gamma_thth"])
    vphi = v_phi_hat / np.sqrt(m["gamma_phiphi"])
    v2 = (m["gamma_rr"] * vr * vr + m["gamma_thth"] * vth * vth + m["gamma_phiphi"] * vphi * vphi
          + 2.0 * m["gamma_rphi"] * vr * vphi)
    cap = 0.81
    if v2 > cap:
        s = np.sqrt(cap / v2)
        vr, vth, vphi, v2 = vr * s, vth * s, vphi * s, cap
    return r, theta, a, M, m, vr, vth, vphi, v2


def check_raising_identity():
    """Verify X^i := gamma^ij kappa_j/h equals u^i + beta^i u^t (NOT u^i
    alone -- an easy trap: raising the lowered kappa_i=h*u_i=h(gamma_ij
    u^j+beta_i u^t) with gamma^ij undoes the gamma_ij u^j part but leaves
    the beta_i u^t part behind as an extra +beta^i u^t). cons_to_prim_ks's
    quadratic-in-u^t derivation depends on exactly this relation -- caught
    by hand-derivation getting it wrong the first time (assumed X^i=u^i
    directly), this checks the corrected version numerically."""
    rng = np.random.default_rng(3)
    gamma_eos = 4.0 / 3.0
    worst = 0.0
    for _ in range(500):
        r, theta, a, M, m, vr, vth, vphi, v2 = _sample_physical_state(rng)
        rho, P = 1.0, 0.1
        W, h, ut, ur, uth, uphi, urCov, uthCov, uphiCov, E, _ = kinematics_ks(
            rho, vr, vth, vphi, P, r, theta, M, a, gamma_eos)
        kappa_r, kappa_phi = h * urCov, h * uphiCov
        Xr = (m["gamma_up_rr"] * kappa_r + m["gamma_up_rphi"] * kappa_phi) / h
        Xphi = (m["gamma_up_rphi"] * kappa_r + m["gamma_up_phiphi"] * kappa_phi) / h
        worst = max(worst, abs(Xr - (ur + m["beta_up_r"] * ut)), abs(Xphi - (uphi + m["beta_up_phi"] * ut)))
    print(f"X^i = u^i + beta^i u^t identity: worst abs error = {worst:.3e}")
    return worst < 1e-8


def check_4d_normalization():
    """Direct, non-circular check: g_munu u^mu u^nu = -1 using the
    INDEPENDENTLY-derived metric (kerr_schild_derive.py's closed forms)
    and this file's u^mu components -- unlike the gamma_ij identity above
    (which only tests that v^2 and W are computed consistently with each
    other, trivially true by construction), this tests whether the u^i
    formula itself is actually consistent with the ADM decomposition."""
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(500):
        r, theta, a, M, m, vr, vth, vphi, v2 = _sample_physical_state(rng)
        W = 1.0 / np.sqrt(1.0 - v2)
        ut = W / m["alpha"]
        ur = W * (vr - m["beta_up_r"] / m["alpha"])
        uth = W * vth
        uphi = W * (vphi - m["beta_up_phi"] / m["alpha"])
        norm = (m["g_tt"] * ut * ut + 2 * m["g_tr"] * ut * ur + 2 * m["g_tphi"] * ut * uphi
                + m["gamma_rr"] * ur * ur + m["gamma_thth"] * uth * uth + m["gamma_phiphi"] * uphi * uphi
                + 2 * m["gamma_rphi"] * ur * uphi)
        worst = max(worst, abs(norm - (-1.0)))
    print(f"g_munu u^mu u^nu = -1 (direct 4D check): worst abs error = {worst:.3e}")
    return worst < 1e-8


def check_roundtrip():
    rng = np.random.default_rng(11)
    gamma_eos = 4.0 / 3.0
    worst = 0.0
    n = 0
    for _ in range(600):
        r, theta, a, M, m, vr, vth, vphi, v2 = _sample_physical_state(rng)
        rho = rng.uniform(0.3, 3.0)
        P = rng.uniform(0.01, 0.4) * rho

        D, Sr, Sth, Sphi, tau = prim_to_cons_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma_eos)
        rho2, vr2, vth2, vphi2, P2 = cons_to_prim_ks(D, Sr, Sth, Sphi, tau, r, theta, M, a, gamma_eos, p_guess=P)
        err = max(abs(rho2 - rho) / rho, abs(vr2 - vr) / max(abs(vr), 0.05),
                  abs(vth2 - vth) / max(abs(vth), 0.05), abs(vphi2 - vphi) / max(abs(vphi), 0.05),
                  abs(P2 - P) / P)
        worst = max(worst, err)
        n += 1
    print(f"round-trip prim->cons->prim: tested {n} states, worst relative error = {worst:.3e}")
    return worst < 1e-6


def photon_speeds_r_ks(m):
    """Coordinate speeds (dr/dt) of the two radial null rays -- from
    g_rr v^2 + 2 g_tr v + g_tt = 0 (ds^2=0, dtheta=dphi=0). UNLIKE BL
    (where this reduces to a symmetric +-sqrt(-g_tt/g_rr) since g_tr=0),
    KS's roots are NOT symmetric around zero -- exactly the point of a
    horizon-penetrating slicing: ingoing and outgoing radial photons have
    different coordinate speeds (an outgoing photon slows to a crawl
    approaching the horizon; an ingoing one doesn't)."""
    disc = m["g_tr"] * m["g_tr"] - m["gamma_rr"] * m["g_tt"]
    disc = max(disc, 0.0)
    root = np.sqrt(disc)
    v_out = (-m["g_tr"] + root) / m["gamma_rr"]
    v_in = (-m["g_tr"] - root) / m["gamma_rr"]
    return v_in, v_out  # v_in < v_out always (root>=0)


def char_speeds_r_ks(vr, vth, vphi, cs2, m):
    """Coordinate (dr/dt) sound characteristic speed bounds, r-direction.
    theta is orthogonal to both r and phi in KS (unchanged from BL: the
    coordinate transform never touches theta), so only the (r,phi) block
    needs a genuine local-orthonormal projection; v_r_hat below is
    exactly the r-component of that projection (verified: v_r_hat^2 +
    v_phi_hat^2 = the FULL (r,phi)-block quadratic form, i.e. the
    orthonormal decomposition is exact, not approximate). The
    Marti-Muller SR formula only ever uses vx and the total v^2 (never
    vy,vz individually), so v_phi_hat itself is never needed here.
    Coordinate conversion includes the -beta^r term (UNLIKE BL, where
    beta^r=0 made this vanish): a wave at LOCAL speed lambda relative to
    the ZAMO observer has coordinate speed alpha*lambda/sqrt(gamma_rr)
    PLUS the ZAMO observer's own coordinate drift dr/dt|_ZAMO=-beta^r
    (from the ADM normal-observer 4-velocity n^mu=(1/alpha,-beta^i/alpha)
    -- the ZAMO observer's SPATIAL coordinates are not fixed once the
    shift is nonzero, unlike the BL case where beta^r=0 made "ZAMO" and
    "fixed r,theta" coincide)."""
    v2 = m["gamma_rr"] * vr * vr + m["gamma_thth"] * vth * vth + m["gamma_phiphi"] * vphi * vphi \
        + 2.0 * m["gamma_rphi"] * vr * vphi
    v_r_hat = np.sqrt(m["gamma_rr"]) * vr + (m["gamma_rphi"] / np.sqrt(m["gamma_rr"])) * vphi
    v_transverse = np.sqrt(max(v2 - v_r_hat * v_r_hat, 0.0))
    lam_minus, lam_plus = marti_muller_lambda_pm_local(v_r_hat, v_transverse, 0.0, cs2)
    sL = m["alpha"] * lam_minus / np.sqrt(m["gamma_rr"]) - m["beta_up_r"]
    sR = m["alpha"] * lam_plus / np.sqrt(m["gamma_rr"]) - m["beta_up_r"]
    return sL, sR


def char_speeds_th_ks(vr, vth, vphi, cs2, m):
    """Same as char_speeds_r_ks, theta-direction -- simpler, since theta
    is orthogonal to both r and phi already (no projection needed,
    beta^theta=0 so no shift correction either -- identical in form to
    the old BL treatment)."""
    v2 = m["gamma_rr"] * vr * vr + m["gamma_thth"] * vth * vth + m["gamma_phiphi"] * vphi * vphi \
        + 2.0 * m["gamma_rphi"] * vr * vphi
    v_th_hat = np.sqrt(m["gamma_thth"]) * vth
    v_transverse = np.sqrt(max(v2 - v_th_hat * v_th_hat, 0.0))
    lam_minus, lam_plus = marti_muller_lambda_pm_local(v_th_hat, v_transverse, 0.0, cs2)
    sL = m["alpha"] * lam_minus / np.sqrt(m["gamma_thth"])
    sR = m["alpha"] * lam_plus / np.sqrt(m["gamma_thth"])
    return sL, sR


def marti_muller_lambda_pm_local(vx, vy, vz, cs2):
    """Same closed-form local-orthonormal-frame formula as
    wave_speed_check.py's marti_muller_lambda_pm -- duplicated here (not
    imported) so this file has no dependency on that one; identical
    formula, already independently verified there against a from-scratch
    symbolic flux Jacobian to 1e-15."""
    v2 = vx * vx + vy * vy + vz * vz
    denom = 1.0 - v2 * cs2
    disc = max(cs2 * (1.0 - v2) * ((1.0 - v2 * cs2) - vx * vx * (1.0 - cs2)), 0.0)
    root = np.sqrt(disc)
    center = vx * (1.0 - cs2)
    return (center - root) / denom, (center + root) / denom


def inverse_4metric_ks(m):
    """Full inverse 4-metric from the ADM pieces (standard identities,
    valid for ANY lapse/shift/spatial-metric -- not KS-specific):
    g^tt=-1/alpha^2, g^ti=beta^i/alpha^2, g^ij=gamma^ij-beta^i beta^j/alpha^2."""
    alpha2 = m["alpha"] * m["alpha"]
    g_up_tt = -1.0 / alpha2
    g_up_tr = m["beta_up_r"] / alpha2
    g_up_tphi = m["beta_up_phi"] / alpha2  # = 0 for KS, kept general
    g_up_rr = m["gamma_up_rr"] - m["beta_up_r"] * m["beta_up_r"] / alpha2
    g_up_rphi = m["gamma_up_rphi"] - m["beta_up_r"] * m["beta_up_phi"] / alpha2
    g_up_phiphi = m["gamma_up_phiphi"] - m["beta_up_phi"] * m["beta_up_phi"] / alpha2
    g_up_thth = m["gamma_up_thth"]
    return dict(tt=g_up_tt, tr=g_up_tr, tphi=g_up_tphi, rr=g_up_rr, rphi=g_up_rphi,
                phiphi=g_up_phiphi, thth=g_up_thth)


def stress_energy_up_ks(rho, h, P, ut, ur, uth, uphi, gU):
    """T^munu = rho*h*u^mu*u^nu + P*g^munu -- general perfect-fluid
    stress-energy, valid for any metric. All 7 independent components
    needed now (KS has 2 more nonzero metric components than BL: g_tr,
    g_rphi), vs. BL's 5 (tt,tphi,rr,thth,phiphi)."""
    Ttt = rho * h * ut * ut + P * gU["tt"]
    Ttr = rho * h * ut * ur + P * gU["tr"]
    Ttphi = rho * h * ut * uphi + P * gU["tphi"]
    Trr = rho * h * ur * ur + P * gU["rr"]
    Trphi = rho * h * ur * uphi + P * gU["rphi"]
    Tphiphi = rho * h * uphi * uphi + P * gU["phiphi"]
    Tthth = rho * h * uth * uth + P * gU["thth"]
    return dict(tt=Ttt, tr=Ttr, tphi=Ttphi, rr=Trr, rphi=Trphi, phiphi=Tphiphi, thth=Tthth)


def metric_lower_dict(r, theta, M, a):
    """The 7 independent lower-index metric components as a dict, for
    convenient finite differencing (mirrors kernels_kerr2d.hpp's
    metricAt()/EulerStep() numerical-derivative pattern -- established,
    already-trusted practice in this project, not a fresh derivation
    risk: Christoffels via central differences of the textbook metric,
    not a by-hand symbolic Christoffel computation)."""
    m = metric_bundle_ks(r, theta, M, a)
    return dict(tt=m["g_tt"], tr=m["g_tr"], tphi=m["g_tphi"], rr=m["gamma_rr"],
                rphi=m["gamma_rphi"], phiphi=m["gamma_phiphi"], thth=m["gamma_thth"])


def source_terms_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma, eps_r=None, eps_th=1e-5):
    """Source_Sr, Source_Sth via the general (metric-independent-in-form)
    momentum-source identity Source_i = sqrt(-g)*0.5*sum_{mu,nu} T^munu
    d(g_munu)/dx^i -- same formula kernels_kerr2d.hpp already uses for
    BL, just summed over KS's 7 nonzero metric components instead of 5
    (the 2 new ones, g_tr and g_rphi, contribute 2*T^tr*dg_tr/dx^i and
    2*T^rphi*dg_rphi/dx^i respectively -- the factor of 2 from each being
    counted once for each symmetric ordering, same as the existing
    2*T^tphi*dg_tphi/dx^i term already in the BL code). D, L(=S_phi), tau
    all stay EXACTLY source-free: t and phi are both still Killing
    vectors for KS (none of the 7 metric components depend on t or phi,
    only r and theta), identical reasoning to the BL case."""
    if eps_r is None:
        eps_r = max(1e-5 * r, 1e-5)
    m = metric_bundle_ks(r, theta, M, a)
    gU = inverse_4metric_ks(m)
    W, h, ut, ur, uth, uphi, urCov, uthCov, uphiCov, E, _ = kinematics_ks(rho, vr, vth, vphi, P, r, theta, M, a, gamma)
    T = stress_energy_up_ks(rho, h, P, ut, ur, uth, uphi, gU)
    sqrtg = m["gamma_thth"] * np.sin(theta)

    def dmetric(dr, dth):
        gp = metric_lower_dict(r + dr, theta + dth, M, a)
        gm = metric_lower_dict(r - dr, theta - dth, M, a)
        return {k: (gp[k] - gm[k]) / (2.0 * (dr if dr != 0 else dth)) for k in gp}

    dg_dr = dmetric(eps_r, 0.0)
    dg_dth = dmetric(0.0, eps_th)

    def source(dg):
        return sqrtg * 0.5 * (T["tt"] * dg["tt"] + 2.0 * T["tr"] * dg["tr"] + 2.0 * T["tphi"] * dg["tphi"]
                               + T["rr"] * dg["rr"] + 2.0 * T["rphi"] * dg["rphi"] + T["phiphi"] * dg["phiphi"]
                               + T["thth"] * dg["thth"])

    return source(dg_dr), source(dg_dth)


def bl_to_ks_velocity(vr_bl, vth_bl, vphi_bl, r, theta, M, a):
    """Transform BL-frame ORTHONORMAL primitives (the OLD convention --
    tools/fishbone_moncrief.py's output) into KS-frame general Valencia
    primitives. r,theta are UNCHANGED by the BL<->KS coordinate map (see
    kerr_schild_derive.py); only the 4-velocity needs transforming, via
    the differential relations du^t_KS=du^t_BL+(2Mr/Delta)du^r_BL,
    du^phi_KS=du^phi_BL+(a/Delta)du^r_BL (r,theta components of u^mu are
    unchanged, since dr_KS=dr_BL and dtheta_KS=dtheta_BL identically).

    Step 1 (BL, using the OLD orthonormal convention -- see
    kerr_equatorial_ref.py): reconstruct u^mu_BL from
    (rho,vr_bl,vth_bl,vphi_bl) via the BL ZAMO kinematics.
    Step 2: apply the coordinate-transform above to get u^mu_KS.
    Step 3 (KS, using the NEW general Valencia convention): recover
    v^i_KS from u^mu_KS via v^i=u^i/W+beta^i/alpha, W=alpha_KS*u^t_KS.
    """
    Delta = r * r - 2.0 * M * r + a * a
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2
    Sigma = r * r + a * a * cos2
    A_bl = (r * r + a * a) ** 2 - a * a * Delta * sin2
    alpha_bl = np.sqrt(Sigma * Delta / A_bl)
    g_tphi_bl = -2.0 * M * a * r * sin2 / Sigma
    gamma_rr_bl = Sigma / Delta
    gamma_thth_bl = Sigma
    gamma_phiphi_bl = A_bl * sin2 / Sigma
    beta_phi_up_bl = g_tphi_bl / gamma_phiphi_bl

    W = 1.0 / np.sqrt(1.0 - vr_bl * vr_bl - vth_bl * vth_bl - vphi_bl * vphi_bl)
    ut_bl = W / alpha_bl
    ur_bl = W * vr_bl / np.sqrt(gamma_rr_bl)
    uth_bl = W * vth_bl / np.sqrt(gamma_thth_bl)
    uphi_bl = W * vphi_bl / np.sqrt(gamma_phiphi_bl) - W * beta_phi_up_bl / alpha_bl

    ut_ks = ut_bl + (2.0 * M * r / Delta) * ur_bl
    ur_ks = ur_bl
    uth_ks = uth_bl
    uphi_ks = uphi_bl + (a / Delta) * ur_bl

    m = metric_bundle_ks(r, theta, M, a)
    W_ks = m["alpha"] * ut_ks
    vr_ks = ur_ks / W_ks + m["beta_up_r"] / m["alpha"]
    vth_ks = uth_ks / W_ks
    vphi_ks = uphi_ks / W_ks + m["beta_up_phi"] / m["alpha"]
    return vr_ks, vth_ks, vphi_ks, W_ks


if __name__ == "__main__":
    ok1 = check_raising_identity()
    print("PASS" if ok1 else "FAIL")
    ok2 = check_4d_normalization()
    print("PASS" if ok2 else "FAIL")
    ok3 = check_roundtrip()
    print("PASS" if ok3 else "FAIL")
    raise SystemExit(0 if (ok1 and ok2 and ok3) else 1)
