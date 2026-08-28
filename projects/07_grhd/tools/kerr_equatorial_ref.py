"""
Reference (double-precision) primitive<->conserved conversion for 07_grhd's
Phase 2a (Kerr equatorial GRHD) -- derived and self-checked here in Python
BEFORE being trusted in the GPU shader, given how much more error-prone
Kerr's off-diagonal metric makes this than Phase 0/1's diagonal cases (see
kerr_metric_check.py's docstring: Phase 1 already caught one misremembered
formula, and this derivation caught a sign error in the shift beta^phi
along the way -- see the git history / README for the story).

Physical (ZAMO-frame) radial and azimuthal velocities v_r_hat, v_phi_hat;
W=1/sqrt(1-v_r_hat^2-v_phi_hat^2). Using the general 3+1 relation
u^mu=W(n^mu+v^mu) with n^mu the normal-observer 4-velocity:

    u^t   = W/alpha
    u^r   = W*v_r_hat/sqrt(gamma_rr)                          (beta^r=0)
    u^phi = W*v_phi_hat/sqrt(gamma_phiphi) - W*beta^phi/alpha  (beta^phi=g_tphi/gamma_phiphi)

Lowering with the (diagonal-in-space, but t-phi-coupled) metric collapses
to clean closed forms once beta^phi's sign is right (verified numerically
in kerr_metric_check.py against the standard alpha^2=Sigma*Delta/A):

    u_r   = W*sqrt(gamma_rr)*v_r_hat
    u_phi = W*sqrt(gamma_phiphi)*v_phi_hat
    E    := -u_t = W*(alpha - g_tphi*v_phi_hat/sqrt(gamma_phiphi))

(the missing overall W on the second term was a real bug caught here: it
made prim_to_cons/cons_to_prim disagree by ~0.1-0.3% for a!=0 -- exactly
right at a=0, where g_tphi=0 makes the missing factor invisible -- and was
found by directly comparing this E against the independently-trusted
circular-orbit E from kerr_orbits.py, not by staring at the algebra.)

Conserved variables (mixed-index T^mu_nu = rho*h*u^mu*u_nu + P*delta^mu_nu,
"areal" r^2-inclusive, matching Phase 0/1's per-cell convention):

    D   = r^2 * rho * u^t
    Sr  = r^2 * T^t_r   = r^2 * rho*h*u^t*u_r
    L   = r^2 * T^t_phi = r^2 * rho*h*u^t*u_phi
    tau = -r^2*T^t_t - D = r^2*rho*h*u^t*E - r^2*P - D

D, L, tau are EXACTLY conserved (no source): D from baryon conservation
(always source-free), L and tau=(E-D)-like from the phi- and t-Killing
vectors respectively (Noether's theorem) -- same argument as Phase 1's
source-free tau, now applying twice. Only Sr has a genuine geometric
source term.
"""

import numpy as np
from scipy.optimize import brentq

from kerr_metric_check import g_components


def metric_bundle(r, M, a):
    g_tt, g_tphi, g_rr, g_phiphi = g_components(r, M, a)
    Delta = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * Delta
    alpha2 = r * r * Delta / A
    alpha = np.sqrt(alpha2)
    gamma_rr = g_rr
    gamma_phiphi = g_phiphi
    beta_phi_up = g_tphi / g_phiphi
    return alpha, gamma_rr, gamma_phiphi, g_tphi, beta_phi_up


def prim_to_cons(rho, v_r, v_phi, P, r, M, a, gamma):
    alpha, gamma_rr, gamma_phiphi, g_tphi, _ = metric_bundle(r, M, a)
    W = 1.0 / np.sqrt(1.0 - v_r * v_r - v_phi * v_phi)
    h = 1.0 + gamma * P / ((gamma - 1.0) * rho)
    u_t_contra = W / alpha
    u_r_cov = W * np.sqrt(gamma_rr) * v_r
    u_phi_cov = W * np.sqrt(gamma_phiphi) * v_phi
    E = W * (alpha - g_tphi * v_phi / np.sqrt(gamma_phiphi))

    r2 = r * r
    D = r2 * rho * u_t_contra
    Sr = r2 * rho * h * u_t_contra * u_r_cov
    L = r2 * rho * h * u_t_contra * u_phi_cov
    tau = r2 * rho * h * u_t_contra * E - r2 * P - D
    return D, Sr, L, tau


def cons_to_prim(D, Sr, L, tau, r, M, a, gamma, p_guess, iters=50):
    """Newton iteration on h (specific enthalpy) -- see module docstring
    for the derivation. Returns (rho, v_r, v_phi, P)."""
    alpha, gamma_rr, gamma_phiphi, g_tphi, _ = metric_bundle(r, M, a)
    r2 = r * r
    kappa = Sr / D
    lam = L / D
    K = kappa * kappa / gamma_rr + lam * lam / gamma_phiphi

    def residual(h):
        W = np.sqrt(1.0 + K / (h * h))
        rho = D * alpha / (r2 * W)
        eps = (h - 1.0) / gamma
        P = (gamma - 1.0) * rho * eps
        Q = (tau + D + r2 * P) / D
        E_from_Q = Q / h
        E_from_W = W * alpha - g_tphi * lam / (h * gamma_phiphi)
        return E_from_Q - E_from_W

    # Newton from an enthalpy guess derived from the caller's pressure guess.
    rho_guess = D * alpha / (r2 * np.sqrt(1.0 + K))  # crude, W>=1 always
    h = 1.0 + gamma * max(p_guess, 1e-12) / ((gamma - 1.0) * rho_guess)
    for _ in range(iters):
        f0 = residual(h)
        dh = max(1e-6 * abs(h), 1e-10)
        fPlus = residual(h + dh)
        fMinus = residual(h - dh)
        deriv = (fPlus - fMinus) / (2 * dh)
        if abs(deriv) > 1e-13:
            h = max(h - f0 / deriv, 1.0 + 1e-13)

    W = np.sqrt(1.0 + K / (h * h))
    rho = D * alpha / (r2 * W)
    v_r = kappa / (h * W * np.sqrt(gamma_rr))
    v_phi = lam / (h * W * np.sqrt(gamma_phiphi))
    eps = (h - 1.0) / gamma
    P = (gamma - 1.0) * rho * eps
    return rho, v_r, v_phi, P


def circular_orbit_v_phi_hat(r0, M, a):
    """The physical (ZAMO-frame) azimuthal velocity of the exact circular
    geodesic at r0 -- v_r_hat=0, v_phi_hat=u_phi_cov/(W*sqrt(gamma_phiphi))
    with the COVARIANT u_phi (u_phi=g_tphi*u^t+g_phiphi*u^phi), not the
    contravariant u^phi that circular_orbit()/kerr_orbits.py solves for
    internally -- mixing those up here once already gave a wrong-looking
    (much too small) velocity, caught only because the round-trip test
    below is independent of this helper and still passed."""
    from kerr_orbits import circular_orbit

    E, L, _ = circular_orbit(r0, M, a)
    alpha, gamma_rr, gamma_phiphi, g_tphi, _ = metric_bundle(r0, M, a)
    g_tt = -(1.0 - 2.0 * M / r0)
    A_mat = np.array([[-g_tt, -g_tphi], [g_tphi, gamma_phiphi]])
    u_t_contra, u_phi_contra = np.linalg.solve(A_mat, [E, L])
    u_phi_cov = g_tphi * u_t_contra + gamma_phiphi * u_phi_contra
    W = alpha * u_t_contra
    return u_phi_cov / (W * np.sqrt(gamma_phiphi))


if __name__ == "__main__":
    M = 1.0
    gamma = 4.0 / 3.0
    for a in [0.0, 0.5, 0.9]:
        for r0 in [8.0, 15.0]:
            v_phi_hat = circular_orbit_v_phi_hat(r0, M, a)

            rho0, P0 = 1.0, 0.01  # arbitrary test thermo state, dust-like low pressure
            D, Sr, L, tau = prim_to_cons(rho0, 0.0, v_phi_hat, P0, r0, M, a, gamma)
            rho1, vr1, vphi1, P1 = cons_to_prim(D, Sr, L, tau, r0, M, a, gamma, p_guess=P0)
            print(f"a={a} r={r0}: v_phi_hat(circular)={v_phi_hat:.6f}   "
                  f"round-trip: rho {rho0:.6f}->{rho1:.6f}  v_r {0.0:.6f}->{vr1:.3e}  "
                  f"v_phi {v_phi_hat:.6f}->{vphi1:.6f}  P {P0:.6f}->{P1:.6f}")
