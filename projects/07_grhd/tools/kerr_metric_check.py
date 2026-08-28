"""
Cross-check for 07_grhd's Phase 2a (Kerr equatorial GRHD): verifies that
Gamma^r_munu (needed for the radial-momentum source term) computed via
NUMERICAL differentiation of the equatorial Kerr metric gives exactly zero
radial 4-acceleration for a circular geodesic orbit -- the literal
definition of "circular orbit" (u^r=0 for all time requires
du^r/dtau + Gamma^r_munu u^mu u^nu = 0, and du^r/dtau=0 identically here).

This is deliberately independent of any hand-derived symbolic Christoffel
formula: after Phase 1 caught one plausible-looking-but-wrong remembered
GR formula (see 07_grhd/README.md), Kerr's off-diagonal metric makes the
by-hand Christoffel algebra meaningfully more error-prone, so this project
computes Gamma^r_munu numerically (central finite difference of the
metric, which itself is just the textbook Kerr metric definition -- not a
derived quantity) in both this check AND the actual GPU shader, rather
than trusting a symbolic derivation to be transcribed correctly in two
places (paper and GLSL).
"""

import numpy as np

from kerr_orbits import circular_orbit


def g_components(r, M, a):
    """Equatorial Kerr metric (Boyer-Lindquist, theta=pi/2), independent
    nonzero components. Textbook definition, not a derived quantity."""
    Delta = r * r - 2.0 * M * r + a * a
    g_tt = -(1.0 - 2.0 * M / r)
    g_tphi = -2.0 * M * a / r
    g_rr = r * r / Delta
    g_phiphi = r * r + a * a + 2.0 * M * a * a / r
    return g_tt, g_tphi, g_rr, g_phiphi


def christoffel_r_numeric(r, M, a, h=1e-5):
    """Gamma^r_munu for munu in {tt, tphi, phiphi, rr} via central finite
    difference: Gamma^r_munu = -0.5*g^rr*d(g_munu)/dr (munu!=rr),
    Gamma^r_rr = +0.5*g^rr*d(g_rr)/dr (standard Christoffel formula,
    simplified since g_{mu r}=0 for mu!=r in Boyer-Lindquist)."""
    gtt_p, gtphi_p, grr_p, gphiphi_p = g_components(r + h, M, a)
    gtt_m, gtphi_m, grr_m, gphiphi_m = g_components(r - h, M, a)
    _, _, grr0, _ = g_components(r, M, a)
    g_rr_inv = 1.0 / grr0

    d_gtt = (gtt_p - gtt_m) / (2 * h)
    d_gtphi = (gtphi_p - gtphi_m) / (2 * h)
    d_gphiphi = (gphiphi_p - gphiphi_m) / (2 * h)
    d_grr = (grr_p - grr_m) / (2 * h)

    Gamma_r_tt = -0.5 * g_rr_inv * d_gtt
    Gamma_r_tphi = -0.5 * g_rr_inv * d_gtphi
    Gamma_r_phiphi = -0.5 * g_rr_inv * d_gphiphi
    Gamma_r_rr = 0.5 * g_rr_inv * d_grr
    return Gamma_r_tt, Gamma_r_tphi, Gamma_r_phiphi, Gamma_r_rr


def radial_geodesic_acceleration(r, u_t_contra, u_phi_contra, M, a):
    """Gamma^r_munu u^mu u^nu for a purely (t,phi)-moving worldline
    (u^r=0): = Gamma^r_tt*(u^t)^2 + 2*Gamma^r_tphi*u^t*u^phi +
    Gamma^r_phiphi*(u^phi)^2 (no Gamma^r_rr term since u^r=0)."""
    Gtt, Gtphi, Gphiphi, _ = christoffel_r_numeric(r, M, a)
    return Gtt * u_t_contra ** 2 + 2.0 * Gtphi * u_t_contra * u_phi_contra + Gphiphi * u_phi_contra ** 2


if __name__ == "__main__":
    M = 1.0
    for a in [0.0, 0.5, 0.9]:
        for r0 in [8.0, 15.0]:
            E, L, Omega = circular_orbit(r0, M, a)
            g_tt, g_tphi, g_rr, g_phiphi = g_components(r0, M, a)
            A_mat = np.array([[-g_tt, -g_tphi], [g_tphi, g_phiphi]])
            u_t, u_phi = np.linalg.solve(A_mat, [E, L])
            accel = radial_geodesic_acceleration(r0, u_t, u_phi, M, a)
            print(f"a={a}  r={r0}: radial 4-accel (should be ~0) = {accel:.3e}")
