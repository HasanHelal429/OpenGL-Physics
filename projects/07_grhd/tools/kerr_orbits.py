"""
Exact equatorial circular geodesic orbits in Kerr (Boyer-Lindquist r,
theta=pi/2), used to validate 07_grhd's Phase 2a (Kerr equatorial-plane
GRHD, see 07_grhd/README.md).

Rather than trust a closed-form formula recalled from memory (Bardeen,
Press & Teukolsky 1972 give one, but Phase 1 already caught one
misremembered GR formula that looked plausible until checked -- see
07_grhd/README.md's Physics section -- so this project now derives these
independently instead), circular-orbit energy E and angular momentum L
(per unit rest mass) are found directly from the definition of a circular
orbit: the radial effective potential

    R(r; E, L) = [E*(r^2+a^2) - L*a]^2 - Delta*(r^2 + (L-a*E)^2)
    (dr/dtau)^2 = R(r) / r^2^2         Delta = r^2 - 2*M*r + a^2

(the standard equatorial, Q=0 Kerr radial potential -- Chandrasekhar,
"The Mathematical Theory of Black Holes", or Bardeen/Press/Teukolsky 1972)
must have a *double* root at the orbit radius r_0: R(r_0)=0 (r_0 is a
turning point) and dR/dr(r_0)=0 (an extremum, not just any turning point --
without this, the orbit would be eccentric, not circular). Two equations,
two unknowns (E,L), solved by Newton's method from a Newtonian-limit
initial guess.

Self-checked against the one Schwarzschild (a=0) circular-orbit fact this
project already trusts without re-deriving: the ISCO sits at r=6M with
E=sqrt(8/9), L=2*sqrt(3)*M (universally quoted; also directly verifiable
by hand that R(6M)=0 for exactly these values).
"""

import numpy as np
from scipy.optimize import fsolve


def _R_and_derivs(r, E, L, M, a):
    """R(r,E,L), dR/dr, and d^2R/dr^2 -- needed for both the circular-orbit
    solve (R=0, dR/dr=0) and the ISCO condition (additionally d^2R/dr^2=0)."""
    Delta = r * r - 2.0 * M * r + a * a
    dDelta = 2.0 * r - 2.0 * M
    d2Delta = 2.0
    A = E * (r * r + a * a) - L * a
    dA = E * 2.0 * r
    d2A = E * 2.0
    B = r * r + (L - a * E) ** 2
    dB = 2.0 * r
    d2B = 2.0

    R = A * A - Delta * B
    dR = 2.0 * A * dA - dDelta * B - Delta * dB
    d2R = 2.0 * dA * dA + 2.0 * A * d2A - d2Delta * B - 2.0 * dDelta * dB - Delta * d2B
    return R, dR, d2R


def circular_orbit(r0, M, a, prograde=True):
    """Returns (E, L, Omega) for the circular geodesic at radius r0."""
    # Newtonian-limit initial guess (a good starting point even fairly close
    # to the black hole -- Newton's method converges from here reliably).
    sign = 1.0 if prograde else -1.0
    L0 = sign * np.sqrt(M * r0)
    E0 = 1.0 - 0.5 * M / r0

    def equations(x):
        E, L = x
        R, dR, _ = _R_and_derivs(r0, E, L, M, a)
        return [R, dR]

    E, L = fsolve(equations, [E0, L0], xtol=1e-13)
    R, dR, _ = _R_and_derivs(r0, E, L, M, a)
    if max(abs(R), abs(dR)) > 1e-8 * max(1.0, E * E, L * L):
        raise RuntimeError(f"circular_orbit: fsolve did not converge at r0={r0} (R={R:.3e}, dR={dR:.3e})")

    # Omega = dphi/dt = u^phi/u^t, from E=-u_t=-(g_tt*u^t+g_tphi*u^phi) and
    # L=u_phi=g_tphi*u^t+g_phiphi*u^phi -- solved as a plain 2x2 linear
    # system rather than substituting a separately-recalled closed form for
    # Omega (the whole point of this module is not doing that -- see the
    # docstring).
    g_tt = -(1.0 - 2.0 * M / r0)
    g_tphi = -2.0 * M * a / r0
    g_phiphi = r0 * r0 + a * a + 2.0 * M * a * a / r0
    A_mat = np.array([[-g_tt, -g_tphi], [g_tphi, g_phiphi]])
    u_t_contra, u_phi_contra = np.linalg.solve(A_mat, [E, L])
    Omega = u_phi_contra / u_t_contra
    return E, L, Omega


def isco_radius(M, a, prograde=True):
    """Innermost stable circular orbit: circular orbit where R also has
    d^2R/dr^2=0 (the double root degenerates to a triple root -- marginal
    stability). Root-find over r using the circular_orbit solve at each r."""
    from scipy.optimize import brentq

    def stability(r):
        E, L, _ = circular_orbit(r, M, a, prograde)
        _, _, d2R = _R_and_derivs(r, E, L, M, a)
        return d2R

    # No circular timelike orbit exists inside the photon-sphere-like
    # region (circular_orbit's fsolve fails there, e.g. right at r=2M=
    # the a=0 horizon) -- rather than guess a fixed bracket that only
    # happens to work for one spin, scan outward from a safely-large
    # radius (stable, so d2R<0 there for any reasonable spin) down to
    # wherever a circular orbit stops existing, and bracket the sign
    # change against the last radius that still worked.
    r_grid = np.linspace(10.0 * M, 1.001 * M, 400)
    prev_r, prev_val = None, None
    for r in r_grid:
        try:
            E, L, _ = circular_orbit(r, M, a, prograde)
            _, _, val = _R_and_derivs(r, E, L, M, a)
        except RuntimeError:
            break
        if prev_val is not None and prev_val * val < 0.0:
            return brentq(stability, r, prev_r, xtol=1e-12)
        prev_r, prev_val = r, val
    raise RuntimeError(f"isco_radius: no sign change found scanning inward (M={M}, a={a}, prograde={prograde})")


if __name__ == "__main__":
    # Self-check: Schwarzschild ISCO is the one fact trusted without
    # re-deriving -- r=6M, E=sqrt(8/9), L=2*sqrt(3)*M.
    E, L, Omega = circular_orbit(6.0, 1.0, 0.0)
    print(f"Schwarzschild r=6M: E={E:.6f} (expect {np.sqrt(8/9):.6f})  "
          f"L={L:.6f} (expect {2*np.sqrt(3):.6f})")
    r_isco = isco_radius(1.0, 0.0)
    print(f"Schwarzschild ISCO: r={r_isco:.6f} (expect 6.0)")

    for a in [0.0, 0.5, 0.9, 0.998]:
        r_isco_pro = isco_radius(1.0, a, prograde=True)
        r_isco_ret = isco_radius(1.0, a, prograde=False)
        print(f"a={a}: ISCO prograde={r_isco_pro:.4f}M  retrograde={r_isco_ret:.4f}M")
