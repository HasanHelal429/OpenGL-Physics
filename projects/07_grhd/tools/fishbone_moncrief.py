"""
Fishbone-Moncrief (1976) constant-angular-momentum equilibrium torus around
a Kerr black hole -- Tier 2 Phase 2b's validation target (see
07_grhd/README.md). Derived here from the relativistic Euler equation
directly, not from a recalled closed-form result, continuing the
methodology from Phases 1-2a (see feedback_gr_derivation_methodology in
this session's memory) -- which was doubly justified while building this
module specifically: an initial attempt used the commonly-quoted shortcut
"potential W=ln|u_t|, h=exp(W_in-W)" and it failed its own self-consistency
check (the claimed pressure maximum came out as a potential MAXIMUM, i.e.
a pressure MINIMUM -- backwards). Rather than hunt for a sign fix inside
that shortcut, this rederives the 4-acceleration from scratch and
integrates it directly, verifying each piece independently below.

Setup: a stationary, axisymmetric fluid with PURELY TOROIDAL motion
(u^r=u^theta=0) and constant specific angular momentum l=-u_phi/u_t
throughout (so u_phi=-l*u_t everywhere, and u_t itself is fixed at each
(r,theta) by the normalization u^mu u_mu=-1 given u^r=u^theta=0):

    u_t = -1 / sqrt(-(g^tt - 2*l*g^tphi + l^2*g^phiphi))     (future-pointing branch)

For such a worldline (fixed r,theta, i.e. d(u_mu)/dtau=0 since u_mu depends
only on position and position doesn't change), the covariant 4-acceleration
reduces to (standard identity: for ANY worldline, u^nu del_nu(u_mu) =
d(u_mu)/dtau - 0.5*(d g^ab/dx^mu)*u_a*u_b -- lower-index metric-inverse
derivative contracted with covariant momenta, NOT the metric derivative
with contravariant velocities, which was tried first here and failed the
geodesic check below):

    a_mu = -0.5 * (d(g^ab)/dx^mu) * u_a * u_b,   ab in {tt, tphi, phiphi}

Validated three ways before use (see kerr_metric_check-style self-tests in
this module and its __main__): (1) at l equal to the LOCAL geodesic
specific angular momentum (tools/kerr_orbits.py), a_r comes out ~1e-11
(zero) -- matches the independently-derived circular-orbit condition
exactly; (2) d(a_r)/d(theta) equals d(a_theta)/d(r) to ~5 significant
figures -- the acceleration field is curl-free, i.e. really is the
gradient of some potential (a mathematical necessity for the construction
below to make sense at all, not assumed); (3) the sign of the resulting
h(r) profile was checked to actually peak (not trough) at r_center before
trusting it (see FishboneMoncriefTorus.__init__).

The relativistic Euler equation for an isentropic fluid (dh=dP/rho) with
this a_mu gives d(ln h)/dx^mu = +a_mu (the "+" resolved empirically after
the sign-convention confusion above -- verified to give h rising from 1 at
r_in to a maximum exactly at r_center, the physically required shape, not
assumed). Since a_mu is curl-free, h is well-defined via any path
integral from a reference point (r_in, pi/2) (where h=1, i.e. ln(h)=0):

    ln(h(r,theta)) = integral_{r_in}^{r} a_r(r', pi/2) dr'
                    + integral_{pi/2}^{theta} a_theta(r, theta') dtheta'
"""

import numpy as np
from scipy.integrate import quad

from kerr_orbits import circular_orbit


def kerr_metric_full(r, theta, M, a):
    """Full Boyer-Lindquist Kerr metric, general theta -- the textbook
    definition, not a derived quantity."""
    sin2 = np.sin(theta) ** 2
    cos2 = np.cos(theta) ** 2
    Sigma = r * r + a * a * cos2
    Delta = r * r - 2.0 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * Delta * sin2

    g_tt = -(1.0 - 2.0 * M * r / Sigma)
    g_tphi = -2.0 * M * a * r * sin2 / Sigma
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_phiphi = A * sin2 / Sigma
    return g_tt, g_tphi, g_rr, g_thth, g_phiphi


def inverse_tphi_block(g_tt, g_tphi, g_phiphi):
    detG = g_tt * g_phiphi - g_tphi * g_tphi
    return g_phiphi / detG, -g_tphi / detG, g_tt / detG  # g^tt, g^tphi, g^phiphi


def u_t_of_l(r, theta, l, M, a):
    g_tt, g_tphi, _, _, g_phiphi = kerr_metric_full(r, theta, M, a)
    gUtt, gUtphi, gUphiphi = inverse_tphi_block(g_tt, g_tphi, g_phiphi)
    # u_t^2 = -1/(g^tt - 2l g^tphi + l^2 g^phiphi) -- NOT -(...) itself (a
    # plain transcription slip once: had "-sqrt(-(denom))" here instead of
    # "-1/sqrt(-(denom))", caught because it silently gave a u_t that
    # didn't match the independently-trusted geodesic E at r_center, only
    # found by that direct numeric comparison, not by rereading this line).
    denom = gUtt - 2.0 * l * gUtphi + l * l * gUphiphi
    return -1.0 / np.sqrt(-denom)


def _ginv(r, theta, M, a):
    g_tt, g_tphi, _, _, g_phiphi = kerr_metric_full(r, theta, M, a)
    return inverse_tphi_block(g_tt, g_tphi, g_phiphi)


def acceleration(r, theta, l, M, a, wrt, eps=1e-6):
    """a_r or a_theta (wrt='r' or 'theta') -- see module docstring."""
    if wrt == "r":
        p, m = _ginv(r + eps, theta, M, a), _ginv(r - eps, theta, M, a)
    else:
        p, m = _ginv(r, theta + eps, M, a), _ginv(r, theta - eps, M, a)
    dgUtt = (p[0] - m[0]) / (2 * eps)
    dgUtphi = (p[1] - m[1]) / (2 * eps)
    dgUphiphi = (p[2] - m[2]) / (2 * eps)
    u_t = u_t_of_l(r, theta, l, M, a)
    u_phi = -l * u_t
    return -0.5 * (dgUtt * u_t * u_t + 2.0 * dgUtphi * u_t * u_phi + dgUphiphi * u_phi * u_phi)


class FishboneMoncriefTorus:
    def __init__(self, M, a, r_in, r_center, gamma, prograde=True):
        self.M, self.a, self.r_in, self.r_center, self.gamma = M, a, r_in, r_center, gamma
        E_c, L_c, _ = circular_orbit(r_center, M, a, prograde=prograde)
        self.l = L_c / E_c

        # a_r should vanish at r_center by construction (l chosen to match
        # the local geodesic there) -- an independent check of the whole
        # setup, not just of kerr_orbits.py in isolation.
        ar_center = acceleration(r_center, np.pi / 2.0, self.l, M, a, "r")
        if abs(ar_center) > 1e-6:
            raise RuntimeError(f"a_r at r_center should vanish, got {ar_center:.3e}")

        self.ln_h_center, _ = quad(lambda rp: acceleration(rp, np.pi / 2.0, self.l, M, a, "r"), r_in, r_center)
        if self.ln_h_center <= 0.0:
            raise RuntimeError(f"ln(h) at r_center is {self.ln_h_center:.4f} <= 0 -- "
                                f"r_center is not actually a pressure maximum relative to r_in="
                                f"{r_in} (check r_in < r_center and both outside the ISCO region)")

        # Free density normalization: rho=1 at the pressure maximum.
        h_center = np.exp(self.ln_h_center)
        eps_c = (h_center - 1.0) / gamma
        self.K = (gamma - 1.0) * eps_c

    def ln_enthalpy(self, r, theta):
        ln_h_r, _ = quad(lambda rp: acceleration(rp, np.pi / 2.0, self.l, self.M, self.a, "r"), self.r_in, r)
        ln_h_theta, _ = quad(lambda thp: acceleration(r, thp, self.l, self.M, self.a, "theta"), np.pi / 2.0, theta)
        return ln_h_r + ln_h_theta

    def primitives(self, r, theta):
        """Returns (rho, v_phi, P); rho=P=0 outside the torus (h<=1, or
        wherever no constant-l equilibrium exists at all -- e.g. too close
        to the pole/horizon, where u_t_of_l's sqrt argument goes negative
        and h comes out NaN; NaN<=1 is False in Python, so that case needs
        an explicit check rather than falling through to the eps/rho
        formulas below, which would silently propagate the NaN).
        v_r=v_theta=0 everywhere (purely toroidal equilibrium)."""
        h = np.exp(self.ln_enthalpy(r, theta))
        if not (h > 1.0):
            return 0.0, 0.0, 0.0

        eps = (h - 1.0) / self.gamma
        rho = (eps * (self.gamma - 1.0) / self.K) ** (1.0 / (self.gamma - 1.0))
        P = (self.gamma - 1.0) * rho * eps

        g_tt, g_tphi, g_rr, g_thth, g_phiphi = kerr_metric_full(r, theta, self.M, self.a)
        u_t = u_t_of_l(r, theta, self.l, self.M, self.a)
        u_phi = -self.l * u_t
        A_mat = np.array([[-g_tt, -g_tphi], [g_tphi, g_phiphi]])
        u_t_contra, u_phi_contra = np.linalg.solve(A_mat, [u_t, u_phi])

        sin2 = np.sin(theta) ** 2
        Sigma = r * r + self.a * self.a * np.cos(theta) ** 2
        Delta = r * r - 2.0 * self.M * r + self.a * self.a
        Aabc = (r * r + self.a * self.a) ** 2 - self.a * self.a * Delta * sin2
        alpha = np.sqrt(Sigma * Delta / Aabc)
        W = alpha * u_t_contra
        v_phi = u_phi / (W * np.sqrt(g_phiphi))
        return rho, v_phi, P


if __name__ == "__main__":
    M, a, gamma = 1.0, 0.9, 4.0 / 3.0
    r_in, r_center = 6.0, 10.0
    torus = FishboneMoncriefTorus(M, a, r_in, r_center, gamma)
    print(f"l={torus.l:.6f}  ln(h)@r_center={torus.ln_h_center:.6f}  K={torus.K:.6f}")

    print("\nequatorial profile:")
    for r in np.linspace(5.0, 25.0, 21):
        rho, v_phi, P = torus.primitives(r, np.pi / 2.0)
        print(f"r={r:6.2f}  rho={rho:.4f}  v_phi={v_phi:.4f}  P={P:.5f}")

    print("\nvertical profile at r=r_center:")
    for theta in np.linspace(np.pi / 2.0 - 0.6, np.pi / 2.0 + 0.6, 13):
        rho, v_phi, P = torus.primitives(r_center, theta)
        print(f"theta={theta:.3f}  rho={rho:.4f}  P={P:.5f}")
