"""
Exact steady-state relativistic Bondi accretion solution: a Gamma-law gas
falling radially onto a Schwarzschild black hole (mass M, geometrized units
G=c=1), in Schwarzschild coordinates. Used both to generate 07_grhd's
Phase 1 initial condition and as the reference solution for validation
(tools/plot_bondi.py).

Method: sonic-point shooting (the standard approach -- see e.g. Shapiro &
Teukolsky "Black Holes, White Dwarfs and Neutron Stars" sec. 14.3, or the
Bondi-flow test as used in numerical GRHD code papers, e.g. Gammie,
McKinney & Toth 2003's HARM paper sec. 4.1). The flow has three exactly
conserved quantities along it (from baryon conservation, the stationary
Euler equation's first integral, and the assumption of no shocks):

    Mdot  = r^2 * rho * u^r                        (mass accretion rate)
    Be    = h * alpha * W                          (relativistic Bernoulli constant)
    K     = P / rho^Gamma                          (entropy, isentropic flow)

with alpha=sqrt(1-2M/r) (lapse), W=1/sqrt(1-v^2) (Lorentz factor for the
physical/orthonormal radial velocity v, negative for infall), and
h=1+Gamma*P/((Gamma-1)*rho) (specific enthalpy, Gamma-law EOS).

At the sonic radius r_c, the critical-point conditions come from demanding
a smooth transonic solution: writing du/dr (u=|v|) from the two
conservation laws above gives a ratio N(r,u)/D(r,u), and a finite,
well-defined du/dr at the point where D=0 requires N=0 there too. Working
this out for the Schwarzschild background (D=0 reduces to the physically
expected condition "flow speed equals sound speed", u_c=cs_c -- not
independently obvious in the relativistic equations, worth deriving rather
than assuming -- and N=0 then fixes cs_c^2 itself) gives:

    u_c = cs_c                     (the physical sonic-point condition)
    cs_c^2 = M / (2*r_c - 3*M)

which correctly reduces to the standard Newtonian Bondi critical-point
condition cs_c^2 = M/(2*r_c) as r_c -> infinity, a good consistency check.
This, combined with the Gamma-law identity cs^2 = Gamma*(Gamma-1)*eps /
(1+Gamma*eps), fixes the specific internal energy eps_c, hence h_c, hence
Be (from the critical-point state) -- all independent of the density
normalization, which is a free overall scale (set here via rho_c=1, i.e.
"code units" normalized at the sonic point, matching this project's
G=M_star=R_star=1 convention elsewhere).

Away from r_c, Mdot and Be are known constants, so (rho(r), v(r)) solve two
equations in two unknowns at each r -- reduced to a single 1D root-find in
rho(r) below (see solve_at_radius).
"""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


def eps_of_cs2(cs2, gamma):
    """Gamma-law identity: cs^2 = Gamma*(Gamma-1)*eps / (1+Gamma*eps), solved for eps."""
    return cs2 / (gamma * (gamma - 1.0 - cs2))


class BondiSolution:
    def __init__(self, M, r_c, gamma):
        self.M = M
        self.r_c = r_c
        self.gamma = gamma

        if r_c <= 1.5 * M:
            raise ValueError("r_c too small: 2*r_c - 3*M <= 0 makes cs_c^2 negative/divergent")
        cs_c2 = M / (2.0 * r_c - 3.0 * M)
        v_c2 = cs_c2  # sonic point: |v_c| = cs_c
        eps_c = eps_of_cs2(cs_c2, gamma)

        self.rho_c = 1.0  # free normalization
        self.eps_c = eps_c
        self.P_c = (gamma - 1.0) * self.rho_c * eps_c
        self.K = self.P_c / self.rho_c ** gamma
        self.h_c = 1.0 + gamma * eps_c
        self.v_c = -np.sqrt(v_c2)  # infall: negative
        self.W_c = 1.0 / np.sqrt(1.0 - v_c2)
        self.alpha_c = np.sqrt(1.0 - 2.0 * M / r_c)

        self.Be = self.h_c * self.alpha_c * self.W_c
        u_r_c = self.W_c * self.v_c * self.alpha_c
        self.Mdot_abs = abs(r_c ** 2 * self.rho_c * u_r_c)

    def _residual(self, rho, r):
        """Bernoulli residual at radius r for a trial density rho, having
        already used mass conservation to fix |v| (hence W) from rho."""
        gamma = self.gamma
        eps = self.K * rho ** (gamma - 1.0) / (gamma - 1.0)
        h = 1.0 + gamma * eps
        alpha = np.sqrt(1.0 - 2.0 * self.M / r)
        u = self.Mdot_abs / (r ** 2 * rho * alpha)  # = |v|*W
        W = np.sqrt(1.0 + u * u)
        return h * alpha * W - self.Be

    def solve_at_radius(self, r):
        """Returns (rho, v, P) at radius r (v<0, infall).

        The Bernoulli+mass-conservation system has TWO roots in rho at any
        r != r_c (a subsonic, high-density branch and a supersonic,
        low-density branch -- the residual is positive at both rho->0 and
        rho->infinity, dips negative in between, crossing zero twice), the
        same two-branch structure as the classic Newtonian Bondi/nozzle
        problem. The physical transonic solution takes the supersonic
        (smaller rho) branch for r<r_c and the subsonic (larger rho) branch
        for r>r_c. The two roots coincide exactly at r=r_c and separate
        only slowly nearby (the gap shrinks to ~1e-4-1e-6 in the residual
        within a few percent of r_c), so a fixed grid scan for a sign
        change is unreliable there -- it can step clean over the narrow
        negative dip. Instead find the residual's interior minimum
        directly (bounded scalar minimization, robust regardless of how
        close the two roots are), confirm a real transonic pair exists
        there (residual<0), and bracket each root against that minimum.
        """
        if abs(r - self.r_c) < 1e-12:
            return self.rho_c, self.v_c, self.P_c

        lo_bound, hi_bound = self.rho_c * 1e-8, self.rho_c * 1e8
        res = minimize_scalar(lambda x: self._residual(x, r), bounds=(lo_bound, hi_bound), method="bounded",
                               options={"xatol": 1e-15})
        rho_min, f_min = res.x, res.fun
        if f_min >= 0.0:
            raise RuntimeError(f"no transonic pair of roots at r={r} (min residual {f_min:.3e} >= 0)")

        if r < self.r_c:
            rho = brentq(lambda x: self._residual(x, r), lo_bound, rho_min, xtol=1e-15, rtol=1e-15)
        else:
            rho = brentq(lambda x: self._residual(x, r), rho_min, hi_bound, xtol=1e-15, rtol=1e-15)

        eps = self.K * rho ** (self.gamma - 1.0) / (self.gamma - 1.0)
        alpha = np.sqrt(1.0 - 2.0 * self.M / r)
        u = self.Mdot_abs / (r ** 2 * rho * alpha)
        v = -u / np.sqrt(1.0 + u * u)  # infall
        P = (self.gamma - 1.0) * rho * eps
        return rho, v, P

    def profile(self, r_array):
        rho = np.empty_like(r_array)
        v = np.empty_like(r_array)
        P = np.empty_like(r_array)
        for i, r in enumerate(r_array):
            rho[i], v[i], P[i] = self.solve_at_radius(r)
        return rho, v, P

    def check_conserved(self, r_array):
        """Self-check: Mdot, Be, K evaluated along the constructed profile
        should all equal the sonic-point values to near machine precision.
        Returns (max Mdot rel. error, max Be rel. error, max K rel. error)."""
        rho, v, P = self.profile(r_array)
        gamma = self.gamma
        alpha = np.sqrt(1.0 - 2.0 * self.M / r_array)
        W = 1.0 / np.sqrt(1.0 - v * v)
        u_r = W * v * alpha
        Mdot = np.abs(r_array ** 2 * rho * u_r)
        h = 1.0 + gamma * P / ((gamma - 1.0) * rho)
        Be = h * alpha * W
        K = P / rho ** gamma
        return (np.max(np.abs(Mdot - self.Mdot_abs) / self.Mdot_abs),
                np.max(np.abs(Be - self.Be) / self.Be),
                np.max(np.abs(K - self.K) / self.K))
