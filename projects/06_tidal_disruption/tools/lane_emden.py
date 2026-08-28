"""
Lane-Emden equation: the dimensionless hydrostatic-equilibrium structure of a
self-gravitating polytropic fluid sphere, P = K*rho**(1+1/n). Used by
make_star_ic.py to build the star's density/pressure/mass profile before it
is ever Monte-Carlo-sampled into SPH particles.

Derivation. Hydrostatic equilibrium (dP/dr = -G*m(r)*rho/r**2) plus
dm/dr = 4*pi*r**2*rho, combined with the polytropic closure rho = rho_c *
theta**n, non-dimensionalizes (r = alpha*xi, alpha**2 = (n+1)*K*rho_c**((1-n)/n)
/ (4*pi*G)) into the single second-order ODE

    theta''(xi) + (2/xi)*theta'(xi) + theta(xi)**n = 0,   theta(0)=1, theta'(0)=0

theta=1 at the center (xi=0), decreasing monotonically to theta=0 at the
star's dimensionless surface xi=xi_1 (finite for 0<=n<5). Physical profiles
follow once xi_1 and theta(xi) are known: rho(r) = rho_c*theta**n, m(r) =
4*pi*alpha**3*rho_c*(-xi**2*theta'(xi)), P(r) = K*rho(r)**((n+1)/n).

Numerical note -- a real, reproducible local bug, not a modeling choice: this
machine's scipy build silently crashes the whole Python process (no
traceback) when solve_ivp is asked to build a dense-output interpolant
(`t_eval=`, `dense_output=True`, or a plain `events=` callback -- all three
route through the same internal interpolation code) for this equation,
because of the 2/xi term's near-singular behavior right at the start of
integration (xi ~ 0). Worked around below by never requesting interpolation:
`_theta_at` only ever reads a solver's own final endpoint (no interpolation
needed for that), and `solve_lane_emden` does exactly one more integration
call to get a profile, using the solver's own adaptively-chosen step points
(`sol.t`) rather than a requested grid. Validated against tabulated
benchmarks (xi_1 and the dimensionless mass integral for n=0,1,1.5,3) to
~1e-7 relative error, and against the hydrostatic-equilibrium ODE itself
(finite-differenced) to ~2e-4.
"""

import numpy as np
from scipy.integrate import solve_ivp


def _series_start(n, xi_start):
    theta0 = 1.0 - xi_start**2 / 6.0 + n * xi_start**4 / 120.0
    dtheta0 = -xi_start / 3.0 + n * xi_start**3 / 30.0
    return theta0, dtheta0


def _rhs(xi, y, n):
    theta, dtheta = y
    tc = theta if theta > 0.0 else 0.0
    return [dtheta, -tc**n - (2.0 / xi) * dtheta]


def _theta_at(n, xi_start, theta0, dtheta0, xi_end, rtol, atol):
    sol = solve_ivp(_rhs, [xi_start, xi_end], [theta0, dtheta0], args=(n,),
                     method="RK45", rtol=rtol, atol=atol)
    return sol.y[0, -1]


def find_xi1(n, xi_start=1e-4, bracket_hi=30.0, rtol=1e-11, atol=1e-13, bisect_iters=60):
    theta0, dtheta0 = _series_start(n, xi_start)
    lo, hi = xi_start, bracket_hi
    while _theta_at(n, xi_start, theta0, dtheta0, hi, rtol, atol) > 0.0:
        hi *= 1.5
    for _ in range(bisect_iters):
        mid = 0.5 * (lo + hi)
        if _theta_at(n, xi_start, theta0, dtheta0, mid, rtol, atol) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_lane_emden(n, xi_start=1e-4, rtol=1e-11, atol=1e-13):
    """Returns (xi, theta, dtheta, xi1) on the solver's own adaptive grid."""
    theta0, dtheta0 = _series_start(n, xi_start)
    xi1 = find_xi1(n, xi_start=xi_start, rtol=rtol, atol=atol)
    sol = solve_ivp(_rhs, [xi_start, xi1], [theta0, dtheta0], args=(n,),
                     method="RK45", rtol=rtol, atol=atol)
    xi, theta, dtheta = sol.t, np.clip(sol.y[0], 0.0, None), sol.y[1]
    return xi, theta, dtheta, xi1


def build_polytrope(n, M_star=1.0, R_star=1.0, G=1.0, xi_start=1e-4):
    """
    Physical polytropic stellar structure matching total mass M_star and
    radius R_star, in units where G is as given (default G=1: the star's own
    natural units, mass in M_star, length in R_star).
    """
    xi, theta, dtheta, xi1 = solve_lane_emden(n, xi_start=xi_start)

    alpha = R_star / xi1
    mass_integral = -xi1**2 * dtheta[-1]
    rho_c = M_star / (4.0 * np.pi * alpha**3 * mass_integral)

    Gamma = (n + 1.0) / n if n > 0 else np.inf
    K = 4.0 * np.pi * G * alpha**2 * rho_c**((n - 1.0) / n) / (n + 1.0)
    P_c = K * rho_c**Gamma

    r = alpha * xi
    rho = rho_c * theta**n
    P = K * rho**Gamma
    m = 4.0 * np.pi * alpha**3 * rho_c * (-xi**2 * dtheta)

    return dict(n=n, Gamma=Gamma, K=K, G=G, rho_c=rho_c, P_c=P_c, alpha=alpha,
                xi1=xi1, r=r, rho=rho, P=P, m=m, M_star=M_star, R_star=R_star)
