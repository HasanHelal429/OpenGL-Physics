"""
Exact solution of the classic (Newtonian) 1D Euler-equation Riemann problem,
ideal Gamma-law gas -- the standard iterative pressure solver (Toro,
"Riemann Solvers and Numerical Methods for Fluid Dynamics", Ch. 4). Used to
validate 07_grhd's relativistic solver in its low-velocity/low-pressure
limit (decks/shocktube_newton_limit.toml), where the SRHD equations should
reduce to these Newtonian ones and a real disagreement would mean a bug in
the conservative variables, flux, or primitive recovery -- not just a
relativistic-vs-Newtonian physics difference.
"""

import numpy as np


def _f_k(p, rho_k, p_k, gamma):
    a_k = 2.0 / ((gamma + 1.0) * rho_k)
    b_k = p_k * (gamma - 1.0) / (gamma + 1.0)
    c_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:  # shock
        return (p - p_k) * np.sqrt(a_k / (p + b_k))
    return (2.0 * c_k / (gamma - 1.0)) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _f_k_prime(p, rho_k, p_k, gamma):
    a_k = 2.0 / ((gamma + 1.0) * rho_k)
    b_k = p_k * (gamma - 1.0) / (gamma + 1.0)
    c_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:
        return np.sqrt(a_k / (b_k + p)) * (1.0 - (p - p_k) / (2.0 * (b_k + p)))
    return (1.0 / (rho_k * c_k)) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))


def solve_star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, tol=1e-12, max_iter=100):
    """Newton-Raphson for the star-region pressure/velocity (Toro sec. 4.3-4.5)."""
    p = 0.5 * (p_l + p_r)
    p = max(p, 1e-10)
    for _ in range(max_iter):
        f = _f_k(p, rho_l, p_l, gamma) + _f_k(p, rho_r, p_r, gamma) + (u_r - u_l)
        fp = _f_k_prime(p, rho_l, p_l, gamma) + _f_k_prime(p, rho_r, p_r, gamma)
        p_new = p - f / fp
        p_new = max(p_new, 1e-10)
        if abs(p_new - p) / p < tol:
            p = p_new
            break
        p = p_new
    u_star = 0.5 * (u_l + u_r) + 0.5 * (_f_k(p, rho_r, p_r, gamma) - _f_k(p, rho_l, p_l, gamma))
    return p, u_star


def sample(xi, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, p_star, u_star):
    """Sample the self-similar solution at xi = (x-x0)/t (Toro sec. 4.5-4.6)."""
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)

    if xi <= u_star:  # left of the contact
        rho_k, u_k, p_k, c_k, sign = rho_l, u_l, p_l, c_l, -1.0
    else:
        rho_k, u_k, p_k, c_k, sign = rho_r, u_r, p_r, c_r, 1.0

    if p_star > p_k:  # shock on this side
        pr = p_star / p_k
        s = u_k + sign * c_k * np.sqrt((gamma + 1.0) / (2.0 * gamma) * pr + (gamma - 1.0) / (2.0 * gamma))
        if (sign < 0 and xi > s) or (sign > 0 and xi < s):
            rho_star = rho_k * (pr + (gamma - 1.0) / (gamma + 1.0)) / (pr * (gamma - 1.0) / (gamma + 1.0) + 1.0)
            return rho_star, u_star, p_star
        return rho_k, u_k, p_k
    else:  # rarefaction on this side
        c_star = c_k * (p_star / p_k) ** ((gamma - 1.0) / (2.0 * gamma))
        head = u_k + sign * c_k
        tail = u_star + sign * c_star
        if (sign < 0 and xi < head) or (sign > 0 and xi > head):
            return rho_k, u_k, p_k
        if (sign < 0 and xi > tail) or (sign > 0 and xi < tail):
            rho_star = rho_k * (p_star / p_k) ** (1.0 / gamma)
            return rho_star, u_star, p_star
        # inside the fan (Toro eq. 4.56/4.63, self-similar closed form)
        u_fan = (2.0 / (gamma + 1.0)) * (-sign * c_k + (gamma - 1.0) / 2.0 * u_k + xi)
        c_fan = (2.0 / (gamma + 1.0)) * (c_k - sign * (gamma - 1.0) / 2.0 * (u_k - xi))
        rho_fan = rho_k * (c_fan / c_k) ** (2.0 / (gamma - 1.0))
        p_fan = p_k * (c_fan / c_k) ** (2.0 * gamma / (gamma - 1.0))
        return rho_fan, u_fan, p_fan


def exact_solution(x, t, x0, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma):
    """Vectorized: returns (rho, u, p) arrays sampled at positions x, time t."""
    p_star, u_star = solve_star_state(rho_l, u_l, p_l, rho_r, u_r, p_r, gamma)
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)
    for i, xv in enumerate(x):
        xi = (xv - x0) / t
        rho[i], u[i], p[i] = sample(xi, rho_l, u_l, p_l, rho_r, u_r, p_r, gamma, p_star, u_star)
    return rho, u, p
