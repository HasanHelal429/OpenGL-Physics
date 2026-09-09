"""
Generate a realistic N-body initial condition and write it as a raw binary
file the C++ sim loads directly (scenario.type = "ic_file").

File format (matches 06_tidal_disruption/tools/make_star_ic.py and
nbody_core/src/Scenarios.cpp's IcFileScenario): N records of 7 float32,
    x, y, z, mass, vx, vy, vz
For a 2D run the C++ side reads and discards the z / vz columns.

Models (units G = M_total = scale = 1):
    plummer    - isotropic Plummer sphere (exact DF for velocities)
    hernquist  - Hernquist (1990) sphere, isotropic-Jeans velocities
    king       - King (1966) lowered-isothermal sphere (RK4 on the W ODE)

This is a Python reimplementation of nbody_core/include/ngrav/ic/*.hpp --
kept deliberately independent so a mismatch between the two is a real
signal, not a shared bug. It prints the measured virial ratio 2T/|W| and a
suggested softening length + timestep for the generated cloud.

Usage:
    python make_ic.py --model hernquist --n 20000 --out ic/hernquist_20k.bin
    python make_ic.py --model king --n 20000 --w0 7 --out ic/king_w7_20k.bin
"""

import argparse

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import erf


def _jeans_sigma2(r_query, r_grid, rho, mass):
    """Isotropic spherical Jeans: sigma_r^2(r) = (1/rho) int_r^inf rho G M/r'^2 dr'
    (G = 1). Integrates the tail inward from the outer edge of r_grid."""
    integrand = rho * mass / r_grid**2
    # cumulative integral from the far edge inward
    tail_rev = cumulative_trapezoid(integrand[::-1], -r_grid[::-1], initial=0.0)
    tail = tail_rev[::-1]
    sigma2 = np.where(rho > 1e-300, tail / rho, 0.0)
    return np.interp(r_query, r_grid, sigma2)


def _sphere_directions(n, rng):
    cos_t = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    sin_t = np.sqrt(1.0 - cos_t**2)
    return np.column_stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])


def plummer(n, rng):
    x1 = rng.uniform(0.0, 1.0, n)
    r = np.minimum(1.0 / np.sqrt(x1 ** (-2.0 / 3.0) - 1.0), 20.0)
    pos = _sphere_directions(n, rng) * r[:, None]

    # Exact Plummer DF: reject-sample q = v / v_esc from g(q) = q^2 (1-q^2)^{7/2}.
    q = np.empty(n)
    filled = 0
    while filled < n:
        cand = rng.uniform(0.0, 1.0, n)
        g = rng.uniform(0.0, 0.1, n)
        ok = g < cand**2 * (1.0 - cand**2) ** 3.5
        take = min(n - filled, ok.sum())
        q[filled:filled + take] = cand[ok][:take]
        filled += take
    v_esc = np.sqrt(2.0) * (1.0 + r**2) ** -0.25
    speed = q * v_esc
    vel = _sphere_directions(n, rng) * speed[:, None]
    return pos, vel


def hernquist(n, rng):
    x1 = rng.uniform(0.0, 1.0, n)
    sq = np.sqrt(x1)
    r = np.minimum(sq / (1.0 - sq), 40.0)
    pos = _sphere_directions(n, rng) * r[:, None]

    rr = np.logspace(-4, np.log10(200.0), 4096)
    rho = (1.0 / (2.0 * np.pi)) / (rr * (rr + 1.0) ** 3)
    mass = rr**2 / (rr + 1.0) ** 2
    sigma = np.sqrt(np.clip(_jeans_sigma2(r, rr, rho, mass), 0.0, None))
    vel = rng.normal(size=(n, 3)) * sigma[:, None]
    return pos, vel


def king(n, w0, rng):
    def g(W):
        W = np.clip(W, 0.0, None)
        return np.where(W > 0.0, np.exp(W) * erf(np.sqrt(W)) - np.sqrt(4.0 * W / np.pi) * (1.0 + 2.0 * W / 3.0), 0.0)

    g0 = float(g(np.array([w0]))[0])

    def rhs(x, y):
        W, Wp = y
        acc = -9.0 * float(g(np.array([W]))[0]) / g0 - (2.0 / x) * Wp if x > 1e-12 else -9.0 * 1.0
        return [Wp, acc]

    def hit_zero(x, y):
        return y[0]

    hit_zero.terminal = True
    hit_zero.direction = -1

    x_eps = 1e-4
    sol = solve_ivp(rhs, [x_eps, 1e4], [w0 - 1.5 * x_eps**2, -3.0 * x_eps],
                    events=hit_zero, max_step=2e-3, dense_output=True, rtol=1e-8, atol=1e-10)
    xt = sol.t[-1]
    xs = np.linspace(x_eps, xt, 4000)
    Ws = sol.sol(xs)[0]
    rho_shape = g(Ws) / g0
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rho_shape[1:] * xs[1:] ** 2 + rho_shape[:-1] * xs[:-1] ** 2) * np.diff(xs))])
    cum /= cum[-1]

    u = rng.uniform(0.0, 1.0, n)
    r = np.interp(u, cum, xs)
    pos = _sphere_directions(n, rng) * r[:, None]

    mass_phys = cum  # totalMass = 1, cum is already M(<r)/M_total on the xs grid
    sigma = np.sqrt(np.clip(_jeans_sigma2(r, xs, np.maximum(rho_shape, 1e-30), mass_phys), 0.0, None))
    vel = rng.normal(size=(n, 3)) * sigma[:, None]
    return pos, vel


def virial_ratio(pos, vel, mass, eps=1e-3):
    """2T/|W| for the WHOLE cloud. W is the full O(N^2) pair sum, computed
    row-chunked so the (N,N) distance matrix is never materialized (a naive
    subsample would break this: pair count scales as N^2, so PE among a
    random half is ~1/4 of the total, not 1/2 -- which silently doubles the
    ratio)."""
    n = len(pos)
    T = 0.5 * mass * np.sum(vel**2)
    W = 0.0
    chunk = 512
    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        d = pos[i0:i1, None, :] - pos[None, :, :]
        r = np.sqrt(np.sum(d**2, axis=-1) + eps**2)
        for k in range(i0, i1):
            r[k - i0, : k + 1] = np.inf  # keep only j > i pairs, drop self
        W -= mass * mass * np.sum(1.0 / r)
    return 2.0 * T / abs(W)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["plummer", "hernquist", "king"])
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--w0", type=float, default=6.0, help="King central dimensionless potential")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output .bin path")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.model == "plummer":
        pos, vel = plummer(args.n, rng)
    elif args.model == "hernquist":
        pos, vel = hernquist(args.n, rng)
    else:
        pos, vel = king(args.n, args.w0, rng)

    mass = 1.0 / args.n
    pos -= np.average(pos, axis=0, weights=np.full(args.n, mass))
    vel -= np.average(vel, axis=0, weights=np.full(args.n, mass))

    rec = np.zeros((args.n, 7), dtype=np.float32)
    rec[:, 0:3] = pos
    rec[:, 3] = mass
    rec[:, 4:7] = vel
    rec.tofile(args.out)

    r = np.linalg.norm(pos, axis=1)
    r_half = np.median(r)
    mean_spacing = r_half / args.n ** (1.0 / 3.0)
    eps = max(0.01, 1.5 * mean_spacing)
    vr = virial_ratio(pos, vel, mass)
    t_cross = 2.0 * np.pi
    dt = t_cross / 800.0

    print(f"wrote {args.n} particles -> {args.out}")
    print(f"  half-mass radius   {r_half:.4f}")
    print(f"  virial 2T/|W|      {vr:.3f}")
    print(f"  suggested softening.eps  {eps:.4f}")
    print(f"  suggested time.dt        {dt:.5f}   (t_cross ~ {t_cross:.2f})")
    print("deck: scenario.type = \"ic_file\";  scenario.ic_file = \"" + args.out + "\"")


if __name__ == "__main__":
    main()
