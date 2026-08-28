"""
Build SPH initial conditions for a polytropic star: solve the Lane-Emden
equation (lane_emden.py), Monte-Carlo sample N equal-mass particle positions
from the resulting density profile (inverse-CDF on enclosed mass), and write
a raw binary file the C++ simulation (../src/TdeSim.cpp) loads directly.

File format: N records of 7 float32 each, x,y,z,mass,vx,vy,vz (velocities
are zero -- the star starts at rest in its own frame; any residual jitter
from finite-N Monte Carlo sampling is exactly what the relaxation phase
(decks/star_relax_*.toml) is for).

Usage:
    python make_star_ic.py --n 4000 --index 1.5 --out ../ic/star_n1.5.bin
"""

import argparse

import numpy as np

from lane_emden import build_polytrope


def sample_star(profile, n_particles, rng):
    r_grid, m_grid = profile["r"], profile["m"]
    u = rng.uniform(0.0, m_grid[-1], n_particles)
    r = np.interp(u, m_grid, r_grid)

    cos_theta = rng.uniform(-1.0, 1.0, n_particles)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    return np.column_stack([x, y, z])


def validate_sampling(profile, positions, n_bins=24):
    """Bin the sampled particles radially and compare to the analytic
    density profile -- a real check on the sampler, not just a smoke test."""
    r = np.linalg.norm(positions, axis=1)
    m_per_particle = profile["M_star"] / len(r)
    edges = np.linspace(0.0, profile["R_star"], n_bins + 1)
    counts, _ = np.histogram(r, bins=edges)
    shell_volume = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    rho_empirical = counts * m_per_particle / shell_volume
    r_mid = 0.5 * (edges[1:] + edges[:-1])
    rho_analytic = np.interp(r_mid, profile["r"], profile["rho"])

    # Skip the innermost/outermost bins: low particle count there makes
    # Poisson shot noise dominate, not sampler error.
    keep = (counts > 20) & (r_mid < 0.9 * profile["R_star"])
    rel_err = np.abs(rho_empirical[keep] - rho_analytic[keep]) / rho_analytic[keep]
    return rel_err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=4000, help="particle count")
    ap.add_argument("--index", type=float, default=1.5, help="polytropic index n")
    ap.add_argument("--mass", type=float, default=1.0, help="star mass (code units)")
    ap.add_argument("--radius", type=float, default=1.0, help="star radius (code units)")
    ap.add_argument("--G", type=float, default=1.0, help="gravitational constant (code units)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output .bin path")
    args = ap.parse_args()

    profile = build_polytrope(args.index, M_star=args.mass, R_star=args.radius, G=args.G)
    rng = np.random.default_rng(args.seed)
    positions = sample_star(profile, args.n, rng)

    rel_err = validate_sampling(profile, positions)
    print(f"Lane-Emden n={args.index}: xi1={profile['xi1']:.6f}  K={profile['K']:.6f}  "
          f"Gamma={profile['Gamma']:.6f}  rho_c={profile['rho_c']:.6f}")
    print(f"radial density sampling check: median rel. err={np.median(rel_err):.3f}  "
          f"max rel. err={np.max(rel_err):.3f}  (Poisson noise at N={args.n}, not a bias check)")

    mean_spacing = args.radius / args.n ** (1.0 / 3.0)
    print(f"suggested sph.h_init ~ {2.5 * mean_spacing:.4f}   suggested gravity.softening ~ {0.3 * mean_spacing:.4f}")
    print(f"(mean interparticle spacing ~ {mean_spacing:.4f}, from R/N^(1/3). Softening well below the")
    print(f" adaptive smoothing length the run will converge to in the core -- softening comparable to")
    print(f" or larger than the local SPH resolution measurably weakens self-gravity there, see README.md)")

    mass_col = np.full(args.n, args.mass / args.n, dtype=np.float32)
    vel_cols = np.zeros((args.n, 3), dtype=np.float32)
    record = np.column_stack([positions.astype(np.float32), mass_col, vel_cols]).astype(np.float32)
    record.tofile(args.out)
    print(f"wrote {args.n} particles -> {args.out} ({record.nbytes} bytes)")


if __name__ == "__main__":
    main()
