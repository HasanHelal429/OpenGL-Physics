"""
Visualize a Fishbone-Moncrief equilibrium torus (tools/fishbone_moncrief.py):
a poloidal (r,theta) cross-section density map, in Cartesian
(x=r*sin(theta), z=r*cos(theta)) projection so the torus's actual shape is
visible, plus the equatorial and vertical (at r_center) density profiles.

Usage:
    python plot_fishbone_moncrief.py --M 1.0 --a 0.9 --gamma 1.3333 \
        --r_in 6.0 --r_center 10.0 [--out FILE]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from fishbone_moncrief import FishboneMoncriefTorus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=4.0 / 3.0)
    ap.add_argument("--r_in", type=float, required=True)
    ap.add_argument("--r_center", type=float, required=True)
    ap.add_argument("--r_max", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torus = FishboneMoncriefTorus(args.M, args.a, args.r_in, args.r_center, args.gamma)
    print(f"l={torus.l:.6f}  ln(h)@r_center={torus.ln_h_center:.6f}  K={torus.K:.6f}")

    r_max = args.r_max or 3.0 * args.r_center
    r_grid = np.linspace(args.r_in * 0.8, r_max, 140)
    theta_grid = np.linspace(0.4, np.pi - 0.4, 140)
    RHO = np.zeros((len(theta_grid), len(r_grid)))
    for i, th in enumerate(theta_grid):
        for j, r in enumerate(r_grid):
            rho, _, _ = torus.primitives(r, th)
            RHO[i, j] = rho

    R, TH = np.meshgrid(r_grid, theta_grid)
    X = R * np.sin(TH)
    Z = R * np.cos(TH)

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    pcm = ax[0].pcolormesh(X, Z, np.log10(np.maximum(RHO, 1e-6)), shading="auto", cmap="inferno")
    ax[0].pcolormesh(-X, Z, np.log10(np.maximum(RHO, 1e-6)), shading="auto", cmap="inferno")
    r_horizon = args.M + np.sqrt(max(args.M ** 2 - args.a ** 2, 0.0))
    circle = plt.Circle((0, 0), r_horizon, color="black")
    ax[0].add_patch(circle)
    ax[0].set_aspect("equal")
    ax[0].set_title(f"poloidal cross-section, log10(rho)\na={args.a}M, r_in={args.r_in}, r_center={args.r_center}")
    ax[0].set_xlabel("x = r sin(theta)"); ax[0].set_ylabel("z = r cos(theta)")
    fig.colorbar(pcm, ax=ax[0], fraction=0.046)

    r_eq = np.linspace(args.r_in * 0.9, r_max, 200)
    rho_eq = np.array([torus.primitives(r, np.pi / 2.0)[0] for r in r_eq])
    ax[1].plot(r_eq, rho_eq)
    ax[1].axvline(args.r_in, color="gray", ls="--", lw=1, label="r_in")
    ax[1].axvline(args.r_center, color="k", ls=":", lw=1, label="r_center")
    ax[1].set_title("equatorial density profile"); ax[1].set_xlabel("r"); ax[1].set_ylabel("rho")
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    out = args.out or "fishbone_moncrief.png"
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
