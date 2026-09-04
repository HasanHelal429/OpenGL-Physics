"""
Analytic Larmor / dipole radiation reference (c = eps0 = 1).

  - non-relativistic:  dP/dOmega ~ sin^2(theta),   P = q^2 a^2 / (6 pi)
  - relativistic, a parallel to v (linear accel):
        dP/dOmega ~ sin^2 th / (1 - beta cos th)^5
  - relativistic, a perp to v (circular):
        dP/dOmega ~ [ (1 - beta cos th)^2 - (1 - beta^2) sin^2 th cos^2 ph ]
                    / (1 - beta cos th)^5

Prints P and the forward half-power half-angle, and (with --out) plots the
polar pattern. Use as the overlay reference for plot_pattern.py.

Usage:
    python larmor_ref.py [--beta 0.6] [--mode perp|par|nonrel] [--out FILE]
"""

import argparse
import numpy as np


def pattern(theta, beta, mode):
    ct, st = np.cos(theta), np.sin(theta)
    if mode == "nonrel" or beta == 0.0:
        return st**2
    kappa = 1.0 - beta * ct
    if mode == "par":
        return st**2 / kappa**5
    # perp (circular), averaged over the azimuth ph about the velocity
    return ((kappa**2) - 0.5 * (1.0 - beta**2) * st**2) / kappa**5


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--mode", choices=["perp", "par", "nonrel"], default="perp")
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--a", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    th = np.linspace(0, np.pi, 4001)
    g = pattern(th, args.beta, args.mode)
    g /= g.max()
    half = th[np.argmax(g < 0.5)] if np.any(g < 0.5) else np.pi
    gamma = 1.0 / np.sqrt(1.0 - args.beta**2)

    P = args.q**2 * args.a**2 / (6.0 * np.pi)
    if args.mode == "par":
        P *= gamma**6
    elif args.mode == "perp":
        P *= gamma**4
    print(f"beta={args.beta}  gamma={gamma:.3f}  mode={args.mode}")
    print(f"  P = {P:.6e}   (nonrel Larmor x gamma^{'6' if args.mode=='par' else '4' if args.mode=='perp' else '0'})")
    print(f"  forward half-power half-angle = {half:.4f} rad   (1/gamma = {1/gamma:.4f})")

    if args.out:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="polar")
        full = np.concatenate([th, th + np.pi])
        gg = np.concatenate([g, g[::-1]])
        ax.plot(full, gg, lw=1.6)
        ax.set_title(f"dP/dOmega  (beta={args.beta}, {args.mode})", pad=18)
        fig.savefig(args.out, dpi=130)
        print("wrote", args.out)


if __name__ == "__main__":
    main()
