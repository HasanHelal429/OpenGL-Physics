"""
Plot a headless self-consistent run of 11_retarded_fields:

  - charge trajectories x_c(t), y_c(t) from diagnostics.csv
  - separation(t) with the analytic inspiral_ref overlay
  - the energy budget KE + PE_interaction + E_radiated vs its t = 0 value

Usage:
    python plot_inspiral.py <results_dir> [--q 1.5 --m 1] [--out FILE]
"""

import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--q", type=float, default=1.5)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    dg = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = dg["t"]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))

    # trajectories
    for c, col in ((0, "#d98f00"), (1, "#3a7bd5")):
        xk, yk = f"x{c}", f"y{c}"
        if xk in dg.dtype.names:
            ax[0].plot(dg[xk], dg[yk], color=col, lw=1.0, label=f"charge {c}")
            ax[0].plot(dg[xk][-1], dg[yk][-1], "o", color=col, ms=5)
    ax[0].set_aspect("equal")
    ax[0].set_title("trajectories"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
    ax[0].legend(fontsize=8)

    # separation vs analytic
    sep = dg["separation"]
    ax[1].plot(t, sep, "k", lw=1.5, label="simulation")
    try:
        from inspiral_ref import integrate
        tr, dr, _, _ = integrate(sep[0], args.q, args.m, t[-1], 4000, half=True)
        ax[1].plot(tr, dr, "r--", lw=1.2, label="analytic (half power)")
    except Exception as e:
        print("inspiral_ref overlay skipped:", e)
    ax[1].set_title("separation"); ax[1].set_xlabel("t"); ax[1].set_ylabel("d")
    ax[1].legend(fontsize=8)

    # energy budget
    ke, pe, er = dg["KE"], dg["PE_interaction"], dg["E_radiated"]
    tot = ke + pe + er
    e0 = tot[0]
    ax[2].plot(t, ke - ke[0], label="d KE")
    ax[2].plot(t, pe - pe[0], label="d PE")
    ax[2].plot(t, er, label="E_radiated")
    ax[2].plot(t, tot - e0, "k", lw=2, label="budget error")
    ax[2].set_title("energy budget"); ax[2].set_xlabel("t"); ax[2].set_ylabel("energy - E(0)")
    ax[2].legend(fontsize=8)

    fig.tight_layout()
    out = args.out or os.path.join(d, "inspiral.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)
    print(f"final budget error / |E0| = {abs(tot[-1] - e0) / abs(e0):.3e}")


if __name__ == "__main__":
    main()
