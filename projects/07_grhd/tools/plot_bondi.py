"""
Validation plot for a headless 07_grhd Schwarzschild (Bondi accretion) run:
density/velocity/pressure profiles at t=0 vs. the final frame, overlaid on
the exact analytic Bondi solution, plus the accretion-rate (f*v*D)
uniformity/conservation diagnostic over time.

Usage:
    python plot_bondi.py <results_dir> [--out FILE]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None

from bondi_analytic import BondiSolution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    n = manifest["grid"]["nx"]
    r_min = deck["grid"]["r_min"]
    r_max = deck["grid"]["r_max"]
    dr = (r_max - r_min) / n
    r = r_min + (np.arange(n) + 0.5) * dr
    M = deck["physics"]["M"]
    gamma = deck["physics"]["gamma"]

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("rho_")})
    first, last = frame_ids[0], frame_ids[-1]
    rho0 = np.load(os.path.join(frames_dir, f"rho_{first}.npy")).ravel()
    v0 = np.load(os.path.join(frames_dir, f"v_{first}.npy")).ravel()
    p0 = np.load(os.path.join(frames_dir, f"p_{first}.npy")).ravel()
    rho1 = np.load(os.path.join(frames_dir, f"rho_{last}.npy")).ravel()
    v1 = np.load(os.path.join(frames_dir, f"v_{last}.npy")).ravel()
    p1 = np.load(os.path.join(frames_dir, f"p_{last}.npy")).ravel()

    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    r_c = deck["bondi"]["r_c"]
    sol = BondiSolution(M=M, r_c=r_c, gamma=gamma)
    rho_ex, v_ex, p_ex = sol.profile(r)

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()

    ax[0].plot(r, rho_ex, "k-", lw=1.5, label="analytic Bondi")
    ax[0].plot(r, rho0, "o", ms=3, label=f"t={0:.1f} (IC)")
    ax[0].plot(r, rho1, "s", ms=3, label=f"t={t_end:.1f}")
    ax[0].set_title("density"); ax[0].set_xlabel("r"); ax[0].legend(fontsize=8)

    ax[1].plot(r, v_ex, "k-", lw=1.5, label="analytic Bondi")
    ax[1].plot(r, v0, "o", ms=3, label="t=0 (IC)")
    ax[1].plot(r, v1, "s", ms=3, label=f"t={t_end:.1f}")
    ax[1].set_title("radial velocity"); ax[1].set_xlabel("r"); ax[1].legend(fontsize=8)

    ax[2].plot(r, p_ex, "k-", lw=1.5, label="analytic Bondi")
    ax[2].plot(r, p0, "o", ms=3, label="t=0 (IC)")
    ax[2].plot(r, p1, "s", ms=3, label=f"t={t_end:.1f}")
    ax[2].set_title("pressure"); ax[2].set_xlabel("r"); ax[2].legend(fontsize=8)

    ax[3].plot(t, data["mdot_spread"])
    ax[3].set_title("accretion-rate spread\n(max-min)/mean of f*v*D across the grid")
    ax[3].set_xlabel("t"); ax[3].set_ylabel("relative spread")

    rho_err = np.max(np.abs(rho1 - rho_ex)) / np.max(rho_ex)
    v_err = np.max(np.abs(v1 - v_ex)) / np.max(np.abs(v_ex))
    print(f"final-frame max relative error vs. analytic: rho={rho_err:.4f}  v={v_err:.4f}")

    fig.suptitle(f"{os.path.basename(os.path.abspath(d))}  (r_c={r_c}, M={M}, final t={t_end:.1f})")
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
