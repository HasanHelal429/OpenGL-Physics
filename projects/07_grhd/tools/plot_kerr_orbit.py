"""
Validation plot for a headless 07_grhd Kerr-equatorial circular-orbit run:
v_r/v_phi/rho profiles at t=0 vs. the final frame (should stay close to
the exact circular-orbit v_phi(r), and v_r should stay small), plus
max|v_r| over time (the primary "did it stay circular" diagnostic).

Usage:
    python plot_kerr_orbit.py <results_dir> [--out FILE]
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

from kerr_equatorial_ref import circular_orbit_v_phi_hat


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
    M = deck["physics"]["M"]
    a = deck["physics"]["a"]
    dr = (r_max - r_min) / n
    r = r_min + (np.arange(n) + 0.5) * dr
    v_phi_exact = np.array([circular_orbit_v_phi_hat(ri, M, a) for ri in r])

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("rho_")})
    first, last = frame_ids[0], frame_ids[-1]
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    def load(field, fid):
        return np.load(os.path.join(frames_dir, f"{field}_{fid}.npy")).ravel()

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()

    ax[0].plot(r, load("v_r", first), "o", ms=3, label=f"t=0")
    ax[0].plot(r, load("v_r", last), "s", ms=3, label=f"t={t_end:.0f}")
    ax[0].axhline(0.0, color="k", lw=1)
    ax[0].set_title("radial velocity (should stay ~0)"); ax[0].set_xlabel("r"); ax[0].legend(fontsize=8)

    ax[1].plot(r, v_phi_exact, "k-", lw=1.5, label="exact circular orbit")
    ax[1].plot(r, load("v_phi", first), "o", ms=3, label="t=0")
    ax[1].plot(r, load("v_phi", last), "s", ms=3, label=f"t={t_end:.0f}")
    ax[1].set_title("azimuthal velocity"); ax[1].set_xlabel("r"); ax[1].legend(fontsize=8)

    ax[2].plot(r, load("rho", first), "o", ms=3, label="t=0")
    ax[2].plot(r, load("rho", last), "s", ms=3, label=f"t={t_end:.0f}")
    ax[2].set_title("density (no equilibrium profile assumed --\nsee README's Fishbone-Moncrief note)")
    ax[2].set_xlabel("r"); ax[2].legend(fontsize=8)

    ax[3].plot(data["t"], data["max_abs_vr"])
    ax[3].set_title("max|v_r| across the grid vs. time\n(the main stays-circular diagnostic)")
    ax[3].set_xlabel("t")

    v_phi_err = np.max(np.abs(load("v_phi", last) - v_phi_exact)) / np.max(np.abs(v_phi_exact))
    print(f"final-frame max relative v_phi error vs. exact circular orbit: {v_phi_err:.4f}")
    print(f"final-frame max|v_r|: {np.max(np.abs(load('v_r', last))):.4f}")

    fig.suptitle(f"{os.path.basename(os.path.abspath(d))}  (a={a}M, final t={t_end:.0f})")
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
