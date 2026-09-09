"""Trajectory of a small-N run (kepler / lagrange) from the pos frames, with
apoapsis/periapsis annotated for the two-body case.

Usage:
    python plot_orbit.py <results_dir> [--out orbit.png]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default="orbit.png")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.results_dir, "frames", "pos_*.npy")))
    frames = [np.load(f) for f in files]  # each (N, D)
    n = frames[0].shape[0]
    traj = np.stack(frames, axis=0)  # (T, N, D)

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(n):
        ax.plot(traj[:, i, 0], traj[:, i, 1], lw=0.8)
        ax.plot(traj[0, i, 0], traj[0, i, 1], "o", ms=4)

    if n == 2:
        sep = np.linalg.norm(traj[:, 1] - traj[:, 0], axis=1)
        ax.set_title(f"two-body: r_min={sep.min():.4f}  r_max={sep.max():.4f}")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
