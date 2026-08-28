"""
Overlay two headless encounter runs' diagnostics.csv (same star/orbit,
different black-hole force law) to compare, e.g., a Newtonian point-mass
pericenter passage against a Paczynski-Wiita one.

Usage:
    python compare_encounters.py <dir1> <dir2> [--labels "A,B"] [--out FILE]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dirs", nargs=2)
    ap.add_argument("--labels", default=None, help="comma-separated, defaults to the dir basenames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else [os.path.basename(os.path.abspath(d)) for d in args.dirs]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for d, label in zip(args.dirs, labels):
        data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
        t = data["t"]
        ax[0].plot(t, data["kinetic"], label=label)
        ax[1].plot(t, data["potential_bh"], label=label)
        rmins = []
        frames_dir = os.path.join(d, "frames")
        ids = sorted({fn.split("_")[-1].split(".")[0] for fn in os.listdir(frames_dir) if fn.startswith("pos_mass_")})
        for fid in ids:
            pm = np.load(os.path.join(frames_dir, f"pos_mass_{fid}.npy"))
            rmins.append(np.linalg.norm(pm[:, :3], axis=1).min())
        ax[2].plot(t[: len(rmins)], rmins, label=label)

    ax[0].set_title("kinetic energy"); ax[0].set_xlabel("t"); ax[0].legend(fontsize=8)
    ax[1].set_title("star-BH potential energy"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=8)
    ax[2].set_title("minimum particle-BH separation"); ax[2].set_xlabel("t"); ax[2].legend(fontsize=8)
    fig.suptitle(" vs. ".join(labels))
    fig.tight_layout()
    out = args.out or "compare_encounters.png"
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
