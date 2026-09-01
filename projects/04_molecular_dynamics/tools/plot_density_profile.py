#!/usr/bin/env python3
"""Density profile rho(z) at a few snapshots for a slab-mode
04_molecular_dynamics run (decks/slab_sublimation.toml).

    python plot_density_profile.py <results_dir> [--frames 0 0.25 0.5 1.0] [--out density_profile.png]

--frames are FRACTIONS of the run (0=first frame, 1=last), not frame
indices, so this works regardless of how many frames a given run has.
Reads frames/density_profile_XXXX.npy (see MDSim::Snapshot -- one bin count
converted to a local number density, over the whole box, only written when
system.slab_fraction > 0) plus vapor_fraction from diagnostics.csv.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("plot_density_profile: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--frames", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    profile_files = sorted(glob.glob(os.path.join(d, "frames", "density_profile_*.npy")))
    if not profile_files:
        sys.exit("no frames/density_profile_*.npy found -- was this deck run with system.slab_fraction > 0?")
    frame_ids = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in profile_files]

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    L = data["box_length"][frame_ids[-1]]
    nbins = np.load(profile_files[0]).shape[0]
    z = (np.arange(nbins) + 0.5) * (L / nbins)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_avail = len(profile_files)
    for frac in args.frames:
        k = int(round(frac * (n_avail - 1)))
        fid = frame_ids[k]
        profile = np.load(profile_files[k]).ravel()
        t = data["t"][fid]
        vf = data["vapor_fraction"][fid]
        ax.plot(z, profile, marker="o", ms=3, label=f"t={t:.1f} (vapor_fraction={vf:.3f})")
        print(f"frame {fid} (t={t:.2f}): vapor_fraction={vf:.4f}  peak_density={profile.max():.3f}")

    ax.set_xlabel("z"); ax.set_ylabel("local number density")
    ax.set_title(f"{os.path.basename(os.path.abspath(d))}: density profile rho(z)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(d, "density_profile.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
