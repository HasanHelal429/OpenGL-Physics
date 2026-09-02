"""
Validation plot for a headless 08_compressible_fluid 2D Riemann-problem run
(decks/riemann2d_config3.toml): the final-frame density field as a
heatmap. There's no exact solution for a genuinely 2D Riemann problem, so
this is a qualitative check -- density should show four elementary waves
emanating from the initial quadrant boundaries, meeting near the center in
the characteristic "double shear layer around a central vortical/mixing
region" pattern published for this configuration (Kurganov & Tadmor 2002;
Liska & Wendroff 2003's scheme-comparison figures), with no negative
density/pressure or grid-scale checkerboard noise anywhere.

Usage:
    python plot_riemann2d.py <results_dir> [--out FILE]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    nx, ny = manifest["grid"]["nx"], manifest["grid"]["ny"]
    lx, ly = manifest["grid"]["lx"], manifest["grid"]["ly"]

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("rho_")})
    last = frame_ids[-1]
    rho = np.load(os.path.join(frames_dir, f"rho_{last}.npy"))
    p = np.load(os.path.join(frames_dir, f"p_{last}.npy"))

    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    print(f"t={t_end:.4f}  rho range=[{rho.min():.4f}, {rho.max():.4f}]  p range=[{p.min():.4f}, {p.max():.4f}]")
    if rho.min() <= 0 or p.min() <= 0:
        print("WARNING: non-physical (non-positive) density or pressure present")

    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    extent = [0, lx, 0, ly]
    im0 = ax[0].imshow(rho, origin="lower", extent=extent, cmap="viridis")
    ax[0].set_title(f"density (t={t_end:.3f})")
    fig.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(p, origin="lower", extent=extent, cmap="magma")
    ax[1].set_title("pressure")
    fig.colorbar(im1, ax=ax[1])
    for a in ax:
        a.set_xlabel("x"); a.set_ylabel("y")

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
