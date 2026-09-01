#!/usr/bin/env python3
"""Radial distribution function g(r) from a headless 04_molecular_dynamics run.

    python plot_rdf.py <results_dir> [--frame -1] [--bins 100] [--rmax 5.0] [--out rdf.png]

Computes g(r) directly from the raw pos_XXXX.npy positions of one frame
(default: the last, i.e. presumed-equilibrated) using the run's own
box_length (from diagnostics.csv) and the minimum-image convention -- an
independent computation from MDSystem::ComputeRDF (which the interactive/
ImGui path uses live), so this is also a cross-check that the two agree, not
just a plotting convenience.

For a liquid state point this should show the well-known LJ liquid
structure: a sharp first peak near r ~= 1.1*sigma, decaying oscillations at
the second/third neighbor shells, and g(r) -> 1 at large r -- this script
reports the first-peak location/height as a qualitative sanity check, not a
fitted comparison to a specific literature number.
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
    sys.exit("plot_rdf: needs numpy + matplotlib")


def min_image(d, L):
    return d - L * np.round(d / L)


def compute_rdf(pos, L, nbins, rmax):
    n = len(pos)
    rmax = min(rmax, 0.49 * L)
    # O(N^2) all-pairs -- fine for the few-hundred/thousand-particle decks
    # this project ships; not meant to scale to MDSystem's linked-cell N.
    diffs = pos[:, None, :] - pos[None, :, :]
    diffs = min_image(diffs, L)
    r = np.sqrt(np.sum(diffs**2, axis=-1))
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    r = r[r < rmax]

    hist, edges = np.histogram(r, bins=nbins, range=(0.0, rmax))
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1] - edges[0]
    V = L**3
    total_pairs = 0.5 * n * (n - 1)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = total_pairs * (shell_vol / V)
    g = np.divide(hist, expected, out=np.zeros_like(hist, dtype=float), where=expected > 1e-12)
    return r_mid, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--frame", type=int, default=-1, help="frame index (default: last)")
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--rmax", type=float, default=5.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    frame_ids = sorted(
        os.path.basename(p).split("_")[-1].split(".")[0]
        for p in glob.glob(os.path.join(d, "frames", "pos_*.npy"))
    )
    frame_id = frame_ids[args.frame]
    pos = np.load(os.path.join(d, "frames", f"pos_{frame_id}.npy"))

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    idx = np.where(data["step"] == float(frame_id))[0]
    L = data["box_length"][idx[0]] if len(idx) else data["box_length"][args.frame]
    t_val = data["t"][idx[0]] if len(idx) else data["t"][args.frame]

    r, g = compute_rdf(pos, L, args.bins, args.rmax)

    peak_idx = np.argmax(g)
    print(f"frame {frame_id} (t={t_val:.3f}, L={L:.4f}, N={len(pos)}):")
    print(f"  first peak: r={r[peak_idx]:.3f}  g(r)={g[peak_idx]:.3f}"
          f"  (LJ liquids typically peak near r~1.1*sigma)")
    print(f"  g(r) at r={r[-1]:.2f} (largest bin): {g[-1]:.3f}  (should be ~1 for a bulk liquid)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(r, g)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("r / sigma"); ax.set_ylabel("g(r)")
    ax.set_title(f"{os.path.basename(os.path.abspath(d))}  (frame {frame_id}, t={t_val:.2f})")
    fig.tight_layout()
    out = args.out or os.path.join(d, "rdf.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
