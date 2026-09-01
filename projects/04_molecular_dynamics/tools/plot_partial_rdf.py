#!/usr/bin/env python3
"""Partial radial distribution functions g_AA(r), g_BB(r), g_AB(r) for a
headless binary-mixture 04_molecular_dynamics run (decks/glass_kob_andersen.toml).

    python plot_partial_rdf.py <results_dir> [--frame -1] [--bins 100] [--rmax 3.0] [--out partial_rdf.png]

Computed directly from the raw pos_XXXX.npy positions of one frame plus the
static species_0000.npy labels (species is written only once, on frame 0 --
see MDSim::Snapshot), independently of MDSystem::ComputePartialRDF (which
the C++ side's rdf_peak_aa/bb/ab diagnostic columns come from) -- this
script's console output is meant to be compared against those columns in
diagnostics.csv as a cross-check, not just as a plot.

For the Kob-Andersen parameters (sigma_AA=1.0, sigma_BB=0.88, sigma_AB=0.80,
eps_AA=1.0, eps_BB=0.50, eps_AB=1.50), expect g_AA's first peak near
r~1.0-1.1 (close to the single-species liquid), g_BB's shifted to smaller r
(smaller sigma_BB), and g_AB depressed/broadened relative to both -- the
non-additive eps_AB=1.5 (deeper than either self-interaction) and
sigma_AB=0.80 (smaller than the arithmetic mean of sigma_AA/sigma_BB) are
exactly the deliberate frustration that keeps this mixture from
crystallizing, and should show up as A-B pairs packing measurably closer/
differently than a naive Lorentz-Berthelot mixture would.
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
    sys.exit("plot_partial_rdf: needs numpy + matplotlib")


def min_image(d, L):
    return d - L * np.round(d / L)


def partial_rdf(pos, species, L, nbins, rmax, sa, sb):
    rmax = min(rmax, 0.49 * L)
    idx_a = np.where(species == sa)[0]
    idx_b = np.where(species == sb)[0]
    if len(idx_a) == 0 or len(idx_b) == 0:
        edges = np.linspace(0.0, rmax, nbins + 1)
        return 0.5 * (edges[:-1] + edges[1:]), np.zeros(nbins)

    if sa == sb:
        pa = pos[idx_a]
        diffs = min_image(pa[:, None, :] - pa[None, :, :], L)
        r = np.sqrt(np.sum(diffs**2, axis=-1))
        iu = np.triu_indices(len(idx_a), k=1)
        r = r[iu]
        total_pairs = 0.5 * len(idx_a) * (len(idx_a) - 1)
    else:
        pa, pb = pos[idx_a], pos[idx_b]
        diffs = min_image(pa[:, None, :] - pb[None, :, :], L)
        r = np.sqrt(np.sum(diffs**2, axis=-1)).ravel()
        total_pairs = len(idx_a) * len(idx_b)

    r = r[r < rmax]
    hist, edges = np.histogram(r, bins=nbins, range=(0.0, rmax))
    r_mid = 0.5 * (edges[:-1] + edges[1:])
    V = L**3
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    expected = total_pairs * (shell_vol / V)
    g = np.divide(hist, expected, out=np.zeros_like(hist, dtype=float), where=expected > 1e-12)
    return r_mid, g


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--frame", type=int, default=-1)
    ap.add_argument("--bins", type=int, default=100)
    ap.add_argument("--rmax", type=float, default=3.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    frame_ids = sorted(
        os.path.basename(p).split("_")[-1].split(".")[0]
        for p in glob.glob(os.path.join(d, "frames", "pos_*.npy"))
    )
    frame_id = frame_ids[args.frame]
    pos = np.load(os.path.join(d, "frames", f"pos_{frame_id}.npy"))
    species = np.load(os.path.join(d, "frames", "species_0000.npy")).ravel()

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    idx = np.where(data["step"] == float(frame_id))[0]
    L = data["box_length"][idx[0]] if len(idx) else data["box_length"][args.frame]

    n_a = int(np.sum(species == 0))
    n_b = int(np.sum(species == 1))
    print(f"frame {frame_id}: N_A={n_a}  N_B={n_b}  (fraction_b={n_b / (n_a + n_b):.3f})")

    labels = [("A-A", 0, 0), ("B-B", 1, 1), ("A-B", 0, 1)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, sa, sb in labels:
        r, g = partial_rdf(pos, species, L, args.bins, args.rmax, sa, sb)
        peak_r = r[np.argmax(g)]
        peak_g = g.max()
        print(f"  g_{name}: first peak r={peak_r:.3f}  g={peak_g:.3f}"
              f"  (cross-check against rdf_peak_{name.lower().replace('-', '')} in diagnostics.csv)")
        ax.plot(r, g, label=f"g_{name}(r)")

    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xlabel("r / sigma_AA"); ax.set_ylabel("g(r)")
    ax.set_title(f"{os.path.basename(os.path.abspath(d))}  (frame {frame_id})")
    ax.legend()
    fig.tight_layout()
    out = args.out or os.path.join(d, "partial_rdf.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
