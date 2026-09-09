"""Per-particle force-error histogram, Barnes-Hut vs Direct, over a sweep of
opening angles -- validates the quadrupole walk (Phase 5) and, later, the FMM
solvers (Phase 6/7) against the exact O(N^2) reference.

This reads a run's own solver-vs-solver comparison if the deck records one;
absent that, it's a template for a benchmark-style sweep -- run the app's
in-app "Benchmark solvers" panel, or extend a deck's Snapshot to also record
a reference-solver run, and point this at both.

Usage:
    python plot_force_error.py <results_dir_reference> <results_dir_test> [--out force_error.png]
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_last_frame(results_dir):
    files = sorted(glob.glob(os.path.join(results_dir, "frames", "pos_*.npy")))
    return np.load(files[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_dir", help="Direct (or other exact/high-order) reference run")
    ap.add_argument("test_dir", help="run to measure force error for (requires matching accel_*.npy if present)")
    ap.add_argument("--out", default="force_error.png")
    a = ap.parse_args()

    # This compares final positions as a proxy when acceleration frames
    # aren't recorded; for a true force-error histogram, snapshot 'accel'
    # alongside 'pos'/'vel' in DeckSim::Snapshot and load it here instead.
    pref = load_last_frame(a.reference_dir)
    ptest = load_last_frame(a.test_dir)
    err = np.linalg.norm(ptest - pref, axis=1)
    ref_scale = np.linalg.norm(pref, axis=1)
    rel = err / np.maximum(ref_scale, 1e-12)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(np.maximum(rel, 1e-16)), bins=50)
    ax.set_xlabel("log10(relative position error, final frame)")
    ax.set_ylabel("count")
    ax.set_title(f"mean={rel.mean():.2e}  max={rel.max():.2e}  median={np.median(rel):.2e}")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
