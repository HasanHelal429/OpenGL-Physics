"""
Quick "did it converge" panel for a headless 01_hartree_fock run:

  left    E_total per SCF iteration
  middle  |dE| and integrated |d(rho)| per iteration (log-y)
  right   converged orbital-energy barcode (one stick per (n,l), colored by l)

Usage:
    python plot_diagnostics.py <results_dir> [--out FILE.png]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hfio import load_manifest, n_frames, orbital_labels, label_text, frame_path, L_SYMBOL

L_COLOR = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868", 3: "#c44e52", 4: "#8172b3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    man = load_manifest(d)
    di = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    step = np.atleast_1d(di["step"])
    e_total = np.atleast_1d(di["e_total"])
    de = np.atleast_1d(di["de"])
    dn = np.atleast_1d(di["dn"])

    labels = orbital_labels(d)
    eps = np.load(frame_path(d, "eps", n_frames(d) - 1)).ravel()

    fig, (axE, axC, axB) = plt.subplots(1, 3, figsize=(13, 4), dpi=130)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.86, bottom=0.15, wspace=0.3)

    axE.plot(step, e_total, "-o", ms=3, color="#333")
    axE.set_xlabel("SCF iteration")
    axE.set_ylabel(r"$E_{\rm total}$  (Ha)")
    axE.set_title(f"total energy  ({e_total[np.isfinite(e_total)][-1]:.5f} Ha)")

    axC.semilogy(step, np.abs(de), "-o", ms=3, label=r"$|\Delta E|$")
    axC.semilogy(step, np.abs(dn), "-s", ms=3, label=r"$\int|\Delta\rho|\,4\pi r^2 dr$")
    axC.set_xlabel("SCF iteration")
    axC.set_title("convergence")
    axC.legend(fontsize=8, frameon=False)

    ls_present = sorted({l for _, l in labels})
    for j, (n, l) in enumerate(labels):
        e = eps[j]
        if not np.isfinite(e):
            continue
        axB.vlines(e, 0, 1, color=L_COLOR.get(l, "#555"), lw=2)
        axB.text(e, 1.02, label_text(n, l), rotation=90, ha="center", va="bottom",
                 fontsize=7, color=L_COLOR.get(l, "#555"))
    axB.set_xscale("symlog", linthresh=1.0)
    axB.set_yticks([])
    axB.set_ylim(0, 1.25)
    axB.set_xlabel(r"$\varepsilon_{n\ell}$  (Ha)")
    axB.set_title("orbital-energy barcode")
    axB.invert_xaxis()

    fig.suptitle(man.get("title", "Atomic SCF"))
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
