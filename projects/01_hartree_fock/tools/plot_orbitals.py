"""
Converged radial orbitals from a headless 01_hartree_fock run.

  left    R_nl(r)              the radial wavefunctions
  right   4*pi*r^2 R_nl(r)^2   radial probability density (shell structure)

Usage:
    python plot_orbitals.py <results_dir> [--out FILE.png] [--rmax 6]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hfio import load_manifest, n_frames, orbital_labels, label_text, radial_grid, frame_path

L_COLOR = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868", 3: "#c44e52", 4: "#8172b3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--rmax", type=float, default=None, help="x-axis cutoff in Bohr")
    args = ap.parse_args()
    d = args.results_dir

    man = load_manifest(d)
    r = radial_grid(d)
    labels = orbital_labels(d)
    last = n_frames(d) - 1
    R = np.load(frame_path(d, "orbitals_final", last))          # (k, N)

    # Eigenvector signs are arbitrary -- flip each so its largest lobe is positive.
    for j in range(R.shape[0]):
        if R[j][np.argmax(np.abs(R[j]))] < 0:
            R[j] = -R[j]

    u = r * R                                                    # r R_nl(r); the solver normalizes so int u^2 dr = 1
    prob = u ** 2                                                 # radial probability density, int prob dr = 1
    if args.rmax:
        rmax = args.rmax
    else:
        total = prob.sum(axis=0)
        cdf = np.cumsum(total) / np.sum(total)
        rmax = min(r[-1], 1.3 * r[np.searchsorted(cdf, 0.995)])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=0.22)

    for j, (n, l) in enumerate(labels):
        c = L_COLOR.get(l, "#555")
        axL.plot(r, u[j], color=c, lw=1.5, label=label_text(n, l))
        axR.plot(r, prob[j], color=c, lw=1.5, label=label_text(n, l))

    for a in (axL, axR):
        a.set_xlim(0, rmax)
        a.set_xlabel("r  (Bohr)")
        a.legend(ncol=2, fontsize=8, frameon=False)
    axL.axhline(0, color="#bbb", lw=0.8)
    axL.set_ylabel(r"$r\,R_{n\ell}(r)$")
    axL.set_title("radial wavefunctions  ($u_{n\\ell} = r R_{n\\ell}$)")
    axR.set_ylabel(r"$[r R_{n\ell}(r)]^2$")
    axR.set_title(r"radial probability density  ($\int P\,dr = 1$)")

    fig.suptitle(man.get("title", "Atomic SCF") + "  -  converged orbitals")
    out = args.out or os.path.join(d, "orbitals.png")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
