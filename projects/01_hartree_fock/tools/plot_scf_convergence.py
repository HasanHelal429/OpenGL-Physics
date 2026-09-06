"""
4-panel SCF-convergence movie for a headless 01_hartree_fock run — the C++
counterpart of the Python solver's scf_movie.py:

  top-left     radial density   4*pi*r^2 * rho(r)      (log-r)
  top-right    scaled potential r * V_eff(r)           (log-r)
  bottom-left  orbital energies grouped by l           (symlog-E)
  bottom-right E_total per iteration + a "you are here" marker

One movie frame per SCF iteration. Uses imageio's bundled ffmpeg; falls back
to a PNG sequence, then a single final-frame PNG.

Usage:
    python plot_scf_convergence.py <results_dir> [--out FILE.mp4] [--fps 6]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hfio import load_manifest, load_field, n_frames, orbital_labels, label_text, radial_grid, L_SYMBOL

L_COLOR = {0: "#4c72b0", 1: "#dd8452", 2: "#55a868", 3: "#c44e52", 4: "#8172b3"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()
    d = args.results_dir

    man = load_manifest(d)
    nf = n_frames(d)
    r = radial_grid(d)
    labels = orbital_labels(d)

    di = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    e_total = np.atleast_1d(di["e_total"])
    steps = np.atleast_1d(di["step"])

    rho = [load_field(d, "rho", i).ravel() for i in range(nf)]
    veff = [load_field(d, "v_eff", i).ravel() for i in range(nf)]
    eps = [load_field(d, "eps", i).ravel() for i in range(nf)]

    rho_scaled = [4.0 * np.pi * r ** 2 * x for x in rho]
    veff_scaled = [r * x for x in veff]

    # Scale the density axis to the converged profile, not the wild iteration-1
    # seed spike (which would squash everything else flat).
    dmax = np.nanmax(rho_scaled[-1]) * 1.6
    vlo = min(np.nanmin(x) for x in veff_scaled)
    vhi = max(np.nanmax(x) for x in veff_scaled)
    ls_present = sorted({l for _, l in labels})

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=args.dpi)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.09, hspace=0.32, wspace=0.26)
    (ax_rho, ax_v), (ax_lev, ax_e) = axes
    title = fig.suptitle("")

    def render(i):
        for a in (ax_rho, ax_v, ax_lev, ax_e):
            a.clear()

        ax_rho.semilogx(r, rho_scaled[i], color="#333", lw=1.4)
        ax_rho.fill_between(r, rho_scaled[i], color="#4c72b0", alpha=0.25)
        ax_rho.set_ylim(0, dmax)
        ax_rho.set_xlim(r[0], r[-1])
        ax_rho.set_xlabel("r  (Bohr)")
        ax_rho.set_ylabel(r"$4\pi r^2 \rho(r)$")
        ax_rho.set_title("radial density")

        ax_v.semilogx(r, veff_scaled[i], color="#c44e52", lw=1.4)
        ax_v.set_ylim(vlo * 1.05, max(vhi * 1.05, 1.0))
        ax_v.set_xlim(r[0], r[-1])
        ax_v.set_xlabel("r  (Bohr)")
        ax_v.set_ylabel(r"$r\,V_{\rm eff}(r)$")
        ax_v.set_title("effective potential")

        for j, (n, l) in enumerate(labels):
            e = eps[i][j]
            if not np.isfinite(e):
                continue
            x = l + 0.5
            ax_lev.hlines(e, x - 0.4, x + 0.4, color=L_COLOR.get(l, "#555"), lw=2.2)
            ax_lev.text(x + 0.45, e, label_text(n, l), va="center", fontsize=8,
                        color=L_COLOR.get(l, "#555"))
        ax_lev.set_yscale("symlog", linthresh=1.0)
        ax_lev.set_xticks([l + 0.5 for l in ls_present])
        ax_lev.set_xticklabels([L_SYMBOL[l] for l in ls_present])
        ax_lev.set_xlim(min(ls_present), max(ls_present) + 1.2)
        ax_lev.set_xlabel("angular momentum")
        ax_lev.set_ylabel(r"$\varepsilon_{n\ell}$  (Ha)")
        ax_lev.set_title("orbital energies")
        ax_lev.invert_yaxis()

        ax_e.plot(steps[: i + 1], e_total[: i + 1], "-o", ms=3, color="#333")
        ax_e.plot([steps[i]], [e_total[i]], "o", ms=8, mfc="#dd8452", mec="k", zorder=5)
        ax_e.set_xlim(steps[0] - 0.5, steps[-1] + 0.5)
        finite = e_total[np.isfinite(e_total)]
        if finite.size:
            span = max(np.ptp(finite), 1e-6)
            ax_e.set_ylim(finite.min() - 0.08 * span, finite.max() + 0.08 * span)
        ax_e.set_xlabel("SCF iteration")
        ax_e.set_ylabel(r"$E_{\rm total}$  (Ha)")
        ax_e.set_title("total energy")

        et = e_total[i]
        title.set_text(f"{man.get('title', 'Atomic SCF')}   —   iteration {int(steps[i])}   "
                       + (f"$E$ = {et:.5f} Ha" if np.isfinite(et) else ""))
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[:, :, :3].copy()

    out = args.out or os.path.join(d, "scf_convergence.mp4")
    try:
        import imageio
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8) as w:
            for i in range(nf):
                w.append_data(render(i))
        print(f"wrote {out}")
    except Exception as e:
        print(f"imageio/ffmpeg unavailable ({e}); writing PNGs")
        seq = os.path.join(d, "scf_frames")
        os.makedirs(seq, exist_ok=True)
        for i in range(nf):
            plt.imsave(os.path.join(seq, f"f_{i:04d}.png"), render(i))
        plt.imsave(os.path.join(d, "scf_convergence_final.png"), render(nf - 1))
        print(f"  -> {seq}/ and scf_convergence_final.png")


if __name__ == "__main__":
    main()
