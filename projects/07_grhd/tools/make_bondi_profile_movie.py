"""
Radial-profile movie for a Bondi-on-Kerr-Schild run (decks/kerr_bondi_a*.toml)
-- a 2D poloidal colormap carries no information for this test (the flow is
theta-independent by construction), so instead this plots rho(r) and v_r(r)
directly (averaged over theta, which should already be ~exact) against the
analytic Bondi reference (tools/bondi_analytic.py) overlaid as a dashed
line, animated over time. Directly shows what tools/plot_accretion.py's
Mdot comparison only measured indirectly: the near-domain solution tracking
the analytic profile early on, and the outer-boundary mass-buildup drift
appearing as a growing gap at large r later on.

Pass one results_dir for a single column, or several (e.g. the a=0 and
a=0.9 runs) for a side-by-side comparison.

    python make_bondi_profile_movie.py <results_dir> [<results_dir> ...] \
        [--fps 10] [--out movie.mp4]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("make_bondi_profile_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None

from bondi_analytic import BondiSolution
from kerr_schild_ref import bl_to_ks_velocity


def load_run(d):
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)
    diag = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    manifest = json.load(open(os.path.join(d, "manifest.json")))

    nr = deck["grid"]["nr"]
    r_min, r_max = deck["grid"]["r_min"], deck["grid"]["r_max"]
    M, gamma = deck["physics"]["M"], deck["physics"]["gamma"]
    dr = (r_max - r_min) / nr
    r = r_min + (np.arange(nr) + 0.5) * dr

    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))
    vr_frames = sorted(glob.glob(os.path.join(d, "frames", "v_r_*.npy")))
    t = diag["t"][: len(rho_frames)]
    return {
        "deck": deck, "manifest": manifest, "r": r, "M": M, "gamma": gamma,
        "rho_frames": rho_frames, "vr_frames": vr_frames, "t": t,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--r_c", type=float, default=8.0, help="Bondi sonic radius used to build the IC (matches make_kerr_bondi_ic.py's --r_c)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [load_run(d) for d in args.results_dirs]
    n = len(runs)
    n_frames = min(len(r["rho_frames"]) for r in runs)

    fig, axes = plt.subplots(2, n, figsize=(6.0 * n, 8.4), squeeze=False, dpi=130)  # even pixel dims for libx264

    analytic = {}
    for col, run in enumerate(runs):
        sol = BondiSolution(run["M"], args.r_c, run["gamma"])
        rho_a, v_bl, _ = sol.profile(run["r"])
        # v_bl is the BL-ORTHONORMAL radial velocity (bondi_analytic.py's
        # own convention, matching make_kerr_bondi_ic.py's IC). The
        # simulation instead stores and evolves the KS-frame coordinate
        # v^r (KerrTorusSim's Valencia primitive) -- a genuinely different
        # number even at identical physical state (frame dragging aside,
        # KS's v^r already differs from a BL-orthonormal v by the
        # coordinate-velocity/tetrad relation, non-trivial even at a=0).
        # Transform the analytic reference through the same BL->KS map the
        # actual IC used, so both curves are in the same convention --
        # comparing them directly without this (tried first) silently
        # compares two different velocity definitions, not a real
        # numerical discrepancy.
        v_a, _, _, _ = bl_to_ks_velocity(v_bl, np.zeros_like(v_bl), np.zeros_like(v_bl),
                                          run["r"], np.pi / 2.0, run["M"], run["deck"]["physics"]["a"])
        analytic[col] = (rho_a, v_a)

        title = run["manifest"].get("title", args.results_dirs[col])
        short_title = title.split("--")[-1].strip() if "--" in title else title

        ax_rho, ax_v = axes[0][col], axes[1][col]
        ax_rho.set_yscale("log")
        ax_rho.set_xlabel("r")
        ax_rho.set_ylabel("rho (theta-averaged)")
        ax_rho.set_title(short_title, fontsize=10)
        ax_rho.grid(alpha=0.25)
        ax_v.set_xlabel("r")
        ax_v.set_ylabel("v_r (theta-averaged)")
        ax_v.grid(alpha=0.25)

    lines = []
    for col, run in enumerate(runs):
        rho_a, v_a = analytic[col]
        ax_rho, ax_v = axes[0][col], axes[1][col]
        (l_rho_num,) = ax_rho.plot(run["r"], rho_a, color="#1f5c6e", lw=2, label="numerical (theta-avg)")
        (l_rho_ana,) = ax_rho.plot(run["r"], rho_a, color="#9c5f1a", lw=1.4, ls="--", label="analytic Bondi")
        ax_rho.legend(fontsize=8)
        (l_v_num,) = ax_v.plot(run["r"], v_a, color="#1f5c6e", lw=2)
        (l_v_ana,) = ax_v.plot(run["r"], v_a, color="#9c5f1a", lw=1.4, ls="--")
        txt = axes[0][col].text(0.03, 0.06, "", transform=axes[0][col].transAxes, fontsize=9, family="monospace")
        lines.append((l_rho_num, l_v_num, txt))

    fig.suptitle("Bondi accretion on Kerr-Schild: numerical vs. analytic radial profile", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    def update(i):
        for col, run in enumerate(runs):
            rho = np.load(run["rho_frames"][i]).mean(axis=1)
            vr = np.load(run["vr_frames"][i]).mean(axis=1)
            l_rho_num, l_v_num, txt = lines[col]
            l_rho_num.set_ydata(rho)
            l_v_num.set_ydata(vr)
            txt.set_text(f"t={run['t'][i]:.2f}")
            ax_rho, ax_v = axes[0][col], axes[1][col]
            ax_rho.relim()
            ax_rho.autoscale_view()
            ax_v.relim()
            ax_v.autoscale_view()
        return [x for tpl in lines for x in tpl]

    out = args.out or os.path.join(args.results_dirs[0], "bondi_profile_movie.mp4")
    try:
        import imageio
        writer_kwargs = dict(fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None)
        with imageio.get_writer(out, **writer_kwargs) as writer:
            for i in range(n_frames):
                update(i)
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                writer.append_data(frame)
                if i % 20 == 0:
                    print(f"\rframe {i + 1}/{n_frames}", end="", flush=True)
        print(f"\nwrote {out}")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"imageio/ffmpeg unavailable ({e})")


if __name__ == "__main__":
    main()
