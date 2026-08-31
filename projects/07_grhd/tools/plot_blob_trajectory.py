"""
Turns the free-falling blob run (decks/kerr_blob_infall.toml) into an
actual measured infall trajectory instead of a shrinking blob on a
colormap: tracks the density-weighted centroid (r(t), theta(t)) across
saved frames, then plots r(t) vs t (with the horizon and starting radius
marked) and dr/dt vs r -- a phase-space view that should show the infall
accelerating inward, the qualitative signature of genuine infall rather
than pressure-driven diffusion.

    python plot_blob_trajectory.py <results_dir> [--out fig.png] [--frame_end N]
"""

import argparse
import glob
import json
import os

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("plot_blob_trajectory: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--frame_end", type=int, default=None, help="stop before this frame index (e.g. to exclude a NaN tail)")
    args = ap.parse_args()

    d = args.results_dir
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)
    diag = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)

    nr, nth = deck["grid"]["nr"], deck["grid"]["ntheta"]
    r_min, r_max = deck["grid"]["r_min"], deck["grid"]["r_max"]
    theta_min = deck["grid"]["theta_min"]
    theta_max = np.pi - theta_min
    M, a = deck["physics"]["M"], deck["physics"]["a"]
    r_horizon = M + np.sqrt(max(M * M - a * a, 0.0))
    dr = (r_max - r_min) / nr
    dth = (theta_max - theta_min) / nth
    r = r_min + (np.arange(nr) + 0.5) * dr
    theta = theta_min + (np.arange(nth) + 0.5) * dth
    R, TH = np.meshgrid(r, theta, indexing="ij")

    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))
    hi = args.frame_end if args.frame_end is not None else len(rho_frames)
    rho_frames = rho_frames[:hi]
    t = diag["t"][: len(rho_frames)]

    r_c = np.full(len(rho_frames), np.nan)
    th_c = np.full(len(rho_frames), np.nan)
    rho_peak = np.full(len(rho_frames), np.nan)
    for i, f in enumerate(rho_frames):
        rho = np.load(f)
        if not np.all(np.isfinite(rho)):
            break
        w = rho.sum()
        r_c[i] = (rho * R).sum() / w
        th_c[i] = (rho * TH).sum() / w
        rho_peak[i] = rho.max()
    valid = np.isfinite(r_c)
    t, r_c, th_c, rho_peak = t[valid], r_c[valid], th_c[valid], rho_peak[valid]

    v_c = np.gradient(r_c, t)  # centroid radial velocity (coordinate dr/dt)

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 10), dpi=130, sharex=False)

    ax = axes[0]
    ax.plot(t, r_c, color="#1f5c6e", lw=2, label="density-weighted centroid r(t)")
    ax.axhline(r_horizon, color="#a8433f", lw=1.2, ls="--", label=f"horizon r_+={r_horizon:.3f}")
    ax.axhline(r_c[0], color="#9c5f1a", lw=1.0, ls=":", label=f"start r0={r_c[0]:.2f}")
    ax.set_xlabel("t (M)")
    ax.set_ylabel("r")
    ax.set_title("Blob centroid radius vs time")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(r_c, v_c, color="#3f7a52", lw=2)
    ax.axvline(r_horizon, color="#a8433f", lw=1.2, ls="--")
    ax.invert_xaxis()
    ax.set_xlabel("r")
    ax.set_ylabel("dr/dt (coordinate radial velocity)")
    ax.set_title("Infall phase portrait: does it accelerate inward?")
    ax.grid(alpha=0.25)

    ax = axes[2]
    ax.plot(t, np.degrees(th_c), color="#6b4fa0", lw=2)
    ax.axhline(90.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("t (M)")
    ax.set_ylabel("theta_centroid (deg)")
    ax.set_title("Polar drift (should stay ~90 deg for this equatorial-start blob)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out = args.out or os.path.join(d, "blob_trajectory.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")
    print(f"r: {r_c[0]:.3f} -> {r_c[-1]:.3f} over t=[{t[0]:.2f}, {t[-1]:.2f}]  "
          f"(final dr/dt={v_c[-1]:.4f}, horizon at r_+={r_horizon:.3f})")


if __name__ == "__main__":
    main()
