"""
Movie of a headless 07_grhd Kerr-torus (2D r,theta) run: an animated
poloidal (x=r*sin(theta), z=r*cos(theta)) cross-section of log10(rho),
mirrored left/right for the full meridional slice (same convention as
tools/plot_fishbone_moncrief.py's static plot), with the horizon marked.

Useful both for the validated torus-stability runs and for diagnosing a
crash: pass --frame_start/--frame_end to focus on a specific window (e.g.
the last N frames before a NaN) instead of rendering the whole run.

    python make_torus_movie.py <results_dir> [--fps 10] [--res 300]
        [--frame_start N] [--frame_end N] [--out movie.mp4]
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
    sys.exit("make_torus_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--frame_start", type=int, default=None)
    ap.add_argument("--frame_end", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    nr = deck["grid"]["nr"]
    nth = deck["grid"]["ntheta"]
    r_min = deck["grid"]["r_min"]
    r_max = deck["grid"]["r_max"]
    theta_min = deck["grid"]["theta_min"]
    theta_max = np.pi - theta_min
    M = deck["physics"]["M"]
    a = deck["physics"]["a"]
    dr = (r_max - r_min) / nr
    dth = (theta_max - theta_min) / nth
    r = r_min + (np.arange(nr) + 0.5) * dr
    theta = theta_min + (np.arange(nth) + 0.5) * dth
    R, TH = np.meshgrid(r, theta, indexing="ij")
    X = R * np.sin(TH)
    Z = R * np.cos(TH)
    r_horizon = M + np.sqrt(max(M * M - a * a, 0.0))

    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))
    if not rho_frames:
        sys.exit("no rho_*.npy frames found")
    lo = args.frame_start if args.frame_start is not None else 0
    hi = args.frame_end if args.frame_end is not None else len(rho_frames)
    rho_frames = rho_frames[lo:hi]
    print(f"rendering frames [{lo}:{hi}] of {len(glob.glob(os.path.join(d, 'frames', 'rho_*.npy')))}")

    all_rho = np.stack([np.load(f) for f in rho_frames])
    finite = all_rho[np.isfinite(all_rho) & (all_rho > 0)]
    rho_min = max(float(finite.min()) if finite.size else 1e-8, 1e-8)
    rho_max = max(float(finite.max()) if finite.size else 1.0, rho_min * 10)

    from matplotlib.colors import LogNorm
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad("white")  # NaN cells render as solid white -- unmistakable in the movie
    norm = LogNorm(vmin=rho_min, vmax=rho_max)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0.04, 1, 0.96)

    rho0 = np.ma.masked_invalid(np.load(rho_frames[0]))
    pcm_r = ax.pcolormesh(X, Z, rho0, shading="auto", cmap=cmap, norm=norm)
    pcm_l = ax.pcolormesh(-X, Z, rho0, shading="auto", cmap=cmap, norm=norm)
    ax.add_patch(plt.Circle((0, 0), r_horizon, color="#7fd0ff", fill=False, lw=1.2))
    txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="w", va="top",
                  fontsize=9, family="monospace")

    # Read actual per-frame t from diagnostics.csv rather than recomputing
    # via manifest.dt*substeps_per_frame*frame_idx: the manifest always
    # reflects the DECK's static substeps_per_frame, not a --substeps CLI
    # override used for a specific headless run (e.g. a fine-resolution
    # diagnostic run), so recomputing that way silently mislabels time by
    # whatever factor the override differed by.
    diag = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t_col = diag["t"]
    n_frames = len(rho_frames)

    def update(i):
        rho = np.ma.masked_invalid(np.load(rho_frames[i]))
        pcm_r.set_array(rho.ravel())
        pcm_l.set_array(rho.ravel())
        frame_idx = lo + i
        t = t_col[frame_idx] if frame_idx < len(t_col) else float("nan")
        nan_flag = " -- NaN PRESENT" if np.any(np.isnan(rho)) else ""
        txt.set_text(f"{manifest.get('title', '')}\nframe {frame_idx}  t={t:.3f}{nan_flag}")
        return pcm_r, pcm_l, txt

    out = args.out or os.path.join(d, "torus_movie.mp4")
    try:
        import imageio
        writer_kwargs = dict(fps=args.fps, codec="libx264", quality=8,
                              pixelformat="yuv420p", macro_block_size=None)
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
        print(f"\nimageio/ffmpeg unavailable ({e}); writing PNG sequence instead")
        pdir = os.path.join(d, "frames_png")
        os.makedirs(pdir, exist_ok=True)
        for i in range(n_frames):
            update(i)
            fig.savefig(os.path.join(pdir, f"{i:04d}.png"), facecolor="black")
        print("wrote", pdir)


if __name__ == "__main__":
    main()
