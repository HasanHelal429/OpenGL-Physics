"""
Turn a headless 06_tidal_disruption run into a movie: a 2D (x,y) scatter of
the SPH particles, colored by density on a fixed log scale so brightness is
comparable frame to frame, with a time label.

    python make_movie.py <results_dir> [--fps 30] [--stride 1] [--out movie.mp4]

Reads manifest.json + frames/pos_mass_*.npy + frames/rho_press_*.npy. Uses
imageio (bundled ffmpeg via imageio-ffmpeg, not the system PATH) so this
works even where a system ffmpeg isn't installed; falls back to a PNG
sequence if that import fails.
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
    from matplotlib.colors import LogNorm
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    pos_frames = sorted(glob.glob(os.path.join(d, "frames", "pos_mass_*.npy")))[:: args.stride]
    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_press_*.npy")))[:: args.stride]
    if not pos_frames:
        sys.exit("no pos_mass_*.npy frames found")

    # Fixed axis limits and color scale across the whole run, so the star's
    # apparent size/brightness changes reflect real dynamics, not rescaling.
    all_pos = np.stack([np.load(f) for f in pos_frames])
    all_rho = np.stack([np.load(f)[:, 0] for f in rho_frames])
    r_max = float(np.percentile(np.linalg.norm(all_pos[..., :3], axis=-1), 99.5)) * 1.15
    rho_min = max(float(all_rho[all_rho > 0].min()), float(all_rho.max()) * 1e-4)
    rho_max = float(all_rho.max())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    pm0 = np.load(pos_frames[0])
    rp0 = np.load(rho_frames[0])
    scat = ax.scatter(pm0[:, 0], pm0[:, 1], c=rp0[:, 0], s=3.5, cmap="magma",
                       norm=LogNorm(vmin=rho_min, vmax=rho_max), linewidths=0)
    txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, color="w", va="top",
                  fontsize=9, family="monospace")

    n_frames = len(pos_frames)
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def update(i):
        pm = np.load(pos_frames[i])
        rp = np.load(rho_frames[i])
        scat.set_offsets(pm[:, :2])
        scat.set_array(rp[:, 0])
        t = i * args.stride * substeps * dt
        txt.set_text(f"{manifest.get('title','')}\nt = {t:.2f}   frame {i*args.stride}/{manifest.get('frames','?')}")
        return scat, txt

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    out = args.out or os.path.join(d, "movie.mp4")
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
                    print(f"\rframe {i+1}/{n_frames}", end="", flush=True)
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
