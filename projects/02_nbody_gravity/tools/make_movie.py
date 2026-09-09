"""
Render a headless-run's particle frames to an mp4 (or a PNG sequence if
libx264 is unavailable).

    python make_movie.py <results_dir> [--out collapse.mp4] [--fps 30]
                         [--projection xy|xz|yz] [--lim L] [--dpi 120]

<results_dir> is a directory written by `02_nbody_gravity --deck ... --out
<dir>` -- it must contain manifest.json and frames/pos_*.npy. Colours
particles by distance from the centre of mass so the collapsing core stands
out. Falls back to writing frames/*.png if the mp4 encoder isn't present.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np


def _winpath(p):
    """Turn an msys/cygwin '/c/Users/...' path into 'C:/Users/...' so the
    native-Windows imageio/ffmpeg can open it. No-op elsewhere."""
    m = re.match(r"^/([a-zA-Z])/(.*)$", p)
    return f"{m.group(1).upper()}:/{m.group(2)}" if m and os.name == "nt" else p

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--projection", default="xy", choices=["xy", "xz", "yz"])
    ap.add_argument("--lim", type=float, default=None, help="half-width of the view box (auto from frame 0 if unset)")
    ap.add_argument("--dpi", type=int, default=120)
    ap.add_argument("--size", type=float, default=2.0, help="marker size")
    args = ap.parse_args()

    rd = args.results_dir
    manifest = json.load(open(os.path.join(rd, "manifest.json")))
    frame_files = sorted(glob.glob(os.path.join(rd, "frames", "pos_*.npy")))
    if not frame_files:
        sys.exit(f"make_movie: no frames/pos_*.npy under {rd}")

    ax_i, ax_j = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}[args.projection]
    p0 = np.load(frame_files[0])
    lim = args.lim if args.lim is not None else 1.1 * np.abs(p0[:, [ax_i, ax_j]]).max()

    out = _winpath(args.out or os.path.join(rd, "movie.mp4"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    writer = None
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                    macro_block_size=None)
        png_dir = None
    except Exception as e:
        png_dir = os.path.join(rd, "movie_frames")
        os.makedirs(png_dir, exist_ok=True)
        print(f"make_movie: mp4 encoder unavailable ({e}); writing PNG sequence to {png_dir}")

    fig, ax = plt.subplots(figsize=(6, 6))
    for k, f in enumerate(frame_files):
        pos = np.load(f)
        com = pos.mean(axis=0)
        r = np.linalg.norm(pos - com, axis=1)
        ax.clear()
        ax.scatter(pos[:, ax_i], pos[:, ax_j], s=args.size, c=r, cmap="turbo",
                   vmin=0.0, vmax=np.percentile(np.linalg.norm(p0 - p0.mean(axis=0), axis=1), 95))
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{manifest.get('title', '')}\nframe {k+1}/{len(frame_files)}", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rgba = rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        rgb = rgba[..., :3].copy()
        if writer is not None:
            writer.append_data(rgb)
        else:
            plt.imsave(os.path.join(png_dir, f"frame_{k:04d}.png"), rgb)

    if writer is not None:
        writer.close()
        print(f"wrote {out}  ({len(frame_files)} frames @ {args.fps} fps)")
    else:
        print(f"wrote {len(frame_files)} PNGs to {png_dir}")


if __name__ == "__main__":
    main()
