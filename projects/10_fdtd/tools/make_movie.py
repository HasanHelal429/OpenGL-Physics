"""
Animate the Ez frames of a headless 10_fdtd run (diverging colormap, with
material regions outlined from the frame-0 field if present). Uses imageio's
bundled ffmpeg; PNG-sequence fallback.

Usage:
    python make_movie.py <results_dir> [--out FILE.mp4] [--fps 30] [--field Ez]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--field", default="Ez")
    args = ap.parse_args()
    d = args.results_dir

    fdir = os.path.join(d, "frames")
    ids = sorted(int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(fdir) if f.startswith(args.field + "_"))
    ids = ids[::args.stride]
    frames = [np.load(os.path.join(fdir, f"{args.field}_{i:04d}.npy")) for i in ids]
    stack = np.abs(np.stack(frames))
    vmax = np.percentile(stack, 99.5)

    dcfg = {}
    try:
        with open(os.path.join(d, "deck.toml"), "rb") as fh:
            dcfg = tomllib.load(fh)
    except Exception:
        pass
    ny, nx = frames[0].shape

    fig, ax = plt.subplots(figsize=(6.4, 6.4 * ny / nx), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    signed = args.field in ("Ez", "Hx", "Hy")
    cmap = "RdBu_r" if signed else "magma"
    im = ax.imshow(frames[0], origin="lower", cmap=cmap,
                   vmin=(-vmax if signed else 0), vmax=vmax)

    # outline material shapes from the deck
    for m in dcfg.get("material", []) + dcfg.get("pec", []):
        col = "0.15" if m in dcfg.get("pec", []) else "0.4"
        sh = m.get("shape", "box")
        if sh == "cylinder":
            ax.add_patch(plt.Circle((m["x"], m["y"]), m["radius"], fill=False,
                                    ec=col, lw=1.0))
        elif sh == "box":
            ax.add_patch(plt.Rectangle((m["x"] - m["half_x"], m["y"] - m["half_y"]),
                                       2 * m["half_x"], 2 * m["half_y"], fill=False,
                                       ec=col, lw=1.0))

    def render(k):
        im.set_data(frames[k])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    out = args.out or os.path.join(d, "movie.mp4")
    try:
        import imageio
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8) as wr:
            for k in range(len(frames)):
                wr.append_data(render(k))
        print(f"wrote {out}")
    except Exception as e:
        seq = os.path.join(d, "frames_png")
        os.makedirs(seq, exist_ok=True)
        print(f"imageio unavailable ({e}); PNG sequence -> {seq}")
        for k in range(len(frames)):
            plt.imsave(os.path.join(seq, f"f_{k:05d}.png"), render(k))


if __name__ == "__main__":
    main()
