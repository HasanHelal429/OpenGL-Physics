"""
Animate the field frames of a headless 11_retarded_fields run. By default it
shows |E| times distance from the box centre, so the radiation pattern reads
instead of the 1/r^2 near-field spike (matching the interactive view's
r-weight mode). Uses imageio's bundled ffmpeg; PNG-sequence fallback.

Usage:
    python make_movie.py <results_dir> [--out FILE.mp4] [--fps 30]
                         [--field E_mag] [--no-rweight] [--stride N]
"""

import argparse
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
    ap.add_argument("--field", default="E_mag")
    ap.add_argument("--no-rweight", action="store_true")
    args = ap.parse_args()
    d = args.results_dir

    fdir = os.path.join(d, "frames")
    ids = sorted(int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(fdir) if f.startswith(args.field + "_"))
    ids = ids[::args.stride]
    frames = [np.load(os.path.join(fdir, f"{args.field}_{i:04d}.npy")) for i in ids]

    dcfg = {}
    try:
        with open(os.path.join(d, "deck.toml"), "rb") as fh:
            dcfg = tomllib.load(fh)
    except Exception:
        pass
    lx = dcfg.get("domain", {}).get("lx", frames[0].shape[1])
    ly = dcfg.get("domain", {}).get("ly", frames[0].shape[0])
    ny, nx = frames[0].shape

    signed = args.field in ("Ex", "Ey", "Ez", "Bz")
    rweight = (not args.no_rweight) and not signed
    if rweight:
        xs = (np.arange(nx) + 0.5) / nx * lx - 0.5 * lx
        ys = (np.arange(ny) + 0.5) / ny * ly - 0.5 * ly
        rr = np.hypot(*np.meshgrid(xs, ys))
        rnear = max(1.0, 3.0 * lx / nx)
        w = np.where(rr < rnear, 0.0, rr)
        frames = [f * w for f in frames]

    stack = np.abs(np.stack(frames))
    vmax = np.percentile(stack, 99.5)
    cmap = "RdBu_r" if signed else "magma"

    fig, ax = plt.subplots(figsize=(6.4, 6.4 * ny / nx), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    im = ax.imshow(frames[0], origin="lower", cmap=cmap,
                   vmin=(-vmax if signed else 0), vmax=vmax)

    def render(k):
        im.set_data(frames[k])
        fig.canvas.draw()
        w_, h_ = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8
                             ).reshape(h_, w_, 4)[:, :, :3].copy()

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
