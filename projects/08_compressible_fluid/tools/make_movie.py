"""
Turn a headless 08_compressible_fluid run into a movie of the vorticity
field (dv/dx - du/dy, central differences on the u/v frames), the field
that makes shedding/wake structure visible at a glance in a way raw
velocity or density don't. Works for any 2D run (--2d/--channel/
--taylor-green/--cylinder); the obstacle mask (if the deck has one) is
drawn as a solid disk so the cylinder itself doesn't show up as a raw
zero-velocity artifact.

    python make_movie.py <results_dir> [--fps 30] [--stride 1] [--vmax V] [--out movie.mp4]

Reads manifest.json + frames/u_*.npy + frames/v_*.npy (+ deck.toml's
[cylinder] table, if present, to draw the obstacle). Uses imageio (bundled
ffmpeg via imageio-ffmpeg, not the system PATH) so this works even where a
system ffmpeg isn't installed; falls back to a PNG sequence if that import
fails -- same convention as 06_tidal_disruption/tools/make_movie.py.
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
    from matplotlib.patches import Circle
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vmax", type=float, default=None,
                     help="vorticity color-scale saturation (symmetric, +-vmax); "
                          "default: the 99th percentile magnitude over all frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    u_frames = sorted(glob.glob(os.path.join(d, "frames", "u_*.npy")))[:: args.stride]
    v_frames = sorted(glob.glob(os.path.join(d, "frames", "v_*.npy")))[:: args.stride]
    if not u_frames:
        sys.exit("no u_*.npy frames found -- this tool is for 2D runs (--2d/--channel/"
                  "--taylor-green/--cylinder), not the 1D shock tube")

    nx, ny = manifest["grid"]["nx"], manifest["grid"]["ny"]
    lx, ly = manifest["grid"]["lx"], manifest["grid"]["ly"]
    dx = lx / nx

    deck_path = os.path.join(d, "deck.toml")
    cyl = None
    if tomllib is not None and os.path.exists(deck_path):
        with open(deck_path, "rb") as f:
            deck = tomllib.load(f)
        if "cylinder" in deck:
            c = deck["cylinder"]
            diameter = c.get("diameter", 1.0)
            upstream_d = c.get("upstream_d", 5.0)
            blockage = c.get("blockage", 0.125)
            y_offset_d = c.get("y_offset_d", 0.02)
            cyl = {
                "x": upstream_d * diameter,
                "y": diameter / blockage / 2.0 + y_offset_d * diameter,
                "r": diameter / 2.0,
            }

    def vorticity(u, v):
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dx, axis=0)
        return dvdx - dudy

    if args.vmax is not None:
        vmax = args.vmax
    else:
        sample_idx = np.linspace(0, len(u_frames) - 1, min(20, len(u_frames))).astype(int)
        mags = []
        for i in sample_idx:
            u, v = np.load(u_frames[i]), np.load(v_frames[i])
            mags.append(np.percentile(np.abs(vorticity(u, v)), 99.0))
        vmax = max(float(np.max(mags)), 1e-6)

    fig, ax = plt.subplots(figsize=(12, 12 * ly / lx), dpi=130)
    im = ax.imshow(np.zeros((ny, nx)), origin="lower", cmap="RdBu_r", extent=[0, lx, 0, ly],
                    vmin=-vmax, vmax=vmax)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="vorticity", fraction=0.025, pad=0.01)
    if cyl is not None:
        ax.add_patch(Circle((cyl["x"], cyl["y"]), cyl["r"], facecolor="0.3", edgecolor="k", zorder=5))
    txt = ax.text(0.01, 0.98, "", transform=ax.transAxes, color="k", va="top",
                  fontsize=9, family="monospace",
                  bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    fig.tight_layout()

    n_frames = len(u_frames)
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def update(i):
        u, v = np.load(u_frames[i]), np.load(v_frames[i])
        im.set_data(vorticity(u, v))
        t = i * args.stride * substeps * dt
        txt.set_text(f"{manifest.get('title','')}\nt = {t:.2f}   frame {i*args.stride}/{manifest.get('frames','?')}")
        return im, txt

    out = args.out or os.path.join(d, "movie.mp4")
    try:
        import imageio
        # macro_block_size=2 (not None): libx264 needs even width/height,
        # and this figure's rendered canvas isn't guaranteed to already be
        # even -- None disables imageio's own padding entirely, which
        # fails outright on an odd dimension (hit in practice: a
        # 1560x499 canvas). 2 pads to the nearest even size (minimal
        # padding) rather than 16 (the imageio default, a much coarser,
        # more visible crop/pad for no benefit here).
        writer_kwargs = dict(fps=args.fps, codec="libx264", quality=8,
                             pixelformat="yuv420p", macro_block_size=2)
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
            fig.savefig(os.path.join(pdir, f"{i:04d}.png"))
        print("wrote", pdir)


if __name__ == "__main__":
    main()
