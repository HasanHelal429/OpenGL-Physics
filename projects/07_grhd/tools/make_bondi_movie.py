"""
Turn a headless 07_grhd Schwarzschild (Bondi accretion) run into a movie.
The simulation is 1D radial (spherically symmetric), so this revolves each
frame's rho(r) profile into a 2D density image (pixel color = rho at that
pixel's distance from the center) -- an honest visualization of a
genuinely spherically symmetric flow, in the same density-colored,
black-hole-marked style as 06_tidal_disruption's make_movie.py.

    python make_bondi_movie.py <results_dir> [--fps 20] [--res 400] [--gamma 0.5] [--out movie.mp4]

Reads manifest.json + deck.toml + frames/rho_*.npy. Uses imageio (bundled
ffmpeg), falling back to a PNG sequence if that import fails.
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
    sys.exit("make_bondi_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--res", type=int, default=400, help="output image resolution (res x res pixels)")
    ap.add_argument("--gamma", type=float, default=0.5,
                     help="density brightness curve on the log scale; <1 lifts the dim, "
                          "far-from-the-hole outer disk out of near-invisibility")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    n = manifest["grid"]["nx"]
    r_min = deck["grid"]["r_min"]
    r_max = deck["grid"]["r_max"]
    M = deck["physics"]["M"]
    dr = (r_max - r_min) / n
    r_cells = r_min + (np.arange(n) + 0.5) * dr

    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))
    if not rho_frames:
        sys.exit("no rho_*.npy frames found")
    all_rho = np.stack([np.load(f).ravel() for f in rho_frames])
    rho_min = float(all_rho.min())
    rho_max = float(all_rho.max())
    log_min, log_max = np.log10(rho_min), np.log10(rho_max)
    cmap = plt.get_cmap("magma")

    # Pixel grid spanning [-r_max, r_max]^2; r_pix = distance from center.
    res = args.res
    xs = np.linspace(-r_max, r_max, res)
    xx, yy = np.meshgrid(xs, xs)
    r_pix = np.sqrt(xx * xx + yy * yy)
    inside_horizon = r_pix < 2.0 * M
    inside_grid = (r_pix >= r_min) & (r_pix <= r_max)

    def colorize(rho_1d):
        rho_pix = np.interp(r_pix, r_cells, rho_1d, left=rho_1d[0], right=rho_1d[-1])
        rho_pix = np.clip(rho_pix, rho_min, rho_max)
        t = (np.log10(rho_pix) - log_min) / (log_max - log_min)
        img = cmap(np.clip(t, 0.0, 1.0) ** args.gamma)
        img[~inside_grid] = (0.0, 0.0, 0.0, 1.0)
        img[inside_horizon] = (0.0, 0.0, 0.0, 1.0)
        return img

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    rho0 = np.load(rho_frames[0]).ravel()
    im = ax.imshow(colorize(rho0), extent=(-r_max, r_max, -r_max, r_max), origin="lower")
    horizon_circle = plt.Circle((0, 0), 2.0 * M, color="#fff6d5", fill=False, lw=1.2, zorder=5)
    ax.add_patch(horizon_circle)
    txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, color="w", va="top",
                  fontsize=9, family="monospace")

    n_frames = len(rho_frames)
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def update(i):
        rho = np.load(rho_frames[i]).ravel()
        im.set_data(colorize(rho))
        t = i * substeps * dt
        txt.set_text(f"{manifest.get('title', '')}\nt = {t:.1f}   frame {i}/{n_frames}")
        return im, txt

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
