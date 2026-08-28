"""
Turn a headless 07_grhd Kerr-equatorial run into a movie. The solver is
axisymmetric (nothing depends on phi -- see README.md), so there's no
actual phi-position to visualize directly; this revolves each frame's
rho(r) profile into a 2D density image (as in
06_tidal_disruption/tools/make_movie.py and 07_grhd's own
make_bondi_movie.py), and ADDS a rotating angular brightness modulation at
each radius's own orbital angular velocity Omega(r) (from
tools/kerr_orbits.py) purely as a visualization aid to convey that the gas
is actually orbiting -- the modulation pattern itself is not simulated
data, just phase = Omega(r)*t drawn on top of the real, simulated rho(r).
The event horizon (r_+=M+sqrt(M^2-a^2)) and the equatorial ergosphere
boundary (r=2M, where frame dragging becomes absolute) are marked.

    python make_kerr_orbit_movie.py <results_dir> [--fps 20] [--res 400] [--out movie.mp4]
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
    sys.exit("make_kerr_orbit_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None

from kerr_orbits import circular_orbit


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--res", type=int, default=420)
    ap.add_argument("--gamma", type=float, default=0.6, help="density brightness curve on the log scale")
    ap.add_argument("--spokes", type=int, default=6, help="number of rotating brightness spokes (visualization aid)")
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
    a = deck["physics"]["a"]
    dr = (r_max - r_min) / n
    r_cells = r_min + (np.arange(n) + 0.5) * dr
    r_horizon = M + np.sqrt(max(M * M - a * a, 0.0))

    omega = np.array([circular_orbit(ri, M, a)[2] for ri in r_cells])

    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))
    if not rho_frames:
        sys.exit("no rho_*.npy frames found")
    all_rho = np.stack([np.load(f).ravel() for f in rho_frames])
    rho_min, rho_max = float(all_rho.min()), float(all_rho.max())
    log_min, log_max = np.log10(rho_min), np.log10(max(rho_max, rho_min * 1.001))
    cmap = plt.get_cmap("inferno")

    res = args.res
    xs = np.linspace(-r_max, r_max, res)
    xx, yy = np.meshgrid(xs, xs)
    r_pix = np.sqrt(xx * xx + yy * yy)
    phi_pix = np.arctan2(yy, xx)
    omega_pix = np.interp(r_pix, r_cells, omega, left=omega[0], right=omega[-1])
    inside_horizon = r_pix < r_horizon
    inside_grid = (r_pix >= r_min) & (r_pix <= r_max)

    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def colorize(rho_1d, t):
        rho_pix = np.interp(r_pix, r_cells, rho_1d, left=rho_1d[0], right=rho_1d[-1])
        rho_pix = np.clip(rho_pix, rho_min, rho_max)
        level = (np.log10(rho_pix) - log_min) / (log_max - log_min)
        level = np.clip(level, 0.0, 1.0) ** args.gamma
        # Rotating brightness spokes at each radius's own orbital Omega(r) --
        # a visualization aid (see module docstring), not simulated data.
        spoke = 0.15 * np.cos(args.spokes * (phi_pix - omega_pix * t))
        level = np.clip(level * (1.0 + spoke), 0.0, 1.0)
        img = cmap(level)
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
    im = ax.imshow(colorize(rho0, 0.0), extent=(-r_max, r_max, -r_max, r_max), origin="lower")
    ax.add_patch(plt.Circle((0, 0), r_horizon, color="#fff6d5", fill=False, lw=1.2, zorder=5))
    ax.add_patch(plt.Circle((0, 0), 2.0 * M, color="#7fd0ff", fill=False, lw=0.8, ls="--", zorder=5))
    txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, color="w", va="top",
                  fontsize=9, family="monospace")

    n_frames = len(rho_frames)

    def update(i):
        rho = np.load(rho_frames[i]).ravel()
        t = i * substeps * dt
        im.set_data(colorize(rho, t))
        txt.set_text(f"{manifest.get('title', '')}\nt = {t:.1f}   frame {i}/{n_frames}\n"
                      f"(spokes: rotation aid from Omega(r), not simulated phi-position)")
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
