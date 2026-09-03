"""
Turn a headless 08_compressible_fluid run into a movie. By default renders
the vorticity field (dv/dx - du/dy, central differences on the u/v frames),
the field that makes shedding/wake structure visible at a glance in a way
raw velocity or density don't; --field density renders density instead,
which is the more legible choice for a shock-dominated run (e.g.
decks/riemann2d_config3.toml) where the interesting structure is sharp
density jumps, not rotation. If the run also wrote a passive scalar tracer
field (Prim2D::tracer -- e.g. decks/cylinder_re100.toml and
decks/airfoil_wind_tunnel.toml's inflow dye stripes), a second panel
renders that too (independent of --field), showing how the flow actually
mixes and rolls up fluid from different streamlines. Works for any --scene
2D run; every obstacle in the deck's [[obstacles]] list is drawn (a circle
as a solid disk, an airfoil as its NACA00xx outline rotated by its angle of
attack) so it doesn't show up as a raw zero-velocity artifact.

    python make_movie.py <results_dir> [--field vorticity|density] [--fps 30] [--stride 1] [--vmax V] [--out movie.mp4]

Reads manifest.json + frames/u_*.npy + frames/v_*.npy (+ frames/tracer_*.npy
if present, + deck.toml's [[obstacles]] tables, if present). Uses imageio
(bundled ffmpeg via imageio-ffmpeg, not the system PATH) so this works even
where a system ffmpeg isn't installed; falls back to a PNG sequence if that
import fails -- same convention as 06_tidal_disruption/tools/make_movie.py.
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
    from matplotlib.patches import Circle, Polygon
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def naca00xx_outline(center, chord, thickness, angle_deg, n=40):
    """World-space (x,y) outline of a symmetric NACA00xx section -- same
    formula and rotation convention as CompressibleSimScene.cpp's
    IsInsideAirfoil (`center` is the leading edge, angle_deg positive
    nose-up), just tracing the boundary instead of testing points against
    it."""
    xs = np.linspace(0.0, chord, n)
    xoc = xs / chord
    yt = 5.0 * thickness * chord * (
        0.2969 * np.sqrt(xoc) - 0.1260 * xoc - 0.3516 * xoc**2 + 0.2843 * xoc**3 - 0.1015 * xoc**4
    )
    upper = np.stack([xs, yt], axis=1)
    lower = np.stack([xs[::-1], -yt[::-1]], axis=1)
    body = np.concatenate([upper, lower], axis=0)
    theta = np.radians(angle_deg)
    ca, sa = np.cos(theta), np.sin(theta)
    # Body -> world is the inverse of the C++ mask's world -> body rotation
    # (there, world-frame (dx,dy) is rotated by -angle_deg into body frame).
    world_x = body[:, 0] * ca - body[:, 1] * sa + center[0]
    world_y = body[:, 0] * sa + body[:, 1] * ca + center[1]
    return np.stack([world_x, world_y], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--field", choices=["vorticity", "density"], default="vorticity",
                     help="primary panel field. vorticity (default) is best for "
                          "shedding/wake structure; density is more legible for a "
                          "shock-dominated run (e.g. decks/riemann2d_config3.toml), "
                          "where vorticity is dominated by any vortical mixing region "
                          "and the shocks themselves show up only as faint thin lines")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--vmax", type=float, default=None,
                     help="color-scale saturation: symmetric +-vmax for vorticity, "
                          "or the upper end of [rho_min, vmax] for density; default: "
                          "the 99th percentile magnitude (vorticity) or the observed "
                          "max (density) over a sample of frames")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    u_frames = sorted(glob.glob(os.path.join(d, "frames", "u_*.npy")))[:: args.stride]
    v_frames = sorted(glob.glob(os.path.join(d, "frames", "v_*.npy")))[:: args.stride]
    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_*.npy")))[:: args.stride]
    tracer_frames = sorted(glob.glob(os.path.join(d, "frames", "tracer_*.npy")))[:: args.stride]
    has_tracer = len(tracer_frames) == len(u_frames) and len(tracer_frames) > 0
    if not u_frames:
        sys.exit("no u_*.npy frames found -- this tool is for --scene 2D runs, not the 1D shock tube")
    if args.field == "density" and len(rho_frames) != len(u_frames):
        sys.exit("--field density needs frames/rho_*.npy, matching u_*.npy in count")

    nx, ny = manifest["grid"]["nx"], manifest["grid"]["ny"]
    lx, ly = manifest["grid"]["lx"], manifest["grid"]["ly"]
    dx = lx / nx

    deck_path = os.path.join(d, "deck.toml")
    obstacles = []
    if tomllib is not None and os.path.exists(deck_path):
        with open(deck_path, "rb") as f:
            deck = tomllib.load(f)
        for obs in deck.get("obstacles", []):
            shape = obs.get("shape", "circle")
            center = obs.get("center", [0.0, 0.0])
            if shape == "circle":
                obstacles.append({"shape": "circle", "x": center[0], "y": center[1],
                                   "r": obs.get("radius", 0.0)})
            elif shape == "airfoil":
                obstacles.append({"shape": "airfoil", "center": center, "chord": obs.get("chord", 1.0),
                                   "thickness": obs.get("thickness", 0.12),
                                   "angle_deg": obs.get("angle_deg", 0.0)})
            else:
                print(f"make_movie: unsupported obstacle shape '{shape}', not drawing it", file=sys.stderr)

    def vorticity(u, v):
        dvdx = np.gradient(v, dx, axis=1)
        dudy = np.gradient(u, dx, axis=0)
        return dvdx - dudy

    sample_idx = np.linspace(0, len(u_frames) - 1, min(20, len(u_frames))).astype(int)
    if args.field == "vorticity":
        if args.vmax is not None:
            vmax = args.vmax
        else:
            mags = [np.percentile(np.abs(vorticity(np.load(u_frames[i]), np.load(v_frames[i]))), 99.0)
                    for i in sample_idx]
            vmax = max(float(np.max(mags)), 1e-6)
        main_cmap, main_vmin, main_vmax, main_label = "RdBu_r", -vmax, vmax, "vorticity"
    else:
        rho_max = args.vmax if args.vmax is not None else max(
            float(np.max([np.load(rho_frames[i]).max() for i in sample_idx])), 1e-6)
        rho_min = min(float(np.min([np.load(rho_frames[i]).min() for i in sample_idx])), rho_max - 1e-6)
        main_cmap, main_vmin, main_vmax, main_label = "viridis", rho_min, rho_max, "density"

    n_panels = 2 if has_tracer else 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, n_panels * 12 * ly / lx + 0.6), dpi=130,
                              squeeze=False)
    axes = axes[:, 0]
    ax_main = axes[0]
    im_main = ax_main.imshow(np.zeros((ny, nx)), origin="lower", cmap=main_cmap, extent=[0, lx, 0, ly],
                              vmin=main_vmin, vmax=main_vmax)
    ax_main.set_ylabel("y"); ax_main.set_title(main_label)
    fig.colorbar(im_main, ax=ax_main, label=main_label, fraction=0.025, pad=0.01)

    im_tracer = None
    if has_tracer:
        ax_tracer = axes[1]
        im_tracer = ax_tracer.imshow(np.zeros((ny, nx)), origin="lower", cmap="viridis", extent=[0, lx, 0, ly],
                                      vmin=0.0, vmax=1.0)
        ax_tracer.set_xlabel("x"); ax_tracer.set_ylabel("y"); ax_tracer.set_title("tracer")
        fig.colorbar(im_tracer, ax=ax_tracer, label="tracer", fraction=0.025, pad=0.01)
    else:
        ax_main.set_xlabel("x")

    for a in axes:
        for obs in obstacles:
            if obs["shape"] == "circle":
                a.add_patch(Circle((obs["x"], obs["y"]), obs["r"], facecolor="0.3", edgecolor="k", zorder=5))
            else:
                pts = naca00xx_outline(obs["center"], obs["chord"], obs["thickness"], obs["angle_deg"])
                a.add_patch(Polygon(pts, closed=True, facecolor="0.3", edgecolor="k", zorder=5))
    txt = ax_main.text(0.01, 0.98, "", transform=ax_main.transAxes, color="k", va="top",
                        fontsize=9, family="monospace",
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=2))
    fig.tight_layout()

    n_frames = len(u_frames)
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def update(i):
        if args.field == "vorticity":
            im_main.set_data(vorticity(np.load(u_frames[i]), np.load(v_frames[i])))
        else:
            im_main.set_data(np.load(rho_frames[i]))
        artists = [im_main, txt]
        if has_tracer:
            im_tracer.set_data(np.load(tracer_frames[i]))
            artists.append(im_tracer)
        t = i * args.stride * substeps * dt
        txt.set_text(f"{manifest.get('title','')}\nt = {t:.2f}   frame {i*args.stride}/{manifest.get('frames','?')}")
        return tuple(artists)

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
