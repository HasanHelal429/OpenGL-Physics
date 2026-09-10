"""
Animate the field frames of a headless 11_retarded_fields run. By default it
shows |E| times distance from the box centre, so the radiation pattern reads
instead of the 1/r^2 near-field spike (matching the interactive view's
r-weight mode). Uses imageio's bundled ffmpeg; PNG-sequence fallback.

Each charge is drawn as a small dot (colour by sign: warm for q>0, cool for
q<0) with a full trajectory trail behind it, so the source is as easy to
track as the field it's making. For a "prescribed" deck the trajectory is
the analytic path (ChargePath.hpp's formulas, reproduced here); for a
"self_consistent" deck it's read straight from diagnostics.csv's x_i/y_i
columns.

Usage:
    python make_movie.py <results_dir> [--out FILE.mp4] [--fps 30]
                         [--field E_mag] [--no-rweight] [--stride N]
                         [--no-trail]
"""

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None

POS_COLOR = "#ff6a3d"
NEG_COLOR = "#4fa8ff"


def charge_pos(ch, t):
    """Reproduce ChargePath::r(t) for one [[charge]] table (prescribed mode)."""
    kind = ch.get("path", "circular")
    center = ch.get("center", [0.0, 0.0, 0.0])
    omega = ch.get("omega", 1.0)
    phase = ch.get("phase", 0.0)
    p = omega * t + phase
    if kind == "static":
        return center[0], center[1]
    if kind == "uniform":
        v0 = ch.get("v0", [0.0, 0.0, 0.0])
        return center[0] + v0[0] * t, center[1] + v0[1] * t
    if kind == "linear_oscillator":
        axis = ch.get("axis", [0.0, 1.0, 0.0])
        amp = ch.get("amplitude", 1.0)
        s = amp * math.sin(p)
        return center[0] + axis[0] * s, center[1] + axis[1] * s
    if kind == "figure8":
        radius = ch.get("radius", 1.0)
        amp = ch.get("amplitude", 1.0)
        return (center[0] + radius * math.sin(p),
                center[1] + amp * math.sin(2.0 * p))
    # circular (default)
    radius = ch.get("radius", 1.0)
    return center[0] + radius * math.cos(p), center[1] + radius * math.sin(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--field", default="E_mag")
    ap.add_argument("--no-rweight", action="store_true")
    ap.add_argument("--no-trail", action="store_true",
                     help="draw the charge marker(s) but not the trajectory trail")
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

    # --- charge trajectories: self_consistent reads diagnostics.csv,
    # prescribed evaluates the analytic path at each frame's time ---
    mode = dcfg.get("mode", {})
    self_consistent = (mode.get("type", "prescribed") == "self_consistent"
                        if isinstance(mode, dict) else mode == "self_consistent")
    charges = dcfg.get("charge", [])
    diag = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t_all = np.atleast_1d(diag["t"])

    tracks = []  # list of (q, xs[frame], ys[frame]) for each charge
    if self_consistent:
        cols = diag.dtype.names
        i = 0
        while f"x{i}" in cols:
            xs = np.atleast_1d(diag[f"x{i}"])[ids]
            ys = np.atleast_1d(diag[f"y{i}"])[ids]
            q = charges[i].get("q", 1.0) if i < len(charges) else 1.0
            tracks.append((q, xs, ys))
            i += 1
    else:
        ts = t_all[ids]
        for ch in charges:
            xs = np.empty(len(ts))
            ys = np.empty(len(ts))
            for k, tt in enumerate(ts):
                xs[k], ys[k] = charge_pos(ch, tt)
            tracks.append((ch.get("q", 1.0), xs, ys))

    def world_to_px(x, y):
        return (x + 0.5 * lx) / lx * nx - 0.5, (y + 0.5 * ly) / ly * ny - 0.5

    signed = args.field in ("Ex", "Ey", "Ez", "Bz")
    rweight = (not args.no_rweight) and not signed
    if rweight:
        xs_ = (np.arange(nx) + 0.5) / nx * lx - 0.5 * lx
        ys_ = (np.arange(ny) + 0.5) / ny * ly - 0.5 * ly
        rr = np.hypot(*np.meshgrid(xs_, ys_))
        rnear = max(1.0, 3.0 * lx / nx)
        w = np.where(rr < rnear, 0.0, rr)
        frames = [f * w for f in frames]

    stack = np.abs(np.stack(frames))
    vmax = np.percentile(stack, 99.5)
    cmap = "RdBu_r" if signed else "magma"

    fig, ax = plt.subplots(figsize=(6.4, 6.4 * ny / nx), dpi=110)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(-0.5, ny - 0.5)
    im = ax.imshow(frames[0], origin="lower", cmap=cmap,
                   vmin=(-vmax if signed else 0), vmax=vmax)

    trails = []
    dots = []
    for q, xs, ys in tracks:
        col = POS_COLOR if q >= 0 else NEG_COLOR
        if not args.no_trail:
            trail, = ax.plot([], [], "-", lw=1.1, color=col, alpha=0.55, zorder=4)
            trails.append(trail)
        dot, = ax.plot([], [], "o", ms=6.5, mfc=col, mec="white", mew=0.6, zorder=5)
        dots.append(dot)

    def render(k):
        im.set_data(frames[k])
        for (q, xs, ys), dot in zip(tracks, dots):
            px, py = world_to_px(xs[k], ys[k])
            dot.set_data([px], [py])
        if not args.no_trail:
            for (q, xs, ys), trail in zip(tracks, trails):
                pxs, pys = world_to_px(xs[: k + 1], ys[: k + 1])
                trail.set_data(pxs, pys)
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
