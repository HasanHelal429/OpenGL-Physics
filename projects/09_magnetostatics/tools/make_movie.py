"""
Animate a headless 09_magnetostatics particle run: the frame-0 |B| field as
a background, and each charge's position + growing trail from
diagnostics.csv. Works for any pusher deck (cyclotron / exb_drift /
magnetic_bottle). Uses imageio's bundled ffmpeg (not the system PATH);
falls back to a PNG sequence if that is unavailable.

Usage:
    python make_movie.py <results_dir> [--out FILE.mp4] [--fps 30] [--stride 1]
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


def load_frame(d, name):
    fdir = os.path.join(d, "frames")
    if not os.path.isdir(fdir):
        return None
    hits = [f for f in os.listdir(fdir) if f.startswith(name + "_")]
    return np.load(os.path.join(fdir, sorted(hits)[0])) if hits else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--trail", type=int, default=500, help="trail length (frames)")
    args = ap.parse_args()
    d = args.results_dir

    man = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)
    lx, ly = deck["grid"]["lx"], deck["grid"]["ly"]
    plane = deck.get("grid", {}).get("plane", "xy")

    Bx, By, Bz = (load_frame(d, n) for n in ("Bx", "By", "Bz"))
    Bmag = None
    if Bx is not None:
        Bmag = np.sqrt(Bx ** 2 + By ** 2 + (Bz ** 2 if Bz is not None else 0))

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    cols = data.dtype.names
    nq = 0
    while f"q{nq}_x" in cols:
        nq += 1

    ax_a, ax_b = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}[plane]
    ext = [-lx / 2, lx / 2, -ly / 2, ly / 2]
    Ax = [data[f"q{k}_{ax_a}"] for k in range(nq)]
    Ay = [data[f"q{k}_{ax_b}"] for k in range(nq)]
    t = data["t"]
    colors = plt.cm.autumn(np.linspace(0.0, 0.8, max(nq, 1)))

    fig, ax = plt.subplots(figsize=(6.4, 6.4 * ly / lx), dpi=110)
    fig.subplots_adjust(left=0.11, right=0.98, top=0.95, bottom=0.1)
    if Bmag is not None and np.ptp(Bmag) > 0:
        ax.imshow(Bmag, origin="lower", extent=ext, cmap="magma",
                  vmax=np.percentile(Bmag, 99))
    else:
        ax.set_facecolor("#08080c")
    ax.set_xlim(-lx / 2, lx / 2)
    ax.set_ylim(-ly / 2, ly / 2)
    ax.set_xlabel(ax_a)
    ax.set_ylabel(ax_b)
    ax.set_aspect("equal")
    trails = [ax.plot([], [], "-", lw=1.1, color=colors[k])[0] for k in range(nq)]
    heads = [ax.plot([], [], "o", ms=5, mfc="white", mec=colors[k])[0]
             for k in range(nq)]
    title = ax.set_title("")
    fig.canvas.draw()

    frame_ids = list(range(0, len(t), args.stride))

    def render(i):
        lo = max(0, i - args.trail)
        for k in range(nq):
            trails[k].set_data(Ax[k][lo:i + 1], Ay[k][lo:i + 1])
            heads[k].set_data([Ax[k][i]], [Ay[k][i]])
        title.set_text(f"t = {t[i]:.2f}")
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf.reshape(h, w, 4)[:, :, :3].copy()

    out = args.out or os.path.join(d, "movie.mp4")
    try:
        import imageio
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=8) as writer:
            for n, i in enumerate(frame_ids):
                writer.append_data(render(i))
                if n % 40 == 0:
                    print(f"\r  frame {n}/{len(frame_ids)}", end="", flush=True)
        print(f"\nwrote {out}")
    except Exception as e:
        seq = os.path.join(d, "frames_png")
        os.makedirs(seq, exist_ok=True)
        print(f"\nimageio/ffmpeg unavailable ({e}); PNG sequence -> {seq}")
        for n, i in enumerate(frame_ids):
            plt.imsave(os.path.join(seq, f"f_{n:05d}.png"), render(i))


if __name__ == "__main__":
    main()
