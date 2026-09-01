#!/usr/bin/env python3
"""Turn a headless 04_molecular_dynamics run into a movie: a periodic (x,y)
projection of the 3D box, particles colored by instantaneous speed (default)
or by species for a binary-mixture run (auto-detected from a
frames/species_0000.npy file; override with --color).

    python make_movie.py <results_dir> [--fps 24] [--stride 2] [--color auto|speed|species] [--out movie.mp4]

The box outline is drawn at each frame's OWN box_length, but the axes stay
fixed at the run's LARGEST box_length throughout -- so a barostat run
(decks/npt_liquid.toml) visibly shows the box shrinking/growing within a
constant view, instead of always filling the frame edge-to-edge (which
would hide exactly the thing worth seeing there).

Reads manifest.json + frames/pos_*.npy (+ vel_*.npy for speed coloring,
species_0000.npy if present) + diagnostics.csv (box_length, temperature,
target_t, lindemann -- shown as a text overlay, whichever columns exist).
Uses imageio (bundled ffmpeg via imageio-ffmpeg) so this works even without
a system ffmpeg; falls back to a PNG sequence if that import fails --
same pattern as 06_tidal_disruption/07_grhd's make_movie.py.
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
    sys.exit("make_movie: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame (speeds up long runs)")
    ap.add_argument("--color", choices=["auto", "speed", "species"], default="auto")
    ap.add_argument("--point-size", type=float, default=8.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    pos_files = sorted(glob.glob(os.path.join(d, "frames", "pos_*.npy")))[:: args.stride]
    if not pos_files:
        sys.exit("no pos_*.npy frames found")
    vel_files = sorted(glob.glob(os.path.join(d, "frames", "vel_*.npy")))[:: args.stride]

    species_path = os.path.join(d, "frames", "species_0000.npy")
    has_species = os.path.exists(species_path)
    color_mode = args.color
    if color_mode == "auto":
        color_mode = "species" if has_species else "speed"
    if color_mode == "species" and not has_species:
        sys.exit("make_movie: --color species requested but no frames/species_0000.npy in this run")

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    box_length_all = data["box_length"]
    n_frames = len(pos_files)
    # diagnostics.csv has one row per frame written (not subsampled by
    # --stride), so index by the ACTUAL frame number encoded in the
    # filename, not by position in the strided list.
    frame_idx = [int(os.path.basename(p).split("_")[-1].split(".")[0]) for p in pos_files]
    box_length = box_length_all[frame_idx]
    max_L = float(box_length_all.max())

    species = np.load(species_path).ravel() if has_species else None

    if color_mode == "speed":
        all_vel = np.stack([np.load(f) for f in vel_files])
        speed = np.linalg.norm(all_vel, axis=-1)
        vmax = float(np.percentile(speed, 99.0))  # robust to rare fast outliers, not the raw max
        cmap = plt.get_cmap("plasma")

        def colors_for(i):
            t = np.clip(speed[i] / max(vmax, 1e-9), 0.0, 1.0)
            return cmap(t)
    else:
        palette = np.array([[0.35, 0.55, 1.0, 1.0], [1.0, 0.45, 0.25, 1.0]])  # A: blue, B: orange

        def colors_for(i):
            return palette[species]

    diag_cols = [c for c in ("temperature", "target_t", "lindemann") if c in data.dtype.names]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("#08080c")
    ax.set_facecolor("#08080c")
    half = max_L * 0.55
    center = max_L * 0.5
    ax.set_xlim(center - half, center + half)
    ax.set_ylim(center - half, center + half)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    pos0 = np.load(pos_files[0])
    scat = ax.scatter(pos0[:, 0], pos0[:, 1], c=colors_for(0), s=args.point_size, linewidths=0)
    box_line, = ax.plot([], [], color="#7fa8d9", linewidth=1.0, alpha=0.6)
    txt = ax.text(0.03, 0.97, "", transform=ax.transAxes, color="w", va="top", fontsize=9, family="monospace")

    title = manifest.get("title", "")

    def update(i):
        pos = np.load(pos_files[i])
        scat.set_offsets(pos[:, :2])
        scat.set_facecolor(colors_for(i))
        L = box_length[i]
        box_line.set_data([0, L, L, 0, 0], [0, 0, L, L, 0])
        t = data["t"][frame_idx[i]]  # read, not recomputed -- avoids re-deriving dt*substeps_per_frame here
        lines = [title, f"t = {t:.2f}   frame {frame_idx[i]}/{manifest.get('frames', '?')}   L = {L:.3f}"]
        for c in diag_cols:
            lines.append(f"{c} = {data[c][frame_idx[i]]:.4f}")
        txt.set_text("\n".join(lines))
        return scat, box_line, txt

    out = args.out or os.path.join(d, "movie.mp4")
    try:
        import imageio
        writer_kwargs = dict(fps=args.fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=None)
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
            fig.savefig(os.path.join(pdir, f"{i:04d}.png"), facecolor=fig.get_facecolor())
        print("wrote", pdir)


if __name__ == "__main__":
    main()
