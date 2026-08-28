#!/usr/bin/env python3
"""Turn a headless 05_tdse_gpu run into a movie.

    python make_movie.py <results_dir> [--mode density|phase|real]
                         [--fps 30] [--exposure 6] [--stride 1] [--out movie.mp4]

Reads manifest.json + frames/psi_*.npy (complex64) + frames/potential_0000.npy.
Writes an mp4 if ffmpeg is available, otherwise a PNG sequence in <dir>/frames_png/.
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
    from matplotlib.colors import hsv_to_rgb
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib (pip install -r requirements.txt)")


def colorize(psi, mode, gain, gamma, ref, pot, vref):
    # Brightness = probability density auto-scaled to a reference peak, then a
    # gamma < 1 lifts the low end so tails and fringe minima stay visible.
    mag2 = np.abs(psi) ** 2
    rel = np.clip(gain * mag2 / max(ref, 1e-30), 0.0, 1.0)
    dens = rel ** gamma
    if mode == "phase":
        h = (np.angle(psi) / (2 * np.pi) + 0.5) % 1.0
        rgb = hsv_to_rgb(np.stack([h, 0.82 * np.ones_like(h), dens], axis=-1))
    elif mode == "real":
        s = np.clip(gain * psi.real / np.sqrt(max(ref, 1e-30)), -1.0, 1.0)
        s = np.sign(s) * np.abs(s) ** gamma
        blue = np.array([0.15, 0.30, 0.90])
        red = np.array([0.95, 0.25, 0.20])
        t = (0.5 + 0.5 * s)[..., None]
        rgb = blue * (1 - t) + red * t
        rgb *= (0.25 + 0.75 * np.abs(s))[..., None]
    else:  # density
        rgb = plt.cm.magma(dens)[..., :3]

    if pot is not None and vref > 0:
        tint = np.clip(np.abs(pot) / vref, 0, 1)[..., None]
        rgb = rgb * (1 - 0.25 * tint) + np.array([0.55, 0.55, 0.60]) * (0.25 * tint)
    return np.clip(rgb, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--mode", choices=["density", "phase", "real"], default="phase")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gain", type=float, default=1.15, help="linear push before the gamma curve")
    ap.add_argument("--gamma", type=float, default=0.5,
                    help="density brightness curve; <1 lifts tails/fringe minima")
    ap.add_argument("--normalize", choices=["frame", "global"], default="frame",
                    help="frame: auto-scale each frame to its own peak (matches the live view); "
                         "global: one peak for the whole run (packet visibly dims as it spreads)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    frames = sorted(glob.glob(os.path.join(d, "frames", "psi_*.npy")))[:: args.stride]
    if not frames:
        sys.exit("no psi_*.npy frames found")

    pot_path = os.path.join(d, "frames", "potential_0000.npy")
    pot = np.load(pot_path) if os.path.exists(pot_path) else None
    vref = float(np.abs(pot).max()) if pot is not None else 0.0

    gmax = 0.0
    if args.normalize == "global":
        for f in frames:
            gmax = max(gmax, float((np.abs(np.load(f)) ** 2).max()))

    def ref_for(psi):
        return gmax if args.normalize == "global" else float((np.abs(psi) ** 2).max())

    lx = manifest["grid"]["lx"]
    ly = manifest["grid"]["ly"]
    fig, ax = plt.subplots(figsize=(6, 6 * ly / lx), dpi=110)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    psi0 = np.load(frames[0])
    im = ax.imshow(colorize(psi0, args.mode, args.gain, args.gamma, ref_for(psi0), pot, vref),
                   origin="lower", extent=[-lx / 2, lx / 2, -ly / 2, ly / 2], interpolation="nearest")
    txt = ax.text(0.02, 0.97, "", transform=ax.transAxes, color="w", va="top", fontsize=9,
                  family="monospace")

    def update(i):
        psi = np.load(frames[i])
        im.set_data(colorize(psi, args.mode, args.gain, args.gamma, ref_for(psi), pot, vref))
        txt.set_text(f"{manifest.get('title','')}\nframe {i*args.stride}/{manifest.get('frames','?')}")
        return im, txt

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    out = args.out or os.path.join(d, f"movie_{args.mode}.mp4")
    try:
        anim.save(out, writer=animation.FFMpegWriter(fps=args.fps, bitrate=4000))
        print("wrote", out)
    except Exception as e:  # noqa: BLE001
        print("ffmpeg unavailable (%s); writing PNG sequence instead" % e)
        pdir = os.path.join(d, "frames_png")
        os.makedirs(pdir, exist_ok=True)
        for i in range(len(frames)):
            update(i)
            fig.savefig(os.path.join(pdir, f"{i:04d}.png"))
        print("wrote", pdir)


if __name__ == "__main__":
    main()
