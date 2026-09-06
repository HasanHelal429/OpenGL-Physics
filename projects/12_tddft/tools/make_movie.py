"""Watch the high-harmonic comb build up: an MP4 from a 12_tddft --h-hhg run.
Three panels -- the laser E(t) and dipole d(t), the running ionized fraction,
and the harmonic spectrum computed from the pulse *so far*, growing as time
advances.

    python make_movie.py <run_dir> [--out FILE.mp4] [--fps 20]

Uses imageio's bundled ffmpeg; falls back to a PNG strip.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(args.run_dir, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = float(v)
    wL, f0, f1, Ip = meta["omega_L"], meta["flat0"], meta["flat1"], meta["Ip"]
    E0 = meta.get("E0", 0.06)

    d = np.genfromtxt(os.path.join(args.run_dir, "dipole.csv"), delimiter=",", names=True)
    t = np.atleast_1d(d["t"])
    dz = np.atleast_1d(d["dz"])
    dt = t[1] - t[0]

    # reconstruct the flat-top pulse for display
    tc = 2 * np.pi / wL
    tup, tflat = f0, f1 - f0
    tpulse = 2 * tup + tflat

    def E(tt):
        tt = np.asarray(tt)
        env = np.where(tt < tup, np.sin(np.pi * tt / (2 * tup)) ** 2,
                       np.where(tt < tup + tflat, 1.0,
                                np.sin(np.pi * np.clip(tpulse - tt, 0, None) / (2 * tup)) ** 2))
        env = np.where((tt < 0) | (tt > tpulse), 0.0, env)
        return E0 * env * np.sin(wL * tt)

    Up = E0 ** 2 / (4 * wL ** 2)
    cut_pred = (Ip + 3.17 * Up) / wL

    dmax = np.abs(dz).max()
    fig, (ax, axs) = plt.subplots(2, 1, figsize=(7, 6), dpi=110)
    frames = list(range(max(20, len(t) // 200), len(t), max(1, len(t) // 160)))

    def render(i):
        for a in (ax, axs):
            a.clear()
        ax.plot(t[:i], E(t[:i]) / E0 * dmax, color="#bbb", lw=0.7, label="E(t) (scaled)")
        ax.plot(t[:i], dz[:i], color="#4c72b0", lw=0.8, label=r"$d_z(t)$")
        ax.axvspan(f0, f1, color="#dd8452", alpha=0.1, label="flat-top")
        ax.set_xlim(t[0], t[-1]); ax.set_ylim(-1.15 * dmax, 1.15 * dmax)
        ax.set_xlabel("t  (a.u.)"); ax.set_ylabel(r"$\langle z\rangle$")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        ax.set_title(f"12_tddft (GPU) -- H HHG   t = {t[i]:.0f} a.u.")

        m = (t[:i] >= f0) & (t[:i] <= min(f1, t[i - 1]))
        if m.sum() > 32:
            tw2, dw2 = t[:i][m], dz[:i][m]
            a2 = np.gradient(np.gradient(dw2, dt), dt) * np.hanning(len(dw2))
            npad = 1
            while npad < 8 * len(a2):
                npad *= 2
            P = np.abs(np.fft.rfft(a2, n=npad)) ** 2
            hn = 2 * np.pi * np.fft.rfftfreq(npad, d=dt) / wL
            axs.semilogy(hn, P / P.max() + 1e-12, color="#333", lw=0.8)
        axs.axvline(cut_pred, color="#c44e52", ls="--", lw=1)
        for k in range(1, 25, 2):
            axs.axvline(k, color="#eee", lw=0.5, zorder=0)
        axs.set_xlim(0, 22); axs.set_ylim(1e-9, 3)
        axs.set_xlabel(r"harmonic order  $\omega/\omega_L$"); axs.set_ylabel(r"$|a(\omega)|^2$")
        fig.tight_layout()
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[:, :, :3].copy()

    out = args.out or os.path.join(args.run_dir, "hhg_movie.mp4")
    try:
        import imageio
        with imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8, macro_block_size=8) as wtr:
            for i in frames:
                wtr.append_data(render(i))
        print(f"wrote {out}")
    except Exception as e:
        seq = os.path.join(args.run_dir, "movie_frames")
        os.makedirs(seq, exist_ok=True)
        print(f"imageio/ffmpeg unavailable ({e}); PNG strip -> {seq}")
        for j, i in enumerate(frames[::8]):
            plt.imsave(os.path.join(seq, f"f_{j:03d}.png"), render(i))


if __name__ == "__main__":
    main()
