"""Genuine far-field angular pattern from a headless 11_retarded_fields run,
rendered as a clean filled polar shape (dark bg, no axes/legend/title) --
same measurement as tools/plot_pattern.py (sample |E| on a circle of radius
r0, time-average (|E| r0)^2 over the second half of frames to drop the
fill-in transient), just restyled as a render instead of a data plot.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import tomllib
except ImportError:
    import tomli as tomllib

ap = argparse.ArgumentParser()
ap.add_argument("results_dir")
ap.add_argument("out")
ap.add_argument("--r0", type=float, default=None)
ap.add_argument("--field", default="E_mag")
args = ap.parse_args()
d = args.results_dir

with open(os.path.join(d, "deck.toml"), "rb") as f:
    deck = tomllib.load(f)
lx = deck["domain"]["lx"]
ly = deck["domain"].get("ly", lx)

fdir = os.path.join(d, "frames")
ids = sorted(int(f.split("_")[-1].split(".")[0])
             for f in os.listdir(fdir) if f.startswith(args.field + "_"))
frames = [np.load(os.path.join(fdir, f"{args.field}_{i:04d}.npy")) for i in ids]
ny, nx = frames[0].shape

r0 = args.r0 or 0.42 * min(lx, ly)
ph = np.linspace(0, 2 * np.pi, 721)
px = r0 * np.cos(ph)
py = r0 * np.sin(ph)
ix = np.clip(((px + 0.5 * lx) / lx * nx).astype(int), 0, nx - 1)
iy = np.clip(((py + 0.5 * ly) / ly * ny).astype(int), 0, ny - 1)

half = len(frames) // 2
acc = np.zeros_like(ph)
for fr in frames[half:]:
    acc += (fr[iy, ix] * r0) ** 2
acc /= len(frames[half:])
acc /= acc.max()

fig = plt.figure(figsize=(6.4, 6.4), dpi=150)
fig.patch.set_facecolor("#0a0a12")
ax = fig.add_subplot(111, projection="polar")
ax.set_facecolor("#0a0a12")
ax.fill(ph, acc, color="#ffb04d", alpha=0.35, zorder=2)
ax.plot(ph, acc, color="#ffb04d", lw=2.2, zorder=3)
ax.plot(0, 0, "o", ms=5, color="#ffe8b0", zorder=4)
ax.set_rticks([])
ax.set_xticks([])
ax.spines["polar"].set_visible(False)
ax.grid(False)
fig.savefig(args.out, facecolor=fig.get_facecolor())
print("wrote", args.out, " r0 =", r0)
