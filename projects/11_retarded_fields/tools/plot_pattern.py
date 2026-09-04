"""
Angular radiation pattern from a headless 11_retarded_fields run.

Samples |E| on a circle of radius r0 (a few wavelengths, well inside the
box), time-averages (|E| r0)^2 over the available frames, and plots the
polar pattern. Overlays:
    --dipole BETA     larmor_ref perp/par/nonrel pattern
    --array DECK      array_factor_ref pattern for that deck

Usage:
    python plot_pattern.py <results_dir> [--r0 R] [--dipole 0.6]
                           [--array decks/dipole_array.toml] [--out FILE]
"""

import argparse
import os

import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--r0", type=float, default=None)
    ap.add_argument("--field", default="E_mag")
    ap.add_argument("--dipole", action="store_true",
                    help="overlay the sin^2(theta) dipole pattern about the "
                         "first charge's oscillation axis")
    ap.add_argument("--array", default=None, help="overlay array_factor_ref for this deck")
    ap.add_argument("--out", default=None)
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
    # nearest-cell sampling
    ix = np.clip(((px + 0.5 * lx) / lx * nx).astype(int), 0, nx - 1)
    iy = np.clip(((py + 0.5 * ly) / ly * ny).astype(int), 0, ny - 1)

    half = len(frames) // 2
    acc = np.zeros_like(ph)
    for fr in frames[half:]:                    # skip the fill-in transient
        acc += (fr[iy, ix] * r0) ** 2
    acc /= len(frames[half:])
    acc /= acc.max()

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.5, 5.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(ph, acc, lw=1.8, label="simulation  <(|E| r0)^2>")

    if args.dipole:
        ax_v = np.array(deck["charge"][0].get("axis", [0, 1, 0])[:2], float)
        ax_v /= np.linalg.norm(ax_v)
        nhat = np.stack([np.cos(ph), np.sin(ph)], axis=1)
        g = 1.0 - (nhat @ ax_v) ** 2                      # sin^2 from the axis
        ax.plot(ph, g / g.max(), "r--", lw=1.2, label="sin^2(theta)  (non-rel dipole)")
    if args.array is not None:
        with open(args.array, "rb") as f:
            adk = tomllib.load(f)
        ch = adk["charge"]
        k = ch[0].get("omega", 1.0) / adk.get("domain", {}).get("c", 1.0)
        pos = np.array([cc.get("center", [0, 0, 0]) for cc in ch])[:, :2]
        psi = np.array([cc.get("phase", 0.0) for cc in ch])
        axis = np.array(ch[0].get("axis", [0, 1, 0])[:2], float)
        axis /= np.linalg.norm(axis)
        nhat = np.stack([np.cos(ph), np.sin(ph)], axis=1)
        af = np.abs(np.sum(np.exp(1j * (k * nhat @ pos.T + psi)), axis=1)) ** 2
        patt = (1 - (nhat @ axis) ** 2) * af
        ax.plot(ph, patt / patt.max(), "g--", lw=1.2, label="array factor")

    ax.set_title(f"radiation pattern at r0 = {r0:.1f}", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(d, "pattern.png")
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()
