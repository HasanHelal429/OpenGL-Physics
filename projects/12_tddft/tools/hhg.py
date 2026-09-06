"""Harmonic-comb figure from a 12_tddft --h-hhg run (dipole.csv + meta.txt).
The dipole-acceleration spectrum |a(w)|^2, analysed over the flat portion of
the pulse -- the same as the Stage-1 Python validate.py --phase 5.

    python hhg.py <run_dir> [--out FILE.png]
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
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(args.run_dir, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = float(v)
    wL, f0, f1, Ip = meta["omega_L"], meta["flat0"], meta["flat1"], meta["Ip"]

    d = np.genfromtxt(os.path.join(args.run_dir, "dipole.csv"), delimiter=",", names=True)
    t = np.atleast_1d(d["t"])
    dz = np.atleast_1d(d["dz"])
    dt = t[1] - t[0]

    m = (t >= f0) & (t <= f1)
    tw, dw = t[m], dz[m]
    a = np.gradient(np.gradient(dw, dt), dt)
    a *= np.hanning(len(a))
    n = 1
    while n < 8 * len(a):
        n *= 2
    A = np.fft.rfft(a, n=n)
    w = 2 * np.pi * np.fft.rfftfreq(n, d=dt)
    P = np.abs(A) ** 2
    hn = w / wL

    def peak(h):
        i = int(np.argmin(np.abs(hn - h)))
        return float(np.max(P[max(0, i - 3):i + 4]))

    odd = np.mean([peak(h) for h in (3, 5, 7)])
    even = np.mean([peak(h) for h in (2, 4, 6)])
    plateau = np.median([peak(h) for h in (3, 5)])
    band = (hn > 2) & (hn < 25)
    above = hn[band][P[band] > plateau * 1e-3]
    cutoff = above.max() if above.size else np.nan
    Up = 0.06 ** 2 / (4 * wL ** 2)

    print(f"  odd/even peak ratio = {odd / even:.0f}")
    print(f"  plateau cutoff      = {cutoff:.1f} harmonics")
    print(f"  I_p + 3.17 U_p      = {(Ip + 3.17 * Up) / wL:.1f} harmonics")
    print(f"  (Stage-1 Python: odd-only comb, plateau + sharp cutoff, cutoff")
    print(f"   above I_p+3.17U_p in the over-the-barrier regime)")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=130)
    ax.semilogy(hn, P / np.max(P), color="#333", lw=0.9)
    ax.axvline((Ip + 3.17 * Up) / wL, color="#c44e52", ls="--", lw=1,
               label=f"$I_p + 3.17 U_p$ = {(Ip + 3.17 * Up) / wL:.1f}")
    if np.isfinite(cutoff):
        ax.axvline(cutoff, color="#4c72b0", lw=1, label=f"cutoff {cutoff:.1f}")
    for k in range(1, 25, 2):
        ax.axvline(k, color="#eee", lw=0.5, zorder=0)
    ax.set_xlim(0, 24); ax.set_ylim(1e-11, 3)
    ax.set_xlabel(r"harmonic order  $\omega / \omega_L$")
    ax.set_ylabel(r"$|a(\omega)|^2$  (norm.)")
    ax.set_title(f"12_tddft (GPU) -- H high-harmonic spectrum  (odd/even = {odd/even:.0f})")
    ax.legend(frameon=False, fontsize=8)
    out = args.out or os.path.join(args.run_dir, "hhg.png")
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
