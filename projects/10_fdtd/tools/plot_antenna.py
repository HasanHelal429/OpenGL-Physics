"""
Half-wave PEC dipole radiation pattern for 10_fdtd
(decks/dipole_antenna.toml): a near-to-far transform of the steady-state
radiated phasor field, against the analytic pattern of a centre-fed
half-wave dipole along y,

    P(phi) ~ | cos((pi/2) sin phi) / cos phi |^2

(broadside peak along +/- x, nulls along the dipole axis).

Usage:
    python plot_antenna.py <results_dir> [--out FILE]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def load_phasor(d, name, f0):
    man = json.load(open(os.path.join(d, "manifest.json")))
    dt = man["dt"] * man.get("substeps_per_frame", 1)
    fdir = os.path.join(d, "frames")
    ids = sorted(int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(fdir) if f.startswith(name + "_"))
    per = max(4, int(round(1.0 / f0 / dt)))
    use = ids[-8 * per:]
    acc = 0.0
    for fid in use:
        acc = acc + np.load(os.path.join(fdir, f"{name}_{fid:04d}.npy")) * \
            np.exp(1j * 2 * np.pi * f0 * fid * dt)
    return 2.0 * acc / len(use)


def ntff(Ez, Hx, Hy, cx, cy, half, k, phis):
    out = np.zeros(len(phis), complex)
    xs = np.arange(cx - half, cx + half + 1, dtype=int)
    ys = np.arange(cy - half, cy + half + 1, dtype=int)
    edges = [
        (np.full_like(ys, cx + half), ys, (1.0, 0.0)),
        (np.full_like(ys, cx - half), ys, (-1.0, 0.0)),
        (xs, np.full_like(xs, cy + half), (0.0, 1.0)),
        (xs, np.full_like(xs, cy - half), (0.0, -1.0)),
    ]
    for px, py, (nx_, ny_) in edges:
        ez, hx, hy = Ez[py, px], Hx[py, px], Hy[py, px]
        Jz = nx_ * hy - ny_ * hx
        for m, phi in enumerate(phis):
            fx, fy = np.cos(phi), np.sin(phi)
            Mphi = ez * (nx_ * fx + ny_ * fy)
            ph = np.exp(-1j * k * ((px - cx) * fx + (py - cy) * fy))
            out[m] += np.sum((Jz - Mphi) * ph)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    with open(os.path.join(d, "deck.toml"), "rb") as f:
        cfg = tomllib.load(f)
    f0 = cfg["source"][0]["f0"]
    cx = int(cfg["source"][0]["x"])
    cy = int(cfg["source"][0]["y"])
    pml = cfg["boundary"]["pml_cells"]
    k = 2 * np.pi * f0

    Ez = load_phasor(d, "Ez", f0)
    Hx = load_phasor(d, "Hx", f0)
    Hy = load_phasor(d, "Hy", f0)

    half = min(cx, cy, Ez.shape[1] - cx, Ez.shape[0] - cy) - pml - 8
    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    P = np.abs(ntff(Ez, Hx, Hy, cx, cy, half, k, phis)) ** 2
    P /= P.max()

    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.abs(np.cos(0.5 * np.pi * np.sin(phis)) / np.cos(phis)) ** 2
    A[~np.isfinite(A)] = 0.0
    A /= A.max()

    corr = np.abs(np.vdot(P, A)) / (np.linalg.norm(P) * np.linalg.norm(A))
    print(f"pattern correlation vs analytic half-wave dipole: {corr:.4f}")

    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(phis, A, "k-", lw=1, label=r"$|\cos(\frac{\pi}{2}\sin\phi)/\cos\phi|^2$")
    ax.plot(phis, P, ".", ms=3, label="FDTD NTFF")
    ax.set_title("half-wave dipole (along y) radiation pattern")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
    fig.tight_layout()
    out = args.out or os.path.join(d, "antenna_pattern.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
