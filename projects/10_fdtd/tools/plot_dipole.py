"""
2D Green's-function validation for a headless 10_fdtd run of decks/dipole.toml:
extract the steady-state Ez phasor by a single-bin DFT at the source
frequency, then compare the radial cut A(r) to the 2D scalar Green's
function. The phasor is taken in the e^{-i w t} convention (Ez ~
Re[A e^{-i w t}]), so the outgoing wave is H0^(1): G(r) = (i/4) H0^(1)(k r).

Usage:
    python plot_dipole.py <results_dir> [--out FILE]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    man = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)
    nx, ny = man["grid"]["nx"], man["grid"]["ny"]
    dt = man["dt"]
    sub = man.get("substeps_per_frame", 1)
    frame_dt = dt * sub
    f0 = deck["source"][0]["f0"]
    c = deck.get("physics", {}).get("c", 1.0)
    k = 2.0 * np.pi * f0 / c
    sx = int(round(deck["source"][0]["x"]))
    sy = int(round(deck["source"][0]["y"]))

    fdir = os.path.join(d, "frames")
    ids = sorted(int(fn.split("_")[-1].split(".")[0])
                 for fn in os.listdir(fdir) if fn.startswith("Ez_"))
    # use the last ~6 periods for the phasor fit
    period_frames = max(4, int(round(1.0 / f0 / frame_dt)))
    use = ids[-6 * period_frames:]
    acc = np.zeros((ny, nx), dtype=complex)
    for fid in use:
        a = np.load(os.path.join(fdir, f"Ez_{fid:04d}.npy"))
        t = fid * frame_dt
        acc += a * np.exp(1j * 2.0 * np.pi * f0 * t)
    phasor = 2.0 * acc / len(use)     # Ez ~ Re[phasor * exp(-i w t)]

    ys, xs = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xs - sx, ys - sy)

    # radial cut: azimuthally averaged |phasor| and phase vs r
    pml = deck["boundary"].get("pml_cells", 10)
    rmax = min(sx, sy, nx - sx, ny - sy) - pml - 4
    rbins = np.arange(3, rmax)

    # azimuthally-averaged complex phasor vs r, and the Green's-function model
    num = np.array([phasor[(r >= rr) & (r < rr + 1)].mean() for rr in rbins])
    G = 0.25j * hankel1(0, k * rbins)   # e^{-i w t} convention -> outgoing = H0^(1)

    # one complex scale factor (source strength) fitted over the mid-range
    lo, hi = int(0.25 * len(rbins)), int(0.85 * len(rbins))
    scale = np.vdot(G[lo:hi], num[lo:hi]) / np.vdot(G[lo:hi], G[lo:hi])
    model = scale * G
    rel = np.abs(num[lo:hi] - model[lo:hi]) / np.abs(model[lo:hi])
    corr = np.abs(np.vdot(num[lo:hi], model[lo:hi])) / (
        np.linalg.norm(num[lo:hi]) * np.linalg.norm(model[lo:hi]))
    print(f"k = {k:.4f}   fit region kr in [{k*rbins[lo]:.1f}, {k*rbins[hi]:.1f}]")
    print(f"  |Ez| vs |G|:  median rel err {np.median(rel):.4f}  max {np.max(rel):.4f}")
    print(f"  complex correlation (amp+phase): {corr:.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    m = np.abs(phasor).max()
    ax[0].imshow(phasor.real, origin="lower", cmap="RdBu_r", vmin=-0.3*m, vmax=0.3*m)
    ax[0].set_title("Re(Ez phasor)"); ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].loglog(rbins, np.abs(num), ".", ms=4, label="FDTD |Ez|")
    ax[1].loglog(rbins, np.abs(model), "k-", lw=1, label=r"$|(i/4)H_0^{(1)}(kr)|$ (fit)")
    ax[1].loglog(rbins, np.abs(model[hi]) * np.sqrt(rbins[hi] / rbins), "--",
                 color="0.6", lw=0.8, label=r"$1/\sqrt{r}$")
    ax[1].set_xlabel("r (cells)"); ax[1].set_ylabel("|Ez|")
    ax[1].set_title("radial amplitude"); ax[1].legend(fontsize=8)

    ax[2].plot(rbins, np.unwrap(np.angle(num)), ".", ms=4, label="FDTD phase")
    ax[2].plot(rbins, np.unwrap(np.angle(model)), "k-", lw=1, label="Green's fn phase")
    ax[2].set_xlabel("r (cells)"); ax[2].set_ylabel("arg(Ez)")
    ax[2].set_title("radial phase"); ax[2].legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "dipole_validation.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
