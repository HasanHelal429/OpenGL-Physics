"""
Validation plot for a headless 09_magnetostatics run of decks/helmholtz.toml
(3D Biot-Savart, xz slice): the on-axis field profile and the uniform-region
map for a Helmholtz pair (two loops of radius R separated by R). Checks:
  - centre field vs (4/5)^{3/2} mu0 I / R
  - d^2 B/dz^2 ~ 0 at the centre
  - the 1e-3 uniformity contour of |B - B_centre| / B_centre

Usage:
    python plot_helmholtz.py <results_dir> [--out FILE]
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


def load_frame(d, name):
    fdir = os.path.join(d, "frames")
    ids = sorted({fn.split("_")[-1].split(".")[0]
                  for fn in os.listdir(fdir) if fn.startswith(name + "_")})
    return np.load(os.path.join(fdir, f"{name}_{ids[0]}.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    nx, ny = manifest["grid"]["nx"], manifest["grid"]["ny"]
    lx, ly = deck["grid"]["lx"], deck["grid"]["ly"]
    mu0 = deck.get("physics", {}).get("mu0", 1.0)
    coil = deck["coil"][0]
    R, I = coil["radius"], coil["current"]

    Bx = load_frame(d, "Bx")
    By = load_frame(d, "By")   # world z-component
    Bz = load_frame(d, "Bz")
    Bmag = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)

    xs = -0.5 * lx + (np.arange(nx) + 0.5) * lx / nx
    zs = -0.5 * ly + (np.arange(ny) + 0.5) * ly / ny
    ix0 = int(np.argmin(np.abs(xs)))
    iz0 = int(np.argmin(np.abs(zs)))

    B_axis = By[:, ix0]
    b0 = Bmag[iz0, ix0]
    b0_exact = (4.0 / 5.0) ** 1.5 * mu0 * I / R
    dz = zs[1] - zs[0]
    d2 = (B_axis[iz0 + 1] - 2 * B_axis[iz0] + B_axis[iz0 - 1]) / dz ** 2

    print(f"centre |B| = {b0:.5f}   analytic (4/5)^3/2 mu0 I / R = {b0_exact:.5f}   "
          f"rel err {abs(b0 - b0_exact) / b0_exact:.2e}")
    print(f"d^2 B/dz^2 at centre = {d2:+.3e}   "
          f"(|.| R^2 / B0 = {abs(d2) * R * R / b0:.2e})")

    dev = np.abs(Bmag - b0) / b0

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(zs, B_axis, "-", lw=1.3, label="|B| on axis")
    ax[0].axhline(b0_exact, color="k", ls="--", lw=1, label=r"$(4/5)^{3/2}\mu_0 I/R$")
    ax[0].axvspan(-0.3 * R, 0.3 * R, color="0.85", zorder=0)
    ax[0].set_xlabel("z (on axis)"); ax[0].set_ylabel("|B|")
    ax[0].set_title("on-axis profile"); ax[0].legend(fontsize=9)
    ax[0].set_xlim(-2 * R, 2 * R)

    ext = [-lx / 2, lx / 2, -ly / 2, ly / 2]
    im = ax[1].imshow(np.clip(dev, 0, 0.02), origin="lower", extent=ext,
                      cmap="viridis")
    cs = ax[1].contour(dev, levels=[1e-3, 5e-3, 1e-2], colors="w",
                       linewidths=0.8, extent=ext)
    ax[1].clabel(cs, fmt="%.0e", fontsize=7)
    ax[1].set_xlim(-1.6 * R, 1.6 * R); ax[1].set_ylim(-1.6 * R, 1.6 * R)
    ax[1].set_xlabel("x"); ax[1].set_ylabel("z")
    ax[1].set_title(r"$|B - B_0|/B_0$")
    fig.colorbar(im, ax=ax[1], shrink=0.8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "helmholtz_validation.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
