"""
Validation plot for a headless 09_magnetostatics run of decks/loop.toml (3D
Biot-Savart, xz slice): the on-axis field against the analytic current-loop
result
    B_z(z) = mu0 * I * R^2 / [2 (R^2 + z^2)^{3/2}]
On the xz slice, grid axis 1 (the "By" field) is world z, so B along the axis
lives in the By frame at the x = 0 column.

Usage:
    python plot_loop.py <results_dir> [--out FILE]
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

    By = load_frame(d, "By")   # world z-component on the xz slice
    Bmag = np.sqrt(load_frame(d, "Bx") ** 2 + By ** 2 + load_frame(d, "Bz") ** 2)

    xs = -0.5 * lx + (np.arange(nx) + 0.5) * lx / nx
    zs = -0.5 * ly + (np.arange(ny) + 0.5) * ly / ny
    ix0 = int(np.argmin(np.abs(xs)))

    Bz_axis = By[:, ix0]
    analytic = mu0 * I * R ** 2 / (2.0 * (R ** 2 + zs ** 2) ** 1.5)

    good = (np.abs(zs) > 0.3) & (np.abs(zs) < 0.4 * ly)  # skip the wire, the far box
    rel = np.abs(Bz_axis[good] - analytic[good]) / np.abs(analytic[good])
    print(f"on-axis B_z vs mu0 I R^2 / 2(R^2+z^2)^3/2 :  "
          f"median rel err {np.median(rel):.4f}   max {np.max(rel):.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(zs, Bz_axis, ".", ms=4, label="09_magnetostatics (GPU Biot-Savart)")
    ax[0].plot(zs, analytic, "k-", lw=1, label=r"$\mu_0 I R^2 / 2(R^2+z^2)^{3/2}$")
    ax[0].set_xlabel("z (on axis)"); ax[0].set_ylabel("B_z")
    ax[0].set_title("on-axis field"); ax[0].legend(fontsize=9)

    im = ax[1].imshow(Bmag, origin="lower", extent=[-lx/2, lx/2, -ly/2, ly/2],
                      cmap="magma")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("z"); ax[1].set_title("|B| (xz slice)")
    fig.colorbar(im, ax=ax[1], shrink=0.8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "loop_validation.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
