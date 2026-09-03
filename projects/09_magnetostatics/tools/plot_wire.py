"""
Validation plot for a headless 09_magnetostatics run of decks/wire.toml: the
solved field magnitude |B|(r) against the analytic infinite-wire result
    B_phi(r) = mu0 * I / (2*pi*r)
sampled on an annulus well inside the (square, A_z = 0) domain boundary, plus
a check that B circulates around the wire (B . rhat ~ 0).

Usage:
    python plot_wire.py <results_dir> [--out FILE]
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
    frames_dir = os.path.join(d, "frames")
    ids = sorted({fn.split("_")[-1].split(".")[0]
                  for fn in os.listdir(frames_dir) if fn.startswith(name + "_")})
    return np.load(os.path.join(frames_dir, f"{name}_{ids[0]}.npy"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    nx = manifest["grid"]["nx"]
    ny = manifest["grid"]["ny"]
    lx = deck["grid"]["lx"]
    ly = deck["grid"]["ly"]
    mu0 = deck.get("physics", {}).get("mu0", 1.0)
    wire = deck["wire"][0]
    I = wire["current"]
    x0, y0 = wire.get("x", 0.0), wire.get("y", 0.0)

    Bx = load_frame(d, "Bx")
    By = load_frame(d, "By")

    dx, dy = lx / nx, ly / ny
    xs = -0.5 * lx + (np.arange(nx) + 0.5) * dx
    ys = -0.5 * ly + (np.arange(ny) + 0.5) * dy
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    R = np.hypot(X - x0, Y - y0)
    Bmag = np.hypot(Bx, By)

    # Radial component (should be ~0: the field circulates).
    rhat_x = (X - x0) / np.where(R > 0, R, 1)
    rhat_y = (Y - y0) / np.where(R > 0, R, 1)
    Br = Bx * rhat_x + By * rhat_y

    r_lo, r_hi = 0.1 * min(lx, ly), 0.25 * min(lx, ly)
    ann = (R > r_lo) & (R < r_hi)

    r_flat = R[ann]
    b_flat = Bmag[ann]
    b_exact = mu0 * I / (2.0 * np.pi * r_flat)
    rel_err = np.abs(b_flat - b_exact) / b_exact
    br_rel = np.abs(Br[ann]) / b_exact

    print(f"annulus r in [{r_lo:.3f}, {r_hi:.3f}]  ({ann.sum()} cells)")
    print(f"  |B| vs mu0 I / 2 pi r :  median rel err {np.median(rel_err):.4f}  "
          f"max {np.max(rel_err):.4f}")
    print(f"  |B . rhat| / |B|      :  median {np.median(br_rel):.4f}  "
          f"max {np.max(br_rel):.4f}")

    # Binned radial profile.
    bins = np.linspace(r_lo, r_hi, 40)
    ctr = 0.5 * (bins[1:] + bins[:-1])
    which = np.digitize(r_flat, bins) - 1
    prof = np.array([b_flat[which == k].mean() if np.any(which == k) else np.nan
                     for k in range(len(ctr))])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(ctr, prof, ".", ms=6, label="09_magnetostatics")
    ax[0].plot(ctr, mu0 * I / (2 * np.pi * ctr), "k-", lw=1,
               label=r"$\mu_0 I / 2\pi r$")
    ax[0].set_xlabel("r"); ax[0].set_ylabel("|B|")
    ax[0].set_title("radial field profile"); ax[0].legend(fontsize=9)

    im = ax[1].imshow(Bmag, origin="lower", extent=[-lx/2, lx/2, -ly/2, ly/2],
                      cmap="magma")
    ax[1].set_title("|B|"); fig.colorbar(im, ax=ax[1], shrink=0.8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "wire_validation.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
