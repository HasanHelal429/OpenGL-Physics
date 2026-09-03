"""
Validation plot for a headless 09_magnetostatics run of decks/solenoid.toml:
two opposite out-of-plane current strips. Between the sheets the field should
be uniform with
    |B_x| ~ mu0 * K_s,   K_s = I / width   (surface current density)
and outside it should cancel to ~0.

Usage:
    python plot_solenoid.py <results_dir> [--out FILE]
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

    nx, ny = manifest["grid"]["nx"], manifest["grid"]["ny"]
    lx, ly = deck["grid"]["lx"], deck["grid"]["ly"]
    mu0 = deck.get("physics", {}).get("mu0", 1.0)

    strips = deck["strip"]
    ytop = max(s["y"] for s in strips)
    ybot = min(s["y"] for s in strips)
    width = strips[0]["width"]
    I = abs(strips[0]["current"])
    Ks = I / width
    B_expected = mu0 * Ks

    Bx = load_frame(d, "Bx")
    By = load_frame(d, "By")

    dx, dy = lx / nx, ly / ny
    xs = -0.5 * lx + (np.arange(nx) + 0.5) * dx
    ys = -0.5 * ly + (np.arange(ny) + 0.5) * dy

    ix0 = np.argmin(np.abs(xs))          # centre column
    # Central x-window, well away from the strip ends.
    xwin = np.abs(xs) < 0.15 * width
    # Interior band: between the sheets.
    inside = (ys > ybot + 0.2 * (ytop - ybot)) & (ys < ytop - 0.2 * (ytop - ybot))
    bx_col = Bx[:, ix0]
    bx_inside = Bx[np.ix_(inside, xwin)]

    # Outside band: above the top sheet.
    outside = (ys > ytop + 0.5 * (ytop - ybot)) & (ys < 0.5 * ly - 0.2)
    bx_outside = Bx[np.ix_(outside, xwin)]

    print(f"expected |B_x| between sheets = mu0 K_s = {B_expected:.4f}")
    print(f"  measured (centre column):  mean {np.mean(np.abs(bx_inside)):.4f}  "
          f"std {np.std(bx_inside):.4f}  "
          f"rel err {abs(np.mean(np.abs(bx_inside)) - B_expected) / B_expected:.4f}")
    print(f"  outside (above top sheet): mean |B_x| "
          f"{np.mean(np.abs(bx_outside)):.4f}  "
          f"({np.mean(np.abs(bx_outside)) / B_expected:.3f} of interior)")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(bx_col, ys, "-", lw=1.2, label="B_x (centre column)")
    ax[0].axvline(B_expected, color="k", ls="--", lw=1, label=r"$\mu_0 K_s$")
    ax[0].axvline(-B_expected, color="k", ls="--", lw=1)
    ax[0].axhline(ytop, color="0.6", lw=0.8)
    ax[0].axhline(ybot, color="0.6", lw=0.8)
    ax[0].set_xlabel("B_x"); ax[0].set_ylabel("y")
    ax[0].set_title("field across the gap"); ax[0].legend(fontsize=9)

    im = ax[1].imshow(np.hypot(Bx, By), origin="lower",
                      extent=[-lx/2, lx/2, -ly/2, ly/2], cmap="magma")
    ax[1].set_title("|B|"); fig.colorbar(im, ax=ax[1], shrink=0.8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "solenoid_validation.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
