"""
Validation plot for a headless 07_grhd shock-tube run: final-frame rho/v/p
profiles, plus (for the Newtonian-limit deck) an overlay of the exact
Newtonian Riemann solution (tools/exact_riemann_newtonian.py) -- the real
check that this SRHD solver's conservative variables, HLLE flux, and
primitive recovery correctly reduce to ordinary Newtonian hydrodynamics in
the low-velocity/low-pressure limit.

Usage:
    python plot_shocktube.py <results_dir> [--newtonian] [--out FILE]
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

from exact_riemann_newtonian import exact_solution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--newtonian", action="store_true",
                     help="overlay the exact Newtonian Riemann solution (only valid for a "
                          "deck whose pressures/velocities are deep in the non-relativistic regime)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    deck_path = os.path.join(d, "deck.toml")
    with open(deck_path, "rb") as f:
        deck = tomllib.load(f)

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("rho_")})
    last = frame_ids[-1]
    rho = np.load(os.path.join(frames_dir, f"rho_{last}.npy")).ravel()
    v = np.load(os.path.join(frames_dir, f"v_{last}.npy")).ravel()
    p = np.load(os.path.join(frames_dir, f"p_{last}.npy")).ravel()

    n = manifest["grid"]["nx"]
    length = manifest["grid"]["lx"]
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()
    ax[0].plot(x, rho, ".", ms=3, label="07_grhd (SRHD)")
    ax[1].plot(x, v, ".", ms=3, label="07_grhd (SRHD)")
    ax[2].plot(x, p, ".", ms=3, label="07_grhd (SRHD)")

    if args.newtonian:
        r = deck["riemann"]
        gamma = deck["physics"]["gamma"]
        x0 = r.get("x0", 0.5) * length
        rho_ex, v_ex, p_ex = exact_solution(
            x, t_end, x0, r["rho_l"], r["v_l"], r["p_l"], r["rho_r"], r["v_r"], r["p_r"], gamma)
        ax[0].plot(x, rho_ex, "k-", lw=1, label="exact Newtonian Riemann solution")
        ax[1].plot(x, v_ex, "k-", lw=1, label="exact Newtonian Riemann solution")
        ax[2].plot(x, p_ex, "k-", lw=1, label="exact Newtonian Riemann solution")
        rho_err = np.max(np.abs(rho - rho_ex)) / np.max(rho_ex)
        print(f"max |rho - exact| / max(exact): {rho_err:.4f}")

    ax[0].set_title("density"); ax[0].set_xlabel("x"); ax[0].legend(fontsize=8)
    ax[1].set_title("velocity"); ax[1].set_xlabel("x"); ax[1].legend(fontsize=8)
    ax[2].set_title("pressure"); ax[2].set_xlabel("x"); ax[2].legend(fontsize=8)

    t = data["t"]
    ax[3].plot(t, data["mass_total"] - data["mass_total"][0], label="mass - mass(0)")
    ax[3].plot(t, data["energy_total"] - data["energy_total"][0], label="energy - energy(0)")
    predicted_momentum = -(deck["riemann"]["p_r"] - deck["riemann"]["p_l"]) * t
    ax[3].plot(t, data["momentum_total"] - predicted_momentum, "--",
               label="momentum - predicted(-(p_R-p_L)*t)")
    ax[3].set_title("conservation (open-boundary momentum checked against\nthe predicted "
                     "boundary-pressure forcing, not zero)")
    ax[3].set_xlabel("t"); ax[3].legend(fontsize=7)

    fig.suptitle(f"{os.path.basename(os.path.abspath(d))}  (t={t_end:.3f})")
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
