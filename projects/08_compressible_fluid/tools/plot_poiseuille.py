"""
Validation plot for a headless 08_compressible_fluid Poiseuille-channel run
(decks/poiseuille_channel.toml): the final-frame streamwise-velocity
profile u(y), averaged over the (trivial, periodic) x direction, against
the analytic steady plane-Poiseuille profile
    u(y) = (f / (2*mu)) * y * (H - y)
(f = channel.body_force_x, mu = physics.mu, H = grid.height) -- the
classic exact solution this scheme's viscous (Newtonian shear stress +
Fourier conduction) terms should reproduce once the flow has relaxed from
rest to steady state.

Usage:
    python plot_poiseuille.py <results_dir> [--out FILE]
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    ny = manifest["grid"]["ny"]
    height = deck["grid"]["height"]
    mu = deck["physics"]["mu"]
    force = deck["channel"]["body_force_x"]
    dy = height / ny
    y = (np.arange(ny) + 0.5) * dy

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("u_")})
    last = frame_ids[-1]
    u = np.load(os.path.join(frames_dir, f"u_{last}.npy"))
    u_profile = u.mean(axis=1)  # average over the (periodic, trivial) x direction

    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    u_exact = (force / (2.0 * mu)) * y * (height - y)
    u_max_exact = force * height ** 2 / (8.0 * mu)
    rel_err = np.max(np.abs(u_profile - u_exact)) / u_max_exact
    print(f"t={t_end:.3f}  u_max numeric={u_profile.max():.5f}  u_max exact={u_max_exact:.5f}  "
          f"max|err|/u_max_exact={rel_err:.4f}")

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(u_profile, y, ".", ms=4, label=f"08_compressible_fluid (t={t_end:.2f})")
    ax[0].plot(u_exact, y, "k-", lw=1, label="analytic steady profile")
    ax[0].set_xlabel("u"); ax[0].set_ylabel("y"); ax[0].set_title("velocity profile")
    ax[0].legend(fontsize=8)

    ax[1].plot(data["t"], data["u_max"], label="numeric u_max(t)")
    ax[1].axhline(u_max_exact, color="k", ls="--", lw=1, label="analytic steady u_max")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("u_max"); ax[1].set_title("spin-up to steady state")
    ax[1].legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
