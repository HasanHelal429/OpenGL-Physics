"""
Test-particle diagnostics for a headless 09_magnetostatics run of a pusher
deck (cyclotron.toml / exb_drift.toml / magnetic_bottle.toml): the xz (or xy)
trajectory of each charge and the conservation quantities -- relativistic
kinetic energy (exact in a static B) and, for the bottle, the adiabatic
invariant mu = v_perp^2 / B at the mid-plane crossings.

Usage:
    python plot_orbit.py <results_dir> [--out FILE]
"""

import argparse
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    cols = data.dtype.names
    nq = len({int(m.group(1)) for c in cols
              for m in [re.match(r"q(\d+)_", c)] if m})
    t = data["t"]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(nq, 1)))

    for k in range(nq):
        x, y, z = data[f"q{k}_x"], data[f"q{k}_y"], data[f"q{k}_z"]
        # pick the two axes that actually move
        spans = [np.ptp(x), np.ptp(y), np.ptp(z)]
        a, b = np.argsort(spans)[-2:]
        names = ["x", "y", "z"]
        A = [x, y, z][a]
        B = [x, y, z][b]
        ax[0].plot(A, B, "-", lw=0.7, color=colors[k], label=f"charge {k}")
        ax[0].set_xlabel(names[a]); ax[0].set_ylabel(names[b])

        ke = data[f"q{k}_ke"]
        ax[1].plot(t, ke / ke[0] - 1.0, color=colors[k], label=f"charge {k}")

        if f"q{k}_mu" in cols:
            mu = data[f"q{k}_mu"]
            zc = np.where(np.diff(np.sign(z)) != 0)[0]  # mid-plane crossings
            if len(zc) > 2:
                ax[2].plot(t[zc], mu[zc] / np.mean(mu[zc]) - 1.0, "o-", ms=3,
                           color=colors[k], label=f"charge {k}")

    ax[0].set_aspect("equal", "box"); ax[0].set_title("trajectory")
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel("t"); ax[1].set_ylabel(r"$KE/KE_0 - 1$")
    ax[1].set_title("kinetic energy (should be flat)")
    ax[2].set_xlabel("t"); ax[2].set_ylabel(r"$\mu/\langle\mu\rangle - 1$")
    ax[2].set_title(r"$\mu$ at mid-plane crossings (bottle)")

    # print the headline numbers
    for k in range(nq):
        ke = data[f"q{k}_ke"]
        line = f"charge {k}:  KE drift {np.ptp(ke) / ke[0]:.2e}"
        if f"q{k}_mu" in cols:
            z = data[f"q{k}_z"]
            zc = np.where(np.diff(np.sign(z)) != 0)[0]
            if len(zc) > 2:
                mc = data[f"q{k}_mu"][zc]
                line += f"   mu bounce-to-bounce std/mean {mc.std() / mc.mean():.3f}"
                line += f"   {len(zc)} mid-plane crossings"
        print(line)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "orbit_diagnostics.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
