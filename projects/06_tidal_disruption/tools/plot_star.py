"""
Validation panels for a headless run of 06_tidal_disruption: energy/momentum
conservation over time, plus the SPH radial density profile (first vs. last
frame) against the analytic Lane-Emden profile the initial conditions were
sampled from.

Usage:
    python plot_star.py <results_dir> [--index 1.5] [--mass 1.0] [--radius 1.0] [--out FILE]
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from lane_emden import build_polytrope


def radial_profile(pos_mass, rho, edges):
    r = np.linalg.norm(pos_mass[:, :3], axis=1)
    idx = np.digitize(r, edges) - 1
    r_mid, rho_mean, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < 5:
            continue
        r_mid.append(0.5 * (edges[b] + edges[b + 1]))
        rho_mean.append(rho[mask].mean())
        counts.append(mask.sum())
    return np.array(r_mid), np.array(rho_mean), np.array(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--index", type=float, default=1.5, help="polytropic index n (must match the IC)")
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("pos_mass_")})
    first, last = frame_ids[0], frame_ids[-1]
    pm0 = np.load(os.path.join(frames_dir, f"pos_mass_{first}.npy"))
    rp0 = np.load(os.path.join(frames_dir, f"rho_press_{first}.npy"))
    pm1 = np.load(os.path.join(frames_dir, f"pos_mass_{last}.npy"))
    rp1 = np.load(os.path.join(frames_dir, f"rho_press_{last}.npy"))

    profile = build_polytrope(args.index, M_star=args.mass, R_star=args.radius, G=args.G)
    edges = np.linspace(0.0, args.radius, 13)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ax = ax.ravel()

    ax[0].plot(t, data["energy"] - data["energy"][0])
    ax[0].set_title(f"energy - energy(0)  (E0={data['energy'][0]:.4f})")
    ax[0].set_xlabel("t")

    ax[1].plot(t, data["kinetic"], label="K")
    ax[1].plot(t, data["thermal"], label="U_thermal")
    ax[1].plot(t, data["potential"], label="W")
    virial_resid = 3.0 * (5.0 / 3.0 - 1.0) * data["thermal"] + data["potential"]
    ax[1].plot(t, virial_resid, "--", label="3(g-1)U_th + W (hydrostatic virial residual)")
    ax[1].set_title("energy budget"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=8)

    ax[2].semilogy(t, np.maximum(data["com_speed"], 1e-16))
    ax[2].set_title("|COM velocity| (momentum conservation)"); ax[2].set_xlabel("t")

    r0, rho0, _ = radial_profile(pm0, rp0[:, 0], edges)
    r1, rho1, _ = radial_profile(pm1, rp1[:, 0], edges)
    ax[3].semilogy(profile["r"], profile["rho"], "k-", label="Lane-Emden analytic")
    ax[3].semilogy(r0, rho0, "o", label=f"SPH, frame {first} (t={t[0]:.2f})")
    ax[3].semilogy(r1, rho1, "s", label=f"SPH, frame {last} (t={t[-1]:.2f})")
    # The analytic curve hits exactly 0 at the surface, which would otherwise
    # stretch the log axis over ~20 decades and flatten the 10-60% bulk/edge
    # deviations that are the actual point of this panel -- clip to the
    # range the data spans instead.
    y_lo = 0.5 * min(rho0.min(), rho1.min())
    y_hi = 1.5 * max(rho0.max(), rho1.max())
    ax[3].set_ylim(y_lo, y_hi)
    ax[3].set_title("radial density profile"); ax[3].set_xlabel("r"); ax[3].legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
