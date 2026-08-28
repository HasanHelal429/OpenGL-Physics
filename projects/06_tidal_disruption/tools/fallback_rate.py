"""
Quantitative tidal-disruption fallback-rate diagnostic: turns a headless
06_tidal_disruption encounter run into the classic dM/dt vs. time-since-
pericenter curve (Rees 1988; Lodato, King & Pringle 2009) and checks it
against the analytic t^(-5/3) power law.

Physics: once a debris parcel has cleared the star's own self-gravity/
pressure (a few dynamical times past pericenter), its orbit around the
black hole is Keplerian and its specific orbital energy
    eps = 0.5*|v|^2 - G*M_bh/|r|
is frozen in. A bound parcel (eps<0) is on an ellipse of semi-major axis
a = -G*M_bh/(2*eps) and returns to pericenter after one Kepler period
    t_return = 2*pi*G*M_bh*(-2*eps)^(-3/2).
The classic "frozen-in" argument (a roughly flat dM/deps across the narrow
spread imparted at disruption, deps ~ G*M_bh*R_star/r_p^2) combined with
t(eps) above gives dM/dt ~ t^(-5/3): most of the mass returns fast, then
the rate falls off as a power law as increasingly weakly-bound (long-period)
material trickles back.

This script reads the FINAL frame's per-particle pos_mass/vel (the run
needs to extend well past pericenter for eps to have actually frozen in --
decks/encounter_beta3_long.toml's ~50 dynamical times is enough, its own
~15-dynamical-time decks/encounter_beta3.toml is not), computes eps_i per
particle, bins the bound (eps<0) particles' mass by their Kepler return
time, and plots the resulting dM/dt against the analytic power law.

Usage:
    python fallback_rate.py <results_dir> [--nbins 24] [--out FILE]
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
    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    manifest = json.load(open(os.path.join(d, "manifest.json")))
    deck_path = os.path.join(d, "deck.toml")
    if tomllib is None or not os.path.exists(deck_path):
        raise SystemExit("fallback_rate: needs deck.toml next to the run (tomllib + a copied deck)")
    with open(deck_path, "rb") as f:
        deck = tomllib.load(f)
    G = float(deck.get("gravity", {}).get("G", 1.0))
    bh = deck.get("blackhole", {})
    if not bh.get("enabled", False) or bh.get("type") != "point":
        raise SystemExit("fallback_rate: eps = 0.5v^2 - GM/r assumes a Newtonian point-mass BH "
                          "(uBhType=1) -- this run's deck doesn't have that")
    M_bh = float(bh["mass"])

    frames_dir = os.path.join(d, "frames")
    frame_ids = sorted({fn.split("_")[-1].split(".")[0]
                         for fn in os.listdir(frames_dir) if fn.startswith("pos_mass_")})
    last = frame_ids[-1]
    pm = np.load(os.path.join(frames_dir, f"pos_mass_{last}.npy"))
    vel = np.load(os.path.join(frames_dir, f"vel_{last}.npy"))
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)
    t_end = int(last) * substeps * dt

    r = np.linalg.norm(pm[:, :3], axis=1)
    v2 = np.sum(vel[:, :3] ** 2, axis=1)
    m = pm[:, 3]
    eps = 0.5 * v2 - G * M_bh / r

    # Pericenter time: the mass-weighted BH potential energy is most
    # negative when the bulk of the (still-compact, pre-stretch) star is
    # closest to the black hole -- robust to the stream stretching apart
    # later, unlike tracking a single "center" position.
    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t_peri = float(data["t"][np.argmin(data["potential_bh"])])

    bound = eps < 0.0
    bound_frac = m[bound].sum() / m.sum()
    t_return = 2.0 * np.pi * G * M_bh * np.power(-2.0 * eps[bound], -1.5)

    print(f"run: {d}  (final frame t={t_end:.2f}, pericenter t={t_peri:.2f})")
    print(f"bound mass fraction (eps<0): {bound_frac:.3f}")
    print(f"Kepler return time range (bound particles): {t_return.min():.2f} .. {t_return.max():.2f}")
    print(f"earliest possible return (sim time): {t_peri + t_return.min():.2f}")

    # log-spaced return-time bins -> dM/dt = (mass in bin) / (bin width).
    edges = np.logspace(np.log10(t_return.min()), np.log10(t_return.max()), args.nbins + 1)
    idx = np.digitize(t_return, edges) - 1
    t_mid, dmdt, counts = [], [], []
    for b in range(args.nbins):
        mask = idx == b
        if mask.sum() < 3:
            continue
        width = edges[b + 1] - edges[b]
        t_mid.append(np.sqrt(edges[b] * edges[b + 1]))  # log-bin center
        dmdt.append(m[bound][mask].sum() / width)
        counts.append(mask.sum())
    t_mid, dmdt, counts = np.array(t_mid), np.array(dmdt), np.array(counts)

    # Fit a power law to the well-populated middle of the range: the first
    # bin or two are the freshly-disrupted, still-settling debris (eps not
    # fully frozen in yet) and the last few are shot-noise limited (few
    # particles per log-bin at the longest periods).
    fit_mask = (counts >= 5) & (t_mid > t_return.min() * 2.0) & (t_mid < t_return.max() * 0.6)
    slope, intercept = np.polyfit(np.log10(t_mid[fit_mask]), np.log10(dmdt[fit_mask]), 1)
    print(f"fitted power-law slope: {slope:.3f}  (classic Rees (1988) prediction: -1.667)")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(t_mid, dmdt, "o", ms=5, label="SPH debris (binned by Kepler return time)")
    t_ref = np.array([t_mid[fit_mask].min(), t_mid[fit_mask].max()])
    ax.loglog(t_ref, 10 ** intercept * t_ref ** slope, "-", lw=1.5,
              label=f"fit: $t^{{{slope:.2f}}}$")
    pivot = t_mid[fit_mask][0]
    dmdt_at_pivot = dmdt[fit_mask][0]
    ax.loglog(t_ref, dmdt_at_pivot * (t_ref / pivot) ** (-5.0 / 3.0), "k--", lw=1.5,
              label="classic $t^{-5/3}$")
    ax.set_xlabel("time since pericenter (Kepler return time)")
    ax.set_ylabel("dM/dt (bound debris fallback rate)")
    ax.set_title(f"{os.path.basename(os.path.abspath(d))}: fallback rate vs. t^(-5/3)\n"
                 f"bound fraction={bound_frac:.2f}, fitted slope={slope:.2f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = args.out or os.path.join(d, "fallback_rate.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
