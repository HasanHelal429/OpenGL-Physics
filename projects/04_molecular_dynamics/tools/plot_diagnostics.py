#!/usr/bin/env python3
"""General-purpose validation plot for a headless 04_molecular_dynamics run.

    python plot_diagnostics.py <results_dir> [--out diagnostics.png]

Reads diagnostics.csv (+ deck.toml for the thermostat/barostat targets) and
plots: total energy drift (the NVE conservation check when thermostat="none";
otherwise just a sanity trace, since a thermostat/barostat deliberately adds
or removes energy), temperature vs. its target, pressure (raw and tail-
corrected) vs. its target (barostat runs only), density/box_length vs. time
(barostat runs only), and the Nose-Hoover invariant drift (nose_hoover runs
only -- this is the quantity that SHOULD stay flat, not total_energy).
"""
import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("plot_diagnostics: needs numpy + matplotlib")

try:
    import tomllib  # py3.11+
except ImportError:
    tomllib = None


def load_deck(path):
    if tomllib is None or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    deck = load_deck(os.path.join(d, "deck.toml"))
    thermostat = deck.get("thermostat", {})
    barostat = deck.get("barostat", {})
    is_nve = thermostat.get("type", "berendsen") == "none"
    is_nh = thermostat.get("type", "berendsen") == "nose_hoover"
    has_barostat = barostat.get("enabled", False)

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    ax = ax.ravel()

    e = data["total_energy"]
    ax[0].plot(t, e - e[0])
    title = "total_energy - E(0)  (NVE conservation check)" if is_nve else "total_energy - E(0)"
    ax[0].set_title(f"{title}\nE0={e[0]:.4f}"); ax[0].set_xlabel("t")

    ax[1].plot(t, data["temperature"], label="T(t)")
    target_t = thermostat.get("target_t")
    if target_t is not None:
        ax[1].axhline(target_t, color="k", ls="--", label=f"target_t={target_t}")
    ax[1].set_title("temperature"); ax[1].set_xlabel("t"); ax[1].legend(fontsize=8)

    ax[2].plot(t, data["pressure"], label="raw")
    ax[2].plot(t, data["pressure_tail"], label="tail-corrected")
    target_p = barostat.get("target_p")
    if has_barostat and target_p is not None:
        # The barostat couples to the RAW (cutoff) pressure, not the tail-
        # corrected one -- MDSystem::ApplyBarostat calls Pressure(), and the
        # tail correction is a reporting-only static addend that never feeds
        # back into the dynamics (see MDSystem.hpp). So target_p belongs
        # against the "raw" curve above, not "tail-corrected".
        ax[2].axhline(target_p, color="k", ls="--", label=f"target_p={target_p} (targets raw P)")
    ax[2].set_title("pressure"); ax[2].set_xlabel("t"); ax[2].legend(fontsize=8)

    if has_barostat:
        ax[3].plot(t, data["density"])
        ax[3].set_title("density (barostat run)"); ax[3].set_xlabel("t")
    else:
        ax[3].plot(t, data["kinetic"], label="KE")
        ax[3].plot(t, data["potential"], label="PE")
        ax[3].set_title("energy split"); ax[3].set_xlabel("t"); ax[3].legend(fontsize=8)

    if is_nh:
        inv = data["nh_invariant"]
        ax[4].plot(t, inv - inv[0], label="nh_invariant - inv(0)")
        ax[4].plot(t, e - e[0], "--", label="total_energy - E(0)", alpha=0.6)
        ax[4].set_title("Nose-Hoover conserved quantity vs. raw energy"); ax[4].set_xlabel("t")
        ax[4].legend(fontsize=8)
    else:
        ax[4].plot(t, data["potential_tail"] - data["potential"])
        ax[4].set_title("tail energy correction"); ax[4].set_xlabel("t")

    ax[5].set_visible(False)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)

    print(f"  energy drift (raw):        {abs(e[-1] - e[0]) / abs(e[0]):.3e} (relative)")
    if is_nh:
        inv = data["nh_invariant"]
        print(f"  Nose-Hoover invariant drift: {abs(inv[-1] - inv[0]) / abs(inv[0]):.3e} (relative)")
    print(f"  final temperature: {data['temperature'][-1]:.4f}  (target {target_t})")
    if has_barostat:
        print(f"  final density: {data['density'][-1]:.4f}  final pressure (raw): {data['pressure'][-1]:.4f}"
              f"  (target_p={target_p})")


if __name__ == "__main__":
    main()
