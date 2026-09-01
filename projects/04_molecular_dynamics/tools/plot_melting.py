#!/usr/bin/env python3
"""Melting/freezing hysteresis plot for a headless 04_molecular_dynamics
temperature-ramp run (decks/melting_ramp.toml).

    python plot_melting.py <results_dir> [--split-t 50.0] [--out melting.png]

Plots two order parameters against the (time-varying) target_t:

  - lindemann: rms displacement from each particle's ORIGINAL FCC lattice
    site / nearest-neighbor distance. Sharp, unambiguous melting-onset
    detector (a real solid stays near the classic Lindemann-criterion range
    ~0.1-0.15; melting shows up as a discontinuous jump), but it is NOT a
    two-way indicator -- once melted, atoms diffuse and never return to
    their literal original coordinates even if the system recrystallizes
    elsewhere, so it grows ~monotonically for the rest of the run regardless
    of what happens on the cooling branch. Reported here for what it's
    actually good for: locating the melting point on the heating branch.
  - rdf_peak: the tallest bin of g(r) computed fresh each frame from the
    INSTANTANEOUS configuration only (no dependence on history/reference
    positions) -- a real order parameter for both branches. A gap between
    the heating and cooling curves at the same target_t (cooling below
    heating) is the classic superheating/supercooling hysteresis signature;
    if the cooling branch never recovers to anywhere near the original
    crystal's rdf_peak, that's evidence the cooling rate outran nucleation
    and the run ended in a disordered/glassy state rather than a
    recrystallized one -- both are physically legitimate outcomes, and
    this script reports whichever one actually happened, not an assumed one.
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
    sys.exit("plot_melting: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--split-t", type=float, default=None,
                     help="time separating heating/cooling branches (default: the run's midpoint)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    split_t = args.split_t if args.split_t is not None else 0.5 * (t[0] + t[-1])
    heat = t < split_t
    cool = ~heat

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax = ax.ravel()

    ax[0].plot(t, data["target_t"])
    ax[0].set_title("target_t(t) schedule"); ax[0].set_xlabel("t")

    ax[1].plot(t, data["lindemann"])
    ax[1].axhline(0.10, color="k", ls="--", lw=0.8, label="classic Lindemann criterion ~0.10-0.15")
    ax[1].axhline(0.15, color="k", ls="--", lw=0.8)
    ax[1].set_title("Lindemann parameter vs t\n(one-way melting detector -- see docstring)")
    ax[1].set_xlabel("t"); ax[1].legend(fontsize=7)

    ax[2].plot(data["target_t"][heat], data["lindemann"][heat], ".", label="heating", ms=3)
    ax[2].plot(data["target_t"][cool], data["lindemann"][cool], ".", label="cooling", ms=3)
    ax[2].set_title("lindemann vs target_t"); ax[2].set_xlabel("target_t"); ax[2].legend(fontsize=8)

    ax[3].plot(data["target_t"][heat], data["rdf_peak"][heat], ".", label="heating", ms=3)
    ax[3].plot(data["target_t"][cool], data["rdf_peak"][cool], ".", label="cooling", ms=3)
    ax[3].set_title("rdf_peak vs target_t (two-way order parameter)")
    ax[3].set_xlabel("target_t"); ax[3].legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "melting.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # First crossing above 0.3 (well past the classic ~0.10-0.15 solid-
    # stability range, but before lindemann's post-melting runaway growth)
    # on the heating branch -- more robust than looking for a single-frame
    # discontinuity, since the rise plays out over several frames, not
    # literally one.
    lin_heat = data["lindemann"][heat]
    tt_heat = data["target_t"][heat]
    above = np.where(lin_heat > 0.3)[0]
    if len(above):
        i = above[0]
        print(f"  melting detected: lindemann crosses 0.3 between target_t={tt_heat[i - 1]:.3f}"
              f" (lindemann={lin_heat[i - 1]:.3f}) and target_t={tt_heat[i]:.3f} (lindemann={lin_heat[i]:.3f})")
    else:
        print("  lindemann never exceeds 0.3 on the heating branch (try a wider target_t range)")

    rdf = data["rdf_peak"]
    print(f"  rdf_peak: initial(t=0)={rdf[0]:.2f}  final(t={t[-1]:.1f})={rdf[-1]:.2f}"
          f"  liquid-plateau min={rdf[heat | cool].min():.2f}")
    print("  (final << initial means the cooling branch did not fully recrystallize at this rate --"
          " a real supercooling/glass-formation outcome, not a bug)")


if __name__ == "__main__":
    main()
