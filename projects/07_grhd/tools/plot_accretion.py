"""
Horizon accretion-rate diagnostic for a 07_grhd Kerr-torus run: tests
whether the torus's total-mass change is accounted for by real physical
flux through the domain's radial boundaries (KerrTorusSim's
inner_boundary_flux_cum / outer_boundary_flux_cum, a direct time-integral
of the actual HLLE flux used by EulerStep -- see KerrTorusSim.hpp's header
comment on m_innerFluxAccum) or by something else (the con2prim floor/
fixup branches, which reset state outside the flux-divergence update and
so are NOT reflected in those two columns).

D is source-free (t is a Killing vector), so in exact arithmetic:

    total_mass(t) - total_mass(0) == inner_boundary_flux_cum(t)
                                      - outer_boundary_flux_cum(t)

(up to the theta-boundary flux, which is ~0 by construction since those
domain edges sit in the rho_floor/v=0 region). Any gap between the two
sides is mass lost or gained OUTSIDE the flux update itself.

    python plot_accretion.py <results_dir> [<results_dir> ...] [--out fig.png]
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("plot_accretion: needs numpy + matplotlib")


def load(results_dir):
    diag = np.genfromtxt(os.path.join(results_dir, "diagnostics.csv"), delimiter=",", names=True)
    manifest = json.load(open(os.path.join(results_dir, "manifest.json")))
    return diag, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = [(d, *load(d)) for d in args.results_dirs]

    fig, axes = plt.subplots(3, len(runs), figsize=(6.2 * len(runs), 11), squeeze=False, dpi=130)

    for col, (d, diag, manifest) in enumerate(runs):
        t = diag["t"]
        mass = diag["total_mass"]
        inner_cum = diag["inner_boundary_flux_cum"]
        outer_cum = diag["outer_boundary_flux_cum"]
        m0 = mass[0]
        predicted = m0 + inner_cum - outer_cum
        residual = mass - predicted
        # -d(inner_cum)/dt: instantaneous rate of mass flowing INWARD across
        # the inner boundary (inner_cum itself decreases while accreting,
        # since F^r_D<0 there means motion toward decreasing r -- see
        # KerrTorusSim's header comment on the sign convention).
        mdot = -np.gradient(inner_cum, t)

        title = manifest.get("title", os.path.basename(d))
        short_title = title.split("--")[-1].strip() if "--" in title else title

        ax = axes[0][col]
        ax.plot(t, mass, color="#1f5c6e", lw=2, label="total_mass (actual)")
        ax.plot(t, predicted, color="#9c5f1a", lw=1.4, ls="--",
                 label="total_mass(0) + boundary flux integral (predicted)")
        ax.set_ylabel("mass in domain")
        ax.set_title(short_title, fontsize=10)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.25)

        ax = axes[1][col]
        ax.plot(t, residual, color="#a8433f", lw=1.6)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_ylabel("residual: actual - predicted\n(unaccounted mass; floor/fixup, not flux)")
        ax.grid(alpha=0.25)

        ax = axes[2][col]
        ax.plot(t, mdot, color="#3f7a52", lw=1.6)
        ax.axhline(0, color="black", lw=0.6, alpha=0.4)
        ax.set_xlabel("t (M)")
        ax.set_ylabel("accretion rate  Mdot = -d(inner flux)/dt\n(>0 = inflow through inner boundary)")
        ax.grid(alpha=0.25)

        rel_residual = np.abs(residual[-1]) / max(abs(mass[0] - mass[-1]), 1e-12)
        print(f"{d}: mass change {mass[0]:.4f} -> {mass[-1]:.4f}  "
              f"(delta={mass[-1]-mass[0]:+.4f}); final residual={residual[-1]:+.4f} "
              f"({100*rel_residual:.2f}% of the total mass change)")

    fig.suptitle("Is the torus's mass loss real accretion, or numerical? "
                  "(actual vs. flux-predicted total mass)", fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = args.out or os.path.join(args.results_dirs[0], "accretion_balance.png")
    fig.savefig(out, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
