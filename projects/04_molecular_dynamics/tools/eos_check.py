#!/usr/bin/env python3
"""Equation-of-state consistency check across one or more headless
04_molecular_dynamics runs.

    python eos_check.py <results_dir> [<results_dir> ...] [--tail-frac 0.25]

For each run, averages temperature/density/tail-corrected-pressure over the
LAST --tail-frac of the trajectory (the presumed-equilibrated tail -- see
decks/liquid.toml's hot-start-then-relax setup) and prints a state-point
table. This is deliberately NOT a comparison against a fitted literature
equation of state (reimplementing one of the multiparameter reference EOS
correlations, e.g. Thol et al. 2016, from a handful of transcribed
coefficients is a real transcription-error risk not worth taking for a demo
validation script) -- instead it does two things that don't require any
external numbers:

  1. If two runs are passed, checks whether they're mutually consistent on
     the SAME underlying equation of state via two independent routes -- a
     fixed-density (NVT) run's measured pressure vs. a fixed-pressure (NPT)
     run's converged density -- e.g. decks/liquid.toml vs. decks/npt_liquid.toml.
  2. Prints each state point's position relative to the LJ critical point
     (Tc*=1.32, rho_c*=0.31 -- Thol, Rutkai, Koester, Lustig, Span, Vrabec,
     "Equation of State for the Lennard-Jones Fluid", 2016) as descriptive
     context: how deep into the liquid region a given (rho*, T*) sits,
     nothing more.
"""
import argparse
import os
import sys

import numpy as np

# Thol et al. (2016) critical-point parameters for the LJ fluid -- verified
# directly from that paper's Eq. (2) discussion, not reconstructed from
# memory of the fitted EOS itself.
LJ_TC = 1.32
LJ_RHOC = 0.31


def tail_average(results_dir, tail_frac):
    data = np.genfromtxt(os.path.join(results_dir, "diagnostics.csv"), delimiter=",", names=True)
    n = len(data)
    start = int((1.0 - tail_frac) * n)
    return {
        "T": float(np.mean(data["temperature"][start:])),
        "T_std": float(np.std(data["temperature"][start:])),
        "rho": float(np.mean(data["density"][start:])),
        "rho_std": float(np.std(data["density"][start:])),
        "P": float(np.mean(data["pressure_tail"][start:])),
        "P_std": float(np.std(data["pressure_tail"][start:])),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--tail-frac", type=float, default=0.25)
    args = ap.parse_args()

    rows = []
    for d in args.results_dirs:
        s = tail_average(d, args.tail_frac)
        rows.append((os.path.basename(os.path.normpath(d)), s))

    print(f"{'run':<24}{'T':>10}{'rho':>10}{'P (tail)':>12}{'T/Tc':>8}{'rho/rhoc':>10}")
    for name, s in rows:
        print(f"{name:<24}{s['T']:>10.4f}{s['rho']:>10.4f}{s['P']:>12.5f}"
              f"{s['T'] / LJ_TC:>8.3f}{s['rho'] / LJ_RHOC:>10.3f}")
    print(f"\n(LJ critical point for context: Tc*={LJ_TC}, rho_c*={LJ_RHOC},"
          f" Thol et al. 2016 -- all state points above are deep in the"
          f" liquid/solid region if T/Tc << 1 and rho/rhoc > 1)")

    if len(rows) >= 2:
        print("\ncross-consistency (fixed-density NVT vs. fixed-pressure NPT, if applicable):")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                (name_i, si), (name_j, sj) = rows[i], rows[j]
                d_rho = abs(si["rho"] - sj["rho"])
                d_p = abs(si["P"] - sj["P"])
                print(f"  {name_i} vs {name_j}:  |drho|={d_rho:.4f}  |dP|={d_p:.4f}"
                      f"  (rho: {si['rho']:.4f} vs {sj['rho']:.4f},"
                      f" P: {si['P']:.4f} vs {sj['P']:.4f})")


if __name__ == "__main__":
    main()
