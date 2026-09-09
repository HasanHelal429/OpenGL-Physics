"""Energy and angular-momentum drift vs time from a headless run's
diagnostics.csv -- the offline version of the app's live DriftChart. Overlay
several runs (e.g. one per solver) by passing multiple results dirs.

Usage:
    python plot_conservation.py <results_dir> [<results_dir> ...] [--out conservation.png]
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(d):
    with open(os.path.join(d, "diagnostics.csv")) as f:
        rows = list(csv.DictReader(f))
    t = [float(r["t"]) for r in rows]
    ed = [float(r["energy_drift_pct"]) for r in rows]
    ld = [float(r["L_drift_pct"]) for r in rows]
    return t, ed, ld


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="+")
    ap.add_argument("--out", default="conservation.png")
    a = ap.parse_args()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for d in a.results_dir:
        t, ed, ld = load(d)
        label = os.path.basename(os.path.normpath(d))
        ax0.plot(t, ed, label=label)
        ax1.plot(t, ld, label=label)
    ax0.set_ylabel("energy drift  [%]")
    ax0.axhline(0, color="k", lw=0.5)
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax1.set_ylabel("|L| drift  [%]")
    ax1.set_xlabel("t")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.grid(alpha=0.3)
    fig.suptitle("N-body conservation")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
