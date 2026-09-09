#!/usr/bin/env python3
"""Fit and plot the N-body solver scaling sweep.

Consumes the CSVs written by 02_nbody_scaling / 03_nbody_scaling_2d (see
scripts/perlmutter_nbody_scaling.sbatch) and produces:

  * a log-log wall-time-vs-N panel per dimension, with a power-law exponent
    fitted over the asymptotic tail of each solver's curve;
  * a thread-scaling panel (speedup vs thread count at fixed N);
  * an accuracy panel, because the fast solvers are approximations and a
    speed plot without the error alongside it is only half the story.

On Perlmutter the interpreter that has numpy/matplotlib is the NERSC conda
one -- `module load python`, or point at it directly:

  /global/common/software/nersc/pe/conda-envs/<ver>/python-3.13/nersc-python/bin/python3 \
      scripts/plot_nbody_scaling.py --dir $SCRATCH/nbody-scaling/<jobid>

Usage:
  plot_nbody_scaling.py --dir RESULTS_DIR [--out FIG.png] [--fit-min-n 8000]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no display on a compute node
import matplotlib.pyplot as plt
import numpy as np

# Categorical slots 1-4 of the reference palette, in its documented order,
# assigned per solver and held stable across every panel so the eye can track
# one solver between plots. Order is fixed, not cycled.
COLORS = {
    "direct": "#2a78d6",  # slot 1, blue
    "bh": "#eb6834",      # slot 2, orange
    "fmm": "#1baf7a",     # slot 3, aqua
    "sfmm": "#eda100",    # slot 4, yellow
}
MARKERS = {"direct": "o", "bh": "s", "fmm": "^", "sfmm": "D"}


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        out = []
        for r in csv.DictReader(fh):
            try:
                out.append(
                    {
                        "solver": r["solver"],
                        "label": r["label"],
                        "n": int(r["n"]),
                        "threads": int(r["threads"]),
                        "min_ms": float(r["min_ms"]),
                        "median_ms": float(r["median_ms"]),
                        "std_ms": float(r["std_ms"]),
                        "err": float(r["rel_l2_err"]),
                    }
                )
            except (KeyError, ValueError):
                continue
        return out


def by_solver(rows):
    """Group rows by solver key, each sorted by N."""
    out = {}
    for r in rows:
        out.setdefault(r["solver"], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["n"])
    return out


def fit_exponent(ns, ts, min_n):
    """Least-squares slope of log t vs log N over the asymptotic tail.

    Returns (exponent, stderr, n_points_used). Small N is excluded because
    fixed per-call overhead (allocation, thread spin-up, tree setup) flattens
    the curve there and would bias the exponent down.
    """
    ns = np.asarray(ns, dtype=float)
    ts = np.asarray(ts, dtype=float)
    keep = (ns >= min_n) & (ts > 0)
    if keep.sum() < 3:  # too few points to claim a slope; use all positive ones
        keep = ts > 0
    if keep.sum() < 2:
        return float("nan"), float("nan"), int(keep.sum())

    x = np.log(ns[keep])
    y = np.log(ts[keep])
    n = x.size
    slope, intercept = np.polyfit(x, y, 1)

    resid = y - (slope * x + intercept)
    if n > 2:
        s2 = float(resid @ resid) / (n - 2)
        sxx = float(((x - x.mean()) ** 2).sum())
        stderr = math.sqrt(s2 / sxx) if sxx > 0 else float("nan")
    else:
        stderr = float("nan")
    return float(slope), float(stderr), int(n)


def plot_complexity(ax, rows, title, fit_min_n, fits_out):
    groups = by_solver(rows)
    if not groups:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    all_n, all_t = [], []
    for key, rs in groups.items():
        ns = [r["n"] for r in rs]
        ts = [r["median_ms"] for r in rs]
        es = [r["std_ms"] for r in rs]
        all_n += ns
        all_t += ts

        slope, stderr, npts = fit_exponent(ns, ts, fit_min_n)
        fits_out.append((title, rs[0]["label"], slope, stderr, npts, max(ns)))

        label = f"{rs[0]['label']}  (N^{slope:.2f})"
        ax.errorbar(
            ns, ts, yerr=es, marker=MARKERS.get(key, "o"), ms=5, lw=1.6, capsize=2,
            color=COLORS.get(key, "#333"), label=label,
        )

    # O(N) and O(N^2) guide lines, anchored at the smallest measured point so
    # they sit under the data rather than across it.
    if all_n:
        n0, t0 = min(all_n), min(all_t)
        nn = np.array([min(all_n), max(all_n)], dtype=float)
        for power, style, tag in ((1.0, ":", "O(N)"), (2.0, "--", "O(N²)")):
            ax.plot(nn, t0 * (nn / n0) ** power, style, color="#999", lw=1.0, zorder=0)
            ax.annotate(
                tag, (nn[-1], t0 * (nn[-1] / n0) ** power), color="#777",
                fontsize=8, ha="right", va="bottom",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N particles")
    ax.set_ylabel("force evaluation, ms (median)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper left")


def plot_threads(ax, rows, title):
    groups = by_solver(rows)
    if not groups:
        ax.text(0.5, 0.5, "no thread-scaling data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title(title)
        return

    maxt = 1
    for key, rs in groups.items():
        rs = sorted(rs, key=lambda r: r["threads"])
        ts = [r["threads"] for r in rs]
        base = next((r["median_ms"] for r in rs if r["threads"] == 1), None)
        if base is None:
            continue
        speedup = [base / r["median_ms"] for r in rs]
        maxt = max(maxt, max(ts))
        ax.plot(ts, speedup, marker=MARKERS.get(key, "o"), ms=5, lw=1.6,
                color=COLORS.get(key, "#333"), label=rs[0]["label"])

    ideal = np.array([1, maxt], dtype=float)
    ax.plot(ideal, ideal, "--", color="#999", lw=1.0, zorder=0, label="ideal (linear)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("speedup vs 1 thread")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=8, loc="upper left")


def plot_accuracy(ax, rows3d, rows2d):
    any_data = False
    for rows, ls, tag in ((rows3d, "-", "3D"), (rows2d, "--", "2D")):
        for key, rs in by_solver(rows).items():
            if key == "direct":
                continue  # Direct is the reference; its error is 0 by construction
            pts = [(r["n"], r["err"]) for r in rs if r["err"] > 0]
            if not pts:
                continue
            any_data = True
            ns = [p[0] for p in pts]
            es = [p[1] for p in pts]
            ax.plot(ns, es, ls, marker=MARKERS.get(key, "o"), ms=4, lw=1.4,
                    color=COLORS.get(key, "#333"), label=f"{tag} {rs[0]['label']}")

    if not any_data:
        ax.text(0.5, 0.5, "no accuracy data", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N particles")
    ax.set_ylabel("rel. L2 error vs Direct")
    ax.set_title("Approximation error (the price of the speedup)")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="results directory from the sbatch run")
    ap.add_argument("--out", default=None, help="output PNG (default: <dir>/nbody_scaling.png)")
    ap.add_argument("--fit-min-n", type=int, default=8000,
                    help="exclude N below this from the power-law fit (default 8000)")
    a = ap.parse_args()

    d = a.dir
    full3d = read_rows(os.path.join(d, "scaling3d_full.csv"))
    full2d = read_rows(os.path.join(d, "scaling2d_full.csv"))
    ser3d = read_rows(os.path.join(d, "scaling3d_serial.csv"))
    ser2d = read_rows(os.path.join(d, "scaling2d_serial.csv"))
    thr3d = read_rows(os.path.join(d, "threads3d.csv"))

    if not any((full3d, full2d, ser3d, ser2d, thr3d)):
        print(f"error: no CSVs found in {d}", file=sys.stderr)
        return 1

    threads = full3d[0]["threads"] if full3d else "?"

    fits = []
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    plot_complexity(axes[0][0], ser3d, "3D solvers, 1 thread", a.fit_min_n, fits)
    plot_complexity(axes[0][1], full3d, f"3D solvers, {threads} threads", a.fit_min_n, fits)
    plot_threads(axes[0][2], thr3d, "3D thread scaling at N=32000")
    plot_complexity(axes[1][0], ser2d, "2D solvers, 1 thread", a.fit_min_n, fits)
    plot_complexity(axes[1][1], full2d, f"2D solvers, {threads} threads", a.fit_min_n, fits)
    plot_accuracy(axes[1][2], full3d, full2d)

    fig.suptitle(
        "N-body gravity solver scaling -- Perlmutter CPU node (2x AMD EPYC 7763)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    out = a.out or os.path.join(d, "nbody_scaling.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    # Fitted exponents, so the numbers behind the picture are quotable.
    print()
    print(f"{'panel':<26} {'solver':<16} {'exponent':>12} {'stderr':>9} {'pts':>4} {'maxN':>8}")
    for panel, label, slope, stderr, npts, maxn in fits:
        print(f"{panel:<26} {label:<16} {slope:>12.3f} {stderr:>9.3f} {npts:>4d} {maxn:>8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
