#!/usr/bin/env python3
"""Compare the pre- and post-unification N-body solvers, and the GPU ones.

Reads two result sets:

  --old-dir   the pre-unification sweep (02_nbody_scaling / 03_nbody_scaling_2d
              CSVs: dim,solver,label,n,threads,reps,min_ms,median_ms,...)
  --new-dir   the ngrav sweep driven through 02_nbody_gravity --bench
              (label,threads,solver,n,theta,order,ms_per_eval,mean_rel_err)

Two things to keep straight when reading the output:

  * The two versions do not solve the same problem. The old harness used the
    repo's `cluster` scenario with Plummer eps=0.05; --bench uses a Plummer
    sphere with eps=0.02. For tree/FMM solvers the particle distribution
    changes tree depth, so their old-vs-new ratios mix implementation and
    problem changes. **Direct is the clean control**: its cost is
    distribution-independent and its interaction count is identical in both
    versions, so its ratio isolates the -O3/-march=native + SoA change.
  * The accuracy columns are different metrics -- the old one is a relative
    L2 norm over the whole field, the new one a mean per-particle relative
    error. They are never plotted on the same axes here.

  plot_nbody_old_vs_new.py --old-dir DIR --new-dir DIR [--out FIG.png]
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Reference categorical palette, slots 1-6, assigned in fixed order.
COLORS = {
    "direct": "#2a78d6",
    "barnes_hut": "#eb6834",
    "fmm": "#1baf7a",
    "spherical_fmm": "#eda100",
    "gpu_direct": "#e87ba4",
    "gpu_bh": "#008300",
}
MARKERS = {
    "direct": "o", "barnes_hut": "s", "fmm": "^",
    "spherical_fmm": "D", "gpu_direct": "v", "gpu_bh": "P",
}
# old harness key -> canonical key
OLD_MAP = {"direct": "direct", "bh": "barnes_hut", "fmm": "fmm", "sfmm": "spherical_fmm"}
LABELS = {
    "direct": "Direct",
    "barnes_hut": "Barnes-Hut",
    "fmm": "FMM",
    "spherical_fmm": "Spherical FMM",
    "gpu_direct": "GPU Direct",
    "gpu_bh": "GPU Barnes-Hut",
}


def read_old(path):
    """dim,solver,label,n,threads,reps,min_ms,median_ms,mean_ms,std_ms,rel_l2_err"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "solver": OLD_MAP.get(r["solver"], r["solver"]),
                    "n": int(r["n"]),
                    "threads": int(r["threads"]),
                    "ms": float(r["median_ms"]),
                    "err": float(r["rel_l2_err"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def read_new(path):
    """label,threads,solver,n,theta,order,ms_per_eval,mean_rel_err"""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append({
                    "solver": r["solver"],
                    "n": int(r["n"]),
                    "threads": int(r["threads"]),
                    "ms": float(r["ms_per_eval"]),
                    "err": float(r["mean_rel_err"]),
                })
            except (KeyError, ValueError):
                continue
    return rows


def group(rows):
    out = {}
    for r in rows:
        out.setdefault(r["solver"], []).append(r)
    for k in out:
        out[k].sort(key=lambda r: r["n"])
    return out


def fit_exponent(ns, ts, min_n):
    ns = np.asarray(ns, float)
    ts = np.asarray(ts, float)
    keep = (ns >= min_n) & (ts > 0)
    if keep.sum() < 3:
        keep = ts > 0
    if keep.sum() < 2:
        return float("nan"), float("nan")
    x, y = np.log(ns[keep]), np.log(ts[keep])
    slope, icept = np.polyfit(x, y, 1)
    resid = y - (slope * x + icept)
    n = x.size
    if n > 2:
        sxx = float(((x - x.mean()) ** 2).sum())
        se = math.sqrt((float(resid @ resid) / (n - 2)) / sxx) if sxx > 0 else float("nan")
    else:
        se = float("nan")
    return float(slope), float(se)


def panel_overlay(ax, old_rows, new_rows, title, fit_min_n, fits):
    """New (solid) over old (dashed), same hue per solver."""
    drew = False
    for tag, rows, style, alpha in (("old", old_rows, "--", .55), ("new", new_rows, "-", 1.0)):
        for key, rs in group(rows).items():
            ns = [r["n"] for r in rs]
            ts = [r["ms"] for r in rs]
            if not ns:
                continue
            drew = True
            slope, se = fit_exponent(ns, ts, fit_min_n)
            fits.append((title, tag, LABELS.get(key, key), slope, se, max(ns)))
            ax.plot(ns, ts, style, marker=MARKERS.get(key, "o"), ms=4.5, lw=1.9 if tag == "new" else 1.4,
                    color=COLORS.get(key, "#555"), alpha=alpha,
                    label=f"{LABELS.get(key, key)} {tag} (N^{slope:.2f})")

    if not drew:
        ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("per force evaluation, ms")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.25, lw=.5)
    ax.legend(fontsize=6.5, loc="upper left", ncol=1)


def panel_ratio(ax, old_rows, new_rows, title):
    """old_ms / new_ms at matched N -- >1 means the new code is faster."""
    og, ng = group(old_rows), group(new_rows)
    drew = False
    for key in ng:
        if key not in og:
            continue
        om = {r["n"]: r["ms"] for r in og[key]}
        pts = [(r["n"], om[r["n"]] / r["ms"]) for r in ng[key] if r["n"] in om and r["ms"] > 0]
        if not pts:
            continue
        drew = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", marker=MARKERS.get(key, "o"),
                ms=5, lw=1.9, color=COLORS.get(key, "#555"), label=LABELS.get(key, key))
    ax.axhline(1.0, color="#999", lw=1, ls="--", zorder=0)
    ax.annotate("parity", (1.0, 1.0), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8, color="#777")
    if not drew:
        ax.text(.5, .5, "no overlapping points", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("old ms / new ms  (>1 = new faster)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.25, lw=.5)
    ax.legend(fontsize=7.5, loc="best")


def panel_plain(ax, rows, title, ylabel="per force evaluation, ms", logy=True):
    drew = False
    for key, rs in group(rows).items():
        ns = [r["n"] for r in rs]
        ts = [r["ms"] for r in rs]
        if not ns:
            continue
        drew = True
        ax.plot(ns, ts, "-", marker=MARKERS.get(key, "o"), ms=5, lw=1.9,
                color=COLORS.get(key, "#555"), label=LABELS.get(key, key))
    if not drew:
        ax.text(.5, .5, "no data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.25, lw=.5)
    ax.legend(fontsize=7.5, loc="best")


def panel_threads(ax, old_rows, new_rows, title):
    for tag, rows, style, alpha in (("old", old_rows, "--", .55), ("new", new_rows, "-", 1.0)):
        for key, rs in group(rows).items():
            rs = sorted(rs, key=lambda r: r["threads"])
            base = next((r["ms"] for r in rs if r["threads"] == 1), None)
            if base is None or len(rs) < 2:
                continue
            ax.plot([r["threads"] for r in rs], [base / r["ms"] for r in rs], style,
                    marker=MARKERS.get(key, "o"), ms=4.5, lw=1.9 if tag == "new" else 1.4,
                    color=COLORS.get(key, "#555"), alpha=alpha,
                    label=f"{LABELS.get(key, key)} {tag}")
    lim = 128
    ax.plot([1, lim], [1, lim], ":", color="#999", lw=1, zorder=0, label="ideal")
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xlabel("OpenMP threads"); ax.set_ylabel("speedup vs 1 thread")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.25, lw=.5)
    ax.legend(fontsize=6.5, loc="upper left")


def panel_accuracy(ax, rows, title):
    drew = False
    for key, rs in group(rows).items():
        pts = [(r["n"], r["err"]) for r in rs if r["err"] > 0]
        if not pts:
            continue
        drew = True
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", marker=MARKERS.get(key, "o"),
                ms=5, lw=1.9, color=COLORS.get(key, "#555"), label=LABELS.get(key, key))
    if not drew:
        ax.text(.5, .5, "no accuracy data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N particles"); ax.set_ylabel("mean rel. error vs fp64 Direct")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=.25, lw=.5)
    ax.legend(fontsize=7.5, loc="best")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--new-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fit-min-n", type=int, default=8000)
    a = ap.parse_args()

    old_full = read_old(os.path.join(a.old_dir, "scaling3d_full.csv"))
    old_ser = read_old(os.path.join(a.old_dir, "scaling3d_serial.csv"))
    old_thr = read_old(os.path.join(a.old_dir, "threads3d.csv"))

    new_full = read_new(os.path.join(a.new_dir, "new_cpu_full.csv"))
    new_ser = read_new(os.path.join(a.new_dir, "new_cpu_serial.csv"))
    new_thr = read_new(os.path.join(a.new_dir, "new_cpu_threads.csv"))
    new_gpu = read_new(os.path.join(a.new_dir, "new_gpu.csv"))
    new_gpu_cpu = read_new(os.path.join(a.new_dir, "new_gpu_cpuref.csv"))

    if not any((new_full, new_ser, new_gpu)):
        print(f"error: no new-side CSVs under {a.new_dir}", file=sys.stderr)
        print("       looked for new_cpu_full.csv / new_cpu_serial.csv / new_gpu.csv", file=sys.stderr)
        return 1

    fits = []
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

    panel_overlay(axes[0][0], old_ser, new_ser, "1 thread: old vs new", a.fit_min_n, fits)
    panel_overlay(axes[0][1], old_full, new_full, "128 threads: old vs new", a.fit_min_n, fits)
    panel_ratio(axes[0][2], old_full, new_full, "Speedup from the rewrite (128 threads)")

    # GPU node: GPU solvers plus that node's own CPU solvers, same hardware.
    panel_plain(axes[1][0], new_gpu + new_gpu_cpu, "GPU node: A100 fp32 vs 64-thread CPU fp64")
    panel_threads(axes[1][1], old_thr, new_thr, "Thread scaling at N=32,000")
    panel_accuracy(axes[1][2], new_full + new_gpu, "New-code accuracy (mean rel. err)")

    fig.suptitle(
        "N-body solvers before and after the ngrav unification -- Perlmutter (EPYC 7763 / A100)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, .02, 1, .97))

    out = a.out or os.path.join(a.new_dir, "nbody_old_vs_new.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")

    print()
    print(f"{'panel':<28} {'ver':<4} {'solver':<16} {'exponent':>10} {'stderr':>8} {'maxN':>8}")
    for panel, tag, label, slope, se, maxn in fits:
        print(f"{panel:<28} {tag:<4} {label:<16} {slope:>10.3f} {se:>8.3f} {maxn:>8d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
