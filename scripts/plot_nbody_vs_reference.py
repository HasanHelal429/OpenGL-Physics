#!/usr/bin/env python3
"""Compare the ngrav solvers against open-source references at MATCHED
accuracy: how many x is our implementation off from a well-optimised one?

    plot_nbody_vs_reference.py --ngrav-csv PARETO.csv --ref-csv REF.csv
                              [--out FIG.png] [--targets 1e-2,1e-3]

Both CSVs use the scripts/nbody_bench_sweep.sh schema
    label,threads,solver,n,theta,order,ms_per_eval,mean_rel_err
with SEVERAL rows per (solver, N) -- one per point on that solver's own
accuracy ladder (opening angle for the ngrav tree/FMM solvers via
`--thetas`, requested precision for fmm3d, opening angle for pytreegrav).

Pairing (ours -> reference; run both unsoftened, same N, same threads):
    fmm           -> fmm3d        FMM vs FMM (different family)
    spherical_fmm -> fmm3d        both spherical-harmonic FMM
    barnes_hut    -> pytreegrav   tree vs tree

For each (pair, N, target error): log-log-interpolate each side's ms at
err == target on its own ladder, and report ms_ours / ms_ref. That ratio,
vs N, is the headline gap.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("needs numpy + matplotlib")

PAIRS = [("fmm", "fmm3d"), ("spherical_fmm", "fmm3d"), ("barnes_hut", "pytreegrav")]
LAB = {"fmm": "mutual FMM", "barnes_hut": "Barnes-Hut", "spherical_fmm": "spherical FMM",
       "fmm3d": "FMM3D", "pytreegrav": "pytreegrav"}
COL = {"fmm": "#1baf7a", "barnes_hut": "#eb6834", "spherical_fmm": "#eda100"}


def read(path):
    d = {}
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                key = (r["solver"], int(r["n"]))
                d.setdefault(key, []).append((float(r["mean_rel_err"]), float(r["ms_per_eval"])))
            except (KeyError, ValueError):
                continue
    return d


def ms_at_err(ladder, target):
    """ladder: [(err, ms), ...]. Log-log interpolate ms at err==target.
    Returns (ms, flag) where flag marks extrapolation."""
    pts = sorted((e, m) for e, m in ladder if e > 0 and m > 0)
    if len(pts) < 2:
        return (pts[0][1] if pts else np.nan), "sparse"
    e = np.log([p[0] for p in pts])
    m = np.log([p[1] for p in pts])
    lt = np.log(target)
    o = np.argsort(e)
    flag = "" if e.min() <= lt <= e.max() else ("extrap<" if lt < e.min() else "extrap>")
    return float(np.exp(np.interp(lt, e[o], m[o]))), flag


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ngrav-csv", required=True)
    ap.add_argument("--ref-csv", required=True)
    ap.add_argument("--targets", default="1e-2,3e-3,1e-3")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ours = read(a.ngrav_csv)
    ref = read(a.ref_csv)
    targets = [float(x) for x in a.targets.split(",")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    print(f"\n{'pair':<28} {'N':>8} {'target':>8} {'our ms':>10} {'ref ms':>10} {'ratio':>8}  flags")
    print("-" * 92)

    for our_s, ref_s in PAIRS:
        ns = sorted({n for (s, n) in ours if s == our_s} & {n for (s, n) in ref if s == ref_s})
        if not ns:
            continue
        for ti, tgt in enumerate(targets):
            ratios, keep_n = [], []
            for n in ns:
                om, of = ms_at_err(ours[(our_s, n)], tgt)
                rm, rf = ms_at_err(ref[(ref_s, n)], tgt)
                if not (np.isfinite(om) and np.isfinite(rm) and rm > 0):
                    continue
                ratio = om / rm
                ratios.append(ratio); keep_n.append(n)
                print(f"{LAB[our_s]+' vs '+LAB[ref_s]:<28} {n:>8} {tgt:>8.0e} {om:>10.2f} {rm:>10.2f} "
                      f"{ratio:>8.1f}  {of} {rf}".rstrip())
            if keep_n:
                ls = ["-", "--", ":"][ti % 3]
                axes[1].semilogx(keep_n, ratios, ls, marker="o", color=COL[our_s],
                                 label=f"{LAB[our_s]} @ {tgt:.0e}")

        # panel 0: cost vs N at the tightest target, ours solid / ref dashed
        tgt = targets[-1]
        oc = [(n, ms_at_err(ours[(our_s, n)], tgt)[0]) for n in ns]
        rc = [(n, ms_at_err(ref[(ref_s, n)], tgt)[0]) for n in ns]
        axes[0].loglog([n for n, _ in oc], [m for _, m in oc], "o-", color=COL[our_s],
                       label=f"{LAB[our_s]} (ours)")
        axes[0].loglog([n for n, _ in rc], [m for _, m in rc], "s--", color=COL[our_s], alpha=.65,
                       label=f"{LAB[ref_s]}")

    axes[0].set_xlabel("N"); axes[0].set_ylabel("ms / force eval")
    axes[0].set_title(f"cost vs N at err = {targets[-1]:.0e}")
    axes[0].legend(fontsize=8); axes[0].grid(True, which="both", alpha=.3)

    axes[1].axhline(1.0, color="k", lw=.8)
    axes[1].set_xlabel("N"); axes[1].set_ylabel("our ms / reference ms")
    axes[1].set_title("implementation gap at matched accuracy  (>1 = we are slower)")
    axes[1].legend(fontsize=7); axes[1].grid(True, which="both", alpha=.3)

    # panel 2: accuracy-vs-cost ladders at the largest common N
    all_common = set()
    for our_s, ref_s in PAIRS:
        ns = {n for (s, n) in ours if s == our_s} & {n for (s, n) in ref if s == ref_s}
        all_common |= ns
    if all_common:
        big = max(all_common)
        for our_s, ref_s in PAIRS:
            for s, style, tag in ((our_s, "o-", "ours"), (ref_s, "s--", "ref")):
                key = (s, big)
                src = ours if tag == "ours" else ref
                if key in src:
                    pts = sorted(src[key])
                    axes[2].plot([m for _, m in pts], [e for e, _ in pts], style,
                                 color=COL.get(our_s, "#888"), alpha=.9 if tag == "ours" else .55,
                                 label=f"{LAB.get(s, s)} ({tag})")
        axes[2].set_xscale("log"); axes[2].set_yscale("log")
        axes[2].set_xlabel("ms / force eval"); axes[2].set_ylabel("mean rel force error")
        axes[2].set_title(f"accuracy vs cost at N = {big:,}")
        axes[2].legend(fontsize=7); axes[2].grid(True, which="both", alpha=.3)

    fig.tight_layout()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.ref_csv)), "nbody_vs_reference.png")
    fig.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
