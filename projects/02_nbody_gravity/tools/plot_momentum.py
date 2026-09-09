"""Total system momentum |P| = |sum m_i v_i| vs time -- the headline
diagnostic for the mutual (falcON-style) dual-tree FMM (Phase 6): a
momentum-conserving force solver keeps this at its initial value (to machine
precision, since the *integrator* is already symplectic); the retired
one-directional AdaptiveFmm, or Barnes-Hut's own incidental cancellation,
generally do not, especially through a violent-relaxation collapse+bounce
where per-step force imbalance has the most opportunity to accumulate.

Usage:
    python plot_momentum.py <results_dir> [<results_dir> ...] [--out momentum.png]

Typical use: compare decks/cold_collapse_fmm.toml against
decks/cold_collapse.toml (Barnes-Hut) on the identical IC.
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
    p = [float(r["p_mag"]) for r in rows]
    e = [float(r["energy_drift_pct"]) for r in rows]
    return t, p, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="+")
    ap.add_argument("--out", default="momentum.png")
    a = ap.parse_args()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for d in a.results_dir:
        t, p, e = load(d)
        label = os.path.basename(os.path.normpath(d))
        ax0.semilogy([max(v, 1e-20) for v in p], label=label) if False else ax0.plot(t, p, label=label)
        ax1.plot(t, e, label=label)
    ax0.set_ylabel("|P| = |sum m_i v_i|")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax1.set_ylabel("energy drift  [%]")
    ax1.set_xlabel("t")
    ax1.axhline(0, color="k", lw=0.5)
    ax1.grid(alpha=0.3)
    fig.suptitle("Momentum conservation: mutual FMM vs Barnes-Hut")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
