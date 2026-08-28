#!/usr/bin/env python3
"""Validation plots from a headless 05_tdse_gpu run.

    python plot_diagnostics.py <results_dir> [--out diagnostics.png]

Reads diagnostics.csv (+ deck.toml for analytic overlays) and plots norm and
energy conservation, <x>(t) against the classical/analytic prediction, Var(x)(t)
against free-particle spreading, and transmission(t).
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
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    deck = load_deck(os.path.join(d, "deck.toml"))
    ini = deck.get("initial", {})
    pots = deck.get("potential", [])
    omega = next((p.get("omega") for p in pots if p.get("type") == "harmonic"), None)

    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    ax = ax.ravel()

    ax[0].plot(t, data["norm"] - 1.0)
    ax[0].set_title("norm - 1"); ax[0].set_xlabel("t")

    if "energy" in data.dtype.names:
        e = data["energy"]
        ax[1].plot(t, e - e[0])
        ax[1].set_title(f"energy - energy(0)   (E0={e[0]:.4f})"); ax[1].set_xlabel("t")

    ax[2].plot(t, data["x_mean"], label="sim")
    x0 = ini.get("x0", 0.0)
    if omega:
        ax[2].plot(t, x0 * np.cos(omega * t), "--", label=f"x0 cos(wt), w={omega}")
    elif "kx" in ini:
        ax[2].plot(t, x0 + ini["kx"] * t, "--", label=f"x0 + kx t")
    ax[2].set_title("<x>(t)"); ax[2].set_xlabel("t"); ax[2].legend()

    if "x_var" in data.dtype.names:
        ax[3].plot(t, data["x_var"], label="sim")
        sigma = ini.get("sigma")
        if sigma and not omega:
            var0 = sigma ** 2 / 2.0
            ax[3].plot(t, var0 + t ** 2 / (4.0 * var0), "--", label="free spreading")
        ax[3].set_title("Var(x)(t)"); ax[3].set_xlabel("t"); ax[3].legend()

    if "transmission" in data.dtype.names:
        ax[4].plot(t, data["transmission"])
        ax[4].set_title("transmission (|psi|^2 beyond barrier)"); ax[4].set_xlabel("t")

    if {"kinetic", "potential_energy"} <= set(data.dtype.names):
        ax[5].plot(t, data["kinetic"], label="<T>")
        ax[5].plot(t, data["potential_energy"], label="<V>")
        ax[5].set_title("energy split"); ax[5].set_xlabel("t"); ax[5].legend()

    for a in ax:
        if not (a.lines or a.collections):
            a.set_visible(False)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)

    # Console summary.
    print(f"  norm drift:   {abs(data['norm'][-1] - 1):.3e}")
    if "energy" in data.dtype.names:
        e = data["energy"]
        print(f"  energy drift: {abs(e[-1] - e[0]) / abs(e[0]):.3e} (relative)")


if __name__ == "__main__":
    main()
