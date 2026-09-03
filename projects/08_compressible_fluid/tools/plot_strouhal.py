"""
Strouhal-number analysis for a headless 08_compressible_fluid cylinder run
(decks/cylinder_re100.toml): FFT the downstream velocity probe (probe_v in
diagnostics.csv), find the dominant (shedding) frequency, compute
St = f*D/U_inf, and compare against Roshko's empirical correlation
(St = 0.198*(1-19.7/Re)) and a representative 2D CFD literature value
(St~0.166 at Re=100) -- the same check MAC_Grid_Solver's
Cylinder_Vortex_Shedding.ipynb runs against its own incompressible solver.

Usage:
    python plot_strouhal.py <results_dir> [--transient T] [--out FILE]

--transient discards data before t=T from the FFT window (default: half
the run) -- the flow needs time to develop from its impulsive start (and,
for a symmetric geometry perturbed only slightly off-axis, the shedding
instability itself needs time to grow from its initial seed -- see the
project README for whether this run's window was long enough to reach a
saturated periodic limit cycle or only the earlier exponential-growth
phase, which still carries the same frequency).
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--transient", type=float, default=None,
                     help="discard t < this value from the FFT window (default: half the run's final time)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    with open(os.path.join(d, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)
    # Generic-scene schema: no dedicated [cylinder] convenience table --
    # diameter/u_inf/Reynolds are derived from the raw obstacle/boundary/
    # physics values the deck actually specifies.
    diameter = 2.0 * deck["obstacles"][0]["radius"]
    u_inf = deck["boundary"]["left"]["u"]
    rho0 = deck["boundary"]["left"]["rho"]
    mu = deck["physics"]["mu"]
    reynolds = rho0 * u_inf * diameter / mu

    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    probe_v = data["probe_v"]

    transient = args.transient if args.transient is not None else 0.5 * t[-1]
    mask = t >= transient
    t_win = t[mask]
    v_win = probe_v[mask]

    # Resample onto a uniform time grid before FFT (this project's dt is
    # fixed, so frame times are already uniform, but do it explicitly to be
    # robust regardless).
    t_uniform = np.linspace(t_win[0], t_win[-1], len(t_win))
    v_uniform = np.interp(t_uniform, t_win, v_win)
    dt_uniform = t_uniform[1] - t_uniform[0]

    v_detrended = v_uniform - np.mean(v_uniform)
    spectrum = np.abs(np.fft.rfft(v_detrended * np.hanning(len(v_detrended))))
    freqs = np.fft.rfftfreq(len(v_detrended), d=dt_uniform)

    peak_idx = np.argmax(spectrum[1:]) + 1  # skip the DC bin
    f_shed = freqs[peak_idx]
    st_measured = f_shed * diameter / u_inf
    st_roshko = 0.198 * (1.0 - 19.7 / reynolds)
    st_literature = 0.166  # representative 2D CFD benchmark point at Re=100 (commonly cited range 0.164-0.167)

    print(f"FFT window: t=[{t_win[0]:.2f}, {t_win[-1]:.2f}]  ({len(t_win)} samples, dt={dt_uniform:.4f})")
    print(f"Measured shedding frequency: f = {f_shed:.5f}")
    print(f"Measured Strouhal number:    St = {st_measured:.4f}")
    print(f"Roshko correlation:          St = {st_roshko:.4f}")
    print(f"Literature (2D CFD, Re={reynolds:.0f}): St ~ {st_literature:.3f}")
    print(f"relative error vs Roshko: {abs(st_measured - st_roshko) / st_roshko:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(t, probe_v)
    axes[0].axvline(transient, color="r", ls="--", lw=1, label=f"FFT window starts (t={transient:.1f})")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("probe v"); axes[0].set_title("full probe timeseries")
    axes[0].legend(fontsize=8)

    axes[1].plot(t_win, v_win)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("probe v"); axes[1].set_title("FFT window")

    axes[2].plot(freqs, spectrum)
    axes[2].axvline(f_shed, color="r", ls="--", lw=1, label=f"peak f={f_shed:.4f}")
    axes[2].set_xlim(0, max(0.5, f_shed * 4))
    axes[2].set_xlabel("frequency"); axes[2].set_ylabel("|FFT|")
    axes[2].set_title(f"St={st_measured:.4f} (Roshko={st_roshko:.4f}, lit~{st_literature:.3f})")
    axes[2].legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "strouhal.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
