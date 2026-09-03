"""
Plot the Ez probe time series (and their spectra) from a headless 10_fdtd
run's diagnostics.csv, plus the total-energy history.

Usage:
    python plot_probe.py <results_dir> [--out FILE]
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    man = json.load(open(os.path.join(d, "manifest.json")))
    dt = man.get("dt", 1.0)
    sub = man.get("substeps_per_frame", 1)
    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    probes = [c for c in data.dtype.names if c.startswith("probe") and c.endswith("_Ez")]

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for p in probes:
        ax[0].plot(t, data[p], lw=0.8, label=p.replace("_Ez", ""))
    ax[0].set_xlabel("t"); ax[0].set_ylabel("Ez"); ax[0].set_title("probe time series")
    ax[0].legend(fontsize=8)

    frame_dt = dt * sub
    for p in probes:
        sig = data[p] - data[p].mean()
        spec = np.abs(np.fft.rfft(sig))
        f = np.fft.rfftfreq(len(sig), frame_dt)
        ax[1].plot(f, spec, lw=0.8, label=p.replace("_Ez", ""))
    ax[1].set_xlabel("frequency"); ax[1].set_ylabel("|FFT(Ez)|")
    ax[1].set_title("probe spectra"); ax[1].legend(fontsize=8)
    ax[1].set_xlim(0, f.max() / 2)

    if "energy" in data.dtype.names:
        e = data["energy"]
        ax[2].plot(t, e)
        ax[2].set_xlabel("t"); ax[2].set_ylabel("U")
        ax[2].set_title("total EM energy")
        # the source is off well before the back half of the run
        es = e[len(e) // 2:]
        print(f"energy (back half of run): min/max = {es.min():.6e} / "
              f"{es.max():.6e}  (spread {np.ptp(es) / es.max():.2e})")

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "probe.png")
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
