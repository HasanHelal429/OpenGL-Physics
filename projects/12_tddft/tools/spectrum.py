"""Turn a 12_tddft --he-spectrum run (dipole.csv + meta.txt) into the
absorption strength function S(omega), the TRK f-sum rule, and the lowest
peak -- the same analysis as the Stage-1 Python reference
(Physics Simulations/Quantum Mechanics/TDDFT/response.py).

    python spectrum.py <run_dir> [--out FILE.png]
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HA_TO_EV = 27.211386245988


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = {}
    with open(os.path.join(args.run_dir, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = float(v)
    k = meta["kappa"]
    Ne = meta["N_e"]

    data = np.genfromtxt(os.path.join(args.run_dir, "dipole.csv"), delimiter=",", names=True)
    t = np.atleast_1d(data["t"])
    d = np.atleast_1d(data["dx"])
    dt = t[1] - t[0]
    T = t[-1] - t[0]
    tau = 0.4 * T

    sig = (d - d[0]) * np.exp(-(t - t[0]) / tau)
    n = 1
    while n < 4 * len(sig):
        n *= 2
    pad = np.zeros(n)
    pad[: len(sig)] = sig
    G = np.fft.ifft(pad) * n * dt                 # integral e^{+iwt} f(t) dt
    w = 2 * np.pi * np.fft.fftfreq(n, d=dt)
    pos = w >= 0
    w = w[pos]
    alpha = (1.0 / k) * G[pos]                    # Im alpha > 0 at resonances
    S = (2.0 * w / np.pi) * np.imag(alpha)

    band = w < 4.0
    w, S, alpha = w[band], S[band], alpha[band]
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    n_eff = float(trap(S, w))
    im = np.imag(alpha)
    im_rel = np.min(im[w > 0.1]) / np.max(im[w > 0.1])

    # lowest prominent peak
    loc = (S[1:-1] > S[:-2]) & (S[1:-1] >= S[2:])
    idx = np.where(loc)[0] + 1
    idx = idx[(w[idx] > 0.2) & (w[idx] < 3.0)]
    order = idx[np.argsort(S[idx])[::-1][:5]]
    strong = [i for i in order if S[i] > 0.15 * S[order[0]]]
    low = min(strong, key=lambda i: w[i]) if strong else order[0]
    w_low = w[low]

    print(f"  integral S dw   = {n_eff:.3f}  = {100 * n_eff / Ne:.0f}% of N_e ({Ne:.1f})")
    print(f"  min/max Im alpha= {im_rel:+.2e}  (passivity)")
    print(f"  lowest peak     = {w_low:.3f} Ha = {w_low * HA_TO_EV:.1f} eV")
    print(f"  alpha(0)        = {np.real(alpha[0]):.2f} a.u.")
    print()
    print("  Stage-1 Python reference (He, N=32, L=16): sum rule ~97% of N_e,")
    print("  lowest peak ~0.50 Ha = 13.5 eV.")

    ev = w * HA_TO_EV
    fig, (a, b) = plt.subplots(2, 1, figsize=(7, 6), dpi=130)
    a.plot(ev, S, color="#4c72b0")
    a.axvline(w_low * HA_TO_EV, color="#dd8452", ls="--", lw=1, label=f"lowest peak {w_low*HA_TO_EV:.1f} eV")
    a.axvline(13.5, color="#888", ls=":", lw=1, label="Stage-1 Python 13.5 eV")
    a.set_xlabel(r"$\omega$  (eV)"); a.set_ylabel(r"$S(\omega)$")
    a.set_title(f"12_tddft (GPU) -- He absorption  ($\\int S\\,d\\omega$ = {n_eff:.2f}, $N_e$ = {Ne:.0f})")
    a.set_xlim(0, min(ev[-1], 90)); a.legend(frameon=False, fontsize=8)
    b.plot(ev, np.cumsum(S) * (w[1] - w[0]), color="#55a868")
    b.axhline(Ne, color="#888", ls=":", lw=1)
    b.set_xlabel(r"$\omega$  (eV)"); b.set_ylabel(r"running $N_{\rm eff}$")
    b.set_xlim(0, min(ev[-1], 90))
    out = args.out or os.path.join(args.run_dir, "spectrum.png")
    fig.tight_layout(); fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
