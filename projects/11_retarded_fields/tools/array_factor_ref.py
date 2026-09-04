"""
Array-factor reference for decks/dipole_array.toml.

N identical dipole elements at positions r_n (read from the deck's [[charge]]
tables), each driven with phase psi_n = phase_n, radiate with the combined
far-field pattern

    |E(theta)|^2  ~  sin^2(theta_from_axis)  *  | sum_n exp(i (k rhat.r_n + psi_n)) |^2

with k = omega / c. This prints the main-lobe / grating-lobe angles and,
with --out, plots the pattern; pass it to plot_pattern.py as the overlay.

Usage:
    python array_factor_ref.py <deck.toml> [--out FILE]
"""

import argparse
import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(args.deck, "rb") as f:
        deck = tomllib.load(f)
    ch = deck["charge"]
    c = deck.get("domain", {}).get("c", 1.0)
    omega = ch[0].get("omega", 1.0)
    k = omega / c
    pos = np.array([cc.get("center", [0, 0, 0]) for cc in ch])[:, :2]
    psi = np.array([cc.get("phase", 0.0) for cc in ch])
    axis = np.array(ch[0].get("axis", [0.0, 1.0, 0.0])[:2])
    axis = axis / np.linalg.norm(axis)
    lam = 2 * np.pi / k
    print(f"omega={omega}  k={k:.4f}  lambda={lam:.3f}")
    print(f"element spacing / lambda = "
          f"{np.linalg.norm(pos[0] - pos[-1]) / lam:.3f}")

    ph = np.linspace(0, 2 * np.pi, 3601)
    nhat = np.stack([np.cos(ph), np.sin(ph)], axis=1)
    af = np.abs(np.sum(np.exp(1j * (k * nhat @ pos.T + psi)), axis=1)) ** 2
    elem = 1.0 - (nhat @ axis) ** 2           # sin^2 from the oscillation axis
    patt = elem * af
    patt /= patt.max()

    if args.out:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="polar")
        ax.plot(ph, patt, lw=1.4)
        ax.set_title("array factor x element pattern", pad=18)
        fig.savefig(args.out, dpi=130)
        print("wrote", args.out)
    else:
        peaks = ph[1:-1][(patt[1:-1] > patt[:-2]) & (patt[1:-1] > patt[2:])
                         & (patt[1:-1] > 0.25)]
        print("lobe angles (deg):", np.round(np.degrees(peaks), 1))


if __name__ == "__main__":
    main()
