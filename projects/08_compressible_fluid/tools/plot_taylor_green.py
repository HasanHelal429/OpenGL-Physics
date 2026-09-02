"""
Validation plot for a headless 08_compressible_fluid Taylor-Green vortex run
(decks/taylor_green.toml): u,v = U0*cos(kx)*sin(ky), -U0*sin(kx)*cos(ky)
varies at wavenumber k in BOTH x and y, so its Laplacian picks up a
coefficient of TWO k^2 (d^2/dx^2 + d^2/dy^2, each contributing -k^2*u) --
velocity decays as exp(-2*nu*k^2*t) and kinetic energy (quadratic in
velocity) decays twice as fast, KE(t) = KE(0) * exp(-4*nu*k^2*t)
(nu = mu/rho0, k = 2*pi*wavenumber_multiplier/length) -- the classic
incompressible-NS Taylor-Green decay law. This fits the measured KE(t)
curve's log-slope and compares it to that analytic rate, the real check
that this project's viscous terms (Phase 4) get the *transient* decay
right, not just the steady-state Poiseuille profile already validated.

`measure_decay_rate(results_dir)` is the reusable core (deck + diagnostics
in, measured/exact decay rates out) -- imported directly by
Studies/compressible_fluid/vorticity_decay_vs_viscosity/analyze.py rather
than re-derived there, same pattern as 04_molecular_dynamics/tools/msd.py's
compute_diffusion.

Usage:
    python plot_taylor_green.py <results_dir> [--out FILE]
"""

import argparse
import os

import numpy as np

try:
    import tomllib
except ImportError:
    tomllib = None


def measure_decay_rate(results_dir):
    """Returns a dict: t, ke (arrays), mu, decay_rate_measured, decay_rate_exact,
    relative_error. decay_rate_measured is a least-squares fit of log(KE) vs t
    over the whole run (robust to the small residual compressible wobble on
    top of the dominant exponential trend)."""
    with open(os.path.join(results_dir, "deck.toml"), "rb") as f:
        deck = tomllib.load(f)

    mu = deck["physics"]["mu"]
    rho0 = deck["physics"]["rho0"]
    length = deck["grid"]["length"]
    wavenumber_multiplier = deck["taylor_green"].get("wavenumber_multiplier", 1)
    nu = mu / rho0
    k = 2.0 * np.pi * wavenumber_multiplier / length
    # Velocity's Laplacian picks up a factor of 2*k^2 (d^2/dx^2 + d^2/dy^2,
    # each -k^2*u, since u varies at wavenumber k in both directions), so
    # velocity decays at rate 2*nu*k^2 and kinetic energy (quadratic in
    # velocity) at twice that.
    decay_rate_exact = 4.0 * nu * k * k

    data = np.genfromtxt(os.path.join(results_dir, "diagnostics.csv"), delimiter=",", names=True)
    t = data["t"]
    ke = data["kinetic_energy"]

    coeffs = np.polyfit(t, np.log(ke), 1)
    decay_rate_measured = -coeffs[0]
    rel_err = abs(decay_rate_measured - decay_rate_exact) / decay_rate_exact
    return {
        "t": t,
        "ke": ke,
        "mu": mu,
        "k": k,
        "decay_rate_measured": decay_rate_measured,
        "decay_rate_exact": decay_rate_exact,
        "relative_error": rel_err,
    }


def main():
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    r = measure_decay_rate(d)
    t, ke = r["t"], r["ke"]
    decay_rate_measured, decay_rate_exact, rel_err = (
        r["decay_rate_measured"], r["decay_rate_exact"], r["relative_error"])
    print(f"decay rate: measured={decay_rate_measured:.5f}  exact=4*nu*k^2={decay_rate_exact:.5f}  "
          f"relative error={rel_err:.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(t, ke, ".", ms=4, label="08_compressible_fluid")
    ax[0].plot(t, ke[0] * np.exp(-decay_rate_exact * t), "k-", lw=1, label="exact exp(-4*nu*k^2*t)")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("kinetic energy"); ax[0].legend(fontsize=8)
    ax[0].set_title("kinetic energy decay")

    ax[1].semilogy(t, ke, ".", ms=4, label="08_compressible_fluid")
    ax[1].semilogy(t, ke[0] * np.exp(-decay_rate_exact * t), "k-", lw=1, label="exact exp(-4*nu*k^2*t)")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("kinetic energy (log scale)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"measured rate={decay_rate_measured:.4f}  exact={decay_rate_exact:.4f}  "
                     f"(err={rel_err:.2%})")

    fig.suptitle(os.path.basename(os.path.abspath(d)))
    fig.tight_layout()
    out = args.out or os.path.join(d, "diagnostics.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
