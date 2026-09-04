"""
Analytic two-body inspiral reference (non-relativistic, circular, c = eps0 = 1).

A bound pair of charges +q, -q with equal mass m on a circular orbit of
separation d has

    omega^2 = 2 k q^2 / (m d^3),            k = 1 / 4 pi
    E(d)    = -k q^2 / (2 d)                 (virial: KE = -E, PE = 2E)

and radiates (Larmor dipole, p = q d)

    P(d) = (q omega^2 d)^2 / (6 pi)  = (2/3) k^2 q^6 / (pi m^2 d^4).

With dE/dd = k q^2 / (2 d^2), energy balance dE/dt = -P gives

    dd/dt = -P / (dE/dd) = -(4/3) k q^4 / (pi m^2 d^2)   [* 0.5 with --half]

11_retarded_fields' pairwise retarded solver, run without the Abraham-
Lorentz self-force, carries half of P for the symmetric pair, so --half
matches the simulation.

Usage:
    python inspiral_ref.py --d0 5 --q 1.5 --m 1 --tmax 400 [--half] [--out FILE]
"""

import argparse
import numpy as np


def integrate(d0, q, m, tmax, nsteps, half):
    k = 1.0 / (4.0 * np.pi)
    fac = (4.0 / 3.0) * k * q**4 / (np.pi * m**2)
    if half:
        fac *= 0.5
    t = np.linspace(0.0, tmax, nsteps)
    dt = t[1] - t[0]
    d = np.empty(nsteps)
    d[0] = d0
    for i in range(1, nsteps):
        di = d[i - 1]
        if di <= 0.05:
            d[i:] = di
            break
        # RK4 on dd/dt = -fac / d^2
        f = lambda x: -fac / (x * x)
        k1 = f(di)
        k2 = f(di + 0.5 * dt * k1)
        k3 = f(di + 0.5 * dt * k2)
        k4 = f(di + dt * k3)
        d[i] = di + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    d = np.clip(d, 0.05, None)
    omega = np.sqrt(k * 2.0 * q * q / (m * d**3))
    E = -k * q * q / (2.0 * d)
    return t, d, omega, E


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--d0", type=float, default=5.0)
    ap.add_argument("--q", type=float, default=1.5)
    ap.add_argument("--m", type=float, default=1.0)
    ap.add_argument("--tmax", type=float, default=400.0)
    ap.add_argument("--nsteps", type=int, default=4000)
    ap.add_argument("--half", action="store_true",
                    help="halve the radiated power (no-self-force pair)")
    ap.add_argument("--csv", default=None, help="write t,d,omega,E to this file")
    ap.add_argument("--out", default=None, help="plot file")
    args = ap.parse_args()

    t, d, omega, E = integrate(args.d0, args.q, args.m, args.tmax,
                               args.nsteps, args.half)

    if args.csv:
        np.savetxt(args.csv, np.column_stack([t, d, omega, E]),
                   delimiter=",", header="t,d,omega,E", comments="")
        print("wrote", args.csv)

    if args.out or not args.csv:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(t, d)
        ax[0].set_xlabel("t"); ax[0].set_ylabel("separation d")
        ax[0].set_title("analytic inspiral" + ("  (half power)" if args.half else ""))
        ax[1].plot(t, E)
        ax[1].set_xlabel("t"); ax[1].set_ylabel("mechanical energy")
        fig.tight_layout()
        out = args.out or "inspiral_ref.png"
        fig.savefig(out, dpi=130)
        print("wrote", out)


if __name__ == "__main__":
    main()
