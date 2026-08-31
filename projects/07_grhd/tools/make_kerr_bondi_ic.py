"""
Non-equilibrium initial condition for 07_grhd's KerrTorusSim: the exact
Schwarzschild relativistic Bondi solution (tools/bondi_analytic.py,
already validated against SchwarzschildSim's 1D solver -- see
decks/bondi_schwarzschild.toml) sampled onto the 2D Kerr-Schild grid,
theta-independent, as a zero-angular-momentum (v_phi=0) radial infall.

This is NOT an exact solution of the Kerr equations for a!=0 -- a
genuinely stationary spherically-symmetric accretion flow doesn't exist
once frame dragging is present. It's used here as a well-motivated
non-equilibrium initial condition instead: start the flow in the state
that IS exactly steady for a=0, and watch what a!=0 does to it. Two uses:

  a=0: an independent cross-check of the whole 2D Kerr-Schild pipeline
       (con2prim, reconstruction, HLLE+tight wave speeds, source terms)
       against the DIFFERENT, already-validated 1D Schwarzschild solver --
       if the measured accretion rate matches BondiSolution.Mdot_abs, that's
       real evidence the 2D solver is correct in a regime this project
       hasn't cross-checked before.
  a=0.9: the actual non-equilibrium test -- does frame dragging measurably
         perturb a nominally-radial inflow?

r_min can be placed INSIDE the horizon (only meaningful because of the
Kerr-Schild reformulation), letting the accretion flow be followed
smoothly across it instead of needing an artificial outflow boundary
outside it (the old Boyer-Lindquist-era limitation this project's
bondi_schwarzschild.toml still has, at r_min=3M).

    python make_kerr_bondi_ic.py --M 1.0 --a 0.9 --gamma 1.4 --r_c 8.0 \
        --nr 96 --ntheta 32 --r_min 1.0 --r_max 30.0 --theta_min 0.5 \
        --out ic/kerr_bondi_a09.bin
"""

import argparse
import sys

import numpy as np

from bondi_analytic import BondiSolution


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--r_c", type=float, required=True, help="Bondi sonic radius")
    ap.add_argument("--nr", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=32)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--theta_min", type=float, required=True)
    ap.add_argument("--r_safe", type=float, default=3.0,
                     help="the analytic profile is Schwarzschild-BL-based (alpha=sqrt(1-2M/r), imaginary "
                          "for r<2M) and only evaluated for r>=r_safe; grid cells with r<r_safe (only "
                          "possible with a Kerr-Schild grid.r_min inside the horizon) are held at the "
                          "r_safe state instead of extrapolated -- a well-motivated placeholder, not a "
                          "claimed exact solution there (see module docstring).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 0.1:
        raise SystemExit(f"r_min={args.r_min} must stay well clear of r=0 (the true curvature singularity)")

    sol = BondiSolution(args.M, args.r_c, args.gamma)
    mdot_err, be_err, k_err = sol.check_conserved(np.linspace(max(args.r_min, args.r_safe), args.r_max, 200))
    print(f"Bondi solution self-check over [r_min,r_max]: Mdot rel.err={mdot_err:.2e}  "
          f"Be rel.err={be_err:.2e}  K rel.err={k_err:.2e}  (Mdot_abs={sol.Mdot_abs:.6f})")
    if max(mdot_err, be_err, k_err) > 1e-6:
        sys.exit("Bondi solution failed its own conserved-quantity check -- refusing to write an IC from it")

    theta_max = np.pi - args.theta_min
    dr = (args.r_max - args.r_min) / args.nr
    dth = (theta_max - args.theta_min) / args.ntheta
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    theta = args.theta_min + (np.arange(args.ntheta) + 0.5) * dth

    r_eval = np.maximum(r, args.r_safe)
    rho_r, v_r, p_r = sol.profile(r_eval)  # v_r here is the BL-orthonormal radial velocity (negative, infall)
    n_clamped = int(np.count_nonzero(r < args.r_safe))
    if n_clamped:
        print(f"note: {n_clamped} radial cells have r<r_safe={args.r_safe} and were held at the r_safe state")

    record = np.zeros((args.nr, args.ntheta, 5), dtype=np.float32)
    for j in range(args.ntheta):
        record[:, j, 0] = rho_r
        record[:, j, 1] = v_r  # v_phi=v_theta=0: zero angular momentum, purely radial
        record[:, j, 4] = p_r

    record.reshape(-1, 5).tofile(args.out)
    print(f"grid: {args.nr}x{args.ntheta} = {args.nr*args.ntheta} cells (theta-independent), "
          f"rho range [{rho_r.min():.4f}, {rho_r.max():.4f}], v_r range [{v_r.min():.4f}, {v_r.max():.4f}]")
    print(f"wrote {args.nr*args.ntheta} cells -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] nr={args.nr}  ntheta={args.ntheta}  r_min={args.r_min}  r_max={args.r_max}  "
          f"theta_min={args.theta_min}   [physics] M={args.M}  a={args.a}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
