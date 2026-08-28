"""
Generate a 07_grhd Schwarzschild-geometry initial condition: the exact
relativistic Bondi accretion solution (tools/bondi_analytic.py), sampled
at each grid cell's center. Writes a raw binary file of N*3 float32
(rho, v, P) per cell, matching SchwarzschildSim::UploadInitial's expected
layout -- grid.n/grid.r_min/grid.r_max in the deck must match what this
was generated with (same convention as 06_tidal_disruption's ic/*.bin).

Usage:
    python make_bondi_ic.py --M 1.0 --r_c 8.0 --gamma 1.4 \
        --n 400 --r_min 3.0 --r_max 40.0 --out ic/bondi_rc8.bin
"""

import argparse

import numpy as np

from bondi_analytic import BondiSolution


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--r_c", type=float, required=True, help="sonic radius")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 2.0 * args.M:
        raise SystemExit(f"r_min={args.r_min} must be outside the horizon (2*M={2*args.M})")

    sol = BondiSolution(M=args.M, r_c=args.r_c, gamma=args.gamma)
    dr = (args.r_max - args.r_min) / args.n
    r = args.r_min + (np.arange(args.n) + 0.5) * dr

    mdot_err, be_err, k_err = sol.check_conserved(r)
    print(f"analytic solution self-check: Mdot_err={mdot_err:.2e}  Be_err={be_err:.2e}  K_err={k_err:.2e}")
    if max(mdot_err, be_err, k_err) > 1e-8:
        raise SystemExit("analytic solution failed its own self-consistency check -- refusing to write an IC")

    rho, v, P = sol.profile(r)
    print(f"rho: [{rho.min():.4f}, {rho.max():.4f}]  v: [{v.min():.4f}, {v.max():.4f}]  "
          f"P: [{P.min():.4f}, {P.max():.4f}]")
    print(f"sonic point r_c={args.r_c}  Mdot={sol.Mdot_abs:.6f}  Be={sol.Be:.6f}  K={sol.K:.6f}")

    # The outer edge (r_max) is subsonic (r_max > r_c): one characteristic
    # family there is incoming from outside the domain, so a zero-gradient
    # "outflow" boundary is ill-posed and lets the solution drift (verified
    # the hard way -- an earlier version of this test used self-outflow at
    # both ends and mass visibly piled up at the outer edge over time). Fix
    # it with a Dirichlet ghost state instead: the exact analytic solution
    # evaluated right at r_max, appended as one extra (N+1-th) record so
    # SchwarzschildSim can pin the outer boundary's exterior state to it.
    # The inner edge (r_min < r_c) is supersonic, where pure extrapolation
    # outflow is the physically correct condition -- no ghost needed there.
    rho_ghost, v_ghost, P_ghost = sol.solve_at_radius(args.r_max)
    rho = np.append(rho, rho_ghost)
    v = np.append(v, v_ghost)
    P = np.append(P, P_ghost)

    record = np.column_stack([rho, v, P]).astype(np.float32)
    record.tofile(args.out)
    print(f"wrote {args.n} cells + 1 outer-ghost record -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] n={args.n}  r_min={args.r_min}  r_max={args.r_max}   "
          f"[physics] M={args.M}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
