"""
Two independent Fishbone-Moncrief rings (tools/fishbone_moncrief.py, the
same validated hydrostatic construction as decks/kerr_ring_hydrostatic.toml)
sampled onto one grid: an outer ring detuned inward (the same
r_in=13/r_center=15/vphi_scale=0.9 already validated) and an inner ring
left at exact equilibrium (vphi_scale=1.0, should stay put). As the outer
ring migrates inward it should eventually reach and collide with the
inner one -- genuine two-body hydrodynamic interaction, not just one
ring's own dynamics.

Each ring's (r_in, r_center) must be chosen so their h>1 regions don't
overlap at t=0 (checked below, not assumed) -- if they do, this script
refuses to write the IC rather than silently blend two overlapping
equilibria that were never validated together.

    python make_two_ring_ic.py --M 1.0 --a 0.9 --gamma 1.333333 \
        --r_in_outer 13.0 --r_center_outer 15.0 --vphi_scale_outer 0.9 \
        --r_in_inner 8.0 --r_center_inner 9.5 --vphi_scale_inner 1.0 \
        --nr 96 --ntheta 32 --r_min 4.0 --r_max 30.0 --theta_min 0.5 \
        --out ic/two_rings_a09.bin
"""

import argparse

import numpy as np

from fishbone_moncrief import FishboneMoncriefTorus


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=4.0 / 3.0)
    ap.add_argument("--r_in_outer", type=float, required=True)
    ap.add_argument("--r_center_outer", type=float, required=True)
    ap.add_argument("--vphi_scale_outer", type=float, default=1.0)
    ap.add_argument("--retrograde_outer", action="store_true")
    ap.add_argument("--r_in_inner", type=float, required=True)
    ap.add_argument("--r_center_inner", type=float, required=True)
    ap.add_argument("--vphi_scale_inner", type=float, default=1.0)
    ap.add_argument("--retrograde_inner", action="store_true")
    ap.add_argument("--nr", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=32)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--theta_min", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 0.1:
        raise SystemExit(f"r_min={args.r_min} must stay well clear of r=0 (the true curvature singularity)")

    outer = FishboneMoncriefTorus(args.M, args.a, args.r_in_outer, args.r_center_outer, args.gamma,
                                   prograde=not args.retrograde_outer)
    inner = FishboneMoncriefTorus(args.M, args.a, args.r_in_inner, args.r_center_inner, args.gamma,
                                   prograde=not args.retrograde_inner)
    print(f"outer: l={outer.l:.4f} ln_h_center={outer.ln_h_center:.5f}")
    print(f"inner: l={inner.l:.4f} ln_h_center={inner.ln_h_center:.5f}")

    theta_max = np.pi - args.theta_min
    dr = (args.r_max - args.r_min) / args.nr
    dth = (theta_max - args.theta_min) / args.ntheta
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    theta = args.theta_min + (np.arange(args.ntheta) + 0.5) * dth

    record = np.zeros((args.nr, args.ntheta, 5), dtype=np.float32)
    n_outer = n_inner = n_overlap = 0
    for i, ri in enumerate(r):
        for j, thj in enumerate(theta):
            rho_o, vphi_o, p_o = outer.primitives(ri, thj)
            rho_i, vphi_i, p_i = inner.primitives(ri, thj)
            if rho_o > 0.0 and rho_i > 0.0:
                n_overlap += 1
                continue  # tallied below; refuse to write if this happens anywhere
            if rho_o > 0.0:
                n_outer += 1
                record[i, j, 0] = rho_o
                record[i, j, 3] = vphi_o * args.vphi_scale_outer
                record[i, j, 4] = p_o
            elif rho_i > 0.0:
                n_inner += 1
                record[i, j, 0] = rho_i
                record[i, j, 3] = vphi_i * args.vphi_scale_inner
                record[i, j, 4] = p_i

    if n_overlap > 0:
        raise SystemExit(f"refusing to write: {n_overlap} cells have BOTH rings' h>1 at t=0 -- "
                          f"choose (r_in, r_center) pairs with a genuine radial gap between them")

    print(f"grid: {args.nr}x{args.ntheta} = {args.nr*args.ntheta} cells, "
          f"{n_outer} in the outer ring, {n_inner} in the inner ring "
          f"({100.0*(n_outer+n_inner)/(args.nr*args.ntheta):.1f}% total)")
    record.reshape(-1, 5).tofile(args.out)
    print(f"wrote {args.nr*args.ntheta} cells -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] nr={args.nr}  ntheta={args.ntheta}  r_min={args.r_min}  r_max={args.r_max}  "
          f"theta_min={args.theta_min}   [physics] M={args.M}  a={args.a}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
