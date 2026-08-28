"""
Generate a 07_grhd Kerr-torus initial condition: the validated
Fishbone-Moncrief equilibrium torus (tools/fishbone_moncrief.py), sampled
onto the 2D (r,theta) grid KerrTorusSim expects. Writes a raw binary file
of nr*ntheta*5 float32 (rho, v_r, v_theta, v_phi, P) per cell, row-major
i*ntheta+j (i=radial, j=polar) -- matching KerrTorusSim::UploadInitial's
expected layout. v_r=v_theta=0 everywhere by construction (the torus is a
purely toroidal equilibrium); cells outside the torus (h<=1) get rho=P=0,
which KerrTorusSim replaces with a small floor density at rest (see its
UploadInitial) so ConsToPrim never divides by a literal zero.

Usage:
    python make_fm_torus_ic.py --M 1.0 --a 0.9 --gamma 1.333333 \
        --r_in 6.0 --r_center 10.0 \
        --nr 96 --ntheta 64 --r_min 4.0 --r_max 30.0 --theta_min 0.5 \
        --out ic/fm_torus_a09.bin
"""

import argparse

import numpy as np

from fishbone_moncrief import FishboneMoncriefTorus


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=4.0 / 3.0)
    ap.add_argument("--r_in", type=float, required=True)
    ap.add_argument("--r_center", type=float, required=True)
    ap.add_argument("--nr", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=64)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--theta_min", type=float, required=True,
                     help="grid.theta_min; theta_max is fixed at pi-theta_min (KerrTorusSim's convention)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 2.0 * args.M:
        raise SystemExit(f"r_min={args.r_min} must clear the horizon region (2*M={2*args.M})")

    torus = FishboneMoncriefTorus(args.M, args.a, args.r_in, args.r_center, args.gamma)
    print(f"l={torus.l:.6f}  ln(h)@r_center={torus.ln_h_center:.6f}  K={torus.K:.6f}")

    theta_max = np.pi - args.theta_min
    dr = (args.r_max - args.r_min) / args.nr
    dth = (theta_max - args.theta_min) / args.ntheta
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    theta = args.theta_min + (np.arange(args.ntheta) + 0.5) * dth

    record = np.zeros((args.nr, args.ntheta, 5), dtype=np.float32)
    n_inside = 0
    rho_max = 0.0
    for i, ri in enumerate(r):
        for j, thj in enumerate(theta):
            rho, v_phi, P = torus.primitives(ri, thj)
            record[i, j, 0] = rho
            record[i, j, 3] = v_phi
            record[i, j, 4] = P
            if rho > 0.0:
                n_inside += 1
                rho_max = max(rho_max, rho)

    print(f"grid: {args.nr}x{args.ntheta} = {args.nr*args.ntheta} cells, {n_inside} inside the torus "
          f"({100.0*n_inside/(args.nr*args.ntheta):.1f}%), peak rho={rho_max:.4f}")

    record.reshape(-1, 5).tofile(args.out)
    print(f"wrote {args.nr*args.ntheta} cells -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] nr={args.nr}  ntheta={args.ntheta}  r_min={args.r_min}  r_max={args.r_max}  "
          f"theta_min={args.theta_min}   [physics] M={args.M}  a={args.a}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
