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
    ap.add_argument("--vphi_scale", type=float, default=1.0,
                     help="multiply the equilibrium v_phi by this factor (default 1.0 = unperturbed "
                          "equilibrium). <1 removes centrifugal support (should accrete faster); >1 "
                          "adds excess (should push outward) -- rho/P are left at the equilibrium "
                          "profile, only the rotation is detuned, a standard way to build a "
                          "non-equilibrium torus test from an equilibrium one.")
    ap.add_argument("--retrograde", action="store_true",
                     help="orbit opposite to the black hole's spin (fishbone_moncrief.py's "
                          "prograde=False -- flips the sign of the circular-orbit L used to set the "
                          "torus's specific angular momentum l). Retrograde orbits have a much larger "
                          "ISCO, so r_in/r_center that work prograde may not be valid retrograde -- "
                          "check the printed ln(h) is positive.")
    ap.add_argument("--exclude_below", type=float, default=None,
                     help="treat any h>1 region at r below this radius as vacuum, not torus. "
                          "Fishbone-Moncrief tori (this is a real property of the construction, not a "
                          "bug in it) can have a SECOND, unphysical h>1 branch very close to the black "
                          "hole -- confirmed for a=0.9 retrograde, r_in=13/r_center=15, where ln(h) "
                          "rises back above 0 for r<~4.6 despite a clean ln(h)<0 gap from r~5 to ~12.5 "
                          "separating it from the genuine torus at r~13-17.5. Always check the ln(h) "
                          "profile (e.g. via FishboneMoncriefTorus.ln_enthalpy at several r) before "
                          "trusting a new (a, r_in, r_center, prograde) combination doesn't have this.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # This guard used to require r_min > 2M -- a Boyer-Lindquist-era
    # safety margin, back when KerrTorusSim used BL coordinates (which
    # have a genuine coordinate singularity at the horizon r_+=M+sqrt(M^2-a^2)
    # and can't be evolved through it). KerrTorusSim now uses horizon-
    # penetrating Kerr-Schild coordinates instead (see kernels_kerr2d.hpp's
    # header comment) -- the metric (Sigma=r^2+a^2*cos^2(theta), lapse
    # alpha^2=Sigma/(Sigma+2Mr)) is well-defined for any r>0, including
    # inside the horizon, so this only needs to keep r away from the true
    # curvature singularity at r=0 (a torus deck's own r_in is always well
    # outside the horizon anyway; this just bounds how far the domain's
    # OWN inner edge, distinct from the torus's inner edge, can safely go).
    if args.r_min <= 0.1:
        raise SystemExit(f"r_min={args.r_min} must stay well clear of r=0 (the true curvature singularity)")

    torus = FishboneMoncriefTorus(args.M, args.a, args.r_in, args.r_center, args.gamma,
                                   prograde=not args.retrograde)
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
            if args.exclude_below is not None and ri < args.exclude_below:
                rho, v_phi, P = 0.0, 0.0, 0.0
            record[i, j, 0] = rho
            record[i, j, 3] = v_phi * args.vphi_scale
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
