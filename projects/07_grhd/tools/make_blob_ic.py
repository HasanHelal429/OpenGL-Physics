"""
Non-equilibrium initial condition for 07_grhd's KerrTorusSim: a single
Gaussian density blob at rest (zero angular momentum, ZAMO-comoving --
v_r=v_theta=v_phi=0 in the BL-orthonormal convention KerrTorusSim::
UploadInitial already expects and transforms to Kerr-Schild), sitting in
an otherwise-empty (floor) domain, at some (r0, theta0). Unlike the
Fishbone-Moncrief torus this isn't any kind of equilibrium -- there is
nothing holding the blob up except its own pressure, so gravity simply
pulls it inward. With a grid r_min chosen inside the horizon (only
possible because of the Kerr-Schild reformulation -- see
kernels_kerr2d.hpp's header comment), this lets you literally watch
matter cross the horizon in the code's own coordinates, which an
equilibrium torus (built specifically to NOT do that) can't demonstrate
directly.

--l_frac (default 0.0, pure radial infall) gives the blob some specific
angular momentum instead: a FRACTION of the circular-geodesic value at r0
(tools/kerr_orbits.py's circular_orbit -- the same double-root effective-
potential construction used throughout this project, not a recalled
formula). Note this solver is axisymmetric (no phi direction in the grid
at all), so a nonzero l doesn't make a localized clump orbiting through
space -- it makes a full RING of gas at that (r,theta) with that angular
momentum, which is the only thing "a blob with angular momentum" can mean
on an (r,theta) grid. Starting at rest radially (v_r=v_theta=0) makes r0
a turning point; l_frac<1 means less than the centrifugal support a
circular orbit there would have, so the ring falls inward, and if l_frac
is close enough to 1 it can be a bound, eccentric orbit that swings in
toward a periapsis and back out to r0 (rather than just plunging in) --
genuine radial epicyclic oscillation, visible in tools/plot_blob_trajectory.py's
r(t) trace, not a monotonic infall.

Output format is IDENTICAL to make_fm_torus_ic.py's (nr*ntheta*5 float32:
rho, v_r, v_theta, v_phi, P per cell, row-major i*ntheta+j, BL-orthonormal
velocity convention) -- reuses KerrTorusSim::UploadInitial unchanged.

    python make_blob_ic.py --M 1.0 --a 0.9 --gamma 1.333333 \
        --r0 15.0 --theta0 1.5708 --sigma_r 1.5 --sigma_theta 0.3 --rho_peak 1.0 \
        --l_frac 0.9 \
        --nr 96 --ntheta 32 --r_min 1.0 --r_max 30.0 --theta_min 0.5 \
        --out ic/blob_a09.bin
"""

import argparse

import numpy as np

from fishbone_moncrief import kerr_metric_full, u_t_of_l
from kerr_orbits import circular_orbit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=4.0 / 3.0)
    ap.add_argument("--r0", type=float, required=True, help="blob center radius")
    ap.add_argument("--theta0", type=float, default=np.pi / 2.0, help="blob center polar angle")
    ap.add_argument("--sigma_r", type=float, required=True)
    ap.add_argument("--sigma_theta", type=float, required=True)
    ap.add_argument("--rho_peak", type=float, default=1.0)
    ap.add_argument("--p_over_rho", type=float, default=0.05,
                     help="P = p_over_rho * rho (a modest, not-dynamically-important pressure -- "
                          "this is a gravity-driven infall test, not a pressure-supported equilibrium)")
    ap.add_argument("--rho_cutoff", type=float, default=1e-4,
                     help="below rho_peak*rho_cutoff, treat the cell as vacuum (floor), same convention "
                          "as make_fm_torus_ic.py's rho<=0 branch")
    ap.add_argument("--l_frac", type=float, default=0.0,
                     help="specific angular momentum as a fraction of the circular-orbit value at r0 "
                          "(0.0 = pure radial infall, the original behavior; close to 1.0 = a bound, "
                          "eccentric orbit that oscillates in and out instead of plunging straight in)")
    ap.add_argument("--nr", type=int, default=96)
    ap.add_argument("--ntheta", type=int, default=32)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--theta_min", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 0.1:
        raise SystemExit(f"r_min={args.r_min} must stay well clear of r=0 (the true curvature singularity)")

    vphi_hat = 0.0
    if args.l_frac != 0.0:
        E_c, L_c, _ = circular_orbit(args.r0, args.M, args.a)
        l = args.l_frac * (L_c / E_c)
        u_t = u_t_of_l(args.r0, args.theta0, l, args.M, args.a)
        u_phi = -l * u_t
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = kerr_metric_full(args.r0, args.theta0, args.M, args.a)
        A_mat = np.array([[-g_tt, -g_tphi], [g_tphi, g_phiphi]])
        u_t_contra, u_phi_contra = np.linalg.solve(A_mat, [u_t, u_phi])
        sin2 = np.sin(args.theta0) ** 2
        Sigma = args.r0 * args.r0 + args.a * args.a * np.cos(args.theta0) ** 2
        Delta = args.r0 * args.r0 - 2.0 * args.M * args.r0 + args.a * args.a
        A_bl = (args.r0 * args.r0 + args.a * args.a) ** 2 - args.a * args.a * Delta * sin2
        alpha = np.sqrt(Sigma * Delta / A_bl)
        W = alpha * u_t_contra
        vphi_hat = u_phi / (W * np.sqrt(g_phiphi))
        print(f"l_frac={args.l_frac}: l={l:.4f} (circular l={L_c/E_c:.4f} at r0={args.r0}), "
              f"BL-orthonormal v_phi={vphi_hat:.4f}")

    theta_max = np.pi - args.theta_min
    dr = (args.r_max - args.r_min) / args.nr
    dth = (theta_max - args.theta_min) / args.ntheta
    r = args.r_min + (np.arange(args.nr) + 0.5) * dr
    theta = args.theta_min + (np.arange(args.ntheta) + 0.5) * dth
    R, TH = np.meshgrid(r, theta, indexing="ij")

    rho = args.rho_peak * np.exp(-0.5 * ((R - args.r0) / args.sigma_r) ** 2
                                  - 0.5 * ((TH - args.theta0) / args.sigma_theta) ** 2)
    inside = rho > args.rho_peak * args.rho_cutoff

    record = np.zeros((args.nr, args.ntheta, 5), dtype=np.float32)
    record[..., 0] = np.where(inside, rho, 0.0)
    # v_r=v_theta=0 (at rest radially -- r0 is therefore a turning point,
    # apoapsis if l_frac<1) and v_phi=vphi_hat (0.0 unless --l_frac given).
    # UploadInitial's BlToKsVelocity transform still runs on every rho>0
    # cell, so "at rest" here means at rest relative to a local ZAMO (plus
    # whatever azimuthal motion vphi_hat carries), not literally v=0 in
    # Kerr-Schild coordinates (KS's own v^i=0 would instead mean comoving
    # with an INFALLING ZAMO -- see UploadInitial's header comment on the
    # vacuum floor for the same distinction).
    record[..., 3] = np.where(inside, vphi_hat, 0.0)
    record[..., 4] = np.where(inside, args.p_over_rho * rho, 0.0)

    n_inside = int(np.count_nonzero(inside))
    record.reshape(-1, 5).tofile(args.out)
    print(f"grid: {args.nr}x{args.ntheta} = {args.nr*args.ntheta} cells, {n_inside} inside the blob "
          f"({100.0*n_inside/(args.nr*args.ntheta):.1f}%), peak rho={float(rho.max()):.4f}")
    print(f"wrote {args.nr*args.ntheta} cells -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] nr={args.nr}  ntheta={args.ntheta}  r_min={args.r_min}  r_max={args.r_max}  "
          f"theta_min={args.theta_min}   [physics] M={args.M}  a={args.a}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
