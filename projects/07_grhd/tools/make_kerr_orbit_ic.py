"""
Generate a 07_grhd Kerr-equatorial initial condition: a fluid ring where
every cell sits on its OWN exact circular geodesic orbit (v_r=0,
v_phi=v_phi_hat(r) from tools/kerr_orbits.py's effective-potential
solve) -- the validation target for Tier 2 Phase 2a: does the solver hold
this steady, i.e. does v_r stay ~0 under evolution?

Density and pressure are NOT part of the circular-orbit solution itself
(that's a statement about velocities only, valid for dust or fluid alike);
they're set to constants here, deliberately not a real disk-equilibrium
profile (that needs the Fishbone-Moncrief construction -- deferred, see
README.md).

Usage:
    python make_kerr_orbit_ic.py --M 1.0 --a 0.7 --gamma 1.3333 \
        --n 400 --r_min 6.0 --r_max 20.0 --rho 1.0 --p 0.01 \
        --out ic/kerr_ring_a07.bin
"""

import argparse

import numpy as np

from kerr_equatorial_ref import circular_orbit_v_phi_hat


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--M", type=float, default=1.0)
    ap.add_argument("--a", type=float, required=True)
    ap.add_argument("--gamma", type=float, default=4.0 / 3.0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--r_min", type=float, required=True)
    ap.add_argument("--r_max", type=float, required=True)
    ap.add_argument("--rho", type=float, default=1.0)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.r_min <= 2.0 * args.M:
        raise SystemExit(f"r_min={args.r_min} must be outside the horizon-ish region (2*M={2*args.M})")

    dr = (args.r_max - args.r_min) / args.n
    r = args.r_min + (np.arange(args.n) + 0.5) * dr

    v_phi = np.array([circular_orbit_v_phi_hat(ri, args.M, args.a) for ri in r])
    if np.any(np.abs(v_phi) >= 1.0):
        raise SystemExit("a circular orbit came out superluminal -- r_min is probably inside the ISCO/photon region")

    rho = np.full_like(r, args.rho)
    v_r = np.zeros_like(r)
    P = np.full_like(r, args.p)

    print(f"v_phi range: [{v_phi.min():.4f}, {v_phi.max():.4f}]  "
          f"(innermost orbital speed at r={args.r_min}: {v_phi[0]:.4f})")

    record = np.column_stack([rho, v_r, v_phi, P]).astype(np.float32)
    record.tofile(args.out)
    print(f"wrote {record.shape[0]} cells -> {args.out} ({record.nbytes} bytes)")
    print(f"deck: [grid] n={args.n}  r_min={args.r_min}  r_max={args.r_max}   "
          f"[physics] M={args.M}  a={args.a}  gamma={args.gamma}")


if __name__ == "__main__":
    main()
