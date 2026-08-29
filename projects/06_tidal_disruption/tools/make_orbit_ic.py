"""
Boost a relaxed star's center of mass onto an orbit around a fixed black
hole at the origin, so 06_tidal_disruption's blackhole gravity term
(src/kernels.hpp) can actually disrupt/strip it. The star itself is
untouched -- every particle's position/velocity gets the same rigid offset
-- so this only needs the star's shape (from
tools/extract_ic_from_frame.py's settled state), not its internal
structure.

Two orbit families, selected by --eccentricity (default 1.0, the original
parabolic case; pass <1 for a bound, repeating orbit):

Parabolic (e=1, specific energy=0) -- a one-time disruption encounter, the
original use of this script. Pericenter r_p is set via the tidal radius
r_t = R_star*(M_bh/M_star)^(1/3) and a penetration factor beta=r_t/r_p
(beta>1 -> pericenter inside the tidal radius, real disruption expected).
The star starts at separation r_0=start_factor*r_t, still approaching:

    r(phi) = 2*r_p/(1+cos(phi))                  (parabola, phi=0 at pericenter)
    v_r    = -sqrt(2*G*M_bh/r^2 * (r-r_p))        (radial speed, still infalling: phi<0)
    v_t    = sqrt(2*G*M_bh*r_p)/r                 (tangential speed, L=sqrt(2*G*M_bh*r_p))

Bound (0<=e<1) -- a repeating binary orbit meant to strip the star a little
each periastron passage ("feeding") rather than disrupt it once. r_p is set
the same way (via r_t and beta); the semi-major axis then follows from
r_p=a*(1-e). The star is placed at APOCENTER (r_a=a*(1+e)) with purely
tangential velocity (v_r=0 there by definition, a turning point), found
from vis-viva v^2=G*M_bh*(2/r-1/a) -- simplest correct bound-orbit IC,
avoiding any true-anomaly bookkeeping. --start_factor is ignored for e<1.

Usage:
    python make_orbit_ic.py --star ic/star_n1.5_relaxed.bin --M_bh 1000 \
        --beta 3 --out ic/star_n1.5_encounter_beta3.bin
    python make_orbit_ic.py --star ic/star_n1.5_relaxed.bin --M_bh 10 \
        --beta 1.2 --eccentricity 0.6 --out ic/star_n1.5_binary_beta1p2_e0p6.bin
"""

import argparse

import numpy as np


def parabolic_orbit_ic(M_bh, M_star, R_star, G, beta, start_factor):
    r_t = R_star * (M_bh / M_star) ** (1.0 / 3.0)
    r_p = r_t / beta
    r_0 = start_factor * r_t

    cos_phi0 = 2.0 * r_p / r_0 - 1.0
    phi0 = -np.arccos(np.clip(cos_phi0, -1.0, 1.0))  # negative: before pericenter

    L = np.sqrt(2.0 * G * M_bh * r_p)
    v_r = -np.sqrt(max(2.0 * G * M_bh / r_0**2 * (r_0 - r_p), 0.0))
    v_t = L / r_0

    pos = r_0 * np.array([np.cos(phi0), np.sin(phi0), 0.0])
    vel = (v_r * np.array([np.cos(phi0), np.sin(phi0), 0.0])
           + v_t * np.array([-np.sin(phi0), np.cos(phi0), 0.0]))
    return pos, vel, r_t, r_p, r_0


def bound_orbit_ic(M_bh, M_star, R_star, G, beta, eccentricity):
    r_t = R_star * (M_bh / M_star) ** (1.0 / 3.0)
    r_p = r_t / beta
    a = r_p / (1.0 - eccentricity)
    r_0 = a * (1.0 + eccentricity)  # apocenter -- the start point here

    v_apo = np.sqrt(G * M_bh * (2.0 / r_0 - 1.0 / a))

    pos = np.array([r_0, 0.0, 0.0])
    vel = np.array([0.0, v_apo, 0.0])  # purely tangential: apocenter is a turning point
    period = 2.0 * np.pi * np.sqrt(a**3 / (G * M_bh))
    return pos, vel, r_t, r_p, r_0, a, period


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--star", required=True, help="relaxed star .bin (tools/extract_ic_from_frame.py)")
    ap.add_argument("--M_bh", type=float, required=True, help="black hole mass, star-mass units")
    ap.add_argument("--M_star", type=float, default=1.0)
    ap.add_argument("--R_star", type=float, default=1.0)
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=3.0, help="penetration factor r_t/r_p")
    ap.add_argument("--eccentricity", type=float, default=1.0,
                    help="1.0 = parabolic one-time encounter (default, original behavior); "
                         "0<=e<1 = bound repeating orbit starting at apocenter")
    ap.add_argument("--start_factor", type=float, default=3.0,
                    help="parabolic only: start distance, in units of r_t")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    record = np.fromfile(args.star, dtype=np.float32).reshape(-1, 7)

    if args.eccentricity >= 1.0:
        pos, vel, r_t, r_p, r_0 = parabolic_orbit_ic(
            args.M_bh, args.M_star, args.R_star, args.G, args.beta, args.start_factor)
        print(f"M_bh={args.M_bh}  tidal radius r_t={r_t:.4f}  pericenter r_p={r_p:.4f}  "
              f"start r_0={r_0:.4f} ({args.start_factor}*r_t)  [parabolic]")
    else:
        pos, vel, r_t, r_p, r_0, a, period = bound_orbit_ic(
            args.M_bh, args.M_star, args.R_star, args.G, args.beta, args.eccentricity)
        print(f"M_bh={args.M_bh}  tidal radius r_t={r_t:.4f}  pericenter r_p={r_p:.4f}  "
              f"eccentricity={args.eccentricity}  semi-major axis a={a:.4f}  "
              f"apocenter r_0={r_0:.4f}  period={period:.4f}  [bound orbit, starts at apocenter]")

    print(f"start position={pos}  start velocity={vel}  |v|={np.linalg.norm(vel):.4f}")

    record[:, 0:3] += pos.astype(np.float32)
    record[:, 4:7] += vel.astype(np.float32)
    record.tofile(args.out)
    print(f"wrote {record.shape[0]} particles -> {args.out}")


if __name__ == "__main__":
    main()
