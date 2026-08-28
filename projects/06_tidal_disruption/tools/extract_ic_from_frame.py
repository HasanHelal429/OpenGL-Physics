"""
Extract one frame of a headless run's pos_mass/vel fields into the same raw
binary IC format tools/make_star_ic.py writes (N x 7 float32: x,y,z,mass,
vx,vy,vz), so a damped relaxation run's settled final state can be re-run
undamped as the real test of whether relaxation worked -- damping alone only
proves the *ringing* was suppressed, not that the underlying configuration
is a genuine equilibrium of the undamped equations of motion.

Usage:
    python extract_ic_from_frame.py <results_dir> --frame 0199 --out ic/star_n1.5_relaxed.bin
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--frame", required=True, help="4-digit frame id, e.g. 0199")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames_dir = os.path.join(args.results_dir, "frames")
    pm = np.load(os.path.join(frames_dir, f"pos_mass_{args.frame}.npy"))
    vel = np.load(os.path.join(frames_dir, f"vel_{args.frame}.npy"))

    record = np.column_stack([pm[:, :3], pm[:, 3], vel[:, :3]]).astype(np.float32)
    record.tofile(args.out)
    print(f"wrote {record.shape[0]} particles -> {args.out} ({record.nbytes} bytes)")


if __name__ == "__main__":
    main()
