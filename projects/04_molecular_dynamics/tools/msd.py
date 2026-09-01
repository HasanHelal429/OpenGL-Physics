#!/usr/bin/env python3
"""Mean-squared displacement + self-diffusion coefficient from a headless
04_molecular_dynamics run.

    python msd.py <results_dir> [--skip-frac 0.25] [--fit-frac 0.5] [--out msd.png]

pos_XXXX.npy stores positions WRAPPED into [0, L) (MDSystem periodic-image
convention) -- MSD needs the *unwrapped* trajectory, so this reconstructs it
by minimum-imaging the frame-to-frame displacement (valid as long as no
particle moves more than L/2 between two consecutive written frames, which
holds for any sane substeps_per_frame/dt choice -- a sanity check for that
assumption is printed).

--skip-frac discards the initial fraction of the run (equilibration/thermostat
relaxation transient -- see decks/liquid.toml's hot-start). Of what remains,
the diffusive (long-time, linear) regime is fit over the LAST --fit-frac of
that window, skipping the early ballistic (~t^2) regime where the Einstein
relation D = slope/6 (3D) does not yet apply.
"""
import argparse
import glob
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("msd: needs numpy + matplotlib")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--skip-frac", type=float, default=0.25)
    ap.add_argument("--fit-frac", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = args.results_dir

    frame_ids = sorted(
        os.path.basename(p).split("_")[-1].split(".")[0]
        for p in glob.glob(os.path.join(d, "frames", "pos_*.npy"))
    )
    data = np.genfromtxt(os.path.join(d, "diagnostics.csv"), delimiter=",", names=True)
    L = data["box_length"]

    positions = [np.load(os.path.join(d, "frames", f"pos_{fid}.npy")) for fid in frame_ids]
    n_frames = len(positions)
    n = positions[0].shape[0]

    unwrapped = np.zeros((n_frames, n, 3))
    unwrapped[0] = positions[0]
    max_jump_over_half_box = 0.0
    for k in range(1, n_frames):
        raw_disp = positions[k] - positions[k - 1]
        Lk = L[k]
        disp = raw_disp - Lk * np.round(raw_disp / Lk)
        # Checking the POST-minimum-image displacement against the half-box
        # threshold, not the raw wrapped one: a particle crossing a periodic
        # boundary between two saved frames gives a raw_disp near a full L
        # (min-image correctly folds that back to a small `disp`) -- that's
        # routine, not a failure. The unwrapping assumption actually breaks
        # only if a particle's true displacement in one frame interval
        # exceeds half the box, so min-image folds it to the WRONG small
        # value instead of the right one; `disp` itself approaching 0.5*L is
        # the real warning sign.
        max_jump_over_half_box = max(max_jump_over_half_box, np.max(np.abs(disp)) / (0.5 * Lk))
        unwrapped[k] = unwrapped[k - 1] + disp

    t = data["t"][:n_frames]
    start = int(args.skip_frac * n_frames)
    disp0 = unwrapped[start:] - unwrapped[start]
    msd = np.mean(np.sum(disp0**2, axis=-1), axis=-1)
    t_rel = t[start:] - t[start]

    fit_start = int((1.0 - args.fit_frac) * len(t_rel))
    slope, intercept = np.polyfit(t_rel[fit_start:], msd[fit_start:], 1)
    D = slope / 6.0  # Einstein relation, 3D: MSD(t) ~ 6*D*t at long time

    print(f"frames: {n_frames}  particles: {n}")
    print(f"  max single-frame displacement / (0.5*L): {max_jump_over_half_box:.3f}"
          f"  (unwrapping assumption breaks down if this approaches 1)")
    print(f"  diffusive-regime fit over t in [{t_rel[fit_start]:.3f}, {t_rel[-1]:.3f}]"
          f" (of {t_rel[0]:.3f}..{t_rel[-1]:.3f} post-skip window)")
    print(f"  self-diffusion coefficient D = {D:.5f}  (reduced LJ units, sigma^2/tau)")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(t_rel, msd, label="MSD(t)")
    ax.plot(t_rel[fit_start:], slope * t_rel[fit_start:] + intercept, "--",
            label=f"linear fit -> D={D:.5f}")
    ax.set_xlabel("t (since post-skip start)"); ax.set_ylabel("MSD")
    ax.set_title(f"{os.path.basename(os.path.abspath(d))}: mean-squared displacement")
    ax.legend()
    fig.tight_layout()
    out = args.out or os.path.join(d, "msd.png")
    fig.savefig(out, dpi=120)
    print("wrote", out)


if __name__ == "__main__":
    main()
