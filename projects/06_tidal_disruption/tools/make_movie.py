"""
Turn a headless 06_tidal_disruption run into a movie: a 2D (x,y) scatter of
the SPH particles, colored by density on a fixed log scale (gamma-lifted --
see colorize() below) so brightness is comparable frame to frame, with a
time label. The black hole (if `[blackhole] enabled=true` in the deck) is
drawn as a schematic event-horizon disk (see --bh-radius).

    python make_movie.py <results_dir> [--fps 30] [--stride 1] [--gamma 0.4] \
        [--follow none|star] [--half-extent X] [--bh-radius R] [--out movie.mp4]

--follow star recenters every frame on the star's own COM instead of the
fixed BH-at-origin frame -- needed for a bound, repeating orbit (see
decks/binary_beta0p5_e0p6.toml) whose periastron/apocenter distances differ
by a large factor: no single fixed crop keeps the star both large on screen
and in view for the whole orbit.

Reads manifest.json + frames/pos_mass_*.npy + frames/rho_press_*.npy. Uses
imageio (bundled ffmpeg via imageio-ffmpeg, not the system PATH) so this
works even where a system ffmpeg isn't installed; falls back to a PNG
sequence if that import fails.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
except ImportError:
    sys.exit("make_movie: needs numpy + matplotlib")

try:
    import tomllib
except ImportError:
    tomllib = None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--half-extent", type=float, default=None,
                    help="override the auto-fit axis half-width -- e.g. to zoom in on a "
                         "black hole encounter's disruption/fallback and crop out a far "
                         "escaping majority that would otherwise force a much wider frame")
    ap.add_argument("--follow", choices=["none", "star"], default="none",
                    help="'star' recenters every frame on the star's own (mass-weighted) "
                         "COM instead of the fixed BH-at-origin frame -- needed for a bound, "
                         "repeating orbit whose periastron/apocenter distances differ by a "
                         "large factor, where no single fixed crop keeps the star both "
                         "large on screen and in view for the whole orbit. The BH marker "
                         "moves in-frame accordingly (sweeping past near periastron).")
    ap.add_argument("--gamma", type=float, default=0.4,
                    help="density brightness curve on the log scale; <1 lifts dim/thinned-out "
                         "debris (e.g. fallback streams, orders of magnitude below the intact "
                         "star's peak density) out of near-invisibility")
    ap.add_argument("--bh-radius", type=float, default=None,
                    help="event-horizon disk radius drawn at the BH marker, in the same "
                         "length units as the sim. Default: the deck's own "
                         "blackhole.schwarzschild_radius if it set one (non-zero), else 3%% "
                         "of the frame's half-extent -- this sim's BH is a Newtonian point "
                         "mass or Paczynski-Wiita potential, not an actual metric, so there "
                         "is no physically-derived horizon size to fall back on in general; "
                         "this is a schematic display radius, not a GR calculation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = args.results_dir
    manifest = json.load(open(os.path.join(d, "manifest.json")))
    pos_frames = sorted(glob.glob(os.path.join(d, "frames", "pos_mass_*.npy")))[:: args.stride]
    rho_frames = sorted(glob.glob(os.path.join(d, "frames", "rho_press_*.npy")))[:: args.stride]
    if not pos_frames:
        sys.exit("no pos_mass_*.npy frames found")

    deck_path = os.path.join(d, "deck.toml")
    bh_enabled = False
    bh_rs_deck = 0.0
    if tomllib is not None and os.path.exists(deck_path):
        with open(deck_path, "rb") as f:
            deck = tomllib.load(f)
        bh_enabled = bool(deck.get("blackhole", {}).get("enabled", False))
        bh_rs_deck = float(deck.get("blackhole", {}).get("schwarzschild_radius", 0.0))

    follow_star = args.follow == "star"

    all_pos = np.stack([np.load(f) for f in pos_frames])
    all_rho = np.stack([np.load(f)[:, 0] for f in rho_frames])

    if follow_star:
        # Per-frame mass-weighted COM -- the frame recenters on the star's
        # own position each frame, not the fixed BH-at-origin one below.
        com = np.array([np.average(p[:, :3], axis=0, weights=p[:, 3]) for p in all_pos])
        rel = all_pos[..., :2] - com[:, None, :2]
        # A LOW percentile drives the auto-fit here, deliberately -- a bound
        # repeating orbit sheds real tidal-tail material that legitimately
        # extends tens of units from the core after a periastron passage
        # (the 99th/99.5th percentile distance-from-COM was 10.7/95.9 in a
        # test case whose actual star+near-tail scale was ~1, and whose
        # rare slingshot-ejected outliers reached |x|~5000) -- percentiles
        # anywhere near "all the mass" reintroduce the same "too zoomed
        # out" problem this mode exists to fix. The 97th percentile stayed
        # core-dominated (~1.5x the star's own quiescent radius) in that
        # same test, so it (with a margin factor) is a much more stable
        # choice: most of the extended tail is expected and accepted to
        # drift off-frame here, which is the point -- this view is for
        # watching the star/BH interaction up close, not the full tail
        # (use --half-extent explicitly for a wider, tail-inclusive crop).
        half_extent = args.half_extent if args.half_extent is not None \
            else float(np.percentile(np.linalg.norm(rel, axis=-1), 97.0)) * 3.5
    else:
        com = None
        # Fixed axis limits and color scale across the whole run, so the
        # star's apparent size/brightness changes reflect real dynamics,
        # not rescaling. Tight bounding box (not a radial percentile) so
        # the frame is used efficiently; still necessarily wide when the
        # star starts far from a black hole it later swings close to,
        # which is real orbital geometry, not something a fixed 2D view
        # can hide (see --follow star for a bound-orbit alternative).
        half_extent = args.half_extent if args.half_extent is not None \
            else float(np.max(np.abs(all_pos[..., :2]))) * 1.08

    rho_min = max(float(all_rho[all_rho > 0].min()), float(all_rho.max()) * 1e-6)
    rho_max = float(all_rho.max())

    # Gamma-lifted log color mapping (not a plain LogNorm): a tidal debris
    # stream can be orders of magnitude more dilute than the intact star
    # (measured: returning fallback material ~1e-4 vs. the star's ~2 peak),
    # which a plain log scale still renders as near-black -- exactly the
    # material a longer/fallback-focused run exists to show. gamma<1 lifts
    # the dim end, same idea as 05_tdse_gpu's density view.
    log_min, log_max = np.log10(rho_min), np.log10(rho_max)
    cmap = plt.get_cmap("magma")

    def colorize(rho):
        rho_c = np.clip(rho, rho_min, rho_max)
        t = (np.log10(rho_c) - log_min) / (log_max - log_min)
        return cmap(np.clip(t, 0.0, 1.0) ** args.gamma)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_xlim(-half_extent, half_extent)
    ax.set_ylim(-half_extent, half_extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)

    bh_circle = None
    if bh_enabled:
        bh_radius = args.bh_radius if args.bh_radius is not None \
            else (bh_rs_deck if bh_rs_deck > 0.0 else half_extent * 0.03)
        bh_pos0 = -com[0] if follow_star else np.zeros(2)
        # Event horizon: a black disk (it emits nothing) with a thin bright
        # rim so it reads against the black background instead of
        # disappearing into it. This sim's BH is a Newtonian point mass or
        # Paczynski-Wiita potential, not an actual metric -- bh_radius is a
        # schematic display size (see --bh-radius), not a GR horizon
        # calculation.
        bh_circle = Circle((bh_pos0[0], bh_pos0[1]), bh_radius, facecolor="black",
                           edgecolor="#ffcc66", linewidth=1.2, zorder=5)
        ax.add_patch(bh_circle)

    pm0 = np.load(pos_frames[0])
    rp0 = np.load(rho_frames[0])
    offs0 = (pm0[:, :2] - com[0, :2]) if follow_star else pm0[:, :2]
    scat = ax.scatter(offs0[:, 0], offs0[:, 1], color=colorize(rp0[:, 0]), s=3.5, linewidths=0)
    txt = ax.text(0.03, 0.96, "", transform=ax.transAxes, color="w", va="top",
                  fontsize=9, family="monospace")

    n_frames = len(pos_frames)
    dt = manifest.get("dt", 0.0)
    substeps = manifest.get("substeps_per_frame", 1)

    def update(i):
        pm = np.load(pos_frames[i])
        rp = np.load(rho_frames[i])
        if follow_star:
            scat.set_offsets(pm[:, :2] - com[i, :2])
            if bh_circle is not None:
                bh_circle.center = (-com[i, 0], -com[i, 1])
        else:
            scat.set_offsets(pm[:, :2])
        scat.set_facecolor(colorize(rp[:, 0]))
        t = i * args.stride * substeps * dt
        txt.set_text(f"{manifest.get('title','')}\nt = {t:.2f}   frame {i*args.stride}/{manifest.get('frames','?')}")
        return scat, txt

    from matplotlib import animation
    anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)

    out = args.out or os.path.join(d, "movie.mp4")
    try:
        import imageio
        writer_kwargs = dict(fps=args.fps, codec="libx264", quality=8,
                             pixelformat="yuv420p", macro_block_size=None)
        with imageio.get_writer(out, **writer_kwargs) as writer:
            for i in range(n_frames):
                update(i)
                fig.canvas.draw()
                frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
                writer.append_data(frame)
                if i % 20 == 0:
                    print(f"\rframe {i+1}/{n_frames}", end="", flush=True)
        print(f"\nwrote {out}")
    except Exception as e:  # noqa: BLE001
        print(f"\nimageio/ffmpeg unavailable ({e}); writing PNG sequence instead")
        pdir = os.path.join(d, "frames_png")
        os.makedirs(pdir, exist_ok=True)
        for i in range(n_frames):
            update(i)
            fig.savefig(os.path.join(pdir, f"{i:04d}.png"), facecolor="black")
        print("wrote", pdir)


if __name__ == "__main__":
    main()
