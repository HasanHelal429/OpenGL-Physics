"""
PEC-cylinder scattering validation for 10_fdtd (decks/cylinder_scatter.toml).
Runs the deck with and without the cylinder, isolates the scattered
steady-state phasor field by subtraction, does a 2D near-to-far-field
transform on a rectangular contour around the cylinder, and compares the
bistatic pattern to the 2D PEC Mie series

    P(phi) = | sum_n  J_n(ka) / H_n^(1)(ka)  e^{i n phi} |^2

(the e^{-i w t} phasor convention -> outgoing wave is H^(1)).

Usage:
    python plot_mie.py --exe <binary> [--out FILE]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1, jv

try:
    import tomllib
except ImportError:
    tomllib = None


def load_phasor(d, name, f0):
    man = json.load(open(os.path.join(d, "manifest.json")))
    dt = man["dt"] * man.get("substeps_per_frame", 1)
    fdir = os.path.join(d, "frames")
    ids = sorted(int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(fdir) if f.startswith(name + "_"))
    per = max(4, int(round(1.0 / f0 / dt)))
    use = ids[-8 * per:]
    acc = 0.0
    for fid in use:
        a = np.load(os.path.join(fdir, f"{name}_{fid:04d}.npy"))
        acc = acc + a * np.exp(1j * 2 * np.pi * f0 * fid * dt)
    return 2.0 * acc / len(use)


def ntff(Ez, Hx, Hy, cx, cy, half, k, phis):
    """2D scattered near-to-far transform on a square contour of half-width
    `half` about (cx, cy). Returns the complex far-field amplitude per phi."""
    out = np.zeros(len(phis), complex)
    # four edges: (points, outward normal)
    xs = np.arange(cx - half, cx + half + 1)
    ys = np.arange(cy - half, cy + half + 1)
    edges = [
        (np.full_like(ys, cx + half), ys, (1.0, 0.0)),   # right
        (np.full_like(ys, cx - half), ys, (-1.0, 0.0)),   # left
        (xs, np.full_like(xs, cy + half), (0.0, 1.0)),    # top
        (xs, np.full_like(xs, cy - half), (0.0, -1.0)),   # bottom
    ]
    for ip, (px, py, (nx_, ny_)) in enumerate(edges):
        px = px.astype(int); py = py.astype(int)
        ez = Ez[py, px]
        hx = Hx[py, px]
        hy = Hy[py, px]
        # equivalent surface currents (eta = 1)
        Jz = nx_ * hy - ny_ * hx                       # (n x H)_z
        for m, phi in enumerate(phis):
            fx, fy = np.cos(phi), np.sin(phi)
            Mphi = ez * (nx_ * fx + ny_ * fy)          # (n . phi_hat) Ez
            ph = np.exp(-1j * k * ((px - cx) * fx + (py - cy) * fy))
            out[m] += np.sum((Jz - Mphi) * ph)
    return out


def default_exe():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    name = "10_fdtd.exe" if os.name == "nt" else "10_fdtd"
    return os.path.join(root, "OpenGL Physics", "build", "release", "projects",
                        "10_fdtd", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    exe = args.exe or default_exe()
    if not os.path.exists(exe):
        sys.exit(f"binary not found at {exe}")

    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(here, "..", "decks", "cylinder_scatter.toml")
    deck = open(base).read()
    with open(base, "rb") as f:
        cfg = tomllib.load(f)
    f0 = cfg["source"][0]["f0"]
    cx = int(cfg["pec"][0]["x"]); cy = int(cfg["pec"][0]["y"])
    a = cfg["pec"][0]["radius"]
    k = 2 * np.pi * f0
    ka = k * a

    tmp = tempfile.mkdtemp(prefix="mie_")
    # with cylinder
    d_yes = os.path.join(tmp, "yes")
    subprocess.run([exe, "--deck", base, "--out", d_yes], check=True,
                   capture_output=True)
    # without cylinder: strip the [[pec]] table
    deck_no = deck.split("[[pec]]")[0]
    p_no = os.path.join(tmp, "no.toml")
    open(p_no, "w").write(deck_no)
    d_no = os.path.join(tmp, "no")
    subprocess.run([exe, "--deck", p_no, "--out", d_no], check=True,
                   capture_output=True)

    Ez = load_phasor(d_yes, "Ez", f0) - load_phasor(d_no, "Ez", f0)
    Hx = load_phasor(d_yes, "Hx", f0) - load_phasor(d_no, "Hx", f0)
    Hy = load_phasor(d_yes, "Hy", f0) - load_phasor(d_no, "Hy", f0)

    phis = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # contour a few cells inside the TFSF box, around the cylinder
    half = min(cx, cy, Ez.shape[1] - cx, Ez.shape[0] - cy) - \
        cfg["boundary"]["pml_cells"] - cfg["tfsf"]["margin"] - 6
    F = ntff(Ez, Hx, Hy, cx, cy, half, k, phis)
    P_fdtd = np.abs(F) ** 2
    P_fdtd /= P_fdtd.max()

    # Mie series (2D PEC, TMz)
    N = int(ka + 15)
    ns = np.arange(-N, N + 1)
    an = jv(ns, ka) / hankel1(ns, ka)
    # incidence is along -x (phi = 180); measure phi relative to the incident
    S = np.array([np.sum(an * np.exp(1j * ns * (phi - np.pi))) for phi in phis])
    P_mie = np.abs(S) ** 2
    P_mie /= P_mie.max()

    corr = np.abs(np.vdot(P_fdtd, P_mie)) / (
        np.linalg.norm(P_fdtd) * np.linalg.norm(P_mie))
    rel = np.abs(P_fdtd - P_mie)
    print(f"ka = {ka:.3f}   pattern correlation {corr:.4f}   "
          f"max |dP| {rel.max():.3f}   median {np.median(rel):.3f}")

    fig = plt.figure(figsize=(11, 4.6))
    ax0 = fig.add_subplot(1, 2, 1, projection="polar")
    ax0.plot(phis, P_mie, "k-", lw=1, label="Mie series")
    ax0.plot(phis, P_fdtd, ".", ms=3, label="FDTD NTFF")
    ax0.set_title(f"bistatic pattern, ka = {ka:.2f}")
    ax0.legend(loc="lower right", fontsize=8)

    ax1 = fig.add_subplot(1, 2, 2)
    deg = np.degrees(phis)
    ax1.plot(deg, 10 * np.log10(P_mie + 1e-6), "k-", lw=1, label="Mie")
    ax1.plot(deg, 10 * np.log10(P_fdtd + 1e-6), ".", ms=3, label="FDTD")
    ax1.set_xlabel("scattering angle (deg)"); ax1.set_ylabel("normalised P (dB)")
    ax1.set_xlim(0, 360); ax1.legend(fontsize=8)

    fig.suptitle(os.path.basename(os.path.abspath(d_yes)))
    fig.tight_layout()
    out = args.out or "mie_validation.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
