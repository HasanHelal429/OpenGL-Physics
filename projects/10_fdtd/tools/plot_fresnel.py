"""
s-polarisation Fresnel validation for 10_fdtd. For each incidence angle it
runs the TFSF plane-wave deck twice -- with and without the dielectric
half-space -- and subtracts the steady-state Ez phasors on a probe row in
the vacuum region. The difference is the reflected wave alone, so

    R(theta) = |E_reflected|^2 / |E_incident|^2

with no scattered-field-strip or back-reflection ambiguity. Compared to the
Fresnel coefficient  r_s = (cos t_i - n cos t_t) / (cos t_i + n cos t_t).

Usage:
    python plot_fresnel.py [--exe PATH] [--angles 0 15 30 45 60 70] [--out FILE]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

NX = NY = 340
PML = 14
MARGIN = 10
IFACE = 150            # dielectric fills y < IFACE
F0 = 0.05
N2 = 2.0
PROBE_ROW = IFACE + 8  # just above the interface: the reflected beam still
                       # fully overlaps the incident footprint (no walk-off)


def deck(angle_deg, with_slab):
    mat = (f'\n[[material]]\nshape="halfspace"\naxis="y"\npos={IFACE}\n'
           f'side="lo"\neps_r={N2 * N2}\n') if with_slab else ""
    return f"""
[grid]
nx = {NX}
ny = {NY}
dx = 1.0
courant = 0.5
[boundary]
type = "cpml"
pml_cells = {PML}
[tfsf]
margin = {MARGIN}
[time]
steps = 1600
substeps_per_frame = 5
[[source]]
kind = "tfsf"
waveform = "sine"
f0 = {F0}
amplitude = 1.0
angle_deg = {270.0 - angle_deg}
ramp_cycles = 4
{mat}"""


def phasor_row(results_dir):
    man = json.load(open(os.path.join(results_dir, "manifest.json")))
    dt = man["dt"] * man.get("substeps_per_frame", 1)
    fdir = os.path.join(results_dir, "frames")
    ids = sorted(int(f.split("_")[-1].split(".")[0])
                 for f in os.listdir(fdir) if f.startswith("Ez_"))
    period_frames = max(4, int(round(1.0 / F0 / dt)))
    acc = None
    for fid in ids[-6 * period_frames:]:
        a = np.load(os.path.join(fdir, f"Ez_{fid:04d}.npy"))[PROBE_ROW]
        t = fid * dt
        acc = (0 if acc is None else acc) + a * np.exp(1j * 2 * np.pi * F0 * t)
    return 2.0 * acc / len(ids[-6 * period_frames:])


def default_exe():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    name = "10_fdtd.exe" if os.name == "nt" else "10_fdtd"
    return os.path.join(root, "OpenGL Physics", "build", "release", "projects",
                        "10_fdtd", name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", default=None)
    ap.add_argument("--angles", type=float, nargs="+",
                    default=[0, 15, 30, 45, 55, 65, 75])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    exe = args.exe or default_exe()
    if not os.path.exists(exe):
        sys.exit(f"binary not found at {exe}")

    tmp = tempfile.mkdtemp(prefix="fresnel_")
    # measure the incident amplitude once (no slab, normal incidence)
    inc_dir = os.path.join(tmp, "inc")
    dp = os.path.join(tmp, "inc.toml")
    open(dp, "w").write(deck(0.0, with_slab=False))
    subprocess.run([exe, "--deck", dp, "--out", inc_dir], check=True,
                   capture_output=True)
    Ei = np.abs(phasor_row(inc_dir)[NX // 3:2 * NX // 3]).mean()

    R_fdtd, R_fresnel = [], []
    for ang in args.angles:
        d0 = os.path.join(tmp, f"a{ang:g}_no")
        d1 = os.path.join(tmp, f"a{ang:g}_yes")
        for path, slab, out in [(f"{tmp}/a{ang:g}_no.toml", False, d0),
                                (f"{tmp}/a{ang:g}_yes.toml", True, d1)]:
            open(path, "w").write(deck(ang, slab))
            subprocess.run([exe, "--deck", path, "--out", out], check=True,
                           capture_output=True)
        total = phasor_row(d1)
        incid = phasor_row(d0)
        refl = np.abs(total - incid)
        # the reflected plane wave has ~constant amplitude where it exists; at
        # oblique angles it only covers part of the probe row (it walks off to
        # one side), so average over the cells where it is actually present.
        present = refl > 0.6 * refl.max()
        Er = refl[present].mean()
        R_fdtd.append((Er / Ei) ** 2)

        ti = np.radians(ang)
        ctt = np.sqrt(1 - (np.sin(ti) / N2) ** 2)
        rs = (np.cos(ti) - N2 * ctt) / (np.cos(ti) + N2 * ctt)
        R_fresnel.append(rs ** 2)
        print(f"  theta={ang:4.0f}  R_fdtd={R_fdtd[-1]:.4f}  "
              f"R_fresnel={R_fresnel[-1]:.4f}  err={abs(R_fdtd[-1]-R_fresnel[-1]):.4f}")

    R_fdtd = np.array(R_fdtd)
    R_fresnel = np.array(R_fresnel)
    print(f"\n  max |R_fdtd - R_fresnel| = {np.max(np.abs(R_fdtd - R_fresnel)):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    tt = np.linspace(0, 85, 200)
    ti = np.radians(tt)
    ctt = np.sqrt(1 - (np.sin(ti) / N2) ** 2)
    rs = (np.cos(ti) - N2 * ctt) / (np.cos(ti) + N2 * ctt)
    ax.plot(tt, rs ** 2, "k-", lw=1, label=f"Fresnel $R_s$ (n={N2})")
    ax.plot(args.angles, R_fdtd, "o", ms=6, label="FDTD (TFSF, phasor subtraction)")
    ax.set_xlabel(r"incidence angle $\theta_i$ (deg)")
    ax.set_ylabel("reflectance R")
    ax.set_title("s-polarisation Fresnel reflectance")
    ax.legend()
    fig.tight_layout()
    out = args.out or "fresnel_validation.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
