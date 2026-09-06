"""Shared readback helpers for 01_hartree_fock headless output dirs."""

import json
import os

import numpy as np

L_SYMBOL = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}


def load_manifest(d):
    with open(os.path.join(d, "manifest.json")) as f:
        return json.load(f)


def frame_path(d, name, i):
    return os.path.join(d, "frames", f"{name}_{i:04d}.npy")


def load_field(d, name, i):
    """One frame of a field as a 1-D (or 2-D) array, squeezed on the leading axis."""
    arr = np.load(frame_path(d, name, i))
    return arr


def n_frames(d):
    return load_manifest(d)["frames"]


def orbital_labels(d):
    """[(n, l), ...] in the fixed order the 'eps' / 'orbitals_final' rows use."""
    nl = np.load(frame_path(d, "eps_nl", 0))
    return [(int(n), int(l)) for n, l in zip(nl[0], nl[1])]


def label_text(n, l):
    return f"{n}{L_SYMBOL.get(l, '?')}"


def radial_grid(d):
    return np.load(frame_path(d, "r", 0)).ravel()
