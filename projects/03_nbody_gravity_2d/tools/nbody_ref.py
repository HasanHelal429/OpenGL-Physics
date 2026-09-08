"""Analytic references the plot scripts overlay: Kepler orbits, the Lagrange
equilateral-triangle rate, and Plummer-model profiles.
"""
import numpy as np


def kepler_ellipse(apoapsis, e):
    """Semi-major axis and periapsis for an orbit with a given apoapsis and
    eccentricity (r_apo = a (1 + e))."""
    a = apoapsis / (1.0 + e)
    return a, a * (1.0 - e)


def lagrange_omega_3d(G=1.0, m=1.0, side=1.0):
    """Rigid-rotation rate of the equilateral-triangle solution: Omega^2 =
    3 G m / L^3 (vertices at circumradius L / sqrt(3))."""
    return np.sqrt(3.0 * G * m / side**3)


def plummer_density(r, M=1.0, a=1.0):
    """rho(r) = (3 M / 4 pi a^3) (1 + r^2/a^2)^{-5/2}."""
    return (3.0 * M) / (4.0 * np.pi * a**3) * (1.0 + (r / a) ** 2) ** -2.5


def plummer_mass(r, M=1.0, a=1.0):
    """M(<r) = M r^3 / (r^2 + a^2)^{3/2}."""
    return M * r**3 / (r**2 + a**2) ** 1.5


def plummer_half_mass_radius(a=1.0):
    return a / np.sqrt(0.5 ** (-2.0 / 3.0) - 1.0)  # ~= 1.305 a


def plummer_sigma(r, G=1.0, M=1.0, a=1.0):
    """1D isotropic velocity dispersion: sigma^2 = G M / (6 sqrt(r^2 + a^2))."""
    return np.sqrt(G * M / (6.0 * np.sqrt(r**2 + a**2)))
