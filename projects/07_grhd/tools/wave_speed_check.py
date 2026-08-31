"""
Derivation + independent numerical verification of the HLLE wave-speed
bound that should replace kernels_kerr2d.hpp's current raw-photon-speed
bound (sqrt(-g_tt/g_rr) etc.) -- see 07_grhd/README.md's mass-loss
investigation. Following this project's established methodology (see
kerr_equatorial_ref.py's docstring, kerr_metric_check.py): never trust a
recalled GR/SR formula without an independent numerical cross-check.

Step 1: state the closed-form Marti-Muller-Anile multi-dimensional sound
characteristic speed formula, valid in a LOCAL ORTHONORMAL (ZAMO) frame.
This project's own v_r/v_theta/v_phi convention (see kerr_equatorial_ref.py
and kernels_kerr2d.hpp's kinematics2D) already defines these as physical
(locally-flat, proper-distance-based) velocity components measured by the
ZAMO observer -- W=1/sqrt(1-v_r^2-v_th^2-v_phi^2) is exactly the special-
relativistic Lorentz factor built from them. That is the key simplification
this derivation leans on: in a LOCAL ORTHONORMAL FRAME, spacetime looks
flat, so the SPECIAL RELATIVISTIC (not curved-space) multi-D sound-speed
formula applies directly to these v components -- no GR-specific
characteristic analysis is needed for the local part at all.

Step 2 converts that local-frame result to a COORDINATE speed (dr/dt or
dtheta/dt, as needed by the actual r/theta flux passes), via the lapse and
the metric scale factor: an observer at rest in coordinates measures
proper time dtau=alpha*dt, and proper radial distance dl_r=sqrt(gamma_rr)*dr
(theta, phi held fixed) -- so a LOCAL characteristic speed lambda (proper
distance per proper time) becomes a COORDINATE speed via
    (dx^i/dt)_char = alpha * lambda / sqrt(gamma_ii) - beta^i
beta^r=beta^theta=0 for Kerr in Boyer-Lindquist coordinates (only beta^phi
is nonzero), so no shift correction is needed for either direction this
2D solver actually computes fluxes in.

This script verifies the CLOSED-FORM local-frame formula against an
independent, from-scratch computation of the flux Jacobian dF/dU, built
via the chain rule dF/dU = (dF/dprim) . (dU/dprim)^-1 with BOTH factors
computed by exact SYMBOLIC differentiation (SymPy) of the flux/conserved-
variable formulas directly -- not by finite-differencing through a
numerical root-find (which was tried first and produced eigenvalues with
no vx-multiplicity-3 structure at all, i.e. was itself buggy/unreliable,
exactly the kind of mistake this cross-check step exists to catch).
"""

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Step 1: closed-form local-frame (special-relativistic) characteristic
# speeds along direction "x" (the flux direction), with transverse
# velocity components vy, vz also present (general multi-D flow).
# ---------------------------------------------------------------------------
def sound_speed_sq(rho, P, gamma_eos, h):
    """cs^2 = Gamma*P/(rho*h) for an ideal-gas EOS with adiabatic index
    equal to the EOS's Gamma (standard result, e.g. Font 2008 Living
    Reviews eq. 68, Marti & Muller 2003 eq. 92)."""
    return gamma_eos * P / (rho * h)


def marti_muller_lambda_pm(vx, vy, vz, cs2):
    """Closed-form fast/slow acoustic characteristic speeds, LOCAL
    ORTHONORMAL FRAME, direction x, general transverse (vy,vz).
    (Marti & Muller 2003 "Numerical Hydrodynamics in Special Relativity",
    Living Reviews in Relativity, eq. 95; Font 2008 eq. 78 -- independently
    verified below via a from-scratch symbolic Jacobian, not trusted from
    memory alone.)"""
    v2 = vx * vx + vy * vy + vz * vz
    denom = 1.0 - v2 * cs2
    disc = cs2 * (1.0 - v2) * ((1.0 - v2 * cs2) - vx * vx * (1.0 - cs2))
    disc = np.maximum(disc, 0.0)  # guard tiny negative float noise at disc~0
    root = np.sqrt(disc)
    num_center = vx * (1.0 - cs2)
    lam_plus = (num_center + root) / denom
    lam_minus = (num_center - root) / denom
    return lam_minus, lam_plus


# ---------------------------------------------------------------------------
# Step 2: symbolic SR flux Jacobian, independent of the closed form above --
# built directly from conserved-variable/flux definitions (mirrors
# kernels_kerr2d.hpp's sideState(), flat-metric special case:
# gamma_rr=gamma_thth=gamma_phiphi=1, alpha=1, g_tphi=0, sqrt(-g)=1), via
# exact symbolic differentiation rather than any numerical root-find.
# ---------------------------------------------------------------------------
_rho, _vx, _vy, _vz, _P, _gamma = sp.symbols('rho vx vy vz P gamma', positive=True)

_v2 = _vx**2 + _vy**2 + _vz**2
_W = 1 / sp.sqrt(1 - _v2)
_h = 1 + _gamma * _P / ((_gamma - 1) * _rho)

_D = _rho * _W
_Sx = _rho * _h * _W**2 * _vx
_Sy = _rho * _h * _W**2 * _vy
_Sz = _rho * _h * _W**2 * _vz
_tau = _rho * _h * _W**2 - _P - _D
_Uvec = sp.Matrix([_D, _Sx, _Sy, _Sz, _tau])

_FD = _rho * _W * _vx
_FSx = _rho * _h * _W**2 * _vx**2 + _P
_FSy = _rho * _h * _W**2 * _vx * _vy
_FSz = _rho * _h * _W**2 * _vx * _vz
_Ftau = _rho * _h * _W**2 * _vx - _rho * _W * _vx
_Fvec = sp.Matrix([_FD, _FSx, _FSy, _FSz, _Ftau])

_prims = sp.Matrix([_rho, _vx, _vy, _vz, _P])
_dU_dprim = _Uvec.jacobian(_prims)
_dF_dprim = _Fvec.jacobian(_prims)

# Lambdify the two Jacobians (exact, symbolic derivatives) separately and
# invert numerically per test point -- symbolically inverting a 5x5 matrix
# full of sqrt() terms was too slow to be worth it; the derivatives
# themselves are still exact, only the final invert+multiply is numeric.
_dU_dprim_func = sp.lambdify((_rho, _vx, _vy, _vz, _P, _gamma), _dU_dprim, 'numpy')
_dF_dprim_func = sp.lambdify((_rho, _vx, _vy, _vz, _P, _gamma), _dF_dprim, 'numpy')


def symbolic_jacobian_eigs(rho, vx, vy, vz, P, gamma_eos):
    dU = np.array(_dU_dprim_func(rho, vx, vy, vz, P, gamma_eos), dtype=float)
    dF = np.array(_dF_dprim_func(rho, vx, vy, vz, P, gamma_eos), dtype=float)
    J = dF @ np.linalg.inv(dU)
    eigs = np.linalg.eigvals(J)
    return np.sort(eigs.real)


def run_check():
    rng = np.random.default_rng(42)
    gamma_eos = 4.0 / 3.0
    max_err = 0.0
    n_tested = 0
    for _ in range(400):
        rho = rng.uniform(0.3, 3.0)
        P = rng.uniform(0.01, 0.5) * rho  # keep it a mildly-relativistic gas, not ultra-stiff
        vx = rng.uniform(-0.6, 0.6)
        vy = rng.uniform(-0.3, 0.3)
        vz = rng.uniform(-0.3, 0.3)
        v2 = vx * vx + vy * vy + vz * vz
        if v2 > 0.9:
            continue
        h = 1.0 + gamma_eos * P / ((gamma_eos - 1.0) * rho)
        cs2 = sound_speed_sq(rho, P, gamma_eos, h)

        lam_minus_cf, lam_plus_cf = marti_muller_lambda_pm(vx, vy, vz, cs2)
        eigs = symbolic_jacobian_eigs(rho, vx, vy, vz, P, gamma_eos)
        eig_min, eig_max = eigs[0], eigs[-1]
        # sanity: the middle three should all equal vx (entropy + 2 degenerate
        # shear modes) -- a real, independent structural check, not just
        # "does the min/max match."
        mid_err = np.max(np.abs(eigs[1:4] - vx))

        err = max(abs(eig_min - lam_minus_cf), abs(eig_max - lam_plus_cf), mid_err)
        max_err = max(max_err, err)
        n_tested += 1

    print(f"tested {n_tested} random states")
    print(f"max |closed-form vs symbolic-Jacobian eigenvalue mismatch| (acoustic pair + vx-triple check) = {max_err:.3e}")
    ok = max_err < 1e-8
    print("PASS" if ok else "FAIL")
    return ok



# ---------------------------------------------------------------------------
# Step 2 check: the coordinate-speed conversion
#     (dx^i/dt)_char = alpha * lambda_local / sqrt(gamma_ii)
# (beta^r=beta^theta=0 for Kerr in Boyer-Lindquist -- no shift correction
# needed for either direction this solver actually fluxes). Sanity-checked
# against the FULL 2D Kerr metric already trusted elsewhere in this project
# (fishbone_moncrief.py's kerr_metric_full, identical formula to
# kernels_kerr2d.hpp's metricAt()): for any physical state (|v|<1, cs<1),
# the resulting coordinate characteristic speed must never exceed the
# coordinate photon speed sqrt(-g_tt/g_ii) -- if it ever did, the new bound
# would be UNSAFE (could let a real signal escape the HLLE fan), not just
# tighter than before.
# ---------------------------------------------------------------------------
def check_coordinate_speed_bound():
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from fishbone_moncrief import kerr_metric_full

    rng = np.random.default_rng(7)
    gamma_eos = 4.0 / 3.0
    worst_margin = np.inf  # photon_speed - |coord_lambda|, want this > 0 always
    n_checked = 0
    for _ in range(2000):
        r = rng.uniform(3.0, 25.0)
        theta = rng.uniform(0.3, np.pi - 0.3)
        a = rng.uniform(0.0, 0.998)
        M = 1.0
        g_tt, g_tphi, g_rr, g_thth, g_phiphi = kerr_metric_full(r, theta, M, a)
        Delta = r * r - 2.0 * M * r + a * a
        A = (r * r + a * a) ** 2 - a * a * Delta * np.sin(theta) ** 2
        alpha2 = g_thth * Delta / A
        if alpha2 <= 0.0:
            continue  # inside the horizon -- not a region this solver's grid covers
        alpha = np.sqrt(alpha2)

        vr = rng.uniform(-0.7, 0.7)
        vth = rng.uniform(-0.5, 0.5)
        vphi = rng.uniform(-0.5, 0.5)
        v2 = vr * vr + vth * vth + vphi * vphi
        if v2 > 0.92:
            continue
        rho = rng.uniform(0.3, 3.0)
        P = rng.uniform(0.01, 0.5) * rho
        h = 1.0 + gamma_eos * P / ((gamma_eos - 1.0) * rho)
        cs2 = sound_speed_sq(rho, P, gamma_eos, h)

        # r-direction: vr is "x", vth/vphi transverse.
        lam_r_minus, lam_r_plus = marti_muller_lambda_pm(vr, vth, vphi, cs2)
        coord_r = alpha * np.maximum(np.abs(lam_r_minus), np.abs(lam_r_plus)) / np.sqrt(g_rr)
        photon_r = np.sqrt(-g_tt / g_rr)
        worst_margin = min(worst_margin, photon_r - coord_r)

        # theta-direction: vth is "x", vr/vphi transverse.
        lam_th_minus, lam_th_plus = marti_muller_lambda_pm(vth, vr, vphi, cs2)
        coord_th = alpha * np.maximum(np.abs(lam_th_minus), np.abs(lam_th_plus)) / np.sqrt(g_thth)
        photon_th = np.sqrt(-g_tt / g_thth)
        worst_margin = min(worst_margin, photon_th - coord_th)

        n_checked += 1

    print(f"checked {n_checked} (r,theta,a,state) combinations")
    print(f"worst-case (photon_speed - |coordinate characteristic speed|) = {worst_margin:.4f} "
          f"(must be > 0 -- new bound must never exceed the photon speed)")
    ok = worst_margin > 0.0
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok1 = run_check()
    ok2 = check_coordinate_speed_bound()
    raise SystemExit(0 if (ok1 and ok2) else 1)
