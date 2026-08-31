"""
Phase 1 of the Kerr-Schild (KS) reformulation for 07_grhd's dynamical 2D
Kerr torus solver -- see README.md's mass-loss investigation for why
Boyer-Lindquist (BL) coordinates were abandoned (coordinate singularity at
the horizon makes the inner radial boundary an artificial mid-flow cutoff
that a simple outflow BC can't properly drain; horizon-penetrating
coordinates let the flow continue cleanly toward/through the horizon
instead).

Following this project's standing methodology (never trust a recalled GR
formula without an independent, from-scratch check -- see
kerr_equatorial_ref.py, kerr_metric_check.py): the KS metric is derived
HERE by symbolically substituting the BL<->KS coordinate transformation
into the ALREADY-TRUSTED BL metric (textbook definition, not a derived
quantity -- see fishbone_moncrief.py's kerr_metric_full), rather than
typed in from memory of "the standard KS metric formula."

BL -> KS coordinate transformation (r, theta UNCHANGED -- this is the key
practical fact: the existing (r,theta) grid, IC sampling radii, and
diagnostics all carry over unchanged; only t and phi are redefined):
    dt_KS   = dt_BL   + (2Mr/Delta) dr
    dphi_KS = dphi_BL + (a/Delta)   dr
i.e. dt_BL = dt_KS - (2Mr/Delta) dr, dphi_BL = dphi_KS - (a/Delta) dr.
"""

import sympy as sp

M, a, r, theta = sp.symbols('M a r theta', positive=True)

# --- Trusted BL metric (textbook definition, matches fishbone_moncrief.py
# kerr_metric_full() and kernels_kerr2d.hpp's metricAt() exactly). ---
sin2 = sp.sin(theta) ** 2
cos2 = sp.cos(theta) ** 2
Sigma = r**2 + a**2 * cos2
Delta = r**2 - 2 * M * r + a**2
A_BL = (r**2 + a**2)**2 - a**2 * Delta * sin2

g_tt_BL = -(1 - 2*M*r/Sigma)
g_tphi_BL = -2*M*a*r*sin2/Sigma
g_rr_BL = Sigma/Delta
g_thth_BL = Sigma
g_phiphi_BL = A_BL*sin2/Sigma

# --- Differentials, dt_BL and dphi_BL expressed in terms of the KS
# differentials (dt_KS, dr, dtheta, dphi_KS) -- symbolic 1-forms, using
# sympy symbols to stand in for each KS differential. ---
dt_KS, dr, dtheta, dphi_KS = sp.symbols('dt_KS dr dtheta dphi_KS')

dt_BL = dt_KS - (2*M*r/Delta) * dr
dphi_BL = dphi_KS - (a/Delta) * dr
dr_BL = dr
dtheta_BL = dtheta

# ds^2 in BL differentials, then substitute.
ds2_BL = (g_tt_BL * dt_BL**2
          + 2 * g_tphi_BL * dt_BL * dphi_BL
          + g_rr_BL * dr_BL**2
          + g_thth_BL * dtheta_BL**2
          + g_phiphi_BL * dphi_BL**2)

ds2_KS = sp.expand(ds2_BL)

# Read off each metric component as half the coefficient of the relevant
# differential product (ds^2 = sum_munu g_munu dx^mu dx^nu, so a CROSS
# term dx^mu dx^nu (mu!=nu) carries coefficint g_munu+g_numu=2*g_munu).
def coeff_of(expr, *diffs):
    """Coefficient of the product of the given differential symbols in expr."""
    term = sp.prod(diffs)
    return expr.coeff(term)

g_tt_KS = sp.simplify(coeff_of(ds2_KS, dt_KS, dt_KS))
g_tr_KS = sp.simplify(coeff_of(ds2_KS, dt_KS, dr) / 2)
g_tphi_KS = sp.simplify(coeff_of(ds2_KS, dt_KS, dphi_KS) / 2)
g_rr_KS = sp.simplify(coeff_of(ds2_KS, dr, dr))
g_rphi_KS = sp.simplify(coeff_of(ds2_KS, dr, dphi_KS) / 2)
g_thth_KS = sp.simplify(coeff_of(ds2_KS, dtheta, dtheta))
g_phiphi_KS = sp.simplify(coeff_of(ds2_KS, dphi_KS, dphi_KS))

print("g_tt_KS     =", g_tt_KS)
print("g_tr_KS     =", g_tr_KS)
print("g_tphi_KS   =", g_tphi_KS)
print("g_rr_KS     =", g_rr_KS)
print("g_rphi_KS   =", g_rphi_KS)
print("g_thth_KS   =", g_thth_KS)
print("g_phiphi_KS =", g_phiphi_KS)

# --- Cross-check against the commonly-cited closed forms (stated from
# memory -- this is exactly the kind of claim this script exists to
# verify, not to assert). ---
print("\n--- cross-check against commonly-cited closed forms ---")
claimed_g_tt = -(1 - 2*M*r/Sigma)
claimed_g_tr = 2*M*r/Sigma
claimed_g_tphi = -2*M*a*r*sin2/Sigma
claimed_g_rr = 1 + 2*M*r/Sigma
claimed_g_rphi = -a*sin2*(1 + 2*M*r/Sigma)
claimed_g_thth = Sigma
claimed_g_phiphi = sin2 * (Sigma + a**2*sin2*(1 + 2*M*r/Sigma))

checks = [
    ("g_tt", g_tt_KS, claimed_g_tt),
    ("g_tr", g_tr_KS, claimed_g_tr),
    ("g_tphi", g_tphi_KS, claimed_g_tphi),
    ("g_rr", g_rr_KS, claimed_g_rr),
    ("g_rphi", g_rphi_KS, claimed_g_rphi),
    ("g_thth", g_thth_KS, claimed_g_thth),
    ("g_phiphi", g_phiphi_KS, claimed_g_phiphi),
]
all_ok = True
for name, derived, claimed in checks:
    diff = sp.simplify(derived - claimed)
    ok = diff == 0
    all_ok &= ok
    print(f"{name}: derived-claimed simplifies to {diff}  {'OK' if ok else 'MISMATCH'}")

print("\nALL MATCH" if all_ok else "\nMISMATCH FOUND -- do not trust the claimed closed forms")

# ---------------------------------------------------------------------------
# ADM 3+1 decomposition. Standard relations:
#   g_ti = gamma_ij beta^j =: beta_i  (spatial-metric-lowered shift)
#   g_ij = gamma_ij (spatial metric IS the space-space block of g_munu)
#   g_tt = -alpha^2 + gamma_ij beta^i beta^j = -alpha^2 + beta_i beta^i
# Unlike Boyer-Lindquist, the spatial 3-metric here has a genuine
# off-diagonal (r,phi) block (gamma_rphi != 0) -- theta still decouples
# (gamma_rtheta=gamma_thetaphi=0, unchanged from BL, since the coordinate
# transformation never touches theta), so only a 2x2 block needs
# inverting, not the full 3x3.
# ---------------------------------------------------------------------------
print("\n--- ADM 3+1 decomposition ---")
gamma_rr, gamma_rphi, gamma_phiphi = g_rr_KS, g_rphi_KS, g_phiphi_KS
gamma_thth = g_thth_KS

# Invert the 2x2 (r,phi) spatial block symbolically.
Gamma2 = sp.Matrix([[gamma_rr, gamma_rphi], [gamma_rphi, gamma_phiphi]])
Gamma2_inv = sp.simplify(Gamma2.inv())
gamma_up_rr, gamma_up_rphi = Gamma2_inv[0, 0], Gamma2_inv[0, 1]
gamma_up_phiphi = Gamma2_inv[1, 1]
print("gamma^rr     =", sp.simplify(gamma_up_rr))
print("gamma^rphi   =", sp.simplify(gamma_up_rphi))
print("gamma^phiphi =", sp.simplify(gamma_up_phiphi))

beta_r, beta_phi = g_tr_KS, g_tphi_KS
beta_up_r = sp.simplify(gamma_up_rr * beta_r + gamma_up_rphi * beta_phi)
beta_up_phi = sp.simplify(gamma_up_rphi * beta_r + gamma_up_phiphi * beta_phi)
print("beta^r       =", beta_up_r)
print("beta^phi     =", beta_up_phi)

alpha2 = sp.simplify(-g_tt_KS + beta_r * beta_up_r + beta_phi * beta_up_phi)
print("alpha^2      =", alpha2)

# Cross-check against the commonly-cited closed forms for KS lapse/shift:
#   alpha^2  = Sigma / (Sigma + 2Mr)      (equivalently 1/(1+2Mr/Sigma))
#   beta^r   = 2Mr / (Sigma + 2Mr)
#   beta^phi = 0                          (the famous KS result: despite
#              g_tphi and g_rphi both being nonzero, the frame-dragging
#              shift beta^phi itself vanishes in spherical KS coordinates)
print("\n--- cross-check ADM against commonly-cited closed forms ---")
claimed_alpha2 = Sigma / (Sigma + 2 * M * r)
claimed_beta_up_r = 2 * M * r / (Sigma + 2 * M * r)
claimed_beta_up_phi = sp.Integer(0)

adm_checks = [
    ("alpha^2", alpha2, claimed_alpha2),
    ("beta^r", beta_up_r, claimed_beta_up_r),
    ("beta^phi", beta_up_phi, claimed_beta_up_phi),
]
adm_ok = True
for name, derived, claimed in adm_checks:
    diff = sp.simplify(derived - claimed)
    ok = diff == 0
    adm_ok &= ok
    print(f"{name}: derived-claimed simplifies to {diff}  {'OK' if ok else 'MISMATCH'}")
print("\nALL ADM MATCH" if adm_ok else "\nADM MISMATCH FOUND")
