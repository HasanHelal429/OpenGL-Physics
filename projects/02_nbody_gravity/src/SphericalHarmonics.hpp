#pragma once

#include <complex>
#include <glm/glm.hpp>
#include <vector>

namespace nbody::sh {

// Arbitrary-order 3D multipole/local expansions via complex solid harmonics,
// the standard alternative to hand-derived Cartesian tensors (see this
// project's quadrupole/octupole path in Octree.hpp/AdaptiveFmm.cpp): one
// generic O(p) evaluator and three generic O(p^2)-per-pair translation
// operators (M2M/M2L/L2L) replace a hand-written formula per order.
//
// Formulas follow Dehnen (2014), "A fast multipole method for stellar
// dynamics" (arXiv:1405.2255), eqs. 2a-2b, 3a-3f: regular solid harmonic
// Upsilon_n^m(r) = (-1)^m r^n/(n+m)! P_n^m(cos theta) e^{im phi}, irregular
// solid harmonic Theta_n^m(r) = (-1)^m (n-m)!/r^{n+1} P_n^m(cos theta)
// e^{im phi}, with:
//   P2M: M_n^m(z)  = sum_a mass_a * Upsilon_n^m(pos_a - z)
//   M2M: M_n^m(z') = sum_{k=0}^{n} sum_{l=-k}^{k} Upsilon_k^l(z-z') * M_{n-k}^{m-l}(z)
//   M2L: L_n^m(s)  = sum_{k=0}^{p-n} sum_{l=-k}^{k} conj(M_k^l(z)) * Theta_{n+k}^{m+l}(s-z)
//   L2L: L_n^m(s') = sum_{k=0}^{p-n} sum_{l=-k}^{k} conj(Upsilon_k^l(s-s')) * L_{n+k}^{m+l}(s)
//   L2P: Psi(x)    = sum_{n=0}^{p} sum_{m=-n}^{n} conj(Upsilon_n^m(s-x)) * L_n^m(s)
// P_n^m here is the associated Legendre function *without* its own
// Condon-Shortley phase (the explicit (-1)^m above supplies it) -- see
// LegendreTable's own comment for why that convention was chosen, and
// SphericalHarmonicsTest.cpp for the brute-force validation that would
// catch getting this backwards.
//
// M_n^{-m} = (-1)^m * conj(M_n^m) always holds for real masses (and the
// same for L_n^m, since the resulting potential is real) -- this specific
// sign was the one bug Stage-A validation actually caught: two plausible-
// looking derivations of this symmetry (with vs. without the (-1)^m, and a
// third variant with a stray missing conjugate) were tried, and only this
// one reproduces brute-force gravity to the expected O(p)-convergent
// precision in M2M/M2L/L2L (which all invoke negative orders through this
// symmetry; P2M/L2P alone never do, which is why P2M looked fine under all
// three variants while only this one made the translation operators
// agree). Every coefficient array here stores only m = 0..n per degree n
// (a "triangular" layout, CoeffIndex/CoeffCount below), half the naive
// storage, with the m<0 half reconstructed on demand wherever a sum needs
// it.

using Complex = std::complex<double>;

inline int CoeffIndex(int n, int m) { return n * (n + 1) / 2 + m; }
inline int CoeffCount(int p) { return (p + 1) * (p + 2) / 2; }

// A degree-0..p expansion (multipole or local -- same triangular layout,
// distinguished only by which operators are applied to it), centered at
// `center`. `at(n,m)` accepts any -n<=m<=n, transparently applying the
// conjugate-symmetry above for m<0 -- callers never need to special-case it.
struct Expansion {
    glm::dvec3 center{0.0};
    int order = 0;
    std::vector<Complex> coeff; // size CoeffCount(order), indexed via CoeffIndex(n,m>=0)

    explicit Expansion(int p = 0, glm::dvec3 c = glm::dvec3(0.0)) : center(c), order(p), coeff(CoeffCount(p), Complex{0.0, 0.0}) {}

    // Adds another same-order expansion already centered at this one's own
    // `center` (e.g. summing several children's M2M-shifted multipoles, or
    // accumulating one node's several M2L hits during the dual-tree walk)
    // -- callers are responsible for shifting first; this never re-centers.
    Expansion& operator+=(const Expansion& other) {
        for (size_t i = 0; i < coeff.size(); ++i) coeff[i] += other.coeff[i];
        return *this;
    }

    Complex& raw(int n, int m) { return coeff[static_cast<size_t>(CoeffIndex(n, m))]; }
    const Complex& raw(int n, int m) const { return coeff[static_cast<size_t>(CoeffIndex(n, m))]; }

    // (-1)^m * conj(at m>=0) for m<0, per this file's own header comment.
    // Returns 0 for n outside [0,order] or |m|>n: every translation sum
    // below reaches past the *other* expansion's actual stored degree/order
    // (e.g. M2M's inner (k,l) loop vs. M_{n-k}^{m-l}'s valid range), and a
    // coefficient past an expansion's own degree is mathematically zero by
    // definition (there's no such moment), not an out-of-bounds access to
    // patch around.
    Complex at(int n, int m) const {
        if (n < 0 || n > order || m < -n || m > n) return Complex{0.0, 0.0};
        if (m >= 0) return raw(n, m);
        const int mm = -m; // mm > 0
        const Complex c = std::conj(raw(n, mm));
        return (mm & 1) ? Complex{-c.real(), -c.imag()} : c; // (-1)^mm * conj(raw(n,mm))
    }
};

// Regular (Upsilon) and irregular (Theta) solid harmonics, degree 0..p,
// order (m) 0..n only -- the m<0 half is never stored, only ever
// reconstructed via Expansion::at's conjugate-symmetry at the point of use.
// `r` is the displacement argument each translation operator's own comment
// says to pass (e.g. M2M's `z - z'`, not `z' - z`).
Expansion EvalUpsilon(const glm::dvec3& r, int p);
Expansion EvalTheta(const glm::dvec3& r, int p);

// P2M: builds a degree-p multipole expansion, about `center`, of the given
// point masses (eq. 3c).
Expansion P2M(const glm::dvec3& center, int p, const std::vector<glm::dvec3>& pos, const std::vector<double>& mass,
              const std::vector<int>& indices);

// M2M: re-expands a multipole expansion about a new center (eq. 3d). Output
// order matches `m`'s own order.
Expansion M2M(const Expansion& m, const glm::dvec3& newCenter);

// M2L: converts a multipole expansion (about m.center) into a *local*
// expansion about a different, well-separated `targetCenter` (eq. 3b).
// `p` is the output local expansion's order (independent of the source
// multipole's own order, though in practice both are the same project-wide
// constant).
Expansion M2L(const Expansion& m, const glm::dvec3& targetCenter, int p);

// L2L: shifts a local expansion to a new (child) center (eq. 3e). Output
// order matches `l`'s own order.
Expansion L2L(const Expansion& l, const glm::dvec3& newCenter);

// L2P: evaluates a local expansion's potential and acceleration (-gradient)
// at an exact point (eq. 3a for the potential; the acceleration is a
// central-difference of this same, independently-validated potential
// evaluator rather than a second hand-derived analytic-gradient formula --
// see this module's .cpp for why that trade was made deliberately here).
struct PotentialAndAccel {
    double potential = 0.0;
    glm::dvec3 accel{0.0};
};
PotentialAndAccel EvaluateLocal(const Expansion& l, const glm::dvec3& point, double G);

} // namespace nbody::sh
