#pragma once

#include <complex>
#include <glm/glm.hpp>
#include <vector>

namespace nbody2d::fmm {

// Arbitrary-order 2D multipole/local expansions via complex Laurent series
// -- the 2D analogue of the 3D project's SphericalHarmonics.hpp, and much
// cheaper: 2D's fundamental solution (ln(r), not 1/r) means each degree is
// one plain complex power instead of a whole row of associated-Legendre
// terms, and M2L is a single sum over one index (O(p) per output
// coefficient) instead of 3D's double sum (O(p^2) per output coefficient,
// O(p^4) total per pair) -- see this project's plan doc for why that
// difference is the entire point of building this as its own project
// rather than a 3D solver, and this order (default 10, see ComplexFmmTree)
// deliberately much higher than the 3D solver's p=5.
//
// Formulas below reproduce (independently re-derived and cross-checked,
// not trusted blindly) a reference 2D FMM implementation this project was
// compared against earlier in this session: for point masses q_i at
// z_i = x_i + i*y_i relative to an expansion center z_c,
//   P2M: M_0 = sum q_i;  M_j = -[sum_i q_i (z_i-z_c)^j] / j,  j=1..p
//   M2M (shift to a new center z_c', disp = z_c - z_c'):
//     M'_0 = M_0;  M'_r = -M_0*disp^r/r + sum_{k=1}^r M_k*disp^(r-k)*C(r-1,k-1)
//   M2L (convert a multipole about z_A into a local expansion about a
//        different, well-separated z_B; disp = z_A - z_B):
//     L_0 = M_0*log(-disp) + sum_{k=1}^p (-1)^k M_k/disp^k
//     L_j = disp^-j * [ -M_0/j + sum_{k=1}^p (-1)^k M_k/disp^k * C(j+k-1,k-1) ]
//   L2L (shift a local expansion to a new center, disp = z_old - z_new):
//     the standard in-place "Taylor shift" (repeated synthetic division),
//     not a combinatorial sum -- see ShiftLocal's own comment.
//   L2P: Psi(z) = sum_k L_k*(z-center)^k (Horner); the complex derivative
//     dW/dz of this same polynomial packages into a real 2D force as
//     (Fx,Fy) = (-Re(dW/dz), Im(dW/dz)) -- derived from the Cauchy-Riemann
//     identity dW/dz = dPhi/dx - i*dPhi/dy for an analytic complex
//     potential W with Phi=Re(W), Fx=-dPhi/dx, Fy=-dPhi/dy.
// M_0 and L_0's log term are the only place mass/log enters; every other
// coefficient is a plain complex power series term, which is what makes
// this whole module simpler and cheaper than the 3D one.

using Complex = std::complex<double>;

inline Complex ToComplex(const glm::dvec2& v) { return Complex(v.x, v.y); }
inline glm::dvec2 ToVec2(const Complex& c) { return glm::dvec2(c.real(), c.imag()); }

// A degree-0..p expansion (multipole or local -- same flat layout,
// distinguished only by which operators are applied to it), centered at
// `center`.
struct Expansion {
    Complex center{0.0, 0.0};
    int order = 0;
    std::vector<Complex> coeff; // size order+1, index == degree

    explicit Expansion(int p = 0, Complex c = Complex(0.0, 0.0)) : center(c), order(p), coeff(static_cast<size_t>(p) + 1, Complex{0.0, 0.0}) {}

    Expansion& operator+=(const Expansion& other) {
        for (size_t i = 0; i < coeff.size(); ++i) coeff[i] += other.coeff[i];
        return *this;
    }
};

// P2M: builds a degree-p multipole expansion, about `center`, of the given
// point masses.
Expansion P2M(const Complex& center, int p, const std::vector<glm::dvec2>& pos, const std::vector<double>& mass,
              const std::vector<int>& indices);

// M2M: re-expands a multipole expansion about a new center. Output order
// matches `m`'s own order.
Expansion M2M(const Expansion& m, const Complex& newCenter);

// M2L: converts a multipole expansion (about m.center) into a *local*
// expansion about a different, well-separated `targetCenter`. `p` is the
// output local expansion's order.
Expansion M2L(const Expansion& m, const Complex& targetCenter, int p);

// L2L: shifts a local expansion to a new (child) center. Output order
// matches `l`'s own order.
Expansion L2L(const Expansion& l, const Complex& newCenter);

// L2P: evaluates a local expansion's potential and force at an exact
// point (closed-form derivative -- cheap enough at this order that no
// finite-difference shortcut is needed, unlike the 3D solver's L2P).
struct PotentialAndForce {
    double potential = 0.0;
    glm::dvec2 force{0.0};
};
PotentialAndForce EvaluateLocal(const Expansion& l, const glm::dvec2& point, double G);

} // namespace nbody2d::fmm
