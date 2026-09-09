#pragma once

#include <glm/glm.hpp>

// Dimension-generic vector aliases and traits shared by the 2D and 3D N-body
// gravity projects. `D` is a compile-time parameter (2 or 3): the two
// binaries are separate anyway, and a compile-time dimension keeps
// kChildren<D>, the Green's-function kernel, and the multipole-backend
// selection all zero-cost.
namespace ngrav {

template <int D>
using Vec = glm::vec<D, double, glm::defaultp>;

// Octree (D=3) has 8 children per node, quadtree (D=2) has 4.
template <int D>
inline constexpr int kChildren = 1 << D;

// Angular momentum is a scalar (the z-component, r_x v_y - r_y v_x) in 2D and
// a 3-vector in 3D. The rest of the diagnostics code is written against
// AngMom<D> so it stays dimension-generic.
template <int D>
struct AngMomTraits;
template <>
struct AngMomTraits<2> {
    using type = double;
};
template <>
struct AngMomTraits<3> {
    using type = glm::dvec3;
};
template <int D>
using AngMom = typename AngMomTraits<D>::type;

// z-component accessor that compiles for both dimensions (returns 0 in 2D).
// Lets shared code use a plain runtime `(D == 3) ? ... : 0.0` ternary without
// tripping over glm::vec<2>::z not existing.
inline double Zc(const glm::dvec3& v) { return v.z; }
inline double Zc(const glm::dvec2&) { return 0.0; }

} // namespace ngrav
