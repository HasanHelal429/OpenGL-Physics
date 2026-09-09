#pragma once

#include "Vec.hpp"

namespace ngrav {

// Multipole acceptance criterion. `Geometric` is the classic scale-invariant
// opening angle (node size vs distance). `Relative` targets a fractional
// force error directly (GADGET-2 style): open a node unless its estimated
// multipole error is below alpha * |a_prev|. `Relative` needs the previous
// step's acceleration magnitude, so the first step / a freshly primed system
// falls back to `Geometric`.
enum class MacKind {
    Geometric,
    Relative,
};

struct MacParams {
    MacKind kind = MacKind::Geometric;
    double theta = 0.5;  // opening angle (Geometric) or its reuse as a tolerance handle
    double alpha = 0.001; // fractional-force-error target (Relative)
};

// Geometric test: accept if (2*halfSize)^2 < theta^2 * dist2. This is the
// form both current projects use (node full size vs distance to its COM).
inline bool GeometricAccept(double nodeHalfSize, double dist2, double theta2) {
    const double size = 2.0 * nodeHalfSize;
    return size * size < theta2 * dist2;
}

// Relative / acceleration test (GADGET-2, Springel 2005 eq. 18): with node
// mass M, node size s, separation d, and the target particle's previous
// acceleration magnitude aOld, accept if
//     G * M * s^2 / d^4  <  alpha * aOld
// i.e.  M * s^2  <  alpha * aOld * d^4 / G.
// Passed d2 (= d^2) and s (= 2*halfSize) to avoid a sqrt.
inline bool RelativeAccept(double nodeMass, double nodeSize, double dist2, double G, double alpha, double aOld) {
    if (aOld <= 0.0) return false; // no history -> caller uses GeometricAccept
    const double lhs = nodeMass * nodeSize * nodeSize;
    const double rhs = alpha * aOld * dist2 * dist2 / G;
    return lhs < rhs;
}

} // namespace ngrav
