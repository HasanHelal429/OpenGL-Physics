#pragma once

#include "Vec.hpp"

#include <cmath>

// Softening models. P1 ships Plummer only (matching both projects' current
// behaviour); compact-support spline softening is added in P4 as a second
// `SofteningKind` -- the interface is shaped for that now so nothing
// downstream changes when it lands.
namespace ngrav {

enum class SofteningKind {
    Plummer, // a ~ m r / (r^2 + eps^2)^{3/2}  (3D); biased at all radii
    Spline,  // compact-support cubic spline; force is exactly Newtonian for r > eps  (P4)
};

// Everything the solvers need to evaluate a softened interaction. `eps2` is
// the squared softening length; `kind` selects the profile. Kept as a small
// value type passed by copy into the hot loops.
struct Softening {
    SofteningKind kind = SofteningKind::Plummer;
    double eps = 0.05;
    double eps2 = 0.05 * 0.05;

    static Softening Plummer(double e) {
        Softening s;
        s.kind = SofteningKind::Plummer;
        s.eps = e;
        s.eps2 = e * e;
        return s;
    }
};

} // namespace ngrav
