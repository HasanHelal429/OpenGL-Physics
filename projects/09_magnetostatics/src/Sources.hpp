#pragma once

#include "Grid.hpp"

#include <vector>

namespace fw { class Deck; }

namespace mag {

// A single out-of-plane line current -- a disk of uniform J_z carrying total
// current `current`. Kept as a live spec (not just baked into J_z) so the
// interactive view can move it or change its current and re-solve.
struct WireSpec {
    double x = 0.0;
    double y = 0.0;
    double current = 1.0;
    double radius = 0.0;   // <=0 -> a default ~1.5 cells, filled in at parse time
};

// A rectangle of uniform J_z carrying total `current` -- an out-of-plane
// current sheet. A pair of opposite strips is the 2D idealisation of a
// solenoid / Helmholtz field.
struct StripSpec {
    double x = 0.0, y = 0.0;
    double width = 0.0, height = 0.0;
    double current = 1.0;
};

struct Sources {
    std::vector<WireSpec> wires;
    std::vector<StripSpec> strips;
};

// Parse [[wire]] and [[strip]] tables from the deck (filling each wire's
// default radius from the grid spacing).
Sources ParseSources(const Grid& g, const fw::Deck& deck);

// Rasterise the specs into an out-of-plane current density J_z(x, y),
// row-major, each source's total current preserved exactly.
std::vector<double> RasterizeCurrent(const Grid& g, const Sources& src);

} // namespace mag
