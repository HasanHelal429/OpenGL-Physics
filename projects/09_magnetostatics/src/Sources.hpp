#pragma once

#include "Grid.hpp"

#include <vector>

namespace fw { class Deck; }

namespace mag {

// Out-of-plane current density J_z(x, y), row-major, summed over the deck's
// [[wire]] and [[strip]] tables:
//
//   [[wire]]   x, y, current      (+ optional radius; default ~1.5 cells)
//              a disk of uniform J_z carrying total current `current`.
//   [[strip]]  x, y, width, height, current
//              a rectangle of uniform J_z carrying total `current` -- an
//              out-of-plane current sheet (a pair of opposite strips is the
//              2D idealisation of a solenoid / Helmholtz field).
std::vector<double> BuildCurrent(const Grid& g, const fw::Deck& deck);

} // namespace mag
