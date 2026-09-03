#pragma once

#include <string>
#include <vector>

namespace fw { class Deck; }

namespace fdtd {

// Per-cell material properties rasterised from the deck's [[material]] and
// [[pec]] tables onto the Ez grid (nx*ny, row-major). Shapes:
//   slab      axis = "x"|"y", pos, thickness            (a layer)
//   box       x, y, half_x, half_y
//   cylinder  x, y, radius
//   halfspace axis = "x"|"y", pos, side = "lo"|"hi"      (fills a half-plane)
// Coordinates are in physical units (cells * dx). Later shapes overwrite
// earlier ones where they overlap.
struct Materials {
    int nx = 0, ny = 0;
    std::vector<double> epsR;   // relative permittivity   (default 1)
    std::vector<double> muR;    // relative permeability    (default 1)
    std::vector<double> sigma;  // electric conductivity    (default 0)
    std::vector<unsigned char> pec;  // 1 -> perfect electric conductor (Ez = 0)

    void Build(int nx_, int ny_, double dx, double dy, const fw::Deck& deck);
    bool AnyPec() const;
    bool AnyDielectric() const;   // epsR != 1 or muR != 1 or sigma != 0 anywhere
};

} // namespace fdtd
