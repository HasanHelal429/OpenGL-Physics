#pragma once

#include <cmath>
#include <vector>

namespace hf {

// Logarithmic radial grid r_i = r_min * exp(i*h), i = 0..N-1.
struct LogGrid {
    std::vector<double> r;
    double h = 0.0;
};

inline LogGrid MakeLogGrid(double rMin, double rMax, int nPoints) {
    LogGrid grid;
    grid.h = std::log(rMax / rMin) / (nPoints - 1);
    grid.r.resize(nPoints);
    for (int i = 0; i < nPoints; ++i) {
        grid.r[static_cast<size_t>(i)] = rMin * std::exp(i * grid.h);
    }
    return grid;
}

// Sane log-grid defaults for atomic number Z (core orbitals scale as 1/Z).
// r_min = 1e-5/Z keeps the finite-difference matrices well away from the
// float64 dynamic-range/conditioning issues that show up once r_min is
// chosen independent of Z for heavy-atom cores.
inline LogGrid DefaultGrid(int Z, double rMax = 150.0, int nPoints = 4000) {
    return MakeLogGrid(1e-5 / Z, rMax, nPoints);
}

} // namespace hf
