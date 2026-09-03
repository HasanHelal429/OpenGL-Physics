#pragma once

#include <cstddef>

namespace mag {

// Uniform cell-centred grid on [-lx/2, lx/2] x [-ly/2, ly/2], nx by ny.
// Row-major storage, index = j*nx + i, with i the x axis and j the y axis --
// the same (y, x) layout fw::OutputWriter writes and the Python tooling reads
// back as an array of shape (ny, nx).
struct Grid {
    int nx = 256;
    int ny = 256;
    double lx = 4.0;
    double ly = 4.0;

    double dx() const { return lx / nx; }
    double dy() const { return ly / ny; }
    double x(int i) const { return -0.5 * lx + (i + 0.5) * dx(); }
    double y(int j) const { return -0.5 * ly + (j + 0.5) * dy(); }

    int count() const { return nx * ny; }
    std::size_t idx(int i, int j) const {
        return static_cast<std::size_t>(j) * nx + i;
    }
    bool onBoundary(int i, int j) const {
        return i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
    }
};

} // namespace mag
