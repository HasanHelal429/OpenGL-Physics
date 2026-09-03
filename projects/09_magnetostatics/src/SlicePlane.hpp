#pragma once

#include "Grid.hpp"

#include <glm/glm.hpp>
#include <string>

namespace mag {

// The 2D solver grid is a slice of 3D space -- which slice is a deck choice.
// grid axis 0 (i, "x") and axis 1 (j, "y") map to two world axes; the third
// world coordinate is fixed at `offset`.
//
//   xy : (x(i), y(j), offset)     -- the default, a plane perpendicular to z
//   xz : (x(i), offset, y(j))     -- a plane containing the z axis (Helmholtz)
//   yz : (offset, x(i), y(j))
struct SlicePlane {
    enum Kind { XY, XZ, YZ };
    Kind kind = XY;
    double offset = 0.0;

    static Kind ParseKind(const std::string& s) {
        if (s == "xz") return XZ;
        if (s == "yz") return YZ;
        return XY;
    }

    glm::dvec3 Point(const Grid& g, int i, int j) const {
        const double a = g.x(i), b = g.y(j);
        switch (kind) {
        case XZ: return {a, offset, b};
        case YZ: return {offset, a, b};
        default: return {a, b, offset};
        }
    }

    // World vector -> the two in-plane components (grid x, grid y), for
    // rendering field lines and packing Bx/By the same way the Poisson path does.
    glm::dvec2 InPlane(const glm::dvec3& v) const {
        switch (kind) {
        case XZ: return {v.x, v.z};
        case YZ: return {v.y, v.z};
        default: return {v.x, v.y};
        }
    }
};

} // namespace mag
