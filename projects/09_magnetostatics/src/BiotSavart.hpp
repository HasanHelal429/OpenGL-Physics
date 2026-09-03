#pragma once

#include "Grid.hpp"
#include "SlicePlane.hpp"

#include "framework/ComputeShader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace fw { class Deck; }

namespace mag {

// One current-carrying coil, discretised to a closed polyline. Kept as a live
// spec (like WireSpec) so the interactive view can move it and recompute.
struct CoilSpec {
    std::string shape = "loop";      // "loop" | "solenoid" | "helmholtz"
    glm::dvec3 center{0.0};
    glm::dvec3 axis{0.0, 0.0, 1.0};  // normalised at parse time
    double radius = 1.0;
    double current = 1.0;            // per turn
    int turns = 1;
    double length = 0.0;             // solenoid extent along the axis
    double spacing = 0.0;            // helmholtz gap (0 -> = radius)
    int segmentsPerTurn = 128;
};

std::vector<CoilSpec> ParseCoils(const fw::Deck& deck);

// A straight current element: midpoint and the directed length dl, with the
// per-turn current folded in as dl *= current.
struct WireSegment {
    glm::dvec3 mid{0.0};
    glm::dvec3 dl{0.0};
};

std::vector<WireSegment> BuildSegments(const std::vector<CoilSpec>& coils);

// Direct Biot-Savart sum at one point (CPU reference / analytic-check path):
//   B(r) = (mu0 / 4pi) sum_seg  (I dl) x (r - mid) / |r - mid|^3
glm::dvec3 BiotSavartAt(const std::vector<WireSegment>& segs,
                        const glm::dvec3& r, double mu0);

// GPU evaluation of the same sum over every cell of a slice plane. One
// compute-shader thread per cell, inner loop over all segments in an SSBO.
class BiotSavartField {
public:
    void Init(const Grid& g);
    // Uploads `segs` and evaluates B on the slice; fills bx/by/bz (grid-sized,
    // row-major) with the two in-plane components and the out-of-plane one.
    void Evaluate(const Grid& g, const SlicePlane& plane, double mu0,
                  const std::vector<WireSegment>& segs,
                  std::vector<double>& bx, std::vector<double>& by,
                  std::vector<double>& bz);
    bool Ready() const { return m_ready; }

private:
    fw::ComputeShader m_prog;
    GLuint m_segBuf = 0;   // std430 array of (vec4 mid, vec4 dl)
    GLuint m_outBuf = 0;   // std430 array of vec4 (Bx,By,Bz,0)
    int m_segCap = 0;
    int m_cells = 0;
    bool m_ready = false;
};

} // namespace mag
