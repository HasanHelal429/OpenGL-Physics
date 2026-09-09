#pragma once

#include "ngrav/State.hpp"

namespace ngrav::gpu {

// GPU quadtree + tree-walk gravity solver, 2D -- the D==2 sibling of
// GpuBarnesHut.hpp, ported the same way (from 06_tidal_disruption's
// kernels.hpp, stripped of SPH/TDE specifics), rewritten for a 4-child
// quadtree instead of an 8-child octree: 2D bounding box, a 30-bit 2D
// Morton code (interleaving two 15-bit axes instead of three 10-bit ones),
// 2D quadrant test in place of octantOf, and the 2D 1/r softened-gravity
// force law (a = G m r/(r^2+eps^2), matching ComputeAccelDirect/BarnesHut's
// own D==2 formula) in place of 3D's 1/r^2. The shared-stack cooperative
// walk's barrier-uniformity discipline is otherwise identical.
//
// Plummer softening only; standalone entry point, not wired into
// ngrav::Solver/System<2>'s generic dispatch -- same reasoning as
// GpuDirect.hpp/GpuBarnesHut.hpp. Requires an active GL context.
struct GpuQuadtreeStats {
    int cellCount = 0;
    int levelsUsed = 0;
    double sortMs = 0.0;
    double treeBuildMs = 0.0;
    double massUpsweepMs = 0.0;
    double forcesMs = 0.0;
    double readbackMs = 0.0;
};

void ComputeAccelGpuQuadtree(const PosMassView<2>& pts, double G, double eps2, double theta, SoA<2>& out,
                            GpuQuadtreeStats* stats = nullptr, int treeMaxDepth = 16, double treeMaxCellsFactor = 4.0);

} // namespace ngrav::gpu
