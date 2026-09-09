#pragma once

#include "ngrav/State.hpp"

namespace ngrav::gpu {

// GPU Barnes-Hut octree + tree-walk gravity solver, 3D, ported from
// 06_tidal_disruption's SPH+gravity pipeline (kernels.hpp/TdeSim.cpp) --
// stripped down to just the tree build (bounding box -> Morton sort ->
// level-by-level ClaimSlots/ResolveSlots -> bottom-up mass/COM upsweep) and
// the grouped, shared-stack cooperative gravity walk (`Forces()`'s gravity
// half). Everything SPH-specific (density/pressure/viscosity, the spatial-
// hash neighbor grid, adaptive smoothing length) and everything TDE-
// specific (the central-black-hole point-mass/Paczynski-Wiita term) is
// dropped -- this is a generic N-body tree, not a star-disruption sim.
// Plummer softening only (see GpuDirect.hpp's own comment for why spline
// softening isn't ported to GPU in this pass).
//
// Standalone entry point, like GpuDirect -- not wired into
// ngrav::Solver/System<D>'s generic dispatch (same reasoning as Phase 10:
// that integration is real scope, better taken on once there's an
// interactive consumer for it). Requires an active GL context.
struct GpuBarnesHutStats {
    int cellCount = 0;     // live tree cells after this step's build
    int levelsUsed = 0;    // ClaimSlots/ResolveSlots levels actually needed
    double sortMs = 0.0;
    double treeBuildMs = 0.0;
    double massUpsweepMs = 0.0;
    double forcesMs = 0.0;
    double readbackMs = 0.0;
};

// `treeMaxDepth` bounds the ClaimSlots/ResolveSlots level loop (it also
// exits early via a cell-count readback once no new cells are created, same
// as 06); `treeMaxCellsFactor` sizes the cell pool as
// max(treeMaxCellsFactor * N, 64), also matching 06's default. Buffers are
// lazily (re)allocated on first use and whenever N changes; shaders are
// compiled once and cached, both scoped to the process's current GL context
// (same caveat as GpuDirect.hpp).
void ComputeAccelGpuBarnesHut(const PosMassView<3>& pts, double G, double eps2, double theta, SoA<3>& out,
                             GpuBarnesHutStats* stats = nullptr, int treeMaxDepth = 16,
                             double treeMaxCellsFactor = 4.0);

} // namespace ngrav::gpu
