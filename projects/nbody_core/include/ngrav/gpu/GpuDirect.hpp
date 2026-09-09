#pragma once

#include "ngrav/State.hpp"

namespace ngrav::gpu {

// GPU direct O(N^2) gravity sum: a compute shader, one thread per target
// particle, looping over every source particle -- the naive all-pairs sum,
// the same one ComputeAccelDirect does on the CPU, run as fp32 on the GPU.
//
// Plummer softening only. Spline (compact-support) softening isn't ported
// to GPU in this phase -- a deliberate scope decision, not an oversight:
// this solver exists to validate the GPU pipeline itself (buffer upload/
// dispatch/readback, fp32-vs-fp64 accuracy) at small-to-moderate N, and
// every other GPU-vs-CPU comparison in the repo (06_tidal_disruption's own
// SelfTest) is likewise Plummer-only. Extending it is a small, well-scoped
// follow-on once a caller actually needs it.
//
// Requires an active GL context (e.g. fw::GLContext::CreateHidden) --
// standalone entry point, not wired into ngrav::Solver/System<D>'s generic
// dispatch (that would need every consumer -- decks, NBodyApp's solver
// radio buttons, ScalingSweepWorker -- to handle a GL-context-dependent
// backend; real scope, better taken on once Phase 11's GPU Barnes-Hut
// makes a GPU path actually useful for the interactive app rather than
// only for --gpu-selftest's own verification).
struct GpuDirectStats {
    double uploadMs = 0.0;
    double dispatchMs = 0.0;
    double readbackMs = 0.0;
};

void ComputeAccelGpuDirect(const PosMassView<3>& pts, double G, double eps2, SoA<3>& out,
                          GpuDirectStats* stats = nullptr);

} // namespace ngrav::gpu
