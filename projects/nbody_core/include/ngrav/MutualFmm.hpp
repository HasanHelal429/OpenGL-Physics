#pragma once

#include "Solvers.hpp"
#include "State.hpp"
#include "Tree.hpp"
#include "Vec.hpp"

namespace ngrav {

// falcON-style (Dehnen 2000/2002) momentum-conserving dual-tree FMM, 3D,
// monopole+quadrupole order. Retires AdaptiveFmm's one-directional
// mono+quad-multipole/first-order-local half-measure: this carries the
// SAME multipole order (reusing AdaptiveTree's existing quadrupole upsweep)
// but every accepted pair updates *both* sides' local expansions from one
// shared computation, so net momentum is conserved to machine precision --
// see MutualFmm.cpp's header comment for the physical argument (the pair
// interaction energy depends only on the separation vector, so the force on
// each side is minus the other's by construction, not by a numerical
// coincidence).
//
// Single-visit unordered traversal (a self-pair only emits (child_i,child_i)
// and (child_i,child_j) i<j, not all ordered pairs) -- half the M2L work of
// the old one-directional traversal for the same accuracy. Single-threaded
// for now (a target-only-disjoint partition -- the old parallelization
// scheme -- doesn't hold once a pair writes to *both* sides' local
// expansions; Phase 8 adds an OpenMP-task traversal with per-thread
// expansion accumulation + reduction).
//
// Optional stats, for the O(N) scaling study / Studies suite.
struct MutualFmmStats {
    int nodeCount = 0;
    long m2lPairs = 0;
    long nearPairs = 0;
    double buildMs = 0.0;
    double traverseMs = 0.0;
    double l2lMs = 0.0;
    double nearFieldMs = 0.0;
};

void ComputeAccelMutualFmm(const PosMassView<3>& pts, const StepParams& sp, SoA<3>& out, MutualFmmStats* stats = nullptr);

} // namespace ngrav
