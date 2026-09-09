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
// the old one-directional traversal for the same accuracy.
//
// Phase 8: the traversal is recursive (was an explicit stack) and, above a
// particle-count threshold, spawns OpenMP tasks for the top ~2 recursion
// levels -- a target-only-disjoint partition (the old parallelization
// scheme sketched for a one-directional FMM) doesn't hold here once a pair
// write touches *both* sides' local expansions, so each task accumulates
// into its own thread-private expansion buffer (indexed by
// omp_get_thread_num(), grabbed fresh at every call -- safe because a
// running task never migrates threads mid-execution) and all buffers are
// summed once, after every task completes, into the single combined result.
// Below the threshold the original single-threaded traversal (unchanged,
// still the correctness oracle every other check is measured against) runs
// instead -- see MutualFmm.cpp for why, and for the parallel-vs-serial
// agreement check that validates the new path rather than just leaving it
// unexercised by the existing small-N selftests.
//
// Optional stats, for the O(N) scaling study / Studies suite. `forceSerial`
// is a test/benchmark hook (always take the single-threaded path regardless
// of N) -- production callers never need it.
struct MutualFmmStats {
    int nodeCount = 0;
    long m2lPairs = 0;
    long nearPairs = 0;
    double buildMs = 0.0;
    double traverseMs = 0.0;
    double l2lMs = 0.0;
    double nearFieldMs = 0.0;
};

void ComputeAccelMutualFmm(const PosMassView<3>& pts, const StepParams& sp, SoA<3>& out, MutualFmmStats* stats = nullptr,
                          bool forceSerial = false);

} // namespace ngrav
