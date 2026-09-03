#pragma once

#include "FieldSolver.hpp"
#include "Grid.hpp"

#include <vector>

namespace mag {

// Geometric multigrid V-cycle for  laplacian(u) = rhs  on the cell-centred
// 5-point stencil, with the domain-edge cells held at `fixedValues` (Dirichlet
// -- the finest level clamps them; coarse correction levels are homogeneous).
//
// This is the CPU reference for the algorithm the interactive view runs: RB-GS
// smoothing, full-weighting restriction, bilinear prolongation. It converges in
// a V-cycle count that is (almost) independent of grid size -- the property
// Phase 2's scaling study checks, and what makes live re-solving practical.
//
// Only the "all edge cells fixed" boundary is supported here (interior
// conductor masks fall back to SolvePoisson / RB-GS). Neumann edges are not
// handled by this path.

struct MultigridOptions {
    double tol = 1e-8;      // stop when L-inf residual / rhs-scale < tol
    int maxCycles = 100;
    int nu1 = 2;            // pre-smoothing sweeps
    int nu2 = 2;            // post-smoothing sweeps
    int nuCoarse = 60;      // sweeps on the coarsest level (near-exact solve)
    int minCoarse = 4;      // coarsest dimension (>=4 -> at least a 2x2 interior)
    int gamma = 2;          // 1 = V-cycle, 2 = W-cycle
    double omega = 1.0;     // smoother relaxation (1.0 = plain RB-GS; good damping)
};

struct MultigridResult {
    int cycles = 0;
    double residual = 0.0;
    bool converged = false;
    int levels = 0;
};

MultigridResult SolvePoissonMultigrid(const Grid& g,
                                      const std::vector<double>& rhs,
                                      const std::vector<double>& fixedValues,
                                      std::vector<double>& u,
                                      const MultigridOptions& opt = {});

} // namespace mag
