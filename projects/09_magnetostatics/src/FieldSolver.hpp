#pragma once

#include "Grid.hpp"

#include <vector>

namespace mag {

// Direct analogue of Physics Simulations' poisson.py::solve_poisson, but by
// iterative relaxation rather than a sparse direct solve (the port's whole
// point -- an iterative smoother is what goes on the GPU as multigrid in
// Phase 2). Solves  laplacian(u) = rhs  on the 5-point stencil, with the
// cells flagged in `fixedMask` clamped to `fixedValues` (Dirichlet /
// conductor cells) instead of being relaxed.
//
// For magnetostatics: rhs = -mu0 * Jz, u = A_z, and the domain-edge cells
// are the fixed (A_z = 0) boundary.

struct SolveResult {
    int iterations = 0;
    double residual = 0.0;   // final L-inf residual over the free cells
    bool converged = false;
};

// Domain-edge boundary condition for the cells NOT in `fixedMask`.
//   Dirichlet: every edge cell must be a fixed cell (A_z pinned there) --
//              the "field decays to zero on a large box" condition, right for
//              compact sources like a single wire.
//   Neumann:   zero normal derivative at the edge (natural / free boundary) --
//              right for a uniform field that should not be dragged back to
//              zero at the wall, e.g. a solenoid's between-sheet field. The
//              rhs mean is projected out for solvability and the solution's
//              additive constant is pinned to zero mean.
enum class BC { Dirichlet, Neumann };

struct SolveOptions {
    double tol = 1e-9;       // stop when L-inf residual / rhs-scale < tol
    int maxIterations = 200000;
    double omega = -1.0;     // SOR factor; <0 -> use the optimal-for-Poisson estimate
    int checkEvery = 20;     // residual-check cadence (relaxation sweeps are cheap)
    BC edgeBC = BC::Dirichlet;
};

// `u` is used as the initial guess (pass a zero-filled vector for a cold
// start) and overwritten with the solution. All arrays are grid-sized,
// row-major.
SolveResult SolvePoisson(const Grid& g,
                         const std::vector<double>& rhs,
                         const std::vector<unsigned char>& fixedMask,
                         const std::vector<double>& fixedValues,
                         std::vector<double>& u,
                         const SolveOptions& opt = {});

// Convenience: a fixed mask that is exactly the domain-edge cells, with
// value 0 everywhere -- the standard "field decays to zero on a large box"
// magnetostatics boundary.
void MakeEdgeDirichlet(const Grid& g,
                       std::vector<unsigned char>& fixedMask,
                       std::vector<double>& fixedValues);

// B = curl(A_z zhat) = (dA_z/dy, -dA_z/dx, 0), central differences in the
// interior and one-sided at the edges. Outputs are grid-sized, row-major.
void CurlZ(const Grid& g, const std::vector<double>& Az,
           std::vector<double>& Bx, std::vector<double>& By);

} // namespace mag
