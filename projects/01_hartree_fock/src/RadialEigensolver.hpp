#pragma once

#include <optional>
#include <vector>

namespace hf {

// Generalized eigenproblem H w = E*M w for the radial equation on a log grid
// (see Grid.hpp). Substitution u(r)=r*R(r), then u(r)=sqrt(r)*w(x) with
// x=ln(r) removes the first-derivative term and turns the centrifugal term
// into (l+1/2)^2 (a correct artifact of the transform, not l(l+1)).
struct RadialMatrices {
    std::vector<double> mainDiagH; // length n
    std::vector<double> offDiagH;  // length n-1
    std::vector<double> mDiag;     // length n
};

RadialMatrices BuildRadialMatrices(const std::vector<double>& r, int l, const std::vector<double>& V, double h);

struct RadialSolution {
    std::vector<double> energies;        // length nStates, ascending (most bound first)
    std::vector<std::vector<double>> u;  // nStates vectors of length n, u_nl(r), trapezoid-normalized
};

// Lowest nStates eigenpairs (energies, u_nl(r)) for angular momentum l.
//
// Solved as a sparse generalized eigenproblem via shift-invert Lanczos
// (Spectra::SymGEigsShiftSolver), which factorizes the natural-scale
// tridiagonal (H - sigma*M) directly via a custom O(N) Thomas-algorithm
// operator instead of forming an ill-conditioned 1/r-rescaled matrix.
// u_nl is normalized so trapezoid(u_nl^2, r) == 1 for each state.
//
// Retries with a progressively larger Krylov subspace and perturbed shift
// if the Lanczos iteration fails to converge (mirrors the robustness fix
// the Python solver needed for early, far-from-self-consistent potentials).
RadialSolution SolveRadialChannel(const std::vector<double>& r, int l, const std::vector<double>& V, double h,
                                   int nStates = 8, std::optional<double> sigma = std::nullopt);

} // namespace hf
