#pragma once

#include "Grid.hpp"
#include "Potentials.hpp"
#include "Shells.hpp"

#include <functional>
#include <optional>
#include <stop_token>
#include <vector>

namespace hf {

enum class Method { Xalpha, Lda };

struct OccupiedOrbital {
    int n;
    int l;
    int N;               // occupation
    double eps;           // orbital energy (Ha)
    std::vector<double> u; // u_nl(r), trapezoid-normalized
};

// E_total = sum_nl(N_nl*eps_nl) - E_H - E_x/3 [- E_c doublecount].
//
// Orbital eigenvalues double-count electron-electron interaction: summing
// N_nl*eps_nl over occupied shells counts the Hartree term twice and the
// (nonlinear) exchange term 4/3 times, so both are corrected out (Hartree:
// simple 1/2 factor; exchange: 1/3 factor from Euler's theorem on the
// rho**(4/3) functional). If epsC/Vc (PZ81 correlation) are given, the
// general Kohn-Sham double-counting correction E_c[rho] - integral(Vc*rho)
// is added too (no closed-form shortcut exists for correlation the way it
// does for Slater exchange).
double ComputeTotalEnergy(const std::vector<double>& r, const std::vector<double>& rho, const std::vector<double>& VH,
                           const std::vector<double>& Vx, const std::vector<OccupiedOrbital>& occupied,
                           const std::vector<double>* epsC = nullptr, const std::vector<double>* Vc = nullptr);

struct ScfSnapshot {
    int iteration = 0;
    std::vector<double> rho;
    std::vector<double> V;
    std::vector<double> r;                 // grid, so a snapshot is self-contained for plotting
    std::map<ShellKey, double> orbitalEnergies;
    double eTotal = 0.0;
    double dE = 0.0;
    double dn = 0.0;
};

struct ScfResult {
    int Z = 0;
    Method method = Method::Xalpha;
    LogGrid grid;
    std::vector<double> rho;
    std::vector<double> V;
    std::vector<double> VH;
    std::vector<double> Vx; // exchange piece alone (Xalpha's V_x, or LDA's exact-LDA Slater piece)
    Configuration config;
    std::map<ShellKey, std::vector<double>> orbitals; // R_nl(r) = u_nl(r)/r
    std::map<ShellKey, double> orbitalEnergies;
    double eTotal = 0.0;
    int iterations = 0;
    std::vector<double> history; // E_total per iteration
    bool converged = false;
    bool cancelled = false;
};

struct ScfParams {
    int Z = 1;
    double alpha = kAlphaSchwarz; // ignored when method == Lda
    Method method = Method::Xalpha;
    int maxIter = 200;
    double mixBeta = 0.3;
    double tolE = 1e-6;
    double tolN = 1e-5;
    std::optional<LogGrid> grid; // defaults to DefaultGrid(Z) if unset
};

using SnapshotCallback = std::function<void(const ScfSnapshot&)>;

// Runs the self-consistent field loop for ScfParams::Z to convergence.
// Occupations come from GroundStateConfiguration(Z) and are held fixed for
// the whole run. Converges when both |dE_total| < tolE and the integrated
// density change < tolN hold for 2 consecutive iterations.
//
// If onSnapshot is set, it's invoked after every iteration (not opt-in like
// the Python version's post-hoc record_history — this port streams for live
// visualization instead). If stopToken has a stop requested, the loop exits
// at the next iteration boundary and ScfResult::cancelled is set.
ScfResult RunScf(const ScfParams& params, const SnapshotCallback& onSnapshot = nullptr,
                  std::stop_token stopToken = {});

} // namespace hf
