// Entry point: runs the eigensolver validation (the acid test for the
// custom Spectra shift-invert operator -- see RadialEigensolver.cpp) and a
// quick SCF regression check against the Python solver's reference numbers,
// then opens the live visualization window.
#include "Grid.hpp"
#include "HFVisualizerApp.hpp"
#include "RadialEigensolver.hpp"
#include "ScfSolver.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

// With V(r) = -Z/r only, solveRadialChannel must reproduce the analytic
// hydrogen-like energies E_nl = -Z^2/(2n^2), including the "accidental"
// l-degeneracy. This is the step most likely to expose a sign/convention
// mismatch in the custom shift-invert operator, so it's gated before
// anything else runs.
bool ValidateEigensolver() {
    bool allPass = true;
    std::printf("Eigensolver validation (V(r) = -Z/r, analytic hydrogen-like energies):\n");

    for (int Z : {1, 10, 50, 90}) {
        const hf::LogGrid grid = hf::DefaultGrid(Z);
        std::vector<double> V(grid.r.size());
        for (size_t i = 0; i < grid.r.size(); ++i) V[i] = -static_cast<double>(Z) / grid.r[i];

        for (int l = 0; l <= 3; ++l) {
            constexpr int kStatesToCheck = 3;
            const hf::RadialSolution sol = hf::SolveRadialChannel(grid.r, l, V, grid.h, kStatesToCheck);

            for (int idx = 0; idx < kStatesToCheck; ++idx) {
                const int n = l + 1 + idx;
                const double analytic = -static_cast<double>(Z) * Z / (2.0 * n * n);
                const double numeric = sol.energies[static_cast<size_t>(idx)];
                const double relErr = std::abs((numeric - analytic) / analytic);
                const bool pass = relErr < 1e-3;
                allPass &= pass;
                std::printf("  Z=%3d l=%d n=%d: analytic=%.6f numeric=%.6f relErr=%.2e %s\n", Z, l, n, analytic, numeric,
                            relErr, pass ? "OK" : "FAIL");
            }
        }
    }

    std::printf(allPass ? "Eigensolver validation: ALL PASS\n\n" : "Eigensolver validation: FAILED\n\n");
    return allPass;
}

// He/Ne/Ar total energies from the Python solver (HF_Solver_Plan.md), Xalpha
// method with the default alpha=0.7 (Schwarz). Informational -- grid/
// implementation differences can shift the last digit or two, so this warns
// rather than blocking the app from opening.
void RunScfRegressionCheck() {
    std::printf("SCF regression check (Xalpha, alpha=0.7) vs. Python reference energies:\n");
    struct Ref {
        int Z;
        const char* name;
        double eTotal;
    };
    for (const Ref& ref : {Ref{2, "He", -2.76637}, Ref{10, "Ne", -128.03490}, Ref{18, "Ar", -525.89503}}) {
        hf::ScfParams params;
        params.Z = ref.Z;
        params.method = hf::Method::Xalpha;
        const hf::ScfResult result = hf::RunScf(params);
        const double diff = result.eTotal - ref.eTotal;
        std::printf("  %s (Z=%d): E_total=%.5f Ha  reference=%.5f Ha  diff=%.2e  iterations=%d  %s\n", ref.name, ref.Z,
                    result.eTotal, ref.eTotal, diff, result.iterations, result.converged ? "converged" : "NOT CONVERGED");
    }
    std::printf("\n");
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    if (!ValidateEigensolver()) {
        std::fprintf(stderr, "Eigensolver validation failed -- aborting before opening the window.\n");
        return 1;
    }

    RunScfRegressionCheck();

    hf::HFVisualizerApp app;
    app.Run();
    return 0;
}
