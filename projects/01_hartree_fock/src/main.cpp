// Entry point and CLI dispatch for the atomic Hartree-Fock-Slater / Kohn-Sham
// LDA solver. See Hartree_Fock_Plan.md and README.md.
//
//   01_hartree_fock --selftest                 eigensolver vs analytic hydrogen (CPU, no window)
//   01_hartree_fock --scf-selftest             SCF energies vs Python + literature references
//   01_hartree_fock --deck <f.toml> --out <d>  headless SCF run -> per-iteration frames + diagnostics
//   01_hartree_fock --interactive              the ImGui periodic-table explorer (needs a window)
#include "Grid.hpp"
#include "HFSim.hpp"
#include "HFVisualizerApp.hpp"
#include "RadialEigensolver.hpp"
#include "ScfSolver.hpp"
#include "Shells.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string deck;
    std::string out;
    bool interactive = false;
    bool selftest = false;
    bool scfSelftest = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--scf-selftest") a.scfSelftest = true;
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

// With V(r) = -Z/r only, SolveRadialChannel must reproduce the analytic
// hydrogen-like energies E_nl = -Z^2/(2n^2), including the "accidental"
// l-degeneracy -- the step most likely to expose a sign/convention mismatch
// in the custom shift-invert operator.
bool ValidateEigensolver() {
    bool allPass = true;
    std::printf("eigensolver selftest (V(r) = -Z/r, analytic hydrogen-like energies):\n");

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

    std::printf("eigensolver selftest: %s\n", allPass ? "PASS" : "FAIL");
    return allPass;
}

// SCF total energies vs the Python solver (HF_Solver_Plan.md) and literature
// non-relativistic HF. Grid / implementation differences shift the last digit
// or two, so the tolerance is loose -- this catches gross regressions, not
// method-level disagreement (Xalpha is not exact HF).
bool RunScfRegressionCheck() {
    struct Ref {
        int Z;
        const char* name;
        hf::Method method;
        double ePython;   // the Python solver's own number
        double tol;       // absolute Ha
    };
    // Kept lean so this stays a fast CI gate. The Madelung-exception atoms
    // (Cr, Cu) and the f-shell case (Gd) are covered by decks/ instead --
    // see Hartree_Fock_Plan.md Phase 4.
    const Ref refs[] = {
        {2, "He", hf::Method::Xalpha, -2.76637, 5e-3},
        {10, "Ne", hf::Method::Xalpha, -128.03490, 5e-2},
        {18, "Ar", hf::Method::Xalpha, -525.89503, 5e-2},
        {2, "He", hf::Method::Lda, -2.83418, 5e-3},
    };

    std::printf("scf selftest (vs Python reference energies):\n");
    bool allPass = true;
    for (const Ref& ref : refs) {
        hf::ScfParams params;
        params.Z = ref.Z;
        params.method = ref.method;
        const hf::ScfResult result = hf::RunScf(params);
        const double diff = std::abs(result.eTotal - ref.ePython);
        const bool pass = result.converged && diff < ref.tol;
        allPass &= pass;
        std::printf("  %-2s Z=%3d %-6s: E=%.5f Ha  ref=%.5f  |dif|=%.2e  iters=%d  %s\n", ref.name, ref.Z,
                    ref.method == hf::Method::Lda ? "(lda)" : "(xa)", result.eTotal, ref.ePython, diff,
                    result.iterations, pass ? "OK" : "FAIL");
    }
    std::printf("scf selftest: %s\n", allPass ? "PASS" : "FAIL");
    return allPass;
}

// Headless SCF run: replay every iteration as a frame.
int RunDeck(const std::string& deckPath, const std::string& outDir) {
    fw::Deck deck = fw::Deck::FromFile(deckPath);

    hf::HFSim sim;
    sim.Configure(deck);

    // Final converged orbitals R_nl(r), stacked in the sorted (n, l) order --
    // written as an extra field on the last SCF frame.
    const hf::ScfResult& res = sim.Result();
    const auto& labels = sim.OrbitalLabels();
    const size_t n = res.grid.r.size();
    std::vector<double> stacked(labels.size() * n, 0.0);
    for (size_t k = 0; k < labels.size(); ++k) {
        auto it = res.orbitals.find(labels[k]);
        if (it != res.orbitals.end()) {
            for (size_t i = 0; i < n; ++i) stacked[k * n + i] = it->second[i];
        }
    }

    fw::OutputWriter writer(outDir, sim.Info(), deck);
    const int last = sim.NumIterations() - 1;
    for (int i = 0; i < sim.NumIterations(); ++i) {
        writer.BeginFrame(static_cast<double>(i + 1), i + 1);
        sim.Snapshot(writer);
        if (i == last) {
            writer.WriteField("orbitals_final", stacked.data(), fw::NpyDtype::F8,
                              static_cast<int>(labels.size()), static_cast<int>(n));
        }
        writer.EndFrame();
        sim.Step(1);
    }
    writer.Finish();

    std::printf("%s (Z=%d, %s): E_total = %.6f Ha, %d iterations, %s\n", deck.GetString("title", "atom").c_str(),
                res.Z, res.method == hf::Method::Lda ? "LDA" : "Xalpha", res.eTotal, res.iterations,
                res.converged ? "converged" : "NOT CONVERGED");
    std::printf("  config: %s\n", hf::FormatConfiguration(res.config).c_str());
    return res.converged ? 0 : 1;
}

void PrintUsage() {
    std::fprintf(stderr,
                 "usage:\n"
                 "  01_hartree_fock --selftest\n"
                 "  01_hartree_fock --scf-selftest\n"
                 "  01_hartree_fock --deck <f.toml> --out <dir>\n"
                 "  01_hartree_fock --interactive\n");
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) return ValidateEigensolver() ? 0 : 1;
    if (a.scfSelftest) return RunScfRegressionCheck() ? 0 : 1;

    if (!a.deck.empty() && !a.interactive) {
        if (a.out.empty()) {
            std::fprintf(stderr, "error: --deck needs --out <dir>\n");
            return 2;
        }
        return RunDeck(a.deck, a.out);
    }

    if (a.interactive) {
        hf::HFVisualizerApp app;
        app.Run();
        return 0;
    }

    PrintUsage();
    return 2;
}
