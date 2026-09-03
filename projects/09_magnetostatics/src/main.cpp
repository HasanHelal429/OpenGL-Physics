#include "FieldSolver.hpp"
#include "Grid.hpp"
#include "MagnetostaticsSim.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Args {
    std::string deck;
    std::string out;
    int frames = 0;
    int substeps = 0;
    bool interactive = false;
    bool selftest = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--frames") a.frames = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

// Manufactured-solution convergence check for the field solver: the exact
//   A(x, y) = sin(pi u / Lx) sin(pi v / Ly),   u = x + Lx/2, v = y + Ly/2
// (zero on the domain boundary), with the analytic Laplacian fed in as the
// right-hand side and the exact boundary values imposed as Dirichlet data.
// The interior L2 error should fall as h^2 -- ratio ~4 per grid halving.
bool SelfTest() {
    const double Lx = 2.0, Ly = 2.0;
    const double kx = kPi / Lx, ky = kPi / Ly;
    const int sizes[] = {32, 64, 128, 256};

    auto exact = [&](double x, double y) {
        return std::sin(kx * (x + 0.5 * Lx)) * std::sin(ky * (y + 0.5 * Ly));
    };

    std::printf("selftest: manufactured-solution convergence\n");
    std::printf("   nx      L2 error      ratio\n");

    double prevErr = 0.0;
    bool ok = true;
    for (int idx = 0; idx < 4; ++idx) {
        mag::Grid g;
        g.nx = g.ny = sizes[idx];
        g.lx = Lx;
        g.ly = Ly;

        std::vector<double> rhs(g.count(), 0.0), u(g.count(), 0.0);
        std::vector<unsigned char> mask;
        std::vector<double> fixed;
        mag::MakeEdgeDirichlet(g, mask, fixed);

        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t p = g.idx(i, j);
                const double a = exact(g.x(i), g.y(j));
                rhs[p] = -(kx * kx + ky * ky) * a;   // analytic Laplacian
                if (mask[p]) fixed[p] = a;            // exact Dirichlet data
            }

        // Relative-residual 1e-9 puts the iterative error well below the
        // discretisation error at every size here; plain SOR stalls near
        // 1e-11 at 256^2 anyway (that is what Phase 2 multigrid fixes).
        mag::SolveOptions opt;
        opt.tol = 1e-9;
        opt.maxIterations = 40000;
        const mag::SolveResult r = mag::SolvePoisson(g, rhs, mask, fixed, u, opt);

        double se = 0.0;
        int n = 0;
        for (int j = 1; j < g.ny - 1; ++j)
            for (int i = 1; i < g.nx - 1; ++i) {
                const double e = u[g.idx(i, j)] - exact(g.x(i), g.y(j));
                se += e * e;
                ++n;
            }
        const double l2 = std::sqrt(se / n);
        const double ratio = idx == 0 ? 0.0 : prevErr / l2;
        std::printf("  %4d   %.6e   %s%.3f   (%d iters, res %.1e%s)\n",
                    g.nx, l2, idx == 0 ? "  -  " : "", ratio,
                    r.iterations, r.residual, r.converged ? "" : " NOTCONV");
        if (idx > 0 && (ratio < 3.6 || ratio > 4.4)) ok = false;
        if (r.residual / (kx * kx + ky * ky) > 1e-6) ok = false;
        prevErr = l2;
    }

    std::printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        return SelfTest() ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  09_magnetostatics --deck <f.toml> --out <dir> [--frames N]\n"
                     "  09_magnetostatics --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        std::fprintf(stderr, "interactive view lands in Phase 2\n");
        return 2;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    mag::MagnetostaticsSim sim;
    sim.Configure(deck);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames > 0 ? a.frames : 1;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
