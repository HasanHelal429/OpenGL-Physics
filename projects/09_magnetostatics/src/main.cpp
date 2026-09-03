#include "FieldSolver.hpp"
#include "Grid.hpp"
#include "MagnetostaticsSim.hpp"
#include "Multigrid.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>
#include <stb_image_write.h>   // implementation lives in framework/src/SimApp.cpp

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    bool mgScaling = false;
    std::string renderCheck;   // path for a one-frame offscreen PNG dump
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
        else if (s == "--mg-scaling") a.mgScaling = true;
        else if (s == "--render-check") a.renderCheck = next();
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

// Multigrid scaling check: the same manufactured problem, solved by the
// V-cycle at a range of grid sizes. Text-book multigrid converges in a
// V-cycle count that is essentially independent of the grid -- the property
// that makes live re-solving practical. Also confirms the V-cycle reaches
// the same discrete solution as RB-GS (interior L2 error still ~h^2).
bool MgScaling() {
    const double Lx = 2.0, Ly = 2.0;
    const double kx = kPi / Lx, ky = kPi / Ly;
    const int sizes[] = {64, 128, 256, 512, 1024};

    auto exact = [&](double x, double y) {
        return std::sin(kx * (x + 0.5 * Lx)) * std::sin(ky * (y + 0.5 * Ly));
    };

    std::printf("mg-scaling: V-cycles to relative residual 1e-9\n");
    std::printf("    nx   V-cycles   levels   L2 error     ms\n");

    double prevErr = 0.0;
    bool ok = true;
    for (int idx = 0; idx < 5; ++idx) {
        mag::Grid g;
        g.nx = g.ny = sizes[idx];
        g.lx = Lx;
        g.ly = Ly;

        std::vector<double> rhs(g.count(), 0.0), u(g.count(), 0.0), fixed(g.count(), 0.0);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const std::size_t p = g.idx(i, j);
                const double a = exact(g.x(i), g.y(j));
                rhs[p] = -(kx * kx + ky * ky) * a;
                if (g.onBoundary(i, j)) fixed[p] = a;
            }

        mag::MultigridOptions opt;
        opt.tol = 1e-9;
        opt.maxCycles = 50;
        const auto t0 = std::chrono::steady_clock::now();
        const mag::MultigridResult r =
            mag::SolvePoissonMultigrid(g, rhs, fixed, u, opt);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - t0)
                              .count();

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
        std::printf("  %5d   %6d   %6d   %.3e  %7.1f   %s\n",
                    g.nx, r.cycles, r.levels, l2, ms,
                    idx == 0 ? "" :
                    (ratio > 3.5 && ratio < 4.5 ? "(2nd order)" : "(?? order)"));
        if (!r.converged) ok = false;
        if (r.cycles > 15) ok = false;              // "flat" == small, bounded
        if (idx > 0 && (ratio < 3.3 || ratio > 4.7)) ok = false;
        prevErr = l2;
    }
    std::printf("mg-scaling: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        return SelfTest() ? 0 : 1;
    }
    if (a.mgScaling) {
        return MgScaling() ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  09_magnetostatics --deck <f.toml> --out <dir> [--frames N]\n"
                     "  09_magnetostatics --interactive --deck <f.toml>\n"
                     "  09_magnetostatics --deck <f.toml> --render-check <out.png>\n"
                     "  09_magnetostatics --selftest      (field-solver convergence)\n"
                     "  09_magnetostatics --mg-scaling    (multigrid V-cycle scaling)\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        mag::MagnetostaticsSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "2D Magnetostatics"));
        app.Run();
        return 0;
    }

    // Offscreen one-frame render to PNG -- the only way to eyeball the
    // interactive view in a headless environment (see 08_compressible_fluid's
    // FBO-to-PNG note). Bigger than GLContext::CreateHidden's 64x64 default.
    if (!a.renderCheck.empty()) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const int W = 1100, H = 850;
        GLuint fbo = 0, color = 0, depth = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glGenRenderbuffers(1, &depth);
        glBindRenderbuffer(GL_RENDERBUFFER, depth);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth);
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
            std::fprintf(stderr, "render-check: incomplete FBO\n");
            return 1;
        }
        glViewport(0, 0, W, H);
        mag::MagnetostaticsSim sim;
        sim.Configure(deck);
        sim.Render(W, H);
        sim.Render(W, H);   // second frame: field-lines/vbo now populated
        glFinish();
        std::vector<unsigned char> px(static_cast<size_t>(W) * H * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        std::vector<unsigned char> flip(px.size());
        for (int y = 0; y < H; ++y)
            std::memcpy(&flip[(size_t)y * W * 4], &px[(size_t)(H - 1 - y) * W * 4], (size_t)W * 4);
        stbi_write_png(a.renderCheck.c_str(), W, H, 4, flip.data(), W * 4);
        std::printf("render-check -> %s\n", a.renderCheck.c_str());
        return 0;
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
