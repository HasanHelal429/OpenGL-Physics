#include "NBodyApp.hpp"
#include "NBodySystem.hpp" // RegisterFmmAdaptersOn

#include "ngrav/DeckSim.hpp"
#include "ngrav/SelfTest.hpp"
#include "ngrav/Solvers.hpp"
#include "ngrav/gpu/GpuQuadtree.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <glad/glad.h>
#include <stb_image_write.h>

#include <glm/glm.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string deck;
    std::string out;
    int frames = 0;
    int substeps = 0;
    bool interactive = false;
    bool selftest = false;
    bool dtSelftest = false;
    bool gpuSelftest = false;
    std::string renderCheck;
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
        else if (s == "--dt-selftest") a.dtSelftest = true;
        else if (s == "--gpu-selftest") a.gpuSelftest = true;
        else if (s == "--render-check") a.renderCheck = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

ngrav::DeckSim<2> MakeSim() {
    return ngrav::DeckSim<2>([](ngrav::System<2>& sys) { nbody2d::RegisterFmmAdaptersOn(sys); });
}

// Phase 12: GPU quadtree (ported from 06_tidal_disruption via GpuBarnesHut's
// 3D port, rewritten for a 4-child quadtree + 2D force law -- see
// GpuQuadtree.hpp's own header comment) vs CPU Barnes-Hut, at theta=0 (walk
// forced down to exact leaf-vs-leaf near field everywhere) and theta=0.5
// (06's own selftest idiom's usual operating point).
bool GpuSelftest() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 2000;
    ngrav::SoA<2> in;
    in.Resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        in.x[ii] = u(rng);
        in.y[ii] = u(rng);
        in.m[ii] = 0.5 + 0.5 * (u(rng) + 1.0);
    }
    const ngrav::PosMassView<2> v = ngrav::ViewOf(in);

    const double G = 1.0, eps = 0.02, eps2 = eps * eps;
    bool ok = true;

    for (const double theta : {0.0, 0.5}) {
        ngrav::StepParams sp;
        sp.G = G;
        sp.soft = ngrav::Softening::Plummer(eps);
        sp.mac.theta = theta;
        ngrav::SoA<2> aCpu;
        ngrav::ComputeAccelBarnesHut<2>(v, sp, aCpu);

        ngrav::gpu::GpuQuadtreeStats stats;
        ngrav::SoA<2> aGpu;
        ngrav::gpu::ComputeAccelGpuQuadtree(v, G, eps2, theta, aGpu, &stats);

        double maxRel = 0.0;
        for (int i = 0; i < n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double rx = aCpu.ax[ii], ry = aCpu.ay[ii];
            const double ex = aGpu.ax[ii] - rx, ey = aGpu.ay[ii] - ry;
            const double rel = std::sqrt(ex * ex + ey * ey) / std::max(std::sqrt(rx * rx + ry * ry), 1e-30);
            maxRel = std::max(maxRel, rel);
        }
        const double tol = (theta == 0.0) ? 1e-4 : 0.2;
        const bool thisOk = maxRel < tol;
        std::printf("  GPU quadtree vs CPU Barnes-Hut, theta=%.1f, max rel err (N=%d): %.3e  (tol %.1e)  %s\n", theta,
                    n, maxRel, tol, thisOk ? "ok" : "WRONG");
        std::printf("  GPU cells=%d levels=%d  stage ms: sort=%.2f build=%.2f mass=%.2f forces=%.2f readback=%.2f\n",
                    stats.cellCount, stats.levelsUsed, stats.sortMs, stats.treeBuildMs, stats.massUpsweepMs,
                    stats.forcesMs, stats.readbackMs);
        ok = ok && thisOk;
    }
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    // The interactive window opens ONLY for a bare invocation or explicit
    // --interactive -- never as an implicit fallback, so a batch/render run
    // whose args don't match a mode fails loudly instead of popping a window.
    const bool wantInteractive = (argc == 1) || a.interactive;

    if (a.selftest) {
        const bool ok = ngrav::CoreSelfTest2D();
        std::printf("\nselftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.dtSelftest) {
        // The adaptive-dt integrator is dimension-generic (ngrav::Integrator<D>);
        // its own validation exercises the 3D eccentric-orbit case (2D's 1/r
        // force law has no closed-form vis-viva orbit to check against).
        const bool ok = ngrav::CoreDtSelfTest();
        std::printf("\ndt-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.gpuSelftest) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const bool ok = GpuSelftest();
        std::printf("\ngpu-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    if (wantInteractive) {
        nbody2d::NBodyApp app;
        app.Run();
        return 0;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr, "03_nbody_gravity_2d: no mode selected. Pass --deck <f.toml> (with --out or\n"
                             "  --render-check), --selftest / --dt-selftest / --gpu-selftest, or --interactive\n"
                             "  (a bare invocation with no args launches the interactive tool).\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (!a.renderCheck.empty()) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const int W = 1000, H = 1000;
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
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        ngrav::DeckSim<2> sim = MakeSim();
        sim.Configure(deck);
        const int warm = deck.GetInt("time.substeps_per_frame", 4) * std::min(deck.GetInt("time.frames", 1), 400);
        if (warm > 1) sim.Step(warm);
        sim.Render(W, H);
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
        std::fprintf(stderr,
                     "usage:\n"
                     "  03_nbody_gravity_2d                       (interactive comparison tool)\n"
                     "  03_nbody_gravity_2d --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  03_nbody_gravity_2d --deck <f.toml> --render-check <out.png>\n"
                     "  03_nbody_gravity_2d --selftest | --dt-selftest | --gpu-selftest\n");
        return 2;
    }

    // No GL context here: fw::RunHeadless only steps the sim and writes
    // .npy / .csv, and every deck-driven solver is CPU. Creating a hidden
    // GLFW context we never use just flashes a window on Windows mid-run
    // (the --render-check path above does need one).
    ngrav::DeckSim<2> sim = MakeSim();
    sim.Configure(deck);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
