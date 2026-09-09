#include "NBodyApp.hpp"
#include "NBodySystem.hpp" // RegisterFmmAdaptersOn
#include "SphericalFmm.hpp"

#include "ngrav/DeckSim.hpp"
#include "ngrav/MutualFmm.hpp"
#include "ngrav/SelfTest.hpp"
#include "ngrav/Solvers.hpp"
#include "ngrav/ic/Plummer.hpp"
#include "ngrav/gpu/GpuBarnesHut.hpp"
#include "ngrav/gpu/GpuDirect.hpp"

#include <chrono>

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <glad/glad.h>
#include <stb_image_write.h> // implementation lives in framework/src/SimApp.cpp

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
    bool fmmSelftest = false;
    bool sphericalSelftest = false;
    bool gpuSelftest = false;
    bool rungSelftest = false;
    std::string renderCheck;

    // --bench: force-evaluation timing + accuracy for one solver, for the
    // Studies scaling suite. `--bench <solver> --bench-n <N> [--bench-theta T]
    // [--bench-reps R]`; prints one CSV line to stdout.
    std::string bench;
    int benchN = 10000;
    double benchTheta = 0.5;
    int benchReps = 5;
    int benchOrder = 5; // spherical_fmm expansion order
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
        else if (s == "--fmm-selftest") a.fmmSelftest = true;
        else if (s == "--spherical-selftest") a.sphericalSelftest = true;
        else if (s == "--gpu-selftest") a.gpuSelftest = true;
        else if (s == "--rung-selftest") a.rungSelftest = true;
        else if (s == "--bench") a.bench = next();
        else if (s == "--bench-n") a.benchN = std::atoi(next());
        else if (s == "--bench-theta") a.benchTheta = std::atof(next());
        else if (s == "--bench-reps") a.benchReps = std::atoi(next());
        else if (s == "--bench-order") a.benchOrder = std::atoi(next());
        else if (s == "--render-check") a.renderCheck = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

ngrav::DeckSim<3> MakeSim() {
    return ngrav::DeckSim<3>([](ngrav::System<3>& sys) { nbody::RegisterFmmAdaptersOn(sys); });
}

// Phase 7: runtime expansion order (was compile-time kOrder=5) + in-place
// M2L accumulation. Validates: (a) accuracy improves monotonically as p
// grows from 2 up to where the fixed opening-angle theta's own truncation
// (the theta^(p+1) FMM error bound) starts to dominate over the expansion
// truncation, at which point it should plateau rather than diverge; (b)
// p=5 (the old fixed default) still gives a sane, non-broken result.
bool SphericalSelfTest() {
    bool ok = true;
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 2000;
    std::vector<glm::dvec3> pos(static_cast<size_t>(n));
    std::vector<double> mass(static_cast<size_t>(n));
    for (auto& p : pos) p = {u(rng), u(rng), u(rng)};
    for (auto& m : mass) m = 0.5 + 0.5 * (u(rng) + 1.0);
    const double eps = 0.01, eps2 = eps * eps;

    double refAx = 0.0;
    for (int j = 1; j < n; ++j) {
        const glm::dvec3 d = pos[static_cast<size_t>(j)] - pos[0];
        const double r2 = glm::dot(d, d) + eps2;
        const double invR = 1.0 / std::sqrt(r2), invR3 = invR * invR * invR;
        refAx += mass[static_cast<size_t>(j)] * invR3 * d.x;
    }

    double prevErr = 1e30;
    bool sawImprovement = false;
    for (int p : {2, 4, 6, 8, 10}) {
        std::vector<glm::dvec3> accel;
        nbody::ComputeAccelSphericalFmm(pos, mass, 1.0, eps, 0.5, accel, nullptr, p);
        const double rel = std::abs(accel[0].x - refAx) / std::abs(refAx);
        std::printf("  p=%2d  rel err vs Direct (particle 0) = %.4e\n", p, rel);
        if (rel < prevErr * 0.99) sawImprovement = true; // strictly improving somewhere in the sweep
        prevErr = rel;
    }
    if (!sawImprovement) {
        std::printf("  WRONG: accuracy never improved as p increased\n");
        ok = false;
    }
    std::printf("  monotonic-ish convergence with p: %s\n", sawImprovement ? "ok" : "WRONG");

    // p=5 sanity: not wildly broken (loose bound -- this is a smoke test,
    // the convergence sweep above is the real check).
    {
        std::vector<glm::dvec3> accel;
        nbody::ComputeAccelSphericalFmm(pos, mass, 1.0, eps, 0.5, accel, nullptr, 5);
        const double rel = std::abs(accel[0].x - refAx) / std::abs(refAx);
        const bool good = rel < 1e-2;
        std::printf("  p=5 (default) rel err = %.4e  %s\n", rel, good ? "ok" : "WRONG");
        if (!good) ok = false;
    }
    return ok;
}

// Phase 10: GPU direct O(N^2) sum vs the CPU fp64 reference (ComputeAccelDirect
// -- the exact same formula, term for term, so a mismatch can only come from
// the GPU pipeline or the fp32-vs-fp64 gap, not a second derivation of the
// physics). Requires an active GL context (created by the caller before this
// runs, same as --render-check).
bool GpuSelftest() {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 2000;
    ngrav::SoA<3> in;
    in.Resize(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        in.x[ii] = u(rng);
        in.y[ii] = u(rng);
        in.z[ii] = u(rng);
        in.m[ii] = 0.5 + 0.5 * (u(rng) + 1.0);
    }
    const ngrav::PosMassView<3> v = ngrav::ViewOf(in);

    const double G = 1.0, eps = 0.02, eps2 = eps * eps;

    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(eps);
    ngrav::SoA<3> aCpu;
    ngrav::ComputeAccelDirect<3>(v, sp, aCpu);

    ngrav::gpu::GpuDirectStats stats;
    ngrav::SoA<3> aGpu;
    ngrav::gpu::ComputeAccelGpuDirect(v, G, eps2, aGpu, &stats);

    double maxRel = 0.0;
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        const double rx = aCpu.ax[ii], ry = aCpu.ay[ii], rz = aCpu.az[ii];
        const double ex = aGpu.ax[ii] - rx, ey = aGpu.ay[ii] - ry, ez = aGpu.az[ii] - rz;
        const double rel =
            std::sqrt(ex * ex + ey * ey + ez * ez) / std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30);
        maxRel = std::max(maxRel, rel);
    }
    bool ok = maxRel < 1e-4;
    std::printf("  GPU fp32 vs CPU fp64 direct, max rel err (N=%d): %.3e  (tol 1.0e-04)  %s\n", n, maxRel,
                ok ? "ok" : "WRONG");
    std::printf("  GPU stage ms: upload=%.2f dispatch=%.2f readback=%.2f\n", stats.uploadMs, stats.dispatchMs,
                stats.readbackMs);

    // Phase 11: GPU Barnes-Hut (ported from 06_tidal_disruption's tree +
    // gravity-walk kernels) vs CPU Barnes-Hut, at theta=0 (should force the
    // walk down to exact leaf-vs-leaf near field everywhere, same idiom as
    // the CPU BH theta=0 check elsewhere in this file) and at theta=0.5
    // (06's own selftest idiom's usual operating point -- opening-angle
    // approximation error, not a pipeline bug, dominates here).
    for (const double theta : {0.0, 0.5}) {
        ngrav::StepParams bhSp;
        bhSp.G = G;
        bhSp.soft = ngrav::Softening::Plummer(eps);
        bhSp.mac.theta = theta;
        ngrav::SoA<3> aCpuBh;
        ngrav::ComputeAccelBarnesHut<3>(v, bhSp, aCpuBh);

        ngrav::gpu::GpuBarnesHutStats bhStats;
        ngrav::SoA<3> aGpuBh;
        ngrav::gpu::ComputeAccelGpuBarnesHut(v, G, eps2, theta, aGpuBh, &bhStats);

        double maxRelBh = 0.0;
        for (int i = 0; i < n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double rx = aCpuBh.ax[ii], ry = aCpuBh.ay[ii], rz = aCpuBh.az[ii];
            const double ex = aGpuBh.ax[ii] - rx, ey = aGpuBh.ay[ii] - ry, ez = aGpuBh.az[ii] - rz;
            const double rel =
                std::sqrt(ex * ex + ey * ey + ez * ez) / std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30);
            maxRelBh = std::max(maxRelBh, rel);
        }
        const double tol = (theta == 0.0) ? 1e-4 : 0.2;
        const bool bhOk = maxRelBh < tol;
        std::printf("  GPU Barnes-Hut vs CPU Barnes-Hut, theta=%.1f, max rel err (N=%d): %.3e  (tol %.1e)  %s\n",
                    theta, n, maxRelBh, tol, bhOk ? "ok" : "WRONG");
        std::printf("  GPU BH cells=%d levels=%d  stage ms: sort=%.2f build=%.2f mass=%.2f forces=%.2f readback=%.2f\n",
                    bhStats.cellCount, bhStats.levelsUsed, bhStats.sortMs, bhStats.treeBuildMs, bhStats.massUpsweepMs,
                    bhStats.forcesMs, bhStats.readbackMs);
        ok = ok && bhOk;
    }
    return ok;
}

// --bench: time one solver's force evaluation on an N-body Plummer sphere
// and report its mean per-eval wall time + mean relative force error vs the
// exact Direct sum. One CSV line to stdout (header printed to stderr), for
// Studies/nbody_gravity/scaling_and_crossover. Not a correctness gate --
// that's the --*-selftest flags; this is a measurement tool.
int RunBench(const Args& a) {
    ngrav::ic::PlummerParams pp;
    pp.n = a.benchN;
    pp.seed = 1;
    ngrav::SoA<3> s = ngrav::ic::Plummer<3>(pp);
    const ngrav::PosMassView<3> v = ngrav::ViewOf(s);

    ngrav::StepParams sp;
    sp.G = 1.0;
    sp.soft = ngrav::Softening::Plummer(0.02);
    sp.mac.theta = a.benchTheta;

    // Exact reference (skip for very large N -- O(N^2) gets slow; then the
    // error column is reported as -1).
    ngrav::SoA<3> aRef;
    const bool haveRef = (a.benchN <= 60000);
    if (haveRef) ngrav::ComputeAccelDirect<3>(v, sp, aRef);

    auto timeIt = [&](auto&& fn) {
        ngrav::SoA<3> out;
        fn(out); // warm
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < a.benchReps; ++r) fn(out);
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / a.benchReps;
        double meanRel = -1.0;
        if (haveRef) {
            double sum = 0.0;
            for (int i = 0; i < a.benchN; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                const double rx = aRef.ax[ii], ry = aRef.ay[ii], rz = aRef.az[ii];
                const double ex = out.ax[ii] - rx, ey = out.ay[ii] - ry, ez = out.az[ii] - rz;
                sum += std::sqrt(ex * ex + ey * ey + ez * ez) /
                       std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30);
            }
            meanRel = sum / a.benchN;
        }
        return std::make_pair(ms, meanRel);
    };

    std::pair<double, double> result{0.0, -1.0};
    if (a.bench == "direct")
        result = timeIt([&](ngrav::SoA<3>& out) { ngrav::ComputeAccelDirect<3>(v, sp, out); });
    else if (a.bench == "barnes_hut" || a.bench == "bh")
        result = timeIt([&](ngrav::SoA<3>& out) { ngrav::ComputeAccelBarnesHut<3>(v, sp, out); });
    else if (a.bench == "fmm")
        result = timeIt([&](ngrav::SoA<3>& out) { ngrav::ComputeAccelMutualFmm(v, sp, out); });
    else if (a.bench == "spherical_fmm") {
        result = timeIt([&](ngrav::SoA<3>& out) {
            std::vector<glm::dvec3> pos(static_cast<std::size_t>(a.benchN));
            for (int i = 0; i < a.benchN; ++i)
                pos[static_cast<std::size_t>(i)] =
                    glm::dvec3(s.x[static_cast<std::size_t>(i)], s.y[static_cast<std::size_t>(i)],
                               s.z[static_cast<std::size_t>(i)]);
            std::vector<glm::dvec3> acc;
            nbody::ComputeAccelSphericalFmm(pos, s.m, sp.G, sp.soft.eps, sp.mac.theta, acc, nullptr, a.benchOrder);
            out.ResizeAccel(static_cast<std::size_t>(a.benchN));
            for (int i = 0; i < a.benchN; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                out.ax[ii] = acc[ii].x;
                out.ay[ii] = acc[ii].y;
                out.az[ii] = acc[ii].z;
            }
        });
    } else {
        std::fprintf(stderr, "bench: unknown solver '%s' (direct|barnes_hut|fmm|spherical_fmm)\n", a.bench.c_str());
        return 2;
    }

    std::fprintf(stderr, "solver,n,theta,order,ms_per_eval,mean_rel_err\n");
    std::printf("%s,%d,%.3f,%d,%.4f,%.6e\n", a.bench.c_str(), a.benchN, a.benchTheta, a.benchOrder, result.first,
                result.second);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    // The interactive comparison tool opens a GLFW window and blocks. That
    // must happen ONLY for a bare invocation (no args at all) or an explicit
    // --interactive -- never as an implicit fallback, so that a batch/bench
    // run whose args somehow don't match a mode fails loudly instead of
    // silently popping a window on someone's desktop mid-sweep.
    const bool wantInteractive = (argc == 1) || a.interactive;

    if (a.selftest) {
        const bool okCore = ngrav::CoreSelfTest3D();
        const bool okIc = ngrav::CoreIcSelfTest(); // P13 -- realistic IC generators (3D)
        const bool ok = okCore && okIc;
        std::printf("\nselftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.dtSelftest) {
        const bool ok = ngrav::CoreDtSelfTest();
        std::printf("\ndt-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.fmmSelftest) {
        const bool ok = ngrav::CoreFmmSelfTest();
        std::printf("\nfmm-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.sphericalSelftest) {
        const bool ok = SphericalSelfTest();
        std::printf("\nspherical-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.rungSelftest) {
        const bool ok = ngrav::CoreRungSelfTest();
        std::printf("\nrung-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (a.gpuSelftest) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const bool ok = GpuSelftest();
        std::printf("\ngpu-selftest: %s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }
    if (!a.bench.empty()) return RunBench(a);

    // Interactive comparison tool: bare invocation or explicit --interactive
    // only (see wantInteractive above). The deck, if any, currently just
    // picks the starting scenario via NBodyApp's own hook.
    if (wantInteractive) {
        nbody::NBodyApp app;
        app.Run();
        return 0;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr, "02_nbody_gravity: no mode selected. Pass --deck <f.toml> (with --out or\n"
                             "  --render-check), a --*-selftest flag, --bench, or --interactive\n"
                             "  (a bare invocation with no args launches the interactive tool).\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

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
        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glEnable(GL_DEPTH_TEST);

        ngrav::DeckSim<3> sim = MakeSim();
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
                     "  02_nbody_gravity                        (interactive comparison tool)\n"
                     "  02_nbody_gravity --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  02_nbody_gravity --deck <f.toml> --render-check <out.png>\n"
                     "  02_nbody_gravity --selftest | --dt-selftest | --fmm-selftest | --spherical-selftest |\n"
                     "                   --gpu-selftest | --rung-selftest\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    ngrav::DeckSim<3> sim = MakeSim();
    sim.Configure(deck);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
