#include "ChargePath.hpp"
#include "LwFields.hpp"
#include "RetardedFieldsSim.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using lw::kPi;

struct Args {
    std::string deck, out, renderCheck;
    int steps = 0, substeps = 0;
    bool interactive = false, selftest = false, gpuSelftest = false, cpu = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--steps") a.steps = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--gpu-selftest") a.gpuSelftest = true;
        else if (s == "--cpu") a.cpu = true;
        else if (s == "--render-check") a.renderCheck = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

lw::PathFn AnalyticPath(const lw::ChargePath& p) {
    return [p](double tau) {
        return lw::PathSample{p.r(tau), p.v(tau), p.a(tau)};
    };
}

bool SelfTest() {
    bool ok = true;
    const double c = 1.0, eps0 = 1.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // (1) static charge: exact Coulomb, B = 0.
    {
        lw::ChargePath p;
        p.kind = lw::ChargePath::Static;
        const auto fn = AnalyticPath(p);
        double maxErr = 0.0, maxB = 0.0;
        for (glm::dvec3 x : {glm::dvec3(3, 0, 0), glm::dvec3(0, 5, 2),
                             glm::dvec3(-4, 1, -3), glm::dvec3(2, -2, 1)}) {
            const lw::LwResult r = lw::LwFields(fn, x, 100.0, 1.0, c, eps0, nan);
            const double R = glm::length(x);
            const glm::dvec3 Ecoul = x / (4.0 * kPi * eps0 * R * R * R);
            maxErr = std::max(maxErr, glm::length(r.E - Ecoul) / glm::length(Ecoul));
            maxB = std::max(maxB, glm::length(r.B));
        }
        std::printf("  static Coulomb: max rel err %.2e   max |B| %.2e\n", maxErr, maxB);
        if (maxErr > 1e-12 || maxB > 1e-14) ok = false;
    }

    // (2) uniformly moving charge: matches the present-position closed form
    //     E = q (1-b^2) Rhat_now / [4 pi eps0 (1 - b^2 sin^2 psi)^{3/2} R_now^2].
    {
        const double beta = 0.3;
        lw::ChargePath p;
        p.kind = lw::ChargePath::Uniform;
        p.v0 = glm::dvec3(beta, 0.0, 0.0);
        const auto fn = AnalyticPath(p);
        const double t = 50.0;
        double maxErr = 0.0;
        for (glm::dvec3 x : {glm::dvec3(beta * t + 4, 0, 0),
                             glm::dvec3(beta * t, 3, 0),
                             glm::dvec3(beta * t + 2, 2, 1),
                             glm::dvec3(beta * t - 3, -1, 2)}) {
            const lw::LwResult r = lw::LwFields(fn, x, t, 1.0, c, eps0, nan);
            const glm::dvec3 Rn = x - glm::dvec3(beta * t, 0, 0);
            const double Rnm = glm::length(Rn);
            const glm::dvec3 nn = Rn / Rnm;
            const double sin2 = 1.0 - nn.x * nn.x;   // angle from v (x axis)
            const double b2 = beta * beta;
            const glm::dvec3 Eexact = (1.0 - b2) * nn /
                (4.0 * kPi * eps0 * std::pow(1.0 - b2 * sin2, 1.5) * Rnm * Rnm);
            maxErr = std::max(maxErr, glm::length(r.E - Eexact) / glm::length(Eexact));
        }
        std::printf("  uniform motion (beta=0.3): max rel err %.2e\n", maxErr);
        if (maxErr > 1e-9) ok = false;
    }

    // (3) Larmor: non-relativistic circular motion, Poynting flux over a
    //     Fibonacci sphere at 4 wavelengths vs P = q^2 a^2 / (6 pi eps0 c^3).
    {
        lw::ChargePath p;
        p.kind = lw::ChargePath::Circular;
        p.omega = 1.0;
        p.radius = 5e-4;                      // v/c = omega*R = 5e-4
        const auto fn = AnalyticPath(p);
        const double accel = p.omega * p.omega * p.radius;
        const double Plarmor = accel * accel / (6.0 * kPi);   // q=c=eps0=1

        const double Rs = 4.0 * (2.0 * kPi / p.omega);         // 4 wavelengths
        const double t = Rs + 10.0;
        const int N = 2000;
        const double golden = kPi * (3.0 - std::sqrt(5.0));
        double flux = 0.0;
        for (int k = 0; k < N; ++k) {
            const double z = 1.0 - 2.0 * (k + 0.5) / N;
            const double rad = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double th = golden * k;
            const glm::dvec3 dir(rad * std::cos(th), rad * std::sin(th), z);
            const glm::dvec3 x = Rs * dir;
            const lw::LwResult r = lw::LwFields(fn, x, t, 1.0, 1.0, 1.0, nan);
            flux += glm::dot(glm::cross(r.E, r.B), dir);
        }
        flux *= 4.0 * kPi * Rs * Rs / N;
        const double rel = std::abs(flux - Plarmor) / Plarmor;
        std::printf("  Larmor: sphere flux %.4e vs P=%.4e   rel err %.2e\n",
                    flux, Plarmor, rel);
        if (rel > 5e-3) ok = false;
    }

    std::printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool GpuSelfTest() {
    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    bool ok = true;

    const char* decks[] = {
        R"(
[domain]
nx = 200
ny = 200
lx = 24
ly = 24
[time]
dt = 0.1
steps = 1
[[charge]]
q = 1.0
path = "circular"
radius = 2.0
omega = 0.5
)",
        R"(
[domain]
nx = 200
ny = 200
lx = 24
ly = 24
[time]
dt = 0.1
steps = 1
[[charge]]
q = 1.0
path = "linear_oscillator"
amplitude = 2.0
omega = 0.6
axis = [1, 0, 0]
[[charge]]
q = -1.0
path = "circular"
radius = 3.0
omega = 0.4
phase = 1.5
)"};

    for (const char* dk : decks) {
        lw::RetardedFieldsSim cpu, gpu;
        fw::Deck d1 = fw::Deck::FromString(dk);
        fw::Deck d2 = fw::Deck::FromString(dk);
        cpu.Configure(d1);
        gpu.Configure(d2);
        cpu.ForceCpu(true);

        cpu.Step(30);   gpu.Step(30);          // warm-start the tret buffer
        cpu.ComputeFieldCpu();
        gpu.ComputeField();

        const auto& ax = cpu.Ex();  const auto& bx = gpu.Ex();
        const auto& ay = cpu.Ey();  const auto& by = gpu.Ey();
        double amax = 0.0;
        for (std::size_t k = 0; k < ax.size(); ++k)
            amax = std::max(amax, std::hypot(ax[k], ay[k]));

        // The GPU shader runs in fp32; within a cell or two of a charge the
        // 1/R^2 near field amplifies tiny retarded-time differences without
        // bound. Compare on the radiation-relevant region by masking the
        // near-singularity spike (|E| > 20% of the grid maximum).
        const double cut = 0.20 * amax;
        double num2 = 0.0, den2 = 0.0, relmax = 0.0;
        for (std::size_t k = 0; k < ax.size(); ++k) {
            const double a = std::hypot(ax[k], ay[k]);
            if (a > cut || a == 0.0) continue;
            const double d = std::hypot(ax[k] - bx[k], ay[k] - by[k]);
            num2 += d * d;
            den2 += a * a;
            relmax = std::max(relmax, d / a);
        }
        const double relL2 = std::sqrt(num2 / den2);
        std::printf("  %d charge(s): max|E| %.3e  masked rel-L2 %.2e  masked rel-max %.2e\n",
                    (int)fw::Deck::FromString(dk).GetTables("charge").size(),
                    amax, relL2, relmax);
        if (relL2 > 3e-3 || relmax > 3e-2) ok = false;
    }

    std::printf("gpu-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);
    if (a.selftest) return SelfTest() ? 0 : 1;
    if (a.gpuSelftest) return GpuSelfTest() ? 0 : 1;

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  11_retarded_fields --deck <f.toml> --out <dir> [--cpu]\n"
                     "  11_retarded_fields --interactive --deck <f.toml>\n"
                     "  11_retarded_fields --deck <f.toml> --render-check <png>\n"
                     "  11_retarded_fields --selftest | --gpu-selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        lw::RetardedFieldsSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "Lienard-Wiechert"));
        app.Run();
        return 0;
    }

    if (!a.renderCheck.empty()) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const int W = 1000, H = 1000;
        GLuint fbo = 0, col = 0, dep = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &col);
        glBindTexture(GL_TEXTURE_2D, col);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
        glGenRenderbuffers(1, &dep);
        glBindRenderbuffer(GL_RENDERBUFFER, dep);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dep);
        glViewport(0, 0, W, H);
        lw::RetardedFieldsSim sim;
        sim.Configure(deck);
        sim.Step(deck.GetInt("render_check.warm_steps", 60));
        sim.Render(W, H);
        sim.Render(W, H);
        glFinish();
        std::vector<unsigned char> px((size_t)W * H * 4), flip(px.size());
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
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
    lw::RetardedFieldsSim sim;
    sim.Configure(deck);
    if (a.cpu) sim.ForceCpu(true);

    const int totalSteps = a.steps > 0 ? a.steps : deck.GetInt("time.steps", 400);
    const int substeps = a.substeps > 0 ? a.substeps
                                        : deck.GetInt("time.substeps_per_frame", 1);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = std::max(1, totalSteps / std::max(1, substeps));
    opts.substeps = substeps;
    return fw::RunHeadless(sim, deck, opts);
}
