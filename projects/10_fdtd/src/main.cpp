#include "Fdtd2D.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string deck;
    std::string out;
    int steps = 0;
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
        else if (s == "--steps") a.steps = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

std::string MakeDeck(int n, double courant, const char* boundary,
                     const char* srcKind, double f0, int steps) {
    char buf[900];
    std::snprintf(buf, sizeof(buf), R"(
[grid]
nx = %d
ny = %d
dx = 1.0
courant = %.4f
[boundary]
type = "%s"
[time]
steps = %d
[[source]]
kind = "%s"
x = %.1f
y = %.1f
f0 = %.4f
amplitude = 1.0
)", n, n, courant, boundary, steps, srcKind,
        0.5 * (n - 1), 0.5 * (n - 1), f0);
    return buf;
}

bool SelfTest() {
    bool ok = true;

    // (1) CFL: `courant` is the full Courant number S = c dt sqrt(sum 1/dx^2);
    //     stable for S <= 1, unstable above it.
    for (auto [S, shouldBlow] :
         {std::pair<double, bool>{0.99, false}, {1.02, true}}) {
        fdtd::Fdtd2D sim;
        fw::Deck d = fw::Deck::FromString(
            MakeDeck(120, S, "pec", "gaussian", 0.06, 1));
        sim.Configure(d);
        sim.Step(2500);
        const double u = sim.TotalEnergy();
        const bool blew = !std::isfinite(u) || u > 1e6;
        std::printf("  CFL S=%.2f -> energy %.3e  %s\n", S, u,
                    blew == shouldBlow ? "ok" : "WRONG");
        if (blew != shouldBlow) ok = false;
    }

    // (2) energy conservation: pulse in a closed PEC box, run long past the
    //     source. The exact discrete energy (H at staggered half-steps) is
    //     conserved to round-off.
    {
        fdtd::Fdtd2D sim;
        fw::Deck d = fw::Deck::FromString(
            MakeDeck(160, 0.7, "pec", "gaussian", 0.05, 1));
        sim.Configure(d);
        sim.Step(700);                       // source has fully turned off
        const double u0 = sim.TotalEnergy();
        double lo = u0, hi = u0;
        for (int blk = 0; blk < 60; ++blk) {
            sim.Step(200);
            lo = std::min(lo, sim.TotalEnergy());
            hi = std::max(hi, sim.TotalEnergy());
        }
        const double spread = (hi - lo) / u0;
        std::printf("  PEC box: U=%.6e  exact-energy spread over 12000 steps "
                    "%.2e\n", u0, spread);
        if (spread > 1e-9) ok = false;
    }

    // (3) propagation speed: a Gaussian pulse from the centre, timed between
    //     two probes at different radii (the difference cancels the source's
    //     own turn-on delay). Speed ~ c, a hair under from FDTD numerical
    //     dispersion at this resolution.
    {
        const int n = 320;
        fdtd::Fdtd2D sim;
        fw::Deck d = fw::Deck::FromString(
            MakeDeck(n, 0.6, "mur", "gaussian", 0.06, 1));
        sim.Configure(d);
        const int pi = n / 2, r1 = 55, r2 = 120;
        auto peakTime = [&](int r) {
            const std::size_t p = static_cast<std::size_t>(n / 2 + r) * n + pi;
            return std::make_pair(p, r);
        };
        const auto [p1, _1] = peakTime(r1);
        const auto [p2, _2] = peakTime(r2);
        double v1max = 0, v2max = 0, t1 = 0, t2 = 0;
        for (int k = 0; k < 4000; ++k) {
            sim.Step(1);
            const double a = std::abs(sim.Ez()[p1]);
            const double b = std::abs(sim.Ez()[p2]);
            if (a > v1max) { v1max = a; t1 = sim.Time(); }
            if (b > v2max) { v2max = b; t2 = sim.Time(); }
        }
        const double speed = double(r2 - r1) / (t2 - t1);
        std::printf("  pulse peak: r=%d at t=%.2f, r=%d at t=%.2f -> speed=%.4f "
                    "(c=1)\n", r1, t1, r2, t2, speed);
        if (!(speed > 0.95 && speed <= 1.01)) ok = false;
    }

    std::printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) return SelfTest() ? 0 : 1;

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  10_fdtd --deck <f.toml> --out <dir> [--steps N]\n"
                     "  10_fdtd --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        std::fprintf(stderr, "interactive view lands in Phase 5\n");
        return 2;
    }
    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    fdtd::Fdtd2D sim;
    sim.Configure(deck);

    // The deck gives total FDTD steps; RunHeadless wants a frame count and a
    // per-frame substep count (frame f writes state, then advances substeps).
    const int totalSteps = a.steps > 0 ? a.steps : deck.GetInt("time.steps", 2000);
    const int substeps = a.substeps > 0 ? a.substeps
                                        : deck.GetInt("time.substeps_per_frame", 4);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = std::max(1, totalSteps / std::max(1, substeps));
    opts.substeps = substeps;
    return fw::RunHeadless(sim, deck, opts);
}
