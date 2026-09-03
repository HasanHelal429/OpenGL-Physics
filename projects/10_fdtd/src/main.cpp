#include "Fdtd2D.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct Args {
    std::string deck;
    std::string out;
    int steps = 0;
    int substeps = 0;
    bool interactive = false;
    bool selftest = false;
    bool gpuSelftest = false;
    bool cpmlSelftest = false;
    bool fresnelSelftest = false;
    bool gpu = false;
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
        else if (s == "--cpml-selftest") a.cpmlSelftest = true;
        else if (s == "--fresnel-selftest") a.fresnelSelftest = true;
        else if (s == "--gpu") a.gpu = true;
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

// A vertical strip domain with the source `srcFromTop` cells below the top
// edge and a probe `probeFromTop` cells below the top -- so the near-top
// physics is identical for any ny, and only a reflection off the FAR (bottom)
// boundary distinguishes a short domain from a long reference one.
std::string MakeStripDeck(int nx, int ny, const char* boundary, int pml,
                          int srcFromTop, double f0) {
    char buf[700];
    std::snprintf(buf, sizeof(buf), R"(
[grid]
nx = %d
ny = %d
dx = 1.0
courant = 0.5
[boundary]
type = "%s"
pml_cells = %d
[time]
steps = 1
[[source]]
kind = "gaussian"
x = %.1f
y = %.1f
f0 = %.4f
amplitude = 1.0
)", nx, ny, boundary, pml, 0.5 * (nx - 1), double(ny - 1 - srcFromTop), f0);
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

// GPU compute backend vs the CPU reference: step both with the same deck and
// compare Ez. float32 GPU against float64 CPU, so a small drift accumulates.
bool GpuSelfTest() {
    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    bool ok = true;

    for (const char* boundary : {"pec", "mur"}) {
        const int n = 220;
        std::string deckStr = MakeDeck(n, 0.6, boundary, "gaussian", 0.06, 1);
        deckStr += "[[source]]\nkind=\"gaussian\"\nx=70.0\ny=140.0\nf0=0.09\n";

        fdtd::Fdtd2D cpu, gpu;
        fw::Deck d1 = fw::Deck::FromString(deckStr);
        fw::Deck d2 = fw::Deck::FromString(deckStr);
        cpu.Configure(d1);
        gpu.Configure(d2);
        gpu.ForceBackend(true);

        const int steps = 1200;
        cpu.Step(steps);
        gpu.Step(steps);
        gpu.SyncFromGpu();

        const auto& a = cpu.Ez();
        const auto& b = gpu.Ez();
        double maxAbs = 0.0, maxDiff = 0.0;
        for (std::size_t k = 0; k < a.size(); ++k) {
            maxAbs = std::max(maxAbs, std::abs(a[k]));
            maxDiff = std::max(maxDiff, std::abs(a[k] - b[k]));
        }
        const double rel = maxDiff / maxAbs;
        std::printf("  %-4s  %d steps:  max|Ez_cpu|=%.4e  max|cpu-gpu|=%.2e  "
                    "rel %.2e\n", boundary, steps, maxAbs, maxDiff, rel);
        if (rel > 2e-4) ok = false;
    }

    std::printf("gpu-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// CPML boundary reflection: run a short CPML domain and a long reference
// domain that share their near-top region; the difference at a probe, before
// the reference's own far boundary can respond, is the short domain's
// PML reflection.
bool CpmlSelfTest() {
    bool ok = true;
    const int nx = 160, pml = 12;
    const int srcFromTop = 45, probeFromTop = 75;
    const double f0 = 0.06;

    const int nyShort = 150;     // bottom PML inner edge ~63 cells below probe
    const int nyLong = 600;      // bottom ~525 cells below probe -- silent in-window

    auto run = [&](int ny, int steps, std::vector<double>& trace) {
        fdtd::Fdtd2D sim;
        fw::Deck d = fw::Deck::FromString(
            MakeStripDeck(nx, ny, "cpml", pml, srcFromTop, f0));
        sim.Configure(d);
        const std::size_t pr =
            static_cast<std::size_t>(ny - 1 - probeFromTop) * nx + nx / 2;
        trace.clear();
        for (int k = 0; k < steps; ++k) {
            sim.Step(1);
            trace.push_back(sim.Ez()[pr]);
        }
    };

    const int steps = 700;       // ~ round trip to the short domain's far wall
    std::vector<double> a, b;
    run(nyShort, steps, a);
    run(nyLong, steps, b);

    double incident = 0.0, resid = 0.0;
    for (int k = 0; k < steps; ++k) {
        incident = std::max(incident, std::abs(b[k]));
        resid = std::max(resid, std::abs(a[k] - b[k]));
    }
    const double dB = 20.0 * std::log10(resid / incident);
    std::printf("  CPML (%d cells): incident |Ez|=%.3e  residual=%.3e  "
                "reflection = %.1f dB\n", pml, incident, resid, dB);
    if (dB > -40.0) ok = false;

    std::printf("cpml-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// s-polarisation Fresnel reflectance. For each incidence angle, run the TFSF
// plane-wave deck twice -- with and without the dielectric half-space -- and
// subtract the Ez phasors on a row just above the interface (the reflected
// beam still fully overlaps the incident footprint there). The interface sits
// high in the domain and the measurement window is short, so it closes before
// the transmitted wave's echo off the (vacuum-tuned) CPML in the dielectric
// can return -- the thorough angle sweep is tools/plot_fresnel.py.
bool FresnelSelfTest() {
    bool ok = true;
    const double n2 = 2.0, epsR = n2 * n2, f0 = 0.05;
    const int nx = 320, ny = 320, pml = 12, margin = 10;
    const int iface = 245, probeRow = iface + 8;
    const double w = 2.0 * kPi * f0;

    auto phasorRow = [&](bool slab, double propDeg) {
        char buf[900];
        std::snprintf(buf, sizeof(buf), R"(
[grid]
nx = %d
ny = %d
dx = 1.0
courant = 0.5
[boundary]
type = "cpml"
pml_cells = %d
[tfsf]
margin = %d
[time]
steps = 1
[[source]]
kind = "tfsf"
waveform = "sine"
f0 = %.4f
amplitude = 1.0
angle_deg = %.4f
ramp_cycles = 4
%s)", nx, ny, pml, margin, f0, propDeg,
            slab ? "[[material]]\nshape=\"halfspace\"\naxis=\"y\"\npos=245\n"
                   "side=\"lo\"\neps_r=4.0\n"
                 : "");
        fdtd::Fdtd2D sim;
        fw::Deck d = fw::Deck::FromString(buf);
        sim.Configure(d);
        const int period = static_cast<int>(std::round(1.0 / f0 / sim.Dt()));
        sim.Step(15 * period);                  // short: beat the dielectric-PML echo
        const int measure = 4 * period;
        std::vector<std::complex<double>> row(nx, 0.0);
        for (int k = 0; k < measure; ++k) {
            sim.Step(1);
            const auto& ez = sim.Ez();
            const auto ph = std::exp(std::complex<double>(0.0, w * sim.Time()));
            for (int i = 0; i < nx; ++i)
                row[i] += ez[static_cast<std::size_t>(probeRow) * nx + i] * ph;
        }
        for (auto& v : row) v *= 2.0 / double(measure);
        return row;
    };

    // incident amplitude from a no-slab run at normal incidence
    const auto inc0 = phasorRow(false, 270.0);
    double Ei = 0.0;
    for (int i = nx / 3; i < 2 * nx / 3; ++i) Ei += std::abs(inc0[i]);
    Ei /= (nx / 3);

    std::printf("  theta_i    R_fdtd    R_fresnel   err   "
                "(full sweep: tools/plot_fresnel.py)\n");
    for (double thetaDeg : {0.0, 20.0, 40.0, 55.0}) {
        const double prop = 270.0 - thetaDeg;
        const auto tot = phasorRow(true, prop);
        const auto inc = phasorRow(false, prop);
        std::vector<double> refl(nx);
        double rmax = 0.0;
        for (int i = 0; i < nx; ++i) {
            refl[i] = std::abs(tot[i] - inc[i]);
            rmax = std::max(rmax, refl[i]);
        }
        double er = 0.0;
        int cnt = 0;
        for (int i = 0; i < nx; ++i)
            if (refl[i] > 0.6 * rmax) { er += refl[i]; ++cnt; }
        const double R = (er / cnt / Ei) * (er / cnt / Ei);

        const double thi = thetaDeg * kPi / 180.0;
        const double ctt = std::sqrt(1.0 - std::pow(std::sin(thi) / n2, 2));
        const double rs = (std::cos(thi) - n2 * ctt) / (std::cos(thi) + n2 * ctt);
        const double Rf = rs * rs;
        std::printf("  %5.0f deg   %.4f    %.4f      %.4f\n",
                    thetaDeg, R, Rf, std::abs(R - Rf));
        if (std::abs(R - Rf) > 0.025) ok = false;
    }
    std::printf("fresnel-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) return SelfTest() ? 0 : 1;
    if (a.gpuSelftest) return GpuSelfTest() ? 0 : 1;
    if (a.cpmlSelftest) return CpmlSelfTest() ? 0 : 1;
    if (a.fresnelSelftest) return FresnelSelfTest() ? 0 : 1;

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  10_fdtd --deck <f.toml> --out <dir> [--steps N] [--gpu]\n"
                     "  10_fdtd --selftest       (CFL / energy / wave speed)\n"
                     "  10_fdtd --gpu-selftest   (GPU compute vs CPU reference)\n");
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
    if (a.gpu) sim.ForceBackend(true);

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
