#include "TdseSim.hpp"
#include "kernels.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/OutputWriter.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <random>
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
    bool relax = false;
    bool scf = false;
    int states = 1;
    int relaxSteps = 2500;
    double relaxDtau = 0.0;   // 0 -> auto
    int scfElectrons = 0;     // 0 -> fill all `states`
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
        else if (s == "--relax") a.relax = true;
        else if (s == "--scf") a.scf = true;
        else if (s == "--states") a.states = std::atoi(next());
        else if (s == "--relax-steps") a.relaxSteps = std::atoi(next());
        else if (s == "--relax-dtau") a.relaxDtau = std::atof(next());
        else if (s == "--electrons") a.scfElectrons = std::atoi(next());
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

// Write a set of eigenstates (+ energies + the potential) as frames.
void WriteEigenstates(const std::string& outDir, const fw::Deck& deck, const tdse::Grid& g,
                      const std::vector<float>& potential,
                      const tdse::TdseSim::RelaxResult& r, const char* title) {
    fw::SimInfo info;
    info.title = title;
    info.gridNx = g.n;
    info.gridNy = g.n;
    info.lx = g.lx;
    info.ly = g.ly;
    info.frameFields = {"psi", "potential"};
    info.diagnostics = {"energy"};
    fw::OutputWriter w(outDir, info, deck);
    for (size_t k = 0; k < r.states.size(); ++k) {
        w.BeginFrame(static_cast<double>(k), static_cast<long>(k));
        w.WriteField("psi", r.states[k].data(), fw::NpyDtype::C8, g.n, g.n);
        if (k == 0) w.WriteField("potential", potential.data(), fw::NpyDtype::F4, g.n, g.n);
        w.WriteScalar("energy", r.energies[k]);
        w.EndFrame();
    }
    w.Finish();
    std::printf("energies:");
    for (double e : r.energies) std::printf(" %.4f", e);
    std::printf("\n");
}

// Round-trip and forward-vs-DFT check of the GPU FFT kernel.
bool SelfTest() {
    constexpr int N = 128;
    constexpr int rows = 4;
    fw::ComputeShader fft = fw::ComputeShader::FromSource(tdse::kernels::Fft1D(N));

    std::vector<float> data(N * rows * 2);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : data) v = dist(rng);
    const std::vector<float> original = data;

    std::vector<float> tw(N * 2);
    for (int m = 0; m < N; ++m) {
        tw[2 * m + 0] = std::cos(2.0 * kPi * m / N);
        tw[2 * m + 1] = -std::sin(2.0 * kPi * m / N);
    }

    GLuint bufData = 0, bufTw = 0;
    glCreateBuffers(1, &bufData);
    glCreateBuffers(1, &bufTw);
    glNamedBufferData(bufData, data.size() * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTw, tw.size() * sizeof(float), tw.data(), GL_STATIC_DRAW);

    auto run = [&](bool inverse) {
        fft.Use();
        fft.SetInt("uSign", inverse ? 1 : -1);
        fft.SetFloat("uScale", inverse ? 1.0f / N : 1.0f);
        fw::ComputeShader::BindBuffer(0, bufData);
        fw::ComputeShader::BindBuffer(1, bufTw);
        fft.Dispatch(rows, 1, 1);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    };

    // Forward, capture, then inverse.
    run(false);
    std::vector<float> fwd(data.size());
    glGetNamedBufferSubData(bufData, 0, fwd.size() * sizeof(float), fwd.data());
    run(true);
    std::vector<float> rt(data.size());
    glGetNamedBufferSubData(bufData, 0, rt.size() * sizeof(float), rt.data());

    double rtErr = 0.0;
    for (size_t k = 0; k < rt.size(); ++k) rtErr = std::max(rtErr, std::abs(double(rt[k]) - original[k]));

    // Forward vs direct DFT on row 0.
    double dftErr = 0.0;
    for (int k = 0; k < N; ++k) {
        std::complex<double> acc{0.0, 0.0};
        for (int nn = 0; nn < N; ++nn) {
            const std::complex<double> x(original[2 * nn], original[2 * nn + 1]);
            const double ang = -2.0 * kPi * k * nn / N;
            acc += x * std::complex<double>(std::cos(ang), std::sin(ang));
        }
        dftErr = std::max(dftErr, std::abs(acc.real() - fwd[2 * k]));
        dftErr = std::max(dftErr, std::abs(acc.imag() - fwd[2 * k + 1]));
    }

    glDeleteBuffers(1, &bufData);
    glDeleteBuffers(1, &bufTw);

    std::printf("selftest: round-trip max err = %.3e, forward-vs-DFT max err = %.3e\n", rtErr, dftErr);
    const bool ok = rtErr < 1e-4 && dftErr < 1e-2;
    std::printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        return SelfTest() ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  05_tdse_gpu --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  05_tdse_gpu --interactive --deck <f.toml>\n"
                     "  05_tdse_gpu --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        tdse::TdseSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "2D TDSE (GPU)"));
        app.Run();
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    tdse::TdseSim sim;
    sim.Configure(deck);

    if (a.relax || a.scf) {
        const tdse::Grid& g = sim.GridInfo();
        const double dtau = a.relaxDtau > 0.0 ? a.relaxDtau : 0.4 * (g.lx / g.n) * (g.lx / g.n);
        tdse::TdseSim::RelaxOptions ro;
        ro.states = a.states;
        ro.steps = a.relaxSteps;
        ro.dtau = dtau;

        if (a.relax) {
            tdse::TdseSim::RelaxResult r = sim.Relax(ro);
            WriteEigenstates(a.out, deck, g, sim.StaticPotential(), r,
                             deck.GetString("title", "eigenstates").c_str());
            return 0;
        }
        return sim.RunScf(deck, ro, a.scfElectrons, a.out);  // --scf (Phase G4)
    }

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
