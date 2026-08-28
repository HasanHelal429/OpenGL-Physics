#include "GrhdSim.hpp"
#include "kernels.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// CPU (double-precision) reference for kernels.hpp's primitive recovery and
// HLLE flux, used only to cross-check the GPU compute shaders -- not a
// physics validation (that is decks/*.toml + tools/plot_shocktube.py,
// checked against the Newtonian limit and the classic relativistic shock
// tube; see README.md).
constexpr double kGamma = 5.0 / 3.0;

glm::dvec3 PrimToConsRef(double rho, double v, double p) {
    const double W = 1.0 / std::sqrt(1.0 - v * v);
    const double h = 1.0 + kGamma * p / ((kGamma - 1.0) * rho);
    const double D = rho * W;
    const double S = rho * h * W * W * v;
    const double tau = rho * h * W * W - p - D;
    return glm::dvec3(D, S, tau);
}

double PressureResidualRef(double D, double S, double tau, double P) {
    const double denom = tau + D + P;
    const double v2 = std::clamp((S * S) / (denom * denom), 0.0, 0.999999);
    const double W = 1.0 / std::sqrt(1.0 - v2);
    const double rho = D / W;
    const double h = denom / (D * W);
    const double eps = h - 1.0 - P / rho;
    return (kGamma - 1.0) * rho * eps;
}

glm::dvec3 RecoverPrimitivesRef(double D, double S, double tau, double pGuess, int iters) {
    double P = std::max(pGuess, 1e-10);
    for (int it = 0; it < iters; ++it) {
        const double f0 = PressureResidualRef(D, S, tau, P) - P;
        const double dP = std::max(1e-6 * std::abs(P), 1e-10);
        const double fPlus = PressureResidualRef(D, S, tau, P + dP) - (P + dP);
        const double fMinus = PressureResidualRef(D, S, tau, P - dP) - (P - dP);
        const double deriv = (fPlus - fMinus) / (2.0 * dP);
        if (std::abs(deriv) > 1e-12) P = std::max(P - f0 / deriv, 1e-10);
    }
    const double denom = tau + D + P;
    const double v2 = std::clamp((S * S) / (denom * denom), 0.0, 0.999999);
    const double W = 1.0 / std::sqrt(1.0 - v2);
    const double rho = D / W;
    const double v = S / denom;
    return glm::dvec3(rho, v, P);
}

void SideFluxRef(const glm::dvec3& U, const glm::dvec3& P, glm::dvec3& F, double& lamMinus, double& lamPlus) {
    const double rho = P.x, v = P.y, p = P.z;
    const double h = 1.0 + kGamma * p / ((kGamma - 1.0) * rho);
    const double cs2 = std::clamp(kGamma * p / (rho * h), 0.0, 0.999999);
    const double cs = std::sqrt(cs2);
    F = glm::dvec3(U.x * v, U.y * v + p, (U.z + p) * v);
    lamMinus = (v - cs) / (1.0 - v * cs);
    lamPlus = (v + cs) / (1.0 + v * cs);
}

glm::dvec3 HlleFluxRef(const glm::dvec3& UL, const glm::dvec3& PL, const glm::dvec3& UR, const glm::dvec3& PR) {
    glm::dvec3 FL, FR;
    double lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    SideFluxRef(UL, PL, FL, lamMinusL, lamPlusL);
    SideFluxRef(UR, PR, FR, lamMinusR, lamPlusR);
    const double sL = std::min(0.0, std::min(lamMinusL, lamMinusR));
    const double sR = std::max(0.0, std::max(lamPlusL, lamPlusR));
    if (sL >= 0.0) return FL;
    if (sR <= 0.0) return FR;
    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
}

// Uploads N random mildly-relativistic cells, runs ConsToPrim+Fluxes on the
// GPU, and compares against this same physics computed independently on
// the CPU in double precision.
bool SelfTest() {
    constexpr int N = 40;
    constexpr int ITERS = 25;

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> rhoDist(0.5, 2.0);
    std::uniform_real_distribution<double> vDist(-0.6, 0.6);
    std::uniform_real_distribution<double> pDist(0.5, 2.0);

    std::vector<double> rho0(N), v0(N), p0(N);
    std::vector<glm::vec4> cons(N);
    for (int i = 0; i < N; ++i) {
        rho0[i] = rhoDist(rng);
        v0[i] = vDist(rng);
        p0[i] = pDist(rng);
        const glm::dvec3 U = PrimToConsRef(rho0[i], v0[i], p0[i]);
        cons[i] = glm::vec4(static_cast<float>(U.x), static_cast<float>(U.y), static_cast<float>(U.z), 0.0f);
    }

    // CPU reference primitive recovery, warm-started from p0 itself (as the
    // GPU pass will be, via the Prim buffer's pre-seeded z-component).
    std::vector<glm::dvec3> primRef(N);
    for (int i = 0; i < N; ++i) {
        primRef[i] = RecoverPrimitivesRef(cons[i].x, cons[i].y, cons[i].z, p0[i], ITERS);
    }
    // CPU reference HLLE flux at each interior interface (i between cell i-1, i).
    std::vector<glm::dvec3> fluxRef(N + 1);
    for (int i = 0; i <= N; ++i) {
        const int iL = (i == 0) ? 0 : (i - 1);
        const int iR = (i == N) ? (N - 1) : i;
        const glm::dvec3 UL(cons[iL].x, cons[iL].y, cons[iL].z);
        const glm::dvec3 UR(cons[iR].x, cons[iR].y, cons[iR].z);
        fluxRef[i] = HlleFluxRef(UL, primRef[iL], UR, primRef[iR]);
    }

    // GPU.
    fw::ComputeShader consToPrim = fw::ComputeShader::FromSource(grhd::kernels::ConsToPrim());
    fw::ComputeShader fluxes = fw::ComputeShader::FromSource(grhd::kernels::Fluxes());
    GLuint bufCons = 0, bufPrim = 0, bufFlux = 0;
    glCreateBuffers(1, &bufCons);
    glCreateBuffers(1, &bufPrim);
    glCreateBuffers(1, &bufFlux);
    glNamedBufferData(bufCons, N * sizeof(glm::vec4), cons.data(), GL_STATIC_DRAW);
    std::vector<glm::vec4> primSeed(N);
    for (int i = 0; i < N; ++i) primSeed[i] = glm::vec4(0.0f, 0.0f, static_cast<float>(p0[i]), 0.0f);
    glNamedBufferData(bufPrim, N * sizeof(glm::vec4), primSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFlux, (N + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    const GLuint groupsN = static_cast<GLuint>((N + grhd::kernels::kWorkgroupSize - 1) / grhd::kernels::kWorkgroupSize);
    const GLuint groupsNp1 = static_cast<GLuint>((N + 1 + grhd::kernels::kWorkgroupSize - 1) / grhd::kernels::kWorkgroupSize);

    consToPrim.Use();
    consToPrim.SetInt("uN", N);
    consToPrim.SetFloat("uGamma", static_cast<float>(kGamma));
    consToPrim.SetInt("uIters", ITERS);
    fw::ComputeShader::BindBuffer(0, bufCons);
    fw::ComputeShader::BindBuffer(4, bufPrim);
    consToPrim.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    fluxes.Use();
    fluxes.SetInt("uN", N);
    fluxes.SetFloat("uGamma", static_cast<float>(kGamma));
    fw::ComputeShader::BindBuffer(0, bufCons);
    fw::ComputeShader::BindBuffer(4, bufPrim);
    fw::ComputeShader::BindBuffer(5, bufFlux);
    fluxes.Dispatch(groupsNp1);
    fw::ComputeShader::Barrier();

    std::vector<glm::vec4> primGpu(N), fluxGpu(N + 1);
    glGetNamedBufferSubData(bufPrim, 0, N * sizeof(glm::vec4), primGpu.data());
    glGetNamedBufferSubData(bufFlux, 0, (N + 1) * sizeof(glm::vec4), fluxGpu.data());
    glDeleteBuffers(1, &bufCons);
    glDeleteBuffers(1, &bufPrim);
    glDeleteBuffers(1, &bufFlux);

    double primErr = 0.0, recoveryErr = 0.0, fluxErr = 0.0;
    for (int i = 0; i < N; ++i) {
        const glm::dvec3 gpu(primGpu[i].x, primGpu[i].y, primGpu[i].z);
        primErr = std::max(primErr, glm::length(gpu - primRef[i]) / glm::length(primRef[i]));
        // Recovery correctness: does it get back the original (rho0,v0,p0)
        // the conserved variables were built from, not just agree with the
        // CPU reference (a bug shared by both would pass that check).
        const glm::dvec3 truth(rho0[i], v0[i], p0[i]);
        recoveryErr = std::max(recoveryErr, glm::length(gpu - truth) / glm::length(truth));
    }
    for (int i = 0; i <= N; ++i) {
        const glm::dvec3 gpu(fluxGpu[i].x, fluxGpu[i].y, fluxGpu[i].z);
        const double refNorm = std::max(glm::length(fluxRef[i]), 1e-10);
        fluxErr = std::max(fluxErr, glm::length(gpu - fluxRef[i]) / refNorm);
    }

    std::printf("selftest: max relative error  prim_vs_cpu=%.3e  prim_vs_truth=%.3e  flux=%.3e\n",
                primErr, recoveryErr, fluxErr);
    const bool ok = primErr < 1e-4 && recoveryErr < 1e-4 && fluxErr < 1e-4;
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
                     "  07_grhd --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  07_grhd --interactive --deck <f.toml>\n"
                     "  07_grhd --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        grhd::GrhdSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "GRHD -- 1D SRHD shock tube"));
        app.Run();
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    grhd::GrhdSim sim;
    sim.Configure(deck);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
