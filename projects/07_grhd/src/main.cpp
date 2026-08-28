#include "GrhdSim.hpp"
#include "KerrEquatorialSim.hpp"
#include "KerrTorusSim.hpp"
#include "SchwarzschildSim.hpp"
#include "kernels.hpp"
#include "kernels_kerr.hpp"
#include "kernels_kerr2d.hpp"
#include "kernels_schwarzschild.hpp"

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

// CPU (double-precision) reference for kernels_schwarzschild.hpp's
// primitive recovery and HLLE flux -- cross-checks only, not a physics
// validation (that is decks/bondi_schwarzschild.toml +
// tools/plot_bondi.py, checked against the analytic Bondi solution; see
// README.md).
glm::dvec3 PrimToConsSchRef(double rho, double v, double P, double r, double alpha, double gamma) {
    const double W = 1.0 / std::sqrt(1.0 - v * v);
    const double h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    const double r2 = r * r;
    const double D = r2 * rho * W / alpha;
    const double S = r2 * rho * h * W * W * v;
    const double tau = r2 * (rho * h * W * W - P) - D;
    return glm::dvec3(D, S, tau);
}

double PressureResidualSchRef(double D, double S, double tau, double P, double r, double alpha, double gamma) {
    const double r2 = r * r;
    const double denom = tau + D + r2 * P;
    const double v2 = std::clamp((S * S) / (denom * denom), 0.0, 0.999999);
    const double W = 1.0 / std::sqrt(1.0 - v2);
    const double rho = D * alpha / (r2 * W);
    const double h = denom / (r2 * rho * W * W);
    const double eps = h - 1.0 - P / rho;
    return (gamma - 1.0) * rho * eps;
}

glm::dvec3 RecoverPrimitivesSchRef(double D, double S, double tau, double pGuess, double r, double alpha,
                                    double gamma, int iters) {
    double P = std::max(pGuess, 1e-12);
    for (int it = 0; it < iters; ++it) {
        const double f0 = PressureResidualSchRef(D, S, tau, P, r, alpha, gamma) - P;
        const double dP = std::max(1e-6 * std::abs(P), 1e-12);
        const double fPlus = PressureResidualSchRef(D, S, tau, P + dP, r, alpha, gamma) - (P + dP);
        const double fMinus = PressureResidualSchRef(D, S, tau, P - dP, r, alpha, gamma) - (P - dP);
        const double deriv = (fPlus - fMinus) / (2.0 * dP);
        if (std::abs(deriv) > 1e-12) P = std::max(P - f0 / deriv, 1e-12);
    }
    const double r2 = r * r;
    const double denom = tau + D + r2 * P;
    const double v2 = std::clamp((S * S) / (denom * denom), 0.0, 0.999999);
    const double W = 1.0 / std::sqrt(1.0 - v2);
    const double rho = D * alpha / (r2 * W);
    const double v = S / denom;
    return glm::dvec3(rho, v, P);
}

void SideFluxSchRef(const glm::dvec3& P, double r, double f, double alpha, double gamma,
                     glm::dvec3& U, glm::dvec3& F, double& lamMinus, double& lamPlus) {
    const double rho = P.x, v = P.y, p = P.z;
    const double h = 1.0 + gamma * p / ((gamma - 1.0) * rho);
    const double W = 1.0 / std::sqrt(std::clamp(1.0 - v * v, 1e-12, 1.0));
    const double r2 = r * r;
    const double D = r2 * rho * W / alpha;
    const double S = r2 * rho * h * W * W * v;
    const double tau = r2 * (rho * h * W * W - p) - D;
    U = glm::dvec3(D, S, tau);
    F = glm::dvec3(f * v * D, f * (v * S + r2 * p), f * (S - v * D));
    const double cs2 = std::clamp(gamma * p / (rho * h), 0.0, 0.999999);
    const double cs = std::sqrt(cs2);
    lamMinus = f * (v - cs) / (1.0 - v * cs);
    lamPlus = f * (v + cs) / (1.0 + v * cs);
}

glm::dvec3 HlleFluxSchRef(const glm::dvec3& PL, const glm::dvec3& PR, double r, double f, double alpha, double gamma) {
    glm::dvec3 UL, UR, FL, FR;
    double lamMinusL, lamPlusL, lamMinusR, lamPlusR;
    SideFluxSchRef(PL, r, f, alpha, gamma, UL, FL, lamMinusL, lamPlusL);
    SideFluxSchRef(PR, r, f, alpha, gamma, UR, FR, lamMinusR, lamPlusR);
    const double sL = std::min(0.0, std::min(lamMinusL, lamMinusR));
    const double sR = std::max(0.0, std::max(lamPlusL, lamPlusR));
    if (sL >= 0.0) return FL;
    if (sR <= 0.0) return FR;
    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
}

// Same structure as SelfTest() above: N random mildly-relativistic cells
// on a grid with a real black hole mass, GPU vs. independent CPU
// double-precision reference for both primitive recovery and the HLLE
// flux (including its metric/geometric factors).
bool SelfTestSchwarzschild() {
    constexpr int N = 40;
    constexpr int ITERS = 30;
    constexpr double GAMMA = 4.0 / 3.0;
    constexpr double M = 1.0, R_MIN = 3.0, DR = 0.2; // cells span r in [3, 3+40*0.2)=[3,11)

    std::mt19937 rng(7);
    std::uniform_real_distribution<double> rhoDist(0.5, 2.0);
    std::uniform_real_distribution<double> vDist(-0.5, -0.05); // infall, avoid v=0 (degenerate direction)
    std::uniform_real_distribution<double> pDist(0.05, 0.3);

    std::vector<double> rho0(N), v0(N), p0(N), rCell(N), alphaCell(N);
    std::vector<glm::vec4> cons(N);
    for (int i = 0; i < N; ++i) {
        rho0[i] = rhoDist(rng);
        v0[i] = vDist(rng);
        p0[i] = pDist(rng);
        rCell[i] = R_MIN + (i + 0.5) * DR;
        alphaCell[i] = std::sqrt(1.0 - 2.0 * M / rCell[i]);
        const glm::dvec3 U = PrimToConsSchRef(rho0[i], v0[i], p0[i], rCell[i], alphaCell[i], GAMMA);
        cons[i] = glm::vec4(static_cast<float>(U.x), static_cast<float>(U.y), static_cast<float>(U.z), 0.0f);
    }

    std::vector<glm::dvec3> primRef(N);
    for (int i = 0; i < N; ++i) {
        primRef[i] = RecoverPrimitivesSchRef(cons[i].x, cons[i].y, cons[i].z, p0[i], rCell[i], alphaCell[i], GAMMA, ITERS);
    }
    std::vector<glm::dvec3> fluxRef(N + 1);
    for (int i = 0; i <= N; ++i) {
        const int iL = (i == 0) ? 0 : (i - 1);
        const int iR = (i == N) ? (N - 1) : i;
        const double rFace = R_MIN + i * DR;
        const double f = 1.0 - 2.0 * M / rFace;
        const double alpha = std::sqrt(f);
        fluxRef[i] = HlleFluxSchRef(primRef[iL], primRef[iR], rFace, f, alpha, GAMMA);
    }

    fw::ComputeShader consToPrim = fw::ComputeShader::FromSource(grhd::kernels_sch::ConsToPrim());
    fw::ComputeShader fluxes = fw::ComputeShader::FromSource(grhd::kernels_sch::Fluxes());
    GLuint bufCons = 0, bufPrim = 0, bufFlux = 0;
    glCreateBuffers(1, &bufCons);
    glCreateBuffers(1, &bufPrim);
    glCreateBuffers(1, &bufFlux);
    glNamedBufferData(bufCons, N * sizeof(glm::vec4), cons.data(), GL_STATIC_DRAW);
    std::vector<glm::vec4> primSeed(N);
    for (int i = 0; i < N; ++i) primSeed[i] = glm::vec4(0.0f, 0.0f, static_cast<float>(p0[i]), 0.0f);
    glNamedBufferData(bufPrim, N * sizeof(glm::vec4), primSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFlux, (N + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    const GLuint groupsN = static_cast<GLuint>((N + grhd::kernels_sch::kWorkgroupSize - 1) / grhd::kernels_sch::kWorkgroupSize);
    const GLuint groupsNp1 = static_cast<GLuint>((N + 1 + grhd::kernels_sch::kWorkgroupSize - 1) / grhd::kernels_sch::kWorkgroupSize);

    consToPrim.Use();
    consToPrim.SetInt("uN", N);
    consToPrim.SetFloat("uGamma", static_cast<float>(GAMMA));
    consToPrim.SetFloat("uM", static_cast<float>(M));
    consToPrim.SetFloat("uRmin", static_cast<float>(R_MIN));
    consToPrim.SetFloat("uDr", static_cast<float>(DR));
    consToPrim.SetInt("uIters", ITERS);
    fw::ComputeShader::BindBuffer(0, bufCons);
    fw::ComputeShader::BindBuffer(4, bufPrim);
    consToPrim.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    fluxes.Use();
    fluxes.SetInt("uN", N);
    fluxes.SetFloat("uGamma", static_cast<float>(GAMMA));
    fluxes.SetFloat("uM", static_cast<float>(M));
    fluxes.SetFloat("uRmin", static_cast<float>(R_MIN));
    fluxes.SetFloat("uDr", static_cast<float>(DR));
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
        const glm::dvec3 truth(rho0[i], v0[i], p0[i]);
        recoveryErr = std::max(recoveryErr, glm::length(gpu - truth) / glm::length(truth));
    }
    for (int i = 0; i <= N; ++i) {
        const glm::dvec3 gpu(fluxGpu[i].x, fluxGpu[i].y, fluxGpu[i].z);
        const double refNorm = std::max(glm::length(fluxRef[i]), 1e-10);
        fluxErr = std::max(fluxErr, glm::length(gpu - fluxRef[i]) / refNorm);
    }

    std::printf("selftest (schwarzschild): max relative error  prim_vs_cpu=%.3e  prim_vs_truth=%.3e  flux=%.3e\n",
                primErr, recoveryErr, fluxErr);
    const bool ok = primErr < 1e-4 && recoveryErr < 1e-4 && fluxErr < 1e-4;
    std::printf("selftest (schwarzschild): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// CPU (double-precision) reference for kernels_kerr.hpp's Kerr-equatorial
// primitive recovery and HLLE flux -- cross-checks only, not a physics
// validation (that is decks/kerr_circular_orbit.toml +
// tools/plot_kerr_orbit.py, checked against the independently-derived
// circular-orbit solution; see README.md).
struct KerrMetricRef { double gtt, gtphi, grr, gphiphi; };
KerrMetricRef KerrMetricAt(double r, double M, double a) {
    const double Delta = r * r - 2.0 * M * r + a * a;
    KerrMetricRef m;
    m.gtt = -(1.0 - 2.0 * M / r);
    m.gtphi = -2.0 * M * a / r;
    m.grr = r * r / Delta;
    m.gphiphi = r * r + a * a + 2.0 * M * a * a / r;
    return m;
}
void ZamoAtRef(double r, double M, double a, double& alpha, double& betaPhiUp, double& gammaRr,
               double& gammaPhiphi, double& gTphi) {
    const KerrMetricRef m = KerrMetricAt(r, M, a);
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double A = (r * r + a * a) * (r * r + a * a) - a * a * Delta;
    alpha = std::sqrt(r * r * Delta / A);
    betaPhiUp = m.gtphi / m.gphiphi;
    gammaRr = m.grr;
    gammaPhiphi = m.gphiphi;
    gTphi = m.gtphi;
}
void KinematicsRef(const glm::dvec4& prim, double r, double M, double a, double gamma, double& W, double& h,
                    double& ut, double& ur, double& uPhiContra, double& urCov, double& uPhiCov, double& E) {
    const double rho = prim.x, vr = prim.y, vphi = prim.z, P = prim.w;
    double alpha, betaPhiUp, gammaRr, gammaPhiphi, gTphi;
    ZamoAtRef(r, M, a, alpha, betaPhiUp, gammaRr, gammaPhiphi, gTphi);
    W = 1.0 / std::sqrt(std::clamp(1.0 - vr * vr - vphi * vphi, 1e-10, 1.0));
    h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    ut = W / alpha;
    ur = W * vr / std::sqrt(gammaRr);
    uPhiContra = W * vphi / std::sqrt(gammaPhiphi) - W * betaPhiUp / alpha;
    urCov = W * std::sqrt(gammaRr) * vr;
    uPhiCov = W * std::sqrt(gammaPhiphi) * vphi;
    E = W * (alpha - gTphi * vphi / std::sqrt(gammaPhiphi));
}

glm::dvec4 RecoverPrimitivesKerrRef(double D, double Sr, double L, double tau, double pGuess, double r, double M,
                                     double a, double gamma, int iters) {
    double alpha, betaPhiUp, gammaRr, gammaPhiphi, gTphi;
    ZamoAtRef(r, M, a, alpha, betaPhiUp, gammaRr, gammaPhiphi, gTphi);
    const double r2 = r * r;
    const double kappa = Sr / D, lam = L / D;
    const double K = kappa * kappa / gammaRr + lam * lam / gammaPhiphi;

    auto residual = [&](double h) {
        const double W0 = std::sqrt(1.0 + K / (h * h));
        const double rho0 = D * alpha / (r2 * W0);
        const double eps0 = (h - 1.0) / gamma;
        const double P0 = (gamma - 1.0) * rho0 * eps0;
        const double Q0 = (tau + D + r2 * P0) / D;
        return Q0 / h - (W0 * alpha - gTphi * lam / (h * gammaPhiphi));
    };

    const double rhoGuess = D * alpha / (r2 * std::sqrt(1.0 + K));
    double h = 1.0 + gamma * std::max(pGuess, 1e-12) / ((gamma - 1.0) * rhoGuess);
    for (int it = 0; it < iters; ++it) {
        const double f0 = residual(h);
        const double dh = std::max(1e-6 * std::abs(h), 1e-9);
        const double fPlus = residual(h + dh);
        const double fMinus = residual(h - dh);
        const double deriv = (fPlus - fMinus) / (2.0 * dh);
        if (std::abs(deriv) > 1e-12) h = std::max(h - f0 / deriv, 1.0 + 1e-9);
    }
    const double W = std::sqrt(1.0 + K / (h * h));
    const double rho = D * alpha / (r2 * W);
    const double vr = kappa / (h * W * std::sqrt(gammaRr));
    const double vphi = lam / (h * W * std::sqrt(gammaPhiphi));
    const double eps = (h - 1.0) / gamma;
    const double P = (gamma - 1.0) * rho * eps;
    return glm::dvec4(rho, vr, vphi, P);
}

void SideFluxKerrRef(const glm::dvec4& prim, double r, double M, double a, double gamma, glm::dvec4& U,
                      glm::dvec4& F) {
    double W, h, ut, ur, uPhiContra, urCov, uPhiCov, E;
    KinematicsRef(prim, r, M, a, gamma, W, h, ut, ur, uPhiContra, urCov, uPhiCov, E);
    const double rho = prim.x, P = prim.w;
    const double r2 = r * r;
    const double D = r2 * rho * ut;
    const double Sr = r2 * rho * h * ut * urCov;
    const double L = r2 * rho * h * ut * uPhiCov;
    const double tau = r2 * rho * h * ut * E - r2 * P - D;
    U = glm::dvec4(D, Sr, L, tau);
    const double FD = r2 * rho * ur;
    const double FSr = r2 * (rho * h * ur * urCov + P);
    const double FL = r2 * rho * h * ur * uPhiCov;
    const double Ftau = r2 * rho * ur * (h * E - 1.0);
    F = glm::dvec4(FD, FSr, FL, Ftau);
}

glm::dvec4 HlleFluxKerrRef(const glm::dvec4& primL, const glm::dvec4& primR, double r, double M, double a,
                            double gamma) {
    glm::dvec4 UL, UR, FL, FR;
    SideFluxKerrRef(primL, r, M, a, gamma, UL, FL);
    SideFluxKerrRef(primR, r, M, a, gamma, UR, FR);
    const KerrMetricRef m = KerrMetricAt(r, M, a);
    const double fPhoton = std::sqrt(-m.gtt / m.grr);
    const double sL = -fPhoton, sR = fPhoton;
    return (sR * FL - sL * FR + sL * sR * (UR - UL)) / (sR - sL);
}

// Random mildly-relativistic cells on a grid with real spin, GPU vs. CPU
// double-precision reference, same structure as SelfTest()/
// SelfTestSchwarzschild() above.
bool SelfTestKerrEquatorial() {
    constexpr int N = 40;
    constexpr int ITERS = 40;
    constexpr double GAMMA = 4.0 / 3.0;
    constexpr double M = 1.0, A = 0.7, R_MIN = 6.0, DR = 0.3; // r in [6, 6+40*0.3)=[6,18)

    std::mt19937 rng(11);
    std::uniform_real_distribution<double> rhoDist(0.5, 2.0);
    std::uniform_real_distribution<double> vrDist(-0.2, 0.2);
    std::uniform_real_distribution<double> vphiDist(0.1, 0.4);
    std::uniform_real_distribution<double> pDist(0.02, 0.2);

    std::vector<glm::dvec4> prim0(N);
    std::vector<glm::vec4> cons(N);
    std::vector<double> rCell(N);
    for (int i = 0; i < N; ++i) {
        rCell[i] = R_MIN + (i + 0.5) * DR;
        const double rho = rhoDist(rng), vr = vrDist(rng), vphi = vphiDist(rng), p = pDist(rng);
        prim0[i] = glm::dvec4(rho, vr, vphi, p);
        glm::dvec4 U, F;
        SideFluxKerrRef(prim0[i], rCell[i], M, A, GAMMA, U, F);
        cons[i] = glm::vec4(static_cast<float>(U.x), static_cast<float>(U.y), static_cast<float>(U.z),
                             static_cast<float>(U.w));
    }

    std::vector<glm::dvec4> primRef(N);
    for (int i = 0; i < N; ++i) {
        primRef[i] = RecoverPrimitivesKerrRef(cons[i].x, cons[i].y, cons[i].z, cons[i].w, prim0[i].w, rCell[i], M, A,
                                               GAMMA, ITERS);
    }
    std::vector<glm::dvec4> fluxRef(N + 1);
    for (int i = 0; i <= N; ++i) {
        const int iL = (i == 0) ? 0 : (i - 1);
        const int iR = (i == N) ? (N - 1) : i;
        const double rFace = R_MIN + i * DR;
        fluxRef[i] = HlleFluxKerrRef(primRef[iL], primRef[iR], rFace, M, A, GAMMA);
    }

    fw::ComputeShader consToPrim = fw::ComputeShader::FromSource(grhd::kernels_kerr::ConsToPrim());
    fw::ComputeShader fluxes = fw::ComputeShader::FromSource(grhd::kernels_kerr::Fluxes());
    GLuint bufCons = 0, bufPrim = 0, bufFlux = 0;
    glCreateBuffers(1, &bufCons);
    glCreateBuffers(1, &bufPrim);
    glCreateBuffers(1, &bufFlux);
    glNamedBufferData(bufCons, N * sizeof(glm::vec4), cons.data(), GL_STATIC_DRAW);
    std::vector<glm::vec4> primSeed(N);
    for (int i = 0; i < N; ++i) primSeed[i] = glm::vec4(0.0f, 0.0f, 0.0f, static_cast<float>(prim0[i].w));
    glNamedBufferData(bufPrim, N * sizeof(glm::vec4), primSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFlux, (N + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    const GLuint groupsN = static_cast<GLuint>((N + grhd::kernels_kerr::kWorkgroupSize - 1) / grhd::kernels_kerr::kWorkgroupSize);
    const GLuint groupsNp1 =
        static_cast<GLuint>((N + 1 + grhd::kernels_kerr::kWorkgroupSize - 1) / grhd::kernels_kerr::kWorkgroupSize);

    consToPrim.Use();
    consToPrim.SetInt("uN", N);
    consToPrim.SetFloat("uGamma", static_cast<float>(GAMMA));
    consToPrim.SetFloat("uM", static_cast<float>(M));
    consToPrim.SetFloat("uA", static_cast<float>(A));
    consToPrim.SetFloat("uRmin", static_cast<float>(R_MIN));
    consToPrim.SetFloat("uDr", static_cast<float>(DR));
    consToPrim.SetInt("uIters", ITERS);
    fw::ComputeShader::BindBuffer(0, bufCons);
    fw::ComputeShader::BindBuffer(4, bufPrim);
    consToPrim.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    fluxes.Use();
    fluxes.SetInt("uN", N);
    fluxes.SetFloat("uGamma", static_cast<float>(GAMMA));
    fluxes.SetFloat("uM", static_cast<float>(M));
    fluxes.SetFloat("uA", static_cast<float>(A));
    fluxes.SetFloat("uRmin", static_cast<float>(R_MIN));
    fluxes.SetFloat("uDr", static_cast<float>(DR));
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
        const glm::dvec4 gpu(primGpu[i].x, primGpu[i].y, primGpu[i].z, primGpu[i].w);
        primErr = std::max(primErr, glm::length(gpu - primRef[i]) / glm::length(primRef[i]));
        recoveryErr = std::max(recoveryErr, glm::length(gpu - prim0[i]) / glm::length(prim0[i]));
    }
    for (int i = 0; i <= N; ++i) {
        const glm::dvec4 gpu(fluxGpu[i].x, fluxGpu[i].y, fluxGpu[i].z, fluxGpu[i].w);
        const double refNorm = std::max(glm::length(fluxRef[i]), 1e-10);
        fluxErr = std::max(fluxErr, glm::length(gpu - fluxRef[i]) / refNorm);
    }

    std::printf("selftest (kerr equatorial): max relative error  prim_vs_cpu=%.3e  prim_vs_truth=%.3e  flux=%.3e\n",
                primErr, recoveryErr, fluxErr);
    const bool ok = primErr < 1e-4 && recoveryErr < 1e-4 && fluxErr < 1e-4;
    std::printf("selftest (kerr equatorial): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// CPU (double-precision) reference for kernels_kerr2d.hpp's full-2D
// (r,theta) primitive recovery and HLLE fluxes -- cross-checks only, not a
// physics validation (that is decks/kerr_torus.toml + tools/plot_torus.py,
// checked against the analytic Fishbone-Moncrief torus; see README.md).
// Deliberately mirrors kernels_kerr2d.hpp's structure line-for-line so the
// two are provably the same computation, continuing this project's
// running practice (see feedback_gr_derivation_methodology) of never
// trusting a second, independent transcription of curved-spacetime
// formulas without a direct comparison.
struct Metric2DRef { double gtt, gtphi, grr, gthth, gphiphi; };
Metric2DRef Metric2DAt(double r, double theta, double M, double a) {
    const double s = std::sin(theta), c = std::cos(theta);
    const double sin2 = s * s, cos2 = c * c;
    const double Sigma = r * r + a * a * cos2;
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * sin2;
    Metric2DRef m;
    m.gtt = -(1.0 - 2.0 * M * r / Sigma);
    m.gtphi = -2.0 * M * a * r * sin2 / Sigma;
    m.grr = Sigma / Delta;
    m.gthth = Sigma;
    m.gphiphi = A * sin2 / Sigma;
    return m;
}
double SqrtNegG2D(double r, double theta, double M, double a) {
    const Metric2DRef m = Metric2DAt(r, theta, M, a);
    return m.gthth * std::sin(theta); // sqrt(-g) = Sigma*sin(theta), NOT sqrt(gamma) -- see kernels_kerr2d.hpp
}
void Zamo2DRef(double r, double theta, double M, double a, double& alpha, double& betaPhiUp, double& gammaRr,
               double& gammaThth, double& gammaPhiphi, double& gTphi) {
    const Metric2DRef m = Metric2DAt(r, theta, M, a);
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double sin2 = std::sin(theta) * std::sin(theta);
    const double A = (r * r + a * a) * (r * r + a * a) - a * a * Delta * sin2;
    alpha = std::sqrt(m.gthth * Delta / A);
    betaPhiUp = m.gtphi / m.gphiphi;
    gammaRr = m.grr;
    gammaThth = m.gthth;
    gammaPhiphi = m.gphiphi;
    gTphi = m.gtphi;
}
void Kinematics2DRef(const glm::dvec4& prim, double P, double r, double theta, double M, double a, double gamma,
                      double& W, double& h, double& ut, double& ur, double& uth, double& uPhiContra,
                      double& urCov, double& uthCov, double& uPhiCov, double& E) {
    const double rho = prim.x, vr = prim.y, vth = prim.z, vphi = prim.w;
    double alpha, betaPhiUp, gammaRr, gammaThth, gammaPhiphi, gTphi;
    Zamo2DRef(r, theta, M, a, alpha, betaPhiUp, gammaRr, gammaThth, gammaPhiphi, gTphi);
    W = 1.0 / std::sqrt(std::clamp(1.0 - vr * vr - vth * vth - vphi * vphi, 1e-10, 1.0));
    h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    ut = W / alpha;
    ur = W * vr / std::sqrt(gammaRr);
    uth = W * vth / std::sqrt(gammaThth);
    uPhiContra = W * vphi / std::sqrt(gammaPhiphi) - W * betaPhiUp / alpha;
    urCov = W * std::sqrt(gammaRr) * vr;
    uthCov = W * std::sqrt(gammaThth) * vth;
    uPhiCov = W * std::sqrt(gammaPhiphi) * vphi;
    E = W * (alpha - gTphi * vphi / std::sqrt(gammaPhiphi));
}

// Returns (D,Sr,Sth,tau,L) and (Fr_D,Fr_Sr,Fr_Sth,Fr_tau,Fr_L),
// (Fth_D,Fth_Sr,Fth_Sth,Fth_tau,Fth_L) for one cell's primitives.
void SideState2DRef(const glm::dvec4& prim, double P, double r, double theta, double M, double a, double gamma,
                     glm::dvec4& U, double& Ul, glm::dvec4& Fr, double& Frl, glm::dvec4& Fth, double& Fthl) {
    double W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E;
    Kinematics2DRef(prim, P, r, theta, M, a, gamma, W, h, ut, ur, uth, uPhiContra, urCov, uthCov, uPhiCov, E);
    const double rho = prim.x;
    const double sqrtg = SqrtNegG2D(r, theta, M, a);

    const double D = sqrtg * rho * ut;
    const double Sr = sqrtg * rho * h * ut * urCov;
    const double Sth = sqrtg * rho * h * ut * uthCov;
    const double L = sqrtg * rho * h * ut * uPhiCov;
    const double tau = sqrtg * rho * h * ut * E - sqrtg * P - D;
    U = glm::dvec4(D, Sr, Sth, tau);
    Ul = L;

    const double FrD = sqrtg * rho * ur;
    const double FrSr = sqrtg * (rho * h * ur * urCov + P);
    const double FrSth = sqrtg * rho * h * ur * uthCov;
    const double FrTau = sqrtg * rho * ur * (h * E - 1.0);
    Fr = glm::dvec4(FrD, FrSr, FrSth, FrTau);
    Frl = sqrtg * rho * h * ur * uPhiCov;

    const double FthD = sqrtg * rho * uth;
    const double FthSr = sqrtg * rho * h * uth * urCov;
    const double FthSth = sqrtg * (rho * h * uth * uthCov + P);
    const double FthTau = sqrtg * rho * uth * (h * E - 1.0);
    Fth = glm::dvec4(FthD, FthSr, FthSth, FthTau);
    Fthl = sqrtg * rho * h * uth * uPhiCov;
}

glm::dvec4 RecoverPrimitivesKerr2DRef(double D, double Sr, double Sth, double L, double tau, double pGuess, double r,
                                       double theta, double M, double a, double gamma, int iters) {
    double alpha, betaPhiUp, gammaRr, gammaThth, gammaPhiphi, gTphi;
    Zamo2DRef(r, theta, M, a, alpha, betaPhiUp, gammaRr, gammaThth, gammaPhiphi, gTphi);
    const double sqrtg = SqrtNegG2D(r, theta, M, a);
    const double kappaR = Sr / D, kappaTh = Sth / D, lam = L / D;
    const double K = kappaR * kappaR / gammaRr + kappaTh * kappaTh / gammaThth + lam * lam / gammaPhiphi;

    auto residual = [&](double h) {
        const double W0 = std::sqrt(1.0 + K / (h * h));
        const double rho0 = D * alpha / (sqrtg * W0);
        const double P0 = (gamma - 1.0) * rho0 * (h - 1.0) / gamma;
        const double Q0 = (tau + D + sqrtg * P0) / D;
        return Q0 / h - (W0 * alpha - gTphi * lam / (h * gammaPhiphi));
    };
    // Same safeguards as kernels_kerr2d.hpp's ConsToPrim (float32-safe
    // floor + damped Newton step) -- see that shader's comment for why
    // both are needed, not just a smaller epsilon.
    const double kHFloor = 1.0 + 1e-4;
    const double rhoGuess = D * alpha / (sqrtg * std::sqrt(1.0 + K));
    double h = std::max(1.0 + gamma * std::max(pGuess, 1e-12) / ((gamma - 1.0) * rhoGuess), kHFloor);
    for (int it = 0; it < iters; ++it) {
        const double f0 = residual(h);
        const double dh = std::max(1e-6 * std::abs(h), 1e-9);
        const double fPlus = residual(h + dh);
        const double fMinus = residual(h - dh);
        const double deriv = (fPlus - fMinus) / (2.0 * dh);
        if (std::abs(deriv) > 1e-12) {
            const double step = std::clamp(f0 / deriv, -0.5 * h, 0.5 * h);
            h = std::max(h - step, kHFloor);
        }
    }
    const double W = std::sqrt(1.0 + K / (h * h));
    const double rho = D * alpha / (sqrtg * W);
    const double vr = kappaR / (h * W * std::sqrt(gammaRr));
    const double vth = kappaTh / (h * W * std::sqrt(gammaThth));
    const double vphi = lam / (h * W * std::sqrt(gammaPhiphi));
    const double eps = (h - 1.0) / gamma;
    const double P = (gamma - 1.0) * rho * eps;
    return glm::dvec4(rho, vr, vth, vphi);
}

glm::dvec4 HlleFluxKerr2DRef(const glm::dvec4& fluxL, const glm::dvec4& UL, const glm::dvec4& fluxR,
                              const glm::dvec4& UR, double s) {
    const double sL = -s, sR = s;
    return (sR * fluxL - sL * fluxR + sL * sR * (UR - UL)) / (sR - sL);
}

// N random mildly-relativistic cells spread over a 2D (r,theta) patch,
// off the equator (so v_theta actually matters), GPU vs. CPU
// double-precision reference for ConsToPrim + both flux directions.
bool SelfTestKerrTorus() {
    constexpr int NR = 6, NTH = 7;
    constexpr int N = NR * NTH;
    constexpr int ITERS = 40;
    constexpr double GAMMA = 4.0 / 3.0;
    constexpr double M = 1.0, A = 0.8;
    constexpr double R_MIN = 5.0, DR = 0.6;          // r in [5, 5+6*0.6)=[5,8.6)
    constexpr double TH_MIN = 1.0, DTH = 0.09;        // theta in [1.0, 1.0+7*0.09)=[1.0,1.63) -- off-equator

    std::mt19937 rng(23);
    std::uniform_real_distribution<double> rhoDist(0.5, 2.0);
    std::uniform_real_distribution<double> vDist(-0.15, 0.15);
    std::uniform_real_distribution<double> vphiDist(0.1, 0.35);
    std::uniform_real_distribution<double> pDist(0.02, 0.15);

    std::vector<glm::dvec4> prim0(N);
    std::vector<double> p0(N), rCell(N), thCell(N);
    std::vector<glm::vec4> cons(N);
    std::vector<float> consL(N);
    for (int i = 0; i < NR; ++i) {
        for (int j = 0; j < NTH; ++j) {
            const int idx = i * NTH + j;
            rCell[idx] = R_MIN + (i + 0.5) * DR;
            thCell[idx] = TH_MIN + (j + 0.5) * DTH;
            const double rho = rhoDist(rng), vr = vDist(rng), vth = vDist(rng), vphi = vphiDist(rng), p = pDist(rng);
            prim0[idx] = glm::dvec4(rho, vr, vth, vphi);
            p0[idx] = p;
            glm::dvec4 U, Fr, Fth;
            double Ul, Frl, Fthl;
            SideState2DRef(prim0[idx], p, rCell[idx], thCell[idx], M, A, GAMMA, U, Ul, Fr, Frl, Fth, Fthl);
            cons[idx] = glm::vec4(static_cast<float>(U.x), static_cast<float>(U.y), static_cast<float>(U.z),
                                   static_cast<float>(U.w));
            consL[idx] = static_cast<float>(Ul);
        }
    }

    std::vector<glm::dvec4> primRef(N);
    for (int idx = 0; idx < N; ++idx) {
        primRef[idx] = RecoverPrimitivesKerr2DRef(cons[idx].x, cons[idx].y, cons[idx].z, consL[idx], cons[idx].w,
                                                   p0[idx], rCell[idx], thCell[idx], M, A, GAMMA, ITERS);
    }

    // r-direction fluxes: (NR+1)*NTH interfaces.
    std::vector<glm::dvec4> fluxRRef((NR + 1) * NTH);
    for (int i = 0; i <= NR; ++i) {
        for (int j = 0; j < NTH; ++j) {
            const int iL = (i == 0) ? 0 : (i - 1), iR = (i == NR) ? (NR - 1) : i;
            const double rFace = R_MIN + i * DR;
            const double theta = TH_MIN + (j + 0.5) * DTH;
            const Metric2DRef m = Metric2DAt(rFace, theta, M, A);
            const double s = std::sqrt(-m.gtt / m.grr);
            glm::dvec4 UL, UR, FrL, FrR, FthDummyL, FthDummyR;
            double UlL, UlR, FrlL, FrlR, FthlDummyL, FthlDummyR;
            SideState2DRef(primRef[iL * NTH + j], p0[iL * NTH + j], rFace, theta, M, A, GAMMA, UL, UlL, FrL, FrlL,
                            FthDummyL, FthlDummyL);
            SideState2DRef(primRef[iR * NTH + j], p0[iR * NTH + j], rFace, theta, M, A, GAMMA, UR, UlR, FrR, FrlR,
                            FthDummyR, FthlDummyR);
            fluxRRef[i * NTH + j] = HlleFluxKerr2DRef(FrL, UL, FrR, UR, s);
        }
    }
    // theta-direction fluxes: NR*(NTH+1) interfaces.
    std::vector<glm::dvec4> fluxThRef(NR * (NTH + 1));
    for (int i = 0; i < NR; ++i) {
        for (int j = 0; j <= NTH; ++j) {
            const int jL = (j == 0) ? 0 : (j - 1), jR = (j == NTH) ? (NTH - 1) : j;
            const double r = R_MIN + (i + 0.5) * DR;
            const double thFace = TH_MIN + j * DTH;
            const Metric2DRef m = Metric2DAt(r, thFace, M, A);
            const double s = std::sqrt(-m.gtt / m.gthth);
            glm::dvec4 UL, UR, FrDummyL, FrDummyR, FthL, FthR;
            double UlL, UlR, FrlDummyL, FrlDummyR, FthlL, FthlR;
            SideState2DRef(primRef[i * NTH + jL], p0[i * NTH + jL], r, thFace, M, A, GAMMA, UL, UlL, FrDummyL,
                            FrlDummyL, FthL, FthlL);
            SideState2DRef(primRef[i * NTH + jR], p0[i * NTH + jR], r, thFace, M, A, GAMMA, UR, UlR, FrDummyR,
                            FrlDummyR, FthR, FthlR);
            fluxThRef[i * (NTH + 1) + j] = HlleFluxKerr2DRef(FthL, UL, FthR, UR, s);
        }
    }

    // GPU.
    fw::ComputeShader consToPrim = fw::ComputeShader::FromSource(grhd::kernels_kerr2d::ConsToPrim());
    fw::ComputeShader fluxesR = fw::ComputeShader::FromSource(grhd::kernels_kerr2d::FluxesR());
    fw::ComputeShader fluxesTheta = fw::ComputeShader::FromSource(grhd::kernels_kerr2d::FluxesTheta());

    GLuint bufCons = 0, bufConsL = 0, bufPrimMain = 0, bufPrimP = 0, bufFluxR = 0, bufFluxRL = 0, bufFluxTh = 0,
           bufFluxThL = 0;
    glCreateBuffers(1, &bufCons);
    glCreateBuffers(1, &bufConsL);
    glCreateBuffers(1, &bufPrimMain);
    glCreateBuffers(1, &bufPrimP);
    glCreateBuffers(1, &bufFluxR);
    glCreateBuffers(1, &bufFluxRL);
    glCreateBuffers(1, &bufFluxTh);
    glCreateBuffers(1, &bufFluxThL);
    glNamedBufferData(bufCons, N * sizeof(glm::vec4), cons.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufConsL, N * sizeof(float), consL.data(), GL_STATIC_DRAW);
    std::vector<glm::vec4> primMainSeed(N);
    std::vector<float> primPSeed(N);
    for (int idx = 0; idx < N; ++idx) {
        primMainSeed[idx] = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
        primPSeed[idx] = static_cast<float>(p0[idx]);
    }
    glNamedBufferData(bufPrimMain, N * sizeof(glm::vec4), primMainSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufPrimP, N * sizeof(float), primPSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFluxR, (NR + 1) * NTH * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFluxRL, (NR + 1) * NTH * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFluxTh, NR * (NTH + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufFluxThL, NR * (NTH + 1) * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    auto setCommon = [&](fw::ComputeShader& sh) {
        sh.Use();
        sh.SetInt("uNr", NR);
        sh.SetInt("uNth", NTH);
        sh.SetFloat("uGamma", static_cast<float>(GAMMA));
        sh.SetFloat("uM", static_cast<float>(M));
        sh.SetFloat("uA", static_cast<float>(A));
        sh.SetFloat("uRmin", static_cast<float>(R_MIN));
        sh.SetFloat("uDr", static_cast<float>(DR));
        sh.SetFloat("uThetaMin", static_cast<float>(TH_MIN));
        sh.SetFloat("uDth", static_cast<float>(DTH));
    };

    const GLuint groupsN = static_cast<GLuint>((N + grhd::kernels_kerr2d::kWorkgroupSize - 1) / grhd::kernels_kerr2d::kWorkgroupSize);
    const GLuint groupsFacesR =
        static_cast<GLuint>(((NR + 1) * NTH + grhd::kernels_kerr2d::kWorkgroupSize - 1) / grhd::kernels_kerr2d::kWorkgroupSize);
    const GLuint groupsFacesTh =
        static_cast<GLuint>((NR * (NTH + 1) + grhd::kernels_kerr2d::kWorkgroupSize - 1) / grhd::kernels_kerr2d::kWorkgroupSize);

    setCommon(consToPrim);
    consToPrim.SetInt("uIters", ITERS);
    consToPrim.SetFloat("uRhoFloor", 0.0f); // test cells are all well above any floor -- disable it here
    consToPrim.SetFloat("uPFloor", 0.0f);
    consToPrim.SetFloat("uEntropyFloor", 0.0f);
    fw::ComputeShader::BindBuffer(0, bufCons);
    fw::ComputeShader::BindBuffer(6, bufConsL);
    fw::ComputeShader::BindBuffer(4, bufPrimMain);
    fw::ComputeShader::BindBuffer(10, bufPrimP);
    consToPrim.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    setCommon(fluxesR);
    fw::ComputeShader::BindBuffer(4, bufPrimMain);
    fw::ComputeShader::BindBuffer(10, bufPrimP);
    fw::ComputeShader::BindBuffer(5, bufFluxR);
    fw::ComputeShader::BindBuffer(11, bufFluxRL);
    fluxesR.Dispatch(groupsFacesR);
    fw::ComputeShader::Barrier();

    setCommon(fluxesTheta);
    fw::ComputeShader::BindBuffer(4, bufPrimMain);
    fw::ComputeShader::BindBuffer(10, bufPrimP);
    fw::ComputeShader::BindBuffer(12, bufFluxTh);
    fw::ComputeShader::BindBuffer(13, bufFluxThL);
    fluxesTheta.Dispatch(groupsFacesTh);
    fw::ComputeShader::Barrier();

    std::vector<glm::vec4> primGpu(N), fluxRGpu((NR + 1) * NTH), fluxThGpu(NR * (NTH + 1));
    glGetNamedBufferSubData(bufPrimMain, 0, N * sizeof(glm::vec4), primGpu.data());
    glGetNamedBufferSubData(bufFluxR, 0, (NR + 1) * NTH * sizeof(glm::vec4), fluxRGpu.data());
    glGetNamedBufferSubData(bufFluxTh, 0, NR * (NTH + 1) * sizeof(glm::vec4), fluxThGpu.data());
    GLuint bufs[] = {bufCons, bufConsL, bufPrimMain, bufPrimP, bufFluxR, bufFluxRL, bufFluxTh, bufFluxThL};
    glDeleteBuffers(8, bufs);

    double primErr = 0.0, recoveryErr = 0.0, fluxErr = 0.0;
    for (int idx = 0; idx < N; ++idx) {
        const glm::dvec4 gpu(primGpu[idx].x, primGpu[idx].y, primGpu[idx].z, primGpu[idx].w);
        primErr = std::max(primErr, glm::length(gpu - primRef[idx]) / glm::length(primRef[idx]));
        recoveryErr = std::max(recoveryErr, glm::length(gpu - prim0[idx]) / glm::length(prim0[idx]));
    }
    for (size_t idx = 0; idx < fluxRRef.size(); ++idx) {
        const glm::dvec4 gpu(fluxRGpu[idx].x, fluxRGpu[idx].y, fluxRGpu[idx].z, fluxRGpu[idx].w);
        const double refNorm = std::max(glm::length(fluxRRef[idx]), 1e-10);
        fluxErr = std::max(fluxErr, glm::length(gpu - fluxRRef[idx]) / refNorm);
    }
    for (size_t idx = 0; idx < fluxThRef.size(); ++idx) {
        const glm::dvec4 gpu(fluxThGpu[idx].x, fluxThGpu[idx].y, fluxThGpu[idx].z, fluxThGpu[idx].w);
        const double refNorm = std::max(glm::length(fluxThRef[idx]), 1e-10);
        fluxErr = std::max(fluxErr, glm::length(gpu - fluxThRef[idx]) / refNorm);
    }

    std::printf("selftest (kerr torus 2D): max relative error  prim_vs_cpu=%.3e  prim_vs_truth=%.3e  flux=%.3e\n",
                primErr, recoveryErr, fluxErr);
    const bool ok = primErr < 1e-4 && recoveryErr < 1e-4 && fluxErr < 1e-4;
    std::printf("selftest (kerr torus 2D): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const bool flatOk = SelfTest();
        const bool schOk = SelfTestSchwarzschild();
        const bool kerrOk = SelfTestKerrEquatorial();
        const bool torusOk = SelfTestKerrTorus();
        return (flatOk && schOk && kerrOk && torusOk) ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  07_grhd --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  07_grhd --interactive --deck <f.toml>\n"
                     "  07_grhd --selftest\n"
                     "deck's [geometry] key selects the solver: \"flat\" (default, Phase 0), "
                     "\"schwarzschild\" (Phase 1), \"kerr_equatorial\" (Phase 2a), "
                     "or \"kerr_torus\" (Phase 2b).\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);
    const std::string geometry = deck.GetString("geometry", "flat");
    const bool schwarzschild = (geometry == "schwarzschild");
    const bool kerrEquatorial = (geometry == "kerr_equatorial");
    const bool kerrTorus = (geometry == "kerr_torus");

    if (a.interactive) {
        if (kerrTorus) {
            grhd::KerrTorusSim sim;
            fw::SimApp app(sim, deck, deck.GetString("title", "GRHD -- Kerr Fishbone-Moncrief torus"));
            app.Run();
        } else if (kerrEquatorial) {
            grhd::KerrEquatorialSim sim;
            fw::SimApp app(sim, deck, deck.GetString("title", "GRHD -- Kerr equatorial circular orbit"));
            app.Run();
        } else if (schwarzschild) {
            grhd::SchwarzschildSim sim;
            fw::SimApp app(sim, deck, deck.GetString("title", "GRHD -- Schwarzschild Bondi flow"));
            app.Run();
        } else {
            grhd::GrhdSim sim;
            fw::SimApp app(sim, deck, deck.GetString("title", "GRHD -- 1D SRHD shock tube"));
            app.Run();
        }
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;

    if (kerrTorus) {
        grhd::KerrTorusSim sim;
        sim.Configure(deck);
        return fw::RunHeadless(sim, deck, opts);
    }
    if (kerrEquatorial) {
        grhd::KerrEquatorialSim sim;
        sim.Configure(deck);
        return fw::RunHeadless(sim, deck, opts);
    }
    if (schwarzschild) {
        grhd::SchwarzschildSim sim;
        sim.Configure(deck);
        return fw::RunHeadless(sim, deck, opts);
    }
    grhd::GrhdSim sim;
    sim.Configure(deck);
    return fw::RunHeadless(sim, deck, opts);
}
