#include "TdeSim.hpp"
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

// CPU reference implementation of kernels.hpp's cubic spline + fused
// gravity/SPH force, used only to cross-check the GPU compute shaders.
double RefKernelW(double r, double h) {
    const double q = r / h;
    const double sigma = 1.0 / (M_PI * h * h * h);
    if (q < 1.0) return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    if (q < 2.0) { const double t = 2.0 - q; return sigma * 0.25 * t * t * t; }
    return 0.0;
}
double RefKernelDwDr(double r, double h) {
    const double q = r / h;
    const double sigma = 1.0 / (M_PI * h * h * h);
    if (q < 1.0) return (sigma / h) * (-3.0 * q + 2.25 * q * q);
    if (q < 2.0) { const double t = 2.0 - q; return (sigma / h) * (-0.75 * t * t); }
    return 0.0;
}

// Uploads N random particles, runs one Density+Forces GPU pass, and compares
// against this same physics computed independently on the CPU (double
// precision, plain nested loops). Not a physics validation (that is
// decks/star_relax*.toml + tools/plot_star.py, checked against the analytic
// Lane-Emden profile) -- this only checks the GPU shaders implement the
// documented formulas correctly.
bool SelfTest() {
    constexpr int N = 37;
    constexpr double G = 1.0, SOFT2 = 0.01 * 0.01, K = 1.0, GAMMA = 5.0 / 3.0;
    constexpr double VISC_A = 1.0, VISC_B = 2.0;
    constexpr double ETA = 1.2, H_INIT = 0.3, H_MIN = 0.2 * H_INIT, H_MAX = 8.0 * H_INIT;
    constexpr int H_ITERS = 3;
    constexpr int BH_TYPE = 2; // Paczynski-Wiita -- exercises more new code than the point-mass case
    constexpr double BH_MASS = 500.0, BH_RS = 0.1;
    constexpr double RHO_FLOOR = 1e-8; // see kernels.hpp Density()/Forces() -- P/rho^2 diverges as rho->0

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> posDist(-1.0, 1.0);
    std::uniform_real_distribution<double> massDist(0.5, 1.5);
    std::vector<glm::vec4> posMass(N), vel(N);
    for (int i = 0; i < N; ++i) {
        posMass[i] = glm::vec4(posDist(rng), posDist(rng), posDist(rng), massDist(rng));
        vel[i] = glm::vec4(posDist(rng) * 0.1, posDist(rng) * 0.1, posDist(rng) * 0.1, 0.0f);
    }

    // CPU reference: same adaptive-h fixed-point iteration as Density(),
    // same symmetrized-h pair force as Forces() (see kernels.hpp).
    std::vector<double> hRef(N, H_INIT), rhoRef(N), pressRef(N), csRef(N);
    auto gatherDensity = [&](int i, double h) {
        double rho = 0.0;
        for (int j = 0; j < N; ++j) {
            if (j == i) continue; // see kernels.hpp Density()'s comment on the self-term bias
            const double r = glm::length(glm::dvec3(posMass[i]) - glm::dvec3(posMass[j]));
            rho += posMass[j].w * RefKernelW(r, h);
        }
        return rho;
    };
    for (int i = 0; i < N; ++i) {
        double h = hRef[i], rho = 0.0;
        for (int it = 0; it < H_ITERS; ++it) {
            rho = gatherDensity(i, h);
            h = std::clamp(ETA * std::cbrt(static_cast<double>(posMass[i].w) / rho), H_MIN, H_MAX);
        }
        rho = gatherDensity(i, h);
        hRef[i] = h;
        rhoRef[i] = rho;
        pressRef[i] = (rho > RHO_FLOOR) ? K * std::pow(rho, GAMMA) : 0.0;
        csRef[i] = (rho > RHO_FLOOR) ? std::sqrt(GAMMA * pressRef[i] / rho) : 0.0;
    }
    std::vector<glm::dvec3> accRef(N, glm::dvec3(0.0));
    for (int i = 0; i < N; ++i) {
        const double r = glm::length(glm::dvec3(posMass[i]));
        const double dr = std::max(r - BH_RS, std::sqrt(SOFT2));
        accRef[i] -= G * BH_MASS / (dr * dr) * (glm::dvec3(posMass[i]) / std::max(r, 1e-8));

        for (int j = 0; j < N; ++j) {
            const glm::dvec3 rij = glm::dvec3(posMass[i]) - glm::dvec3(posMass[j]);
            const double r2 = glm::dot(rij, rij);
            accRef[i] -= G * static_cast<double>(posMass[j].w) * std::pow(r2 + SOFT2, -1.5) * rij;

            const double hij = 0.5 * (hRef[i] + hRef[j]);
            const double r = std::sqrt(r2);
            const double dWdr = RefKernelDwDr(r, hij);
            if (dWdr == 0.0) continue;
            const glm::dvec3 gradW = (r > 0.0) ? (dWdr / r) * rij : glm::dvec3(0.0);
            const double rhobar = 0.5 * (rhoRef[i] + rhoRef[j]);
            double piVisc = 0.0;
            if (rhobar > RHO_FLOOR) {
                const glm::dvec3 vij = glm::dvec3(vel[i]) - glm::dvec3(vel[j]);
                const double vijDotRij = glm::dot(vij, rij);
                if (vijDotRij < 0.0) {
                    const double mu = hij * vijDotRij / (r2 + 0.01 * hij * hij);
                    const double cbar = 0.5 * (csRef[i] + csRef[j]);
                    piVisc = (-VISC_A * cbar * mu + VISC_B * mu * mu) / rhobar;
                }
            }
            const double piOverRho2 = (rhoRef[i] > RHO_FLOOR) ? pressRef[i] / (rhoRef[i] * rhoRef[i]) : 0.0;
            const double pjOverRho2 = (rhoRef[j] > RHO_FLOOR) ? pressRef[j] / (rhoRef[j] * rhoRef[j]) : 0.0;
            accRef[i] -= static_cast<double>(posMass[j].w) * (piOverRho2 + pjOverRho2 + piVisc) * gradW;
        }
    }

    // GPU.
    fw::ComputeShader density = fw::ComputeShader::FromSource(tde::kernels::Density());
    fw::ComputeShader forces = fw::ComputeShader::FromSource(tde::kernels::Forces());
    GLuint bufPosMass = 0, bufVel = 0, bufAcc = 0, bufRhoPress = 0, bufH = 0;
    glCreateBuffers(1, &bufPosMass);
    glCreateBuffers(1, &bufVel);
    glCreateBuffers(1, &bufAcc);
    glCreateBuffers(1, &bufRhoPress);
    glCreateBuffers(1, &bufH);
    glNamedBufferData(bufPosMass, N * sizeof(glm::vec4), posMass.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufVel, N * sizeof(glm::vec4), vel.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufAcc, N * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufRhoPress, N * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    const std::vector<float> hSeed(N, static_cast<float>(H_INIT));
    glNamedBufferData(bufH, N * sizeof(float), hSeed.data(), GL_DYNAMIC_DRAW);

    const GLuint groups = static_cast<GLuint>((N + tde::kernels::kWorkgroupSize - 1) / tde::kernels::kWorkgroupSize);
    density.Use();
    density.SetInt("uN", N);
    density.SetFloat("uK", static_cast<float>(K));
    density.SetFloat("uGamma", static_cast<float>(GAMMA));
    density.SetFloat("uEta", static_cast<float>(ETA));
    density.SetInt("uHIters", H_ITERS);
    density.SetFloat("uHMin", static_cast<float>(H_MIN));
    density.SetFloat("uHMax", static_cast<float>(H_MAX));
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(3, bufRhoPress);
    fw::ComputeShader::BindBuffer(4, bufH);
    density.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    forces.Use();
    forces.SetInt("uN", N);
    forces.SetFloat("uG", static_cast<float>(G));
    forces.SetFloat("uSoftening2", static_cast<float>(SOFT2));
    forces.SetFloat("uViscAlpha", static_cast<float>(VISC_A));
    forces.SetFloat("uViscBeta", static_cast<float>(VISC_B));
    forces.SetInt("uBhType", BH_TYPE);
    forces.SetFloat("uBhMass", static_cast<float>(BH_MASS));
    forces.SetFloat("uBhRs", static_cast<float>(BH_RS));
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(1, bufVel);
    fw::ComputeShader::BindBuffer(2, bufAcc);
    fw::ComputeShader::BindBuffer(3, bufRhoPress);
    fw::ComputeShader::BindBuffer(4, bufH);
    forces.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<glm::vec4> rhoPressGpu(N), accGpu(N);
    std::vector<float> hGpu(N);
    glGetNamedBufferSubData(bufRhoPress, 0, N * sizeof(glm::vec4), rhoPressGpu.data());
    glGetNamedBufferSubData(bufAcc, 0, N * sizeof(glm::vec4), accGpu.data());
    glGetNamedBufferSubData(bufH, 0, N * sizeof(float), hGpu.data());
    glDeleteBuffers(1, &bufPosMass);
    glDeleteBuffers(1, &bufVel);
    glDeleteBuffers(1, &bufAcc);
    glDeleteBuffers(1, &bufRhoPress);
    glDeleteBuffers(1, &bufH);

    double hErr = 0.0, rhoErr = 0.0, accErr = 0.0;
    for (int i = 0; i < N; ++i) {
        hErr = std::max(hErr, std::abs(hGpu[i] - hRef[i]) / hRef[i]);
        rhoErr = std::max(rhoErr, std::abs(rhoPressGpu[i].x - rhoRef[i]) / rhoRef[i]);
        const glm::dvec3 da = glm::dvec3(accGpu[i]) - accRef[i];
        accErr = std::max(accErr, glm::length(da) / glm::length(accRef[i]));
    }
    std::printf("selftest: max relative error  h=%.3e  rho=%.3e  accel=%.3e\n", hErr, rhoErr, accErr);
    const bool ok = hErr < 1e-4 && rhoErr < 1e-4 && accErr < 1e-3;
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
                     "  06_tidal_disruption --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  06_tidal_disruption --interactive --deck <f.toml>\n"
                     "  06_tidal_disruption --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        tde::TdeSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "Tidal Disruption -- SPH star"));
        app.Run();
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    tde::TdeSim sim;
    sim.Configure(deck);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
