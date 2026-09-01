#include "MDApp.hpp"
#include "MDSim.hpp"
#include "MDSystem.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

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

double RefLjForceMag(double r, double eps, double sigma) {
    const double sr6 = std::pow(sigma / r, 6);
    return 24.0 * eps * (2.0 * sr6 * sr6 - sr6) / r;
}
double RefLjPotential(double r, double eps, double sigma) {
    const double sr6 = std::pow(sigma / r, 6);
    return 4.0 * eps * (sr6 * sr6 - sr6);
}
glm::dvec3 RefMinImage(glm::dvec3 d, double L) {
    d.x -= L * std::round(d.x / L);
    d.y -= L * std::round(d.y / L);
    d.z -= L * std::round(d.z / L);
    return d;
}

// Independent brute-force reference for the shifted-force-truncated LJ
// potential (same formula as MDSystem::ComputeForces, reimplemented here
// with no knowledge of its linked-cell neighbor list) -- cross-checks that
// the production neighbor search finds exactly the right pairs and applies
// the right force/energy shift.
void RefForcesAndEnergy(const std::vector<glm::dvec3>& pos, double L, double cutoff,
                         std::vector<glm::dvec3>& accelOut, double& energyOut) {
    const int n = static_cast<int>(pos.size());
    accelOut.assign(static_cast<size_t>(n), glm::dvec3(0.0));
    energyOut = 0.0;
    const double Fc = RefLjForceMag(cutoff, 1.0, 1.0);
    const double Uc = RefLjPotential(cutoff, 1.0, 1.0);
    const double cutoff2 = cutoff * cutoff;
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const glm::dvec3 rij = RefMinImage(pos[static_cast<size_t>(i)] - pos[static_cast<size_t>(j)], L);
            const double r2 = glm::dot(rij, rij);
            if (r2 > cutoff2) continue;
            const double r = std::sqrt(r2);
            const double Fmag = RefLjForceMag(r, 1.0, 1.0) - Fc;
            const glm::dvec3 f = (Fmag / r) * rij;
            accelOut[static_cast<size_t>(i)] += f;
            accelOut[static_cast<size_t>(j)] -= f;
            energyOut += RefLjPotential(r, 1.0, 1.0) - Uc + (r - cutoff) * Fc;
        }
    }
}

// Random N particles in an L^3 periodic box: MDSystem's own neighbor-list ->
// force pipeline vs. the independent brute-force reference above. `L` is
// picked per case so cellsPerAxis = floor(L/cutoff) lands on either side of
// the linked-cell code's brute-force-fallback threshold (see
// MDSystem.cpp's BuildCellListPairs) -- both real code paths get exercised,
// not just one.
bool SelfTestForces(int n, double L, double cutoff, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, L);
    std::vector<glm::dvec3> pos(static_cast<size_t>(n));
    for (glm::dvec3& p : pos) p = glm::dvec3(u(rng), u(rng), u(rng));
    const std::vector<glm::dvec3> vel(static_cast<size_t>(n), glm::dvec3(0.0));

    md::MDSystem sys;
    md::MDParams params;
    params.cutoff = cutoff;
    params.skin = 0.3;
    sys.SetParticles(pos, vel, L);
    sys.PrimeForces(params);

    std::vector<glm::dvec3> refAccel;
    double refEnergy = 0.0;
    RefForcesAndEnergy(sys.Positions(), L, std::min(cutoff, 0.49 * L), refAccel, refEnergy);

    double accelErr = 0.0, refNorm = 1e-12;
    const std::vector<glm::dvec3>& accel = sys.Accelerations();
    for (int i = 0; i < n; ++i) {
        accelErr = std::max(accelErr, glm::length(accel[static_cast<size_t>(i)] - refAccel[static_cast<size_t>(i)]));
        refNorm = std::max(refNorm, glm::length(refAccel[static_cast<size_t>(i)]));
    }
    const double relAccelErr = accelErr / refNorm;
    const double relEnergyErr = std::abs(sys.PotentialEnergy() - refEnergy) / std::max(std::abs(refEnergy), 1e-12);

    std::printf("selftest (%s): max relative error  accel=%.3e  energy=%.3e\n", label, relAccelErr, relEnergyErr);
    const bool ok = relAccelErr < 1e-9 && relEnergyErr < 1e-9;
    std::printf("selftest (%s): %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

// Two particles, one pair, no periodic wrap in play: checks that
// MDSystem's analytic force is actually the gradient of the SAME potential
// it reports via PotentialEnergy() (central finite difference on the
// separation) -- independent of whether either matches literature, this
// validates internal force/energy self-consistency of the shifted-force
// implementation itself.
bool SelfTestNumericalGradient() {
    const double L = 20.0;
    const double cutoff = 2.5;
    const double sep = 1.2;
    const double eps = 1e-6;

    auto potentialAt = [&](double dx) {
        md::MDSystem sys;
        md::MDParams params;
        params.cutoff = cutoff;
        params.skin = 0.3;
        const std::vector<glm::dvec3> pos = {glm::dvec3(0.0), glm::dvec3(sep + dx, 0.0, 0.0)};
        const std::vector<glm::dvec3> vel(2, glm::dvec3(0.0));
        sys.SetParticles(pos, vel, L);
        sys.PrimeForces(params);
        return sys.PotentialEnergy();
    };
    const double numForce = -(potentialAt(eps) - potentialAt(-eps)) / (2.0 * eps);

    md::MDSystem sys;
    md::MDParams params;
    params.cutoff = cutoff;
    params.skin = 0.3;
    const std::vector<glm::dvec3> pos = {glm::dvec3(0.0), glm::dvec3(sep, 0.0, 0.0)};
    const std::vector<glm::dvec3> vel(2, glm::dvec3(0.0));
    sys.SetParticles(pos, vel, L);
    sys.PrimeForces(params);
    const double anaForce = sys.Accelerations()[1].x; // mass=1 -> accel == force

    const double relErr = std::abs(anaForce - numForce) / std::max(std::abs(numForce), 1e-12);
    std::printf("selftest (numerical-gradient): analytic=%.6f  numerical=%.6f  relerr=%.3e\n", anaForce, numForce,
                relErr);
    const bool ok = relErr < 1e-4; // finite-difference floor, not machine precision
    std::printf("selftest (numerical-gradient): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool SelfTest() {
    const bool small = SelfTestForces(40, 6.0, 2.5, 1, "small (brute-force fallback)");
    const bool multicell = SelfTestForces(400, 14.0, 2.5, 2, "multicell (linked-cell)");
    const bool gradient = SelfTestNumericalGradient();
    return small && multicell && gradient;
}

} // namespace

int main(int argc, char** argv) {
    if (argc <= 1) {
        // No arguments: legacy ImGui app (presets dropdown, live charts).
        md::MDApp app;
        app.Run();
        return 0;
    }

    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        return SelfTest() ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  04_molecular_dynamics                                   (legacy ImGui app)\n"
                     "  04_molecular_dynamics --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  04_molecular_dynamics --interactive --deck <f.toml>\n"
                     "  04_molecular_dynamics --selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        md::MDSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "Molecular Dynamics -- Lennard-Jones fluid"));
        app.Run();
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    md::MDSim sim;
    sim.Configure(deck);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
