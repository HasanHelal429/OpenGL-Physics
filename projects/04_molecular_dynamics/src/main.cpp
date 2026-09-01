#include "MDApp.hpp"
#include "MDSim.hpp"
#include "MDSystem.hpp"
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

// Same idea as SelfTestForces, but with a random 80:20 A:B species
// assignment and Kob-Andersen-like per-pair sigma/epsilon -- independently
// re-derives the same shifted-force formula with its OWN species lookup
// (not calling into MDSystem's), cross-checking that MDSystem::ComputeForces
// picks the right (species i, species j) parameters for every pair, not
// just that the single-species formula is right (already covered above).
bool SelfTestMixtureForces(int n, double L, double cutoff, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, L);
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    std::vector<glm::dvec3> pos(static_cast<size_t>(n));
    std::vector<int> species(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        pos[static_cast<size_t>(i)] = glm::dvec3(u(rng), u(rng), u(rng));
        species[static_cast<size_t>(i)] = (coin(rng) < 0.8) ? 0 : 1;
    }
    const std::vector<glm::dvec3> vel(static_cast<size_t>(n), glm::dvec3(0.0));

    const double sigma[2][2] = {{1.0, 0.8}, {0.8, 0.88}};
    const double epsilon[2][2] = {{1.0, 1.5}, {1.5, 0.5}};

    md::MDSystem sys;
    md::MDParams params;
    params.cutoff = cutoff;
    params.skin = 0.3;
    sys.SetParticles(pos, vel, L, species);
    sys.SetSpeciesLJParams(sigma[1][1], epsilon[1][1], sigma[0][1], epsilon[0][1]);
    sys.PrimeForces(params);

    const double rc = std::min(cutoff, 0.49 * L);
    double Fc[2][2], Uc[2][2];
    for (int si = 0; si < 2; ++si) {
        for (int sj = 0; sj < 2; ++sj) {
            Fc[si][sj] = RefLjForceMag(rc, epsilon[si][sj], sigma[si][sj]);
            Uc[si][sj] = RefLjPotential(rc, epsilon[si][sj], sigma[si][sj]);
        }
    }
    std::vector<glm::dvec3> refAccel(static_cast<size_t>(n), glm::dvec3(0.0));
    double refEnergy = 0.0;
    const std::vector<glm::dvec3>& refPos = sys.Positions();
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const glm::dvec3 rij = RefMinImage(refPos[static_cast<size_t>(i)] - refPos[static_cast<size_t>(j)], L);
            const double r2 = glm::dot(rij, rij);
            if (r2 > rc * rc) continue;
            const int si = species[static_cast<size_t>(i)];
            const int sj = species[static_cast<size_t>(j)];
            const double r = std::sqrt(r2);
            const double Fmag = RefLjForceMag(r, epsilon[si][sj], sigma[si][sj]) - Fc[si][sj];
            const glm::dvec3 f = (Fmag / r) * rij;
            refAccel[static_cast<size_t>(i)] += f;
            refAccel[static_cast<size_t>(j)] -= f;
            refEnergy += RefLjPotential(r, epsilon[si][sj], sigma[si][sj]) - Uc[si][sj] + (r - rc) * Fc[si][sj];
        }
    }

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

// Cross-checks kernels.hpp's GPU BuildGrid+Forces against MDSystem's own
// (already independently-validated, see SelfTestForces above) CPU
// implementation -- same physics, two completely separate code paths (a
// periodic dense-grid GPU compute shader vs. the CPU's linked-cell
// std::vector list), same role as 06/07's GPU-vs-CPU selftest cases.
// Needs a current GL context (caller's responsibility, see main()).
bool SelfTestGpuForces(int n, double L, double cutoff, unsigned seed, const char* label) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, L);
    std::vector<glm::dvec3> posD(static_cast<size_t>(n));
    for (glm::dvec3& p : posD) p = glm::dvec3(u(rng), u(rng), u(rng));
    const std::vector<glm::dvec3> velD(static_cast<size_t>(n), glm::dvec3(0.0));

    md::MDSystem sys;
    md::MDParams params;
    params.cutoff = cutoff;
    params.skin = 0.3;
    sys.SetParticles(posD, velD, L);
    sys.PrimeForces(params); // CPU reference: MDSystem's own force evaluation

    const int nc = static_cast<int>(std::floor(L / cutoff));
    if (nc < 3) {
        std::printf("selftest (%s): SKIP (uNc=%d < 3, kernels.hpp needs the real-grid regime)\n", label, nc);
        return true;
    }

    std::vector<glm::vec4> posF(static_cast<size_t>(n));
    const std::vector<glm::dvec3>& cpuPos = sys.Positions(); // already in [0,L), generated that way above
    for (int i = 0; i < n; ++i) posF[static_cast<size_t>(i)] = glm::vec4(glm::vec3(cpuPos[static_cast<size_t>(i)]), 0.0f);

    fw::ComputeShader buildGrid = fw::ComputeShader::FromSource(md::kernels::BuildGrid());
    fw::ComputeShader forces = fw::ComputeShader::FromSource(md::kernels::Forces());

    GLuint bufPos = 0, bufAccelEnergy = 0, bufCellHead = 0, bufNextIndex = 0;
    glCreateBuffers(1, &bufPos);
    glCreateBuffers(1, &bufAccelEnergy);
    glCreateBuffers(1, &bufCellHead);
    glCreateBuffers(1, &bufNextIndex);
    glNamedBufferData(bufPos, n * sizeof(glm::vec4), posF.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufAccelEnergy, n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    const int nCells = nc * nc * nc;
    glNamedBufferData(bufCellHead, static_cast<GLsizeiptr>(nCells) * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufNextIndex, n * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    const unsigned int sentinel = 0xFFFFFFFFu;
    glClearNamedBufferData(bufCellHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &sentinel);

    // cellSize MUST be L/nc (not `cutoff` directly) so the uNc cells tile
    // the box exactly with no remainder -- otherwise cellOf() can compute
    // an out-of-[0,uNc) index for a particle near the far edge, corrupting
    // the grid. Matches MDSystem.cpp's BuildCellListPairs exactly (its
    // cellSize is likewise L/nc, not the raw cutoff).
    const float cellSize = static_cast<float>(L / nc);
    const GLuint groups = static_cast<GLuint>((n + md::kernels::kWorkgroupSize - 1) / md::kernels::kWorkgroupSize);

    buildGrid.Use();
    buildGrid.SetInt("uN", n);
    buildGrid.SetInt("uNc", nc);
    buildGrid.SetFloat("uCellSize", cellSize);
    buildGrid.SetFloat("uL", static_cast<float>(L));
    fw::ComputeShader::BindBuffer(0, bufPos);
    fw::ComputeShader::BindBuffer(6, bufCellHead);
    fw::ComputeShader::BindBuffer(7, bufNextIndex);
    buildGrid.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const double rc = std::min(cutoff, 0.49 * L);
    const float Fc = static_cast<float>(RefLjForceMag(rc, 1.0, 1.0));
    const float Uc = static_cast<float>(RefLjPotential(rc, 1.0, 1.0));

    forces.Use();
    forces.SetInt("uN", n);
    forces.SetInt("uNc", nc);
    forces.SetFloat("uCellSize", cellSize);
    forces.SetFloat("uL", static_cast<float>(L));
    forces.SetFloat("uCutoff", static_cast<float>(rc));
    forces.SetFloat("uSigma", 1.0f);
    forces.SetFloat("uEpsilon", 1.0f);
    forces.SetFloat("uFc", Fc);
    forces.SetFloat("uUc", Uc);
    fw::ComputeShader::BindBuffer(0, bufPos);
    fw::ComputeShader::BindBuffer(1, bufAccelEnergy);
    fw::ComputeShader::BindBuffer(6, bufCellHead);
    fw::ComputeShader::BindBuffer(7, bufNextIndex);
    forces.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<glm::vec4> gpuOut(static_cast<size_t>(n));
    glGetNamedBufferSubData(bufAccelEnergy, 0, n * sizeof(glm::vec4), gpuOut.data());
    GLuint bufs[4] = {bufPos, bufAccelEnergy, bufCellHead, bufNextIndex};
    glDeleteBuffers(4, bufs);

    double gpuEnergy = 0.0;
    for (const glm::vec4& v : gpuOut) gpuEnergy += v.w;

    double accelErr = 0.0, refNorm = 1e-12;
    const std::vector<glm::dvec3>& cpuAccel = sys.Accelerations();
    for (int i = 0; i < n; ++i) {
        const glm::dvec3 gpuAccel(gpuOut[static_cast<size_t>(i)]);
        accelErr = std::max(accelErr, glm::length(gpuAccel - cpuAccel[static_cast<size_t>(i)]));
        refNorm = std::max(refNorm, glm::length(cpuAccel[static_cast<size_t>(i)]));
    }
    const double relAccelErr = accelErr / refNorm;
    const double relEnergyErr = std::abs(gpuEnergy - sys.PotentialEnergy()) / std::max(std::abs(sys.PotentialEnergy()), 1e-12);

    std::printf("selftest (%s): max relative error  accel=%.3e  energy=%.3e\n", label, relAccelErr, relEnergyErr);
    // Looser tolerance than the CPU-vs-CPU cases: this compares float32 GPU
    // arithmetic against MDSystem's double-precision CPU path, so the floor
    // is float32 precision (~1e-6 relative), not machine-double precision.
    const bool ok = relAccelErr < 1e-4 && relEnergyErr < 1e-4;
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
    const bool mixture = SelfTestMixtureForces(400, 14.0, 2.5, 3, "mixture (Kob-Andersen params)");
    const bool gpu = SelfTestGpuForces(400, 14.0, 2.5, 4, "gpu (kernels.hpp vs. CPU MDSystem)");
    return small && multicell && gradient && mixture && gpu;
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
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6); // needed for the GPU cross-check case
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
