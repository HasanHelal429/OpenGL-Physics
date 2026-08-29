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

// Uploads N random particles, runs one Density+Forces GPU pass (including
// building the spatial hash grid Density()/Forces() now query for SPH
// neighbors -- see kernels.hpp's header comment), and compares against this
// same physics computed independently on the CPU (double precision, plain
// nested loops). Not a physics validation (that is decks/star_relax*.toml +
// tools/plot_star.py, checked against the analytic Lane-Emden profile) --
// this only checks the GPU shaders implement the documented formulas
// correctly. posRange sets the domain half-width particles are scattered
// over: `small` uses a domain not much bigger than 2h (most particles end
// up sharing one or two grid cells, exercising the "everything nearby"
// path); `gridMulticell` spreads them over a domain several cells wide in
// each direction so the 27-neighbor-cell search, empty buckets, and hash
// collisions across genuinely different cells all actually get exercised
// (the failure mode a same-cell-only test can't catch: a bug in computing
// a NEIGHBORING cell's coordinates, or in the CellStart/CellCount
// indexing, could still "work" if every particle only ever looks at its
// own bucket).
bool SelfTestCase(int N, double posRange, double hInit, double treeTheta, double accelTol, const char* label) {
    const double G = 1.0, SOFT2 = 0.01 * 0.01, K = 1.0, GAMMA = 5.0 / 3.0;
    const double VISC_A = 1.0, VISC_B = 2.0;
    const double ETA = 1.2, H_INIT = hInit, H_MIN = 0.2 * H_INIT, H_MAX = 8.0 * H_INIT;
    const int H_ITERS = 3;
    const int BH_TYPE = 2; // Paczynski-Wiita -- exercises more new code than the point-mass case
    const double BH_MASS = 500.0, BH_RS = 0.1;
    const double RHO_FLOOR = 1e-8; // see kernels.hpp Density()/Forces() -- P/rho^2 diverges as rho->0
    const float cellSize = static_cast<float>(2.0 * H_MAX); // see kernels.hpp -- fixed at 2*hMax
    const int hashTableSize = std::max(2 * N, 64);
    const int maxCells = std::max(4 * N, 64); // see TdeSim.hpp -- gravity.tree_max_cells_factor default
    const int treeMaxDepth = 16;              // see TdeSim.hpp -- gravity.tree_max_depth default

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> posDist(-posRange, posRange);
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
        for (int j = 0; j < N; ++j) { // includes j==i -- see kernels.hpp Density()'s comment on the self-term
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
    fw::ComputeShader buildGrid = fw::ComputeShader::FromSource(tde::kernels::BuildGrid());
    fw::ComputeShader computeBoundingBox = fw::ComputeShader::FromSource(tde::kernels::ComputeBoundingBox());
    fw::ComputeShader initTreeRoot = fw::ComputeShader::FromSource(tde::kernels::InitTreeRoot());
    fw::ComputeShader claimSlots = fw::ComputeShader::FromSource(tde::kernels::ClaimSlots());
    fw::ComputeShader resolveSlots = fw::ComputeShader::FromSource(tde::kernels::ResolveSlots());
    fw::ComputeShader computeTreeCellMass = fw::ComputeShader::FromSource(tde::kernels::ComputeTreeCellMass());
    fw::ComputeShader computeMortonKeys = fw::ComputeShader::FromSource(tde::kernels::ComputeMortonKeys());
    fw::ComputeShader bitonicSortStep = fw::ComputeShader::FromSource(tde::kernels::BitonicSortStep());
    GLuint bufPosMass = 0, bufVel = 0, bufAcc = 0, bufRhoPress = 0, bufH = 0;
    GLuint bufCellHead = 0, bufNextIndex = 0;
    GLuint bufTreeChild = 0, bufTreeCellMass = 0, bufTreeCellGeom = 0, bufTreeCellDepth = 0,
           bufTreeCellCounter = 0, bufTreeBBox = 0;
    GLuint bufTreeParticleCell = 0, bufTreeSlotHead = 0, bufTreeSlotNext = 0;
    GLuint bufTreeSortKey = 0, bufTreeSortIndex = 0;
    glCreateBuffers(1, &bufPosMass);
    glCreateBuffers(1, &bufVel);
    glCreateBuffers(1, &bufAcc);
    glCreateBuffers(1, &bufRhoPress);
    glCreateBuffers(1, &bufH);
    glCreateBuffers(1, &bufCellHead);
    glCreateBuffers(1, &bufNextIndex);
    glCreateBuffers(1, &bufTreeChild);
    glCreateBuffers(1, &bufTreeCellMass);
    glCreateBuffers(1, &bufTreeCellGeom);
    glCreateBuffers(1, &bufTreeCellDepth);
    glCreateBuffers(1, &bufTreeCellCounter);
    glCreateBuffers(1, &bufTreeBBox);
    glCreateBuffers(1, &bufTreeParticleCell);
    glCreateBuffers(1, &bufTreeSlotHead);
    glCreateBuffers(1, &bufTreeSlotNext);
    glCreateBuffers(1, &bufTreeSortKey);
    glCreateBuffers(1, &bufTreeSortIndex);
    glNamedBufferData(bufPosMass, N * sizeof(glm::vec4), posMass.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufVel, N * sizeof(glm::vec4), vel.data(), GL_STATIC_DRAW);
    glNamedBufferData(bufAcc, N * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufRhoPress, N * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    const std::vector<float> hSeed(N, static_cast<float>(H_INIT));
    glNamedBufferData(bufH, N * sizeof(float), hSeed.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(bufCellHead, hashTableSize * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufNextIndex, N * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeChild, 8 * maxCells * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeCellMass, maxCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeCellGeom, maxCells * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeCellDepth, maxCells * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeCellCounter, sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeBBox, 6 * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeParticleCell, N * sizeof(int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeSlotHead, 8 * maxCells * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeSlotNext, N * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);

    // Bonsai-style grouped gravity walk (see kernels.hpp's header comment
    // and TdeSim.hpp's m_paddedN comment) -- same padding rule as TdeSim.
    int paddedN = 1;
    while (paddedN < N) paddedN *= 2;
    paddedN = std::max(paddedN, tde::kernels::kGravityGroupSize);
    glNamedBufferData(bufTreeSortKey, paddedN * sizeof(unsigned int), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(bufTreeSortIndex, paddedN * sizeof(int), nullptr, GL_DYNAMIC_DRAW);

    const GLuint groups = static_cast<GLuint>((N + tde::kernels::kWorkgroupSize - 1) / tde::kernels::kWorkgroupSize);
    const GLuint cellGroups = static_cast<GLuint>((maxCells + tde::kernels::kWorkgroupSize - 1) / tde::kernels::kWorkgroupSize);
    const GLuint paddedGroups = static_cast<GLuint>((paddedN + tde::kernels::kWorkgroupSize - 1) / tde::kernels::kWorkgroupSize);
    const GLuint gravityGroups = static_cast<GLuint>((paddedN + tde::kernels::kGravityGroupSize - 1) / tde::kernels::kGravityGroupSize);
    const GLuint slotGroups = static_cast<GLuint>((8 * maxCells + tde::kernels::kWorkgroupSize - 1) / tde::kernels::kWorkgroupSize);

    // Build the atomic-linked-list spatial grid -- same one-pass scheme as
    // TdeSim::BuildSpatialGrid (see kernels.hpp's header comment).
    const unsigned int sentinel = 0xFFFFFFFFu;
    glClearNamedBufferData(bufCellHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &sentinel);
    buildGrid.Use();
    buildGrid.SetInt("uN", N);
    buildGrid.SetFloat("uCellSize", cellSize);
    buildGrid.SetInt("uHashTableSize", hashTableSize);
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(6, bufCellHead);
    fw::ComputeShader::BindBuffer(7, bufNextIndex);
    buildGrid.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Build the Barnes-Hut octree -- same scheme as
    // TdeSim::BuildBarnesHutTree (see kernels.hpp's header comment):
    // bounding box, root init, then a level-by-level (BFS) ClaimSlots/
    // ResolveSlots loop (no locking/spinning -- see that comment for why
    // an earlier CAS-lock design was abandoned after it deadlocked here).
    const unsigned int bboxSeed[6] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u};
    glNamedBufferSubData(bufTreeBBox, 0, sizeof(bboxSeed), bboxSeed);
    computeBoundingBox.Use();
    computeBoundingBox.SetInt("uN", N);
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(13, bufTreeBBox);
    fw::ComputeShader::BindBuffer(14, bufTreeParticleCell);
    computeBoundingBox.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Bonsai-style grouped gravity walk (see kernels.hpp's header comment
    // and TdeSim::SortParticlesByMorton): Morton-sort particle indices so
    // Forces() can dispatch in spatially-tight groups. Reuses the SAME
    // bounding box just computed above -- no separate pass needed here,
    // unlike TdeSim's per-frame-vs-per-substep amortization concern (this
    // is a one-shot selftest, not a running simulation).
    computeMortonKeys.Use();
    computeMortonKeys.SetInt("uN", N);
    computeMortonKeys.SetInt("uPaddedN", paddedN);
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(13, bufTreeBBox);
    fw::ComputeShader::BindBuffer(17, bufTreeSortKey);
    fw::ComputeShader::BindBuffer(18, bufTreeSortIndex);
    computeMortonKeys.Dispatch(paddedGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    bitonicSortStep.Use();
    fw::ComputeShader::BindBuffer(17, bufTreeSortKey);
    fw::ComputeShader::BindBuffer(18, bufTreeSortIndex);
    for (int k = 2; k <= paddedN; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            bitonicSortStep.SetInt("uPaddedN", paddedN);
            bitonicSortStep.SetInt("uK", k);
            bitonicSortStep.SetInt("uJ", j);
            bitonicSortStep.Dispatch(paddedGroups);
            fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }

    initTreeRoot.Use();
    fw::ComputeShader::BindBuffer(8, bufTreeChild);
    fw::ComputeShader::BindBuffer(10, bufTreeCellGeom);
    fw::ComputeShader::BindBuffer(11, bufTreeCellDepth);
    fw::ComputeShader::BindBuffer(12, bufTreeCellCounter);
    fw::ComputeShader::BindBuffer(13, bufTreeBBox);
    initTreeRoot.Dispatch(1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const unsigned int slotSentinel = 0xFFFFFFFFu;
    unsigned int prevCellCount = 1u; // root only, from initTreeRoot
    int actualMaxDepth = treeMaxDepth;
    for (int level = 0; level <= treeMaxDepth; ++level) {
        glClearNamedBufferData(bufTreeSlotHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &slotSentinel);

        claimSlots.Use();
        claimSlots.SetInt("uN", N);
        fw::ComputeShader::BindBuffer(0, bufPosMass);
        fw::ComputeShader::BindBuffer(10, bufTreeCellGeom);
        fw::ComputeShader::BindBuffer(14, bufTreeParticleCell);
        fw::ComputeShader::BindBuffer(15, bufTreeSlotHead);
        fw::ComputeShader::BindBuffer(16, bufTreeSlotNext);
        claimSlots.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        resolveSlots.Use();
        resolveSlots.SetInt("uN", N);
        resolveSlots.SetInt("uMaxCells", maxCells);
        resolveSlots.SetInt("uLevel", level);
        fw::ComputeShader::BindBuffer(8, bufTreeChild);
        fw::ComputeShader::BindBuffer(10, bufTreeCellGeom);
        fw::ComputeShader::BindBuffer(11, bufTreeCellDepth);
        fw::ComputeShader::BindBuffer(12, bufTreeCellCounter);
        fw::ComputeShader::BindBuffer(14, bufTreeParticleCell);
        fw::ComputeShader::BindBuffer(15, bufTreeSlotHead);
        fw::ComputeShader::BindBuffer(16, bufTreeSlotNext);
        resolveSlots.Dispatch(slotGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        unsigned int curCellCount = 0;
        glGetNamedBufferSubData(bufTreeCellCounter, 0, sizeof(unsigned int), &curCellCount);
        if (curCellCount == prevCellCount) {
            actualMaxDepth = level;
            break;
        }
        prevCellCount = curCellCount;
    }

    computeTreeCellMass.Use();
    computeTreeCellMass.SetInt("uN", N);
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(8, bufTreeChild);
    fw::ComputeShader::BindBuffer(9, bufTreeCellMass);
    fw::ComputeShader::BindBuffer(11, bufTreeCellDepth);
    fw::ComputeShader::BindBuffer(12, bufTreeCellCounter);
    for (int level = actualMaxDepth; level >= 0; --level) {
        computeTreeCellMass.SetInt("uLevel", level);
        computeTreeCellMass.Dispatch(cellGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }

    density.Use();
    density.SetInt("uN", N);
    density.SetFloat("uK", static_cast<float>(K));
    density.SetFloat("uGamma", static_cast<float>(GAMMA));
    density.SetFloat("uEta", static_cast<float>(ETA));
    density.SetInt("uHIters", H_ITERS);
    density.SetFloat("uHMin", static_cast<float>(H_MIN));
    density.SetFloat("uHMax", static_cast<float>(H_MAX));
    density.SetFloat("uCellSize", cellSize);
    density.SetInt("uHashTableSize", hashTableSize);
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(3, bufRhoPress);
    fw::ComputeShader::BindBuffer(4, bufH);
    fw::ComputeShader::BindBuffer(6, bufCellHead);
    fw::ComputeShader::BindBuffer(7, bufNextIndex);
    density.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    forces.Use();
    forces.SetInt("uN", N);
    forces.SetInt("uPaddedN", paddedN);
    forces.SetFloat("uG", static_cast<float>(G));
    forces.SetFloat("uSoftening2", static_cast<float>(SOFT2));
    forces.SetFloat("uViscAlpha", static_cast<float>(VISC_A));
    forces.SetFloat("uViscBeta", static_cast<float>(VISC_B));
    forces.SetInt("uBhType", BH_TYPE);
    forces.SetFloat("uBhMass", static_cast<float>(BH_MASS));
    forces.SetFloat("uBhRs", static_cast<float>(BH_RS));
    forces.SetFloat("uCellSize", cellSize);
    forces.SetInt("uHashTableSize", hashTableSize);
    forces.SetFloat("uTreeTheta", static_cast<float>(treeTheta));
    fw::ComputeShader::BindBuffer(0, bufPosMass);
    fw::ComputeShader::BindBuffer(1, bufVel);
    fw::ComputeShader::BindBuffer(2, bufAcc);
    fw::ComputeShader::BindBuffer(3, bufRhoPress);
    fw::ComputeShader::BindBuffer(4, bufH);
    fw::ComputeShader::BindBuffer(6, bufCellHead);
    fw::ComputeShader::BindBuffer(7, bufNextIndex);
    fw::ComputeShader::BindBuffer(8, bufTreeChild);
    fw::ComputeShader::BindBuffer(9, bufTreeCellMass);
    fw::ComputeShader::BindBuffer(10, bufTreeCellGeom);
    fw::ComputeShader::BindBuffer(18, bufTreeSortIndex);
    forces.Dispatch(gravityGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    std::vector<glm::vec4> rhoPressGpu(N), accGpu(N);
    std::vector<float> hGpu(N);
    glGetNamedBufferSubData(bufRhoPress, 0, N * sizeof(glm::vec4), rhoPressGpu.data());
    glGetNamedBufferSubData(bufAcc, 0, N * sizeof(glm::vec4), accGpu.data());
    glGetNamedBufferSubData(bufH, 0, N * sizeof(float), hGpu.data());
    GLuint allBufs[18] = {bufPosMass,          bufVel,             bufAcc,            bufRhoPress,
                          bufH,                bufCellHead,        bufNextIndex,      bufTreeChild,
                          bufTreeCellMass,     bufTreeCellGeom,    bufTreeCellDepth,  bufTreeCellCounter,
                          bufTreeBBox,         bufTreeParticleCell, bufTreeSlotHead,  bufTreeSlotNext,
                          bufTreeSortKey,      bufTreeSortIndex};
    glDeleteBuffers(18, allBufs);

    double hErr = 0.0, rhoErr = 0.0, accErr = 0.0;
    for (int i = 0; i < N; ++i) {
        hErr = std::max(hErr, std::abs(hGpu[i] - hRef[i]) / hRef[i]);
        rhoErr = std::max(rhoErr, std::abs(rhoPressGpu[i].x - rhoRef[i]) / rhoRef[i]);
        const glm::dvec3 da = glm::dvec3(accGpu[i]) - accRef[i];
        accErr = std::max(accErr, glm::length(da) / glm::length(accRef[i]));
    }
    std::printf("selftest (%s): max relative error  h=%.3e  rho=%.3e  accel=%.3e  (theta=%.2f)\n",
                label, hErr, rhoErr, accErr, treeTheta);
    const bool ok = hErr < 1e-4 && rhoErr < 1e-4 && accErr < accelTol;
    std::printf("selftest (%s): %s\n", label, ok ? "PASS" : "FAIL");
    return ok;
}

bool SelfTest() {
    // "small"/"grid-multicell": as before (see SelfTestCase's comment on
    // what each exercises for the SPH grid), now run at theta=0 -- this
    // forces the Barnes-Hut tree walk to descend exhaustively to every
    // leaf with no approximation at all, so it validates the OCTREE
    // MACHINERY ITSELF (build + bottom-up mass pass + tree walk) against
    // the brute-force CPU reference with the same tight tolerance as
    // before, independent of any Barnes-Hut approximation error -- a
    // real, if reordered, sum over the same particles should agree to
    // float-precision-and-reordering noise, not to some looser
    // "reasonable approximation" bound.
    const bool small = SelfTestCase(37, 1.0, 0.3, 0.0, 1e-3, "small");
    const bool gridMulticell = SelfTestCase(2000, 5.0, 0.15, 0.0, 1e-3, "grid-multicell");
    // "tree-theta": a THIRD case at the classic Barnes-Hut default
    // (theta=0.5, what production decks actually use), reporting the real
    // approximation error the tree introduces once it's allowed to
    // actually approximate distant cells -- gated on a much looser bound
    // (this is a sanity check that the approximation is in the expected
    // ballpark, not an exactness check).
    const bool treeTheta = SelfTestCase(2000, 5.0, 0.15, 0.5, 0.2, "tree-theta");
    return small && gridMulticell && treeTheta;
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
