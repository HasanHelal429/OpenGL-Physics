#include "TdeSim.hpp"
#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace tde {

namespace {
GLuint MakeVec4Buffer(GLsizei count) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferStorage(buf, count * static_cast<GLsizeiptr>(sizeof(glm::vec4)), nullptr,
                         GL_DYNAMIC_STORAGE_BIT);
    return buf;
}
GLuint MakeFloatBuffer(GLsizei count) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferStorage(buf, count * static_cast<GLsizeiptr>(sizeof(float)), nullptr,
                         GL_DYNAMIC_STORAGE_BIT);
    return buf;
}
GLuint MakeUintBuffer(GLsizei count) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferStorage(buf, count * static_cast<GLsizeiptr>(sizeof(unsigned int)), nullptr,
                         GL_DYNAMIC_STORAGE_BIT);
    return buf;
}
} // namespace

TdeSim::~TdeSim() {
    GLuint bufs[6] = {m_posMass, m_vel, m_acc, m_rhoPress, m_h, m_accretedMass};
    glDeleteBuffers(6, bufs);
    GLuint gridBufs[2] = {m_cellHead, m_nextIndex};
    glDeleteBuffers(2, gridBufs);
    GLuint treeBufs[9] = {m_treeChild,        m_treeCellMass,    m_treeCellGeom,
                          m_treeCellDepth,    m_treeCellCounter, m_treeBBox,
                          m_treeParticleCell, m_treeSlotHead,    m_treeSlotNext};
    glDeleteBuffers(9, treeBufs);
    GLuint sortBufs[2] = {m_treeSortKey, m_treeSortIndex};
    glDeleteBuffers(2, sortBufs);
}

void TdeSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "Tidal Disruption -- SPH star");

    const std::string icFile = deck.GetString("star.ic_file", "");
    if (icFile.empty()) throw std::runtime_error("deck missing [star] ic_file");

    std::ifstream f(icFile, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open ic_file: " + icFile);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    constexpr int kFloatsPerParticle = 7; // x,y,z,mass,vx,vy,vz -- tools/make_star_ic.py
    m_n = static_cast<int>(bytes / (kFloatsPerParticle * static_cast<std::streamsize>(sizeof(float))));
    if (m_n <= 0) throw std::runtime_error("ic_file has no particles: " + icFile);

    std::vector<float> raw(static_cast<size_t>(m_n) * kFloatsPerParticle);
    f.read(reinterpret_cast<char*>(raw.data()), bytes);

    m_initial.resize(m_n);
    for (int i = 0; i < m_n; ++i) {
        const float* p = &raw[static_cast<size_t>(i) * kFloatsPerParticle];
        m_initial[i].pos = glm::vec3(p[0], p[1], p[2]);
        m_initial[i].mass = p[3];
        m_initial[i].vel = glm::vec3(p[4], p[5], p[6]);
    }

    m_G = deck.GetDouble("gravity.G", 1.0);
    m_softening = deck.GetDouble("gravity.softening", 0.05);
    // Barnes-Hut self-gravity (see kernels.hpp's header comment). theta=0
    // forces exhaustive descent (no approximation, equivalent to brute
    // force just via a different summation order); ~0.5 is the classic
    // default trading a small, bounded force error for O(N log N).
    m_treeTheta = deck.GetDouble("gravity.theta", 0.5);
    // See TdeSim.hpp's member comment for how these two defaults were
    // measured against a real stellar density profile, not guessed.
    m_treeMaxDepth = deck.GetInt("gravity.tree_max_depth", 16);
    m_treeMaxCellsFactor = deck.GetDouble("gravity.tree_max_cells_factor", 4.0);

    m_hInit = deck.GetDouble("sph.h_init", 0.1);
    m_eta = deck.GetDouble("sph.eta", 1.2);
    m_hIters = deck.GetInt("sph.h_iters", 3);
    m_hMin = deck.GetDouble("sph.h_min_factor", 0.2) * m_hInit;
    m_hMax = deck.GetDouble("sph.h_max_factor", 8.0) * m_hInit;
    m_K = deck.GetDouble("sph.K", 1.0);
    m_gamma = deck.GetDouble("sph.gamma", 5.0 / 3.0);
    m_viscAlpha = deck.GetDouble("sph.visc_alpha", 1.0);
    m_viscBeta = deck.GetDouble("sph.visc_beta", 2.0);
    m_damping = deck.GetDouble("time.damping", 0.0);

    if (deck.GetBool("blackhole.enabled", false)) {
        const std::string type = deck.GetString("blackhole.type", "point");
        m_bhType = (type == "paczynski_wiita") ? 2 : 1;
        m_bhMass = deck.GetDouble("blackhole.mass", 1000.0);
        m_bhRs = deck.GetDouble("blackhole.schwarzschild_radius", 0.0);
        m_accretionRadius = deck.GetDouble("blackhole.accretion_radius", m_bhRs);
    } else {
        m_bhType = 0;
        m_accretionRadius = 0.0;
    }
    m_bhMassInitial = m_bhMass;
    // Accretion (mass capture -> BH growth, see kernels::Accretion) only
    // makes sense against PW's actual horizon-like length scale r_s -- a
    // Newtonian point mass has no such scale, so it stays off there even
    // if accretion_radius were set.
    m_accretionEnabled = (m_bhType == 2) && (m_accretionRadius > 0.0);
    m_totalAccreted = 0.0;

    m_dt = deck.GetDouble("time.dt", 1e-3);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_diagNames = {"kinetic", "thermal", "potential", "potential_bh", "energy", "virial_2T_over_W",
                   "com_x", "com_y", "com_z", "com_speed", "bh_mass", "total_accreted", "n_particles"};

    m_density = fw::ComputeShader::FromSource(kernels::Density());
    m_forces = fw::ComputeShader::FromSource(kernels::Forces());
    m_kick = fw::ComputeShader::FromSource(kernels::Kick());
    m_drift = fw::ComputeShader::FromSource(kernels::Drift());
    m_accretion = fw::ComputeShader::FromSource(kernels::Accretion());
    m_buildGrid = fw::ComputeShader::FromSource(kernels::BuildGrid());
    m_computeBoundingBox = fw::ComputeShader::FromSource(kernels::ComputeBoundingBox());
    m_initTreeRoot = fw::ComputeShader::FromSource(kernels::InitTreeRoot());
    m_claimSlots = fw::ComputeShader::FromSource(kernels::ClaimSlots());
    m_resolveSlots = fw::ComputeShader::FromSource(kernels::ResolveSlots());
    m_computeTreeCellMass = fw::ComputeShader::FromSource(kernels::ComputeTreeCellMass());
    m_computeMortonKeys = fw::ComputeShader::FromSource(kernels::ComputeMortonKeys());
    m_bitonicSortStep = fw::ComputeShader::FromSource(kernels::BitonicSortStep());

    // Fixed for the whole run (h is always clamped to <=m_hMax by
    // Density()'s own clamp) -- see kernels.hpp's header comment for why
    // this exact bound makes the grid's 27-cell neighbor search exact.
    m_cellSize = static_cast<float>(2.0 * m_hMax);

    // Reusable spawn template: position/velocity relative to THIS file's own
    // mass-weighted COM, independent of whatever orbit its particles already
    // carry -- SpawnStar() rigidly re-boosts a fresh copy onto a new orbit
    // (see TdeSim.hpp's member comment).
    m_templateCount = m_n;
    m_templatePos.resize(m_n);
    m_templateVel.resize(m_n);
    {
        glm::dvec3 comPos(0.0), comVel(0.0);
        double totalMass = 0.0;
        for (int i = 0; i < m_n; ++i) {
            totalMass += m_initial[i].mass;
            comPos += static_cast<double>(m_initial[i].mass) * glm::dvec3(m_initial[i].pos);
            comVel += static_cast<double>(m_initial[i].mass) * glm::dvec3(m_initial[i].vel);
        }
        comPos /= totalMass;
        comVel /= totalMass;
        for (int i = 0; i < m_n; ++i) {
            m_templatePos[i] = m_initial[i].pos - glm::vec3(comPos);
            m_templateVel[i] = m_initial[i].vel - glm::vec3(comVel);
        }
        m_templateMassPerParticle = m_initial[0].mass; // equal-mass particles, project-wide convention
        m_templateTotalMass = totalMass;
    }
    m_templateRStar = deck.GetDouble("spawn.r_star", 1.0);
    m_pendingBeta = deck.GetDouble("spawn.beta", 0.47);
    m_pendingEccentricity = deck.GetDouble("spawn.eccentricity", 0.0);

    m_nInitial = m_n;
    CreateBuffers();
    UploadInitial();
    SortParticlesByMorton(); // Forces()'s grouped gravity walk needs a sorted TreeSortIndex before its first use
    ComputeDensityAndForces(); // acc(t=0), needed before Step's first half-kick

    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = 4.0f;
    m_camera.pitch = 20.0f;

    // See TdeSim.hpp's comment on m_particles/m_text: must be constructed
    // here, not as default-constructed members, since a valid GL context is
    // only guaranteed to exist once Configure() runs.
    m_particles = std::make_unique<fw::ParticleCloud>();
    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);
}

void TdeSim::CreateBuffers() {
    m_posMass = MakeVec4Buffer(m_n);
    m_vel = MakeVec4Buffer(m_n);
    m_acc = MakeVec4Buffer(m_n);
    m_rhoPress = MakeVec4Buffer(m_n);
    m_h = MakeFloatBuffer(m_n);
    m_accretedMass = MakeFloatBuffer(m_n);

    // Hash table sized a few times N -- a handful of empty/lightly-loaded
    // buckets is a fine trade for keeping collision chains short; not
    // performance-critical to tune tightly (see kernels.hpp's header
    // comment). Floored so a tiny spawn template still gets a workable table.
    m_hashTableSize = std::max(2 * m_n, 64);
    m_cellHead = MakeUintBuffer(m_hashTableSize);
    m_nextIndex = MakeUintBuffer(m_n);

    // Barnes-Hut octree cell pool. A perfectly balanced octree needs at
    // most N-1 internal cells for N leaves, but real (non-uniform)
    // distributions can need more -- m_treeMaxCellsFactor gives headroom;
    // if the build ever needs more cells than this, ResolveSlots's own
    // bounds check (see kernels.hpp) just drops those particles from the
    // tree for this one substep rather than overflow the buffer -- a
    // rare, bounded, self-correcting case, not a crash.
    m_maxCells = std::max(static_cast<int>(m_treeMaxCellsFactor * m_n), 64);
    m_treeChild = MakeUintBuffer(8 * m_maxCells);
    m_treeCellMass = MakeVec4Buffer(m_maxCells);
    m_treeCellGeom = MakeVec4Buffer(m_maxCells);
    m_treeCellDepth = MakeUintBuffer(m_maxCells);
    m_treeCellCounter = MakeUintBuffer(1);
    m_treeBBox = MakeUintBuffer(6);
    m_treeParticleCell = MakeUintBuffer(m_n);
    m_treeSlotHead = MakeUintBuffer(8 * m_maxCells);
    m_treeSlotNext = MakeUintBuffer(m_n);

    // Bonsai-style grouped gravity walk (see kernels.hpp's header comment
    // and TdeSim.hpp's m_paddedN comment): N rounded up to a power of 2
    // (BitonicSortStep's compare-exchange network requirement), then
    // further up to at least kGravityGroupSize (also a power of 2, so
    // this is just taking the larger of two powers of 2 -- always exact,
    // never needs a second rounding pass).
    int paddedN = 1;
    while (paddedN < m_n) paddedN *= 2;
    m_paddedN = std::max(paddedN, kernels::kGravityGroupSize);
    m_treeSortKey = MakeUintBuffer(m_paddedN);
    m_treeSortIndex = MakeUintBuffer(m_paddedN);
}

void TdeSim::ResizeBuffersTo(int newN) {
    GLuint bufs[6] = {m_posMass, m_vel, m_acc, m_rhoPress, m_h, m_accretedMass};
    glDeleteBuffers(6, bufs);
    GLuint gridBufs[2] = {m_cellHead, m_nextIndex};
    glDeleteBuffers(2, gridBufs);
    GLuint treeBufs[9] = {m_treeChild,        m_treeCellMass,    m_treeCellGeom,
                          m_treeCellDepth,    m_treeCellCounter, m_treeBBox,
                          m_treeParticleCell, m_treeSlotHead,    m_treeSlotNext};
    glDeleteBuffers(9, treeBufs);
    GLuint sortBufs[2] = {m_treeSortKey, m_treeSortIndex};
    glDeleteBuffers(2, sortBufs);
    m_n = newN;
    CreateBuffers();
}

void TdeSim::UploadInitial() {
    std::vector<glm::vec4> posMass(m_n), vel(m_n);
    for (int i = 0; i < m_n; ++i) {
        posMass[i] = glm::vec4(m_initial[i].pos, m_initial[i].mass);
        vel[i] = glm::vec4(m_initial[i].vel, 0.0f);
    }
    glNamedBufferSubData(m_posMass, 0, m_n * sizeof(glm::vec4), posMass.data());
    glNamedBufferSubData(m_vel, 0, m_n * sizeof(glm::vec4), vel.data());
    const std::vector<glm::vec4> zero(m_n, glm::vec4(0.0f));
    glNamedBufferSubData(m_acc, 0, m_n * sizeof(glm::vec4), zero.data());
    glNamedBufferSubData(m_rhoPress, 0, m_n * sizeof(glm::vec4), zero.data());
    const std::vector<float> hInit(m_n, static_cast<float>(m_hInit));
    glNamedBufferSubData(m_h, 0, m_n * sizeof(float), hInit.data());
}

void TdeSim::Reset() {
    if (m_n != m_nInitial) ResizeBuffersTo(m_nInitial);
    UploadInitial();
    SortParticlesByMorton(); // see Configure()'s identical call for why this must precede ComputeDensityAndForces()
    ComputeDensityAndForces();
    m_bhMass = m_bhMassInitial;
    m_totalAccreted = 0.0;
    m_spawnAngle = 0.0;
}

void TdeSim::SpawnStar(double beta, double eccentricity) {
    // Accretion requires PW (see Configure()); spawning new material to
    // feed only makes sense in the same setup, so keep them coupled rather
    // than allowing orbits to be computed against a black hole with no
    // actual capture mechanism.
    if (m_templateCount <= 0 || m_bhType == 0) return;

    const double rT = m_templateRStar * std::cbrt(m_bhMass / m_templateTotalMass);
    const double rP = rT / beta;
    const double e = std::clamp(eccentricity, 0.0, 0.9);
    const double a = rP / (1.0 - e);
    const double r0 = a * (1.0 + e); // apocenter -- the spawn point, a turning point (purely tangential v)
    const double vApo = std::sqrt(m_G * m_bhMass * (2.0 / r0 - 1.0 / a));

    const double ang = m_spawnAngle;
    m_spawnAngle += glm::radians(137.508); // golden angle: successive spawns fan out, don't stack
    const glm::vec3 orbitPos(static_cast<float>(r0 * std::cos(ang)), static_cast<float>(r0 * std::sin(ang)), 0.0f);
    const glm::vec3 orbitVel(static_cast<float>(-vApo * std::sin(ang)), static_cast<float>(vApo * std::cos(ang)), 0.0f);

    const int oldN = m_n;
    const int addN = m_templateCount;
    const int newN = oldN + addN;

    // Read back the live state BEFORE recreating the (immutable-size) GPU
    // buffers at the new count -- ResizeBuffersTo deletes the old ones.
    std::vector<glm::vec4> posMass(newN), vel(newN), acc(newN), rhoPress(newN);
    std::vector<float> h(newN);
    glGetNamedBufferSubData(m_posMass, 0, oldN * sizeof(glm::vec4), posMass.data());
    glGetNamedBufferSubData(m_vel, 0, oldN * sizeof(glm::vec4), vel.data());
    glGetNamedBufferSubData(m_acc, 0, oldN * sizeof(glm::vec4), acc.data());
    glGetNamedBufferSubData(m_rhoPress, 0, oldN * sizeof(glm::vec4), rhoPress.data());
    glGetNamedBufferSubData(m_h, 0, oldN * sizeof(float), h.data());

    for (int k = 0; k < addN; ++k) {
        posMass[oldN + k] = glm::vec4(orbitPos + m_templatePos[k], static_cast<float>(m_templateMassPerParticle));
        vel[oldN + k] = glm::vec4(orbitVel + m_templateVel[k], 0.0f);
        acc[oldN + k] = glm::vec4(0.0f);
        rhoPress[oldN + k] = glm::vec4(0.0f);
        h[oldN + k] = static_cast<float>(m_hInit);
    }

    ResizeBuffersTo(newN);
    glNamedBufferSubData(m_posMass, 0, newN * sizeof(glm::vec4), posMass.data());
    glNamedBufferSubData(m_vel, 0, newN * sizeof(glm::vec4), vel.data());
    glNamedBufferSubData(m_acc, 0, newN * sizeof(glm::vec4), acc.data());
    glNamedBufferSubData(m_rhoPress, 0, newN * sizeof(glm::vec4), rhoPress.data());
    glNamedBufferSubData(m_h, 0, newN * sizeof(float), h.data());

    SortParticlesByMorton(); // ResizeBuffersTo just recreated TreeSortIndex at the new N -- see Configure()'s identical call
    ComputeDensityAndForces();
    std::printf("[spawn] beta=%.2f e=%.2f  r_p=%.3f r_apo=%.3f  N=%d  BH_mass=%.3f\n",
                beta, e, rP, r0, newN, m_bhMass);
}

void TdeSim::BuildSpatialGrid() {
    if (m_n <= 0) return;
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);

    // Atomic-linked-list grid, one parallel pass, no scan (see
    // kernels.hpp's header comment -- two earlier counting-sort attempts,
    // CPU-scan and single-GPU-thread-scan, were both measured net SLOWER
    // than plain brute force).
    const unsigned int sentinel = 0xFFFFFFFFu;
    glClearNamedBufferData(m_cellHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &sentinel);

    m_buildGrid.Use();
    m_buildGrid.SetInt("uN", m_n);
    m_buildGrid.SetFloat("uCellSize", m_cellSize);
    m_buildGrid.SetInt("uHashTableSize", m_hashTableSize);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(6, m_cellHead);
    fw::ComputeShader::BindBuffer(7, m_nextIndex);
    m_buildGrid.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdeSim::SortParticlesByMorton() {
    if (m_n <= 0) return;
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const GLuint paddedGroups = static_cast<GLuint>((m_paddedN + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);

    // Fresh bounding box for keying (BuildBarnesHutTree recomputes its own
    // a moment later this same substep -- a small, harmless redundancy,
    // not correctness-relevant: this one is only paid once per FRAME,
    // since this whole function runs once at the top of Step(), not once
    // per substep -- see kernels.hpp's header comment on why the sort
    // itself must be amortized this way).
    const unsigned int bboxSeed[6] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u};
    glNamedBufferSubData(m_treeBBox, 0, sizeof(bboxSeed), bboxSeed);
    m_computeBoundingBox.Use();
    m_computeBoundingBox.SetInt("uN", m_n);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(13, m_treeBBox);
    fw::ComputeShader::BindBuffer(14, m_treeParticleCell); // scratch write, harmless -- BuildBarnesHutTree overwrites it again moments later
    m_computeBoundingBox.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_computeMortonKeys.Use();
    m_computeMortonKeys.SetInt("uN", m_n);
    m_computeMortonKeys.SetInt("uPaddedN", m_paddedN);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(13, m_treeBBox);
    fw::ComputeShader::BindBuffer(17, m_treeSortKey);
    fw::ComputeShader::BindBuffer(18, m_treeSortIndex);
    m_computeMortonKeys.Dispatch(paddedGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Standard bitonic sort network: log2(paddedN)*(log2(paddedN)+1)/2
    // barrier-separated compare-exchange passes -- see kernels.hpp's
    // BitonicSortStep comment for why this needs no atomics (unlike the
    // octree build's level loop, no invocation ever shares a slot with
    // another WITHIN one dispatch).
    m_bitonicSortStep.Use();
    fw::ComputeShader::BindBuffer(17, m_treeSortKey);
    fw::ComputeShader::BindBuffer(18, m_treeSortIndex);
    for (int k = 2; k <= m_paddedN; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            m_bitonicSortStep.SetInt("uPaddedN", m_paddedN);
            m_bitonicSortStep.SetInt("uK", k);
            m_bitonicSortStep.SetInt("uJ", j);
            m_bitonicSortStep.Dispatch(paddedGroups);
            fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
}

void TdeSim::BuildBarnesHutTree() {
    if (m_n <= 0) return;
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const GLuint cellGroups = static_cast<GLuint>((m_maxCells + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    // ResolveSlots is dispatched over 8*maxCells (one thread per potential
    // (cell,octant) slot), not maxCells -- a distinct group count from
    // ComputeTreeCellMass's per-cell dispatch below.
    const GLuint slotGroups = static_cast<GLuint>((8 * m_maxCells + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);

    // Seed the bounding-box reduction: [0,1,2] (flipped min x,y,z) to the
    // largest possible uint so any real value lowers it via atomicMin;
    // [3,4,5] (flipped max x,y,z) to 0 so any real value raises it via
    // atomicMax. A tiny 6-uint upload, not a readback -- no stall.
    const unsigned int bboxSeed[6] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u};
    glNamedBufferSubData(m_treeBBox, 0, sizeof(bboxSeed), bboxSeed);

    m_computeBoundingBox.Use();
    m_computeBoundingBox.SetInt("uN", m_n);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(13, m_treeBBox);
    fw::ComputeShader::BindBuffer(14, m_treeParticleCell);
    m_computeBoundingBox.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_initTreeRoot.Use();
    fw::ComputeShader::BindBuffer(8, m_treeChild);
    fw::ComputeShader::BindBuffer(10, m_treeCellGeom);
    fw::ComputeShader::BindBuffer(11, m_treeCellDepth);
    fw::ComputeShader::BindBuffer(12, m_treeCellCounter);
    fw::ComputeShader::BindBuffer(13, m_treeBBox);
    m_initTreeRoot.Dispatch(1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Level-by-level (BFS) build -- see kernels.hpp's header comment for
    // why this replaced an earlier lock-based design that deadlocked on
    // this hardware. Each level is ClaimSlots (particles claim a slot in
    // their current cell) then ResolveSlots (each slot resolves its
    // candidate list into either a direct leaf, a leaf group, or a new
    // cell needing further subdivision), separated by a barrier since
    // ResolveSlots depends on ClaimSlots having fully finished writing
    // this level's slot lists.
    //
    // Early exit: Ncrit=8 leaf grouping (ResolveSlots' own comment) means
    // a real stellar density profile stops growing the tree after only
    // ~log8(N) levels -- measured at ~4-6 for N=4000-24000, far below
    // uTreeMaxDepth's safety headroom (16). Blindly dispatching every
    // remaining level anyway is NOT free the way a single per-invocation
    // depth check is inside a shader: each iteration here is still a
    // buffer clear + 2 kernel launches + 2 barriers, and that fixed
    // per-level GPU submission cost, multiplied by 250+ substeps/frame,
    // was measured to dominate over the O(N log N) win this tree is
    // supposed to provide (a 24000-particle benchmark ran SLOWER with the
    // tree than brute-force gravity until this loop stopped once no new
    // cells appear). One 4-byte synchronous readback of the cell counter
    // per level actually run (not per uTreeMaxDepth iteration) is the
    // cost of detecting that -- cheap next to a whole extra level's worth
    // of GPU submissions, and it also drives ComputeTreeCellMass's own
    // loop bound below so that one stops early too.
    const unsigned int slotSentinel = 0xFFFFFFFFu;
    unsigned int prevCellCount = 1u; // root only, from InitTreeRoot
    int actualMaxDepth = m_treeMaxDepth;
    for (int level = 0; level <= m_treeMaxDepth; ++level) {
        glClearNamedBufferData(m_treeSlotHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &slotSentinel);

        m_claimSlots.Use();
        m_claimSlots.SetInt("uN", m_n);
        fw::ComputeShader::BindBuffer(0, m_posMass);
        fw::ComputeShader::BindBuffer(10, m_treeCellGeom);
        fw::ComputeShader::BindBuffer(14, m_treeParticleCell);
        fw::ComputeShader::BindBuffer(15, m_treeSlotHead);
        fw::ComputeShader::BindBuffer(16, m_treeSlotNext);
        m_claimSlots.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        m_resolveSlots.Use();
        m_resolveSlots.SetInt("uN", m_n);
        m_resolveSlots.SetInt("uMaxCells", m_maxCells);
        m_resolveSlots.SetInt("uLevel", level);
        fw::ComputeShader::BindBuffer(8, m_treeChild);
        fw::ComputeShader::BindBuffer(10, m_treeCellGeom);
        fw::ComputeShader::BindBuffer(11, m_treeCellDepth);
        fw::ComputeShader::BindBuffer(12, m_treeCellCounter);
        fw::ComputeShader::BindBuffer(14, m_treeParticleCell);
        fw::ComputeShader::BindBuffer(15, m_treeSlotHead);
        fw::ComputeShader::BindBuffer(16, m_treeSlotNext);
        m_resolveSlots.Dispatch(slotGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        unsigned int curCellCount = 0;
        glGetNamedBufferSubData(m_treeCellCounter, 0, sizeof(unsigned int), &curCellCount);
        if (curCellCount == prevCellCount) {
            actualMaxDepth = level; // no cell exists past this depth -- stop
            break;
        }
        prevCellCount = curCellCount;
    }

    // Bottom-up mass/center-of-mass pass, once per depth level from the
    // deepest level actually reached (not uTreeMaxDepth -- see above) down
    // to the root -- see kernels.hpp's header comment for why this
    // depth-driven dispatch order is correct without needing per-cell
    // "children ready" tracking.
    m_computeTreeCellMass.Use();
    m_computeTreeCellMass.SetInt("uN", m_n);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(8, m_treeChild);
    fw::ComputeShader::BindBuffer(9, m_treeCellMass);
    fw::ComputeShader::BindBuffer(11, m_treeCellDepth);
    fw::ComputeShader::BindBuffer(12, m_treeCellCounter);
    for (int level = actualMaxDepth; level >= 0; --level) {
        m_computeTreeCellMass.SetInt("uLevel", level);
        m_computeTreeCellMass.Dispatch(cellGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
}

void TdeSim::ComputeDensityAndForces() {
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    // Forces() dispatches over the Morton-sorted slot array in groups of
    // kernels::kGravityGroupSize (NOT kWorkgroupSize) -- see kernels.hpp's
    // header comment on the Bonsai-style grouped gravity walk.
    const GLuint gravityGroups = static_cast<GLuint>((m_paddedN + kernels::kGravityGroupSize - 1) / kernels::kGravityGroupSize);

    BuildSpatialGrid();    // positions just moved (or this is t=0) -- rebuild before either pass queries it
    BuildBarnesHutTree();  // same reason -- Forces()'s gravity tree walk needs this substep's tree

    m_density.Use();
    m_density.SetInt("uN", m_n);
    m_density.SetFloat("uK", static_cast<float>(m_K));
    m_density.SetFloat("uGamma", static_cast<float>(m_gamma));
    m_density.SetFloat("uEta", static_cast<float>(m_eta));
    m_density.SetInt("uHIters", m_hIters);
    m_density.SetFloat("uHMin", static_cast<float>(m_hMin));
    m_density.SetFloat("uHMax", static_cast<float>(m_hMax));
    m_density.SetFloat("uCellSize", m_cellSize);
    m_density.SetInt("uHashTableSize", m_hashTableSize);
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(3, m_rhoPress);
    fw::ComputeShader::BindBuffer(4, m_h);
    fw::ComputeShader::BindBuffer(6, m_cellHead);
    fw::ComputeShader::BindBuffer(7, m_nextIndex);
    m_density.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_forces.Use();
    m_forces.SetInt("uN", m_n);
    m_forces.SetInt("uPaddedN", m_paddedN);
    m_forces.SetFloat("uG", static_cast<float>(m_G));
    m_forces.SetFloat("uSoftening2", static_cast<float>(m_softening * m_softening));
    m_forces.SetFloat("uViscAlpha", static_cast<float>(m_viscAlpha));
    m_forces.SetFloat("uViscBeta", static_cast<float>(m_viscBeta));
    m_forces.SetInt("uBhType", m_bhType);
    m_forces.SetFloat("uBhMass", static_cast<float>(m_bhMass));
    m_forces.SetFloat("uBhRs", static_cast<float>(m_bhRs));
    m_forces.SetFloat("uCellSize", m_cellSize);
    m_forces.SetInt("uHashTableSize", m_hashTableSize);
    m_forces.SetFloat("uTreeTheta", static_cast<float>(m_treeTheta));
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(1, m_vel);
    fw::ComputeShader::BindBuffer(2, m_acc);
    fw::ComputeShader::BindBuffer(3, m_rhoPress);
    fw::ComputeShader::BindBuffer(4, m_h);
    fw::ComputeShader::BindBuffer(6, m_cellHead);
    fw::ComputeShader::BindBuffer(7, m_nextIndex);
    fw::ComputeShader::BindBuffer(8, m_treeChild);
    fw::ComputeShader::BindBuffer(9, m_treeCellMass);
    fw::ComputeShader::BindBuffer(10, m_treeCellGeom);
    fw::ComputeShader::BindBuffer(18, m_treeSortIndex);
    m_forces.Dispatch(gravityGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdeSim::RunAccretion() {
    if (!m_accretionEnabled || m_n <= 0) return;
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    m_accretion.Use();
    m_accretion.SetInt("uN", m_n);
    m_accretion.SetFloat("uAccretionRadius", static_cast<float>(m_accretionRadius));
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(5, m_accretedMass);
    m_accretion.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_accretedCpu.resize(m_n);
    glGetNamedBufferSubData(m_accretedMass, 0, m_n * sizeof(float), m_accretedCpu.data());
    double gained = 0.0;
    for (float m : m_accretedCpu) gained += m;
    if (gained > 0.0) {
        m_bhMass += gained;
        m_totalAccreted += gained;
    }
}

void TdeSim::Step(int substeps) {
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const float halfDt = static_cast<float>(0.5 * m_dt);
    const float dampingFactor = static_cast<float>(std::exp(-m_damping * halfDt));

    // Once per FRAME, not once per substep -- see kernels.hpp's header
    // comment on Forces()'s Bonsai-style grouped gravity walk for why the
    // Morton sort's own O(log^2 paddedN) dispatch count needs to be
    // amortized over a whole frame's worth of substeps rather than paid
    // every single one.
    SortParticlesByMorton();

    auto kick = [&]() {
        m_kick.Use();
        m_kick.SetInt("uN", m_n);
        m_kick.SetFloat("uHalfDt", halfDt);
        m_kick.SetFloat("uDampingFactor", dampingFactor);
        fw::ComputeShader::BindBuffer(1, m_vel);
        fw::ComputeShader::BindBuffer(2, m_acc);
        m_kick.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    auto drift = [&]() {
        m_drift.Use();
        m_drift.SetInt("uN", m_n);
        m_drift.SetFloat("uDt", static_cast<float>(m_dt));
        fw::ComputeShader::BindBuffer(0, m_posMass);
        fw::ComputeShader::BindBuffer(1, m_vel);
        m_drift.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    for (int s = 0; s < substeps; ++s) {
        kick();                    // v(t+dt/2) = v(t) + (dt/2) a(t)
        drift();                   // x(t+dt) = x(t) + dt v(t+dt/2)
        RunAccretion();            // consume anything now inside the horizon, grow the BH
        ComputeDensityAndForces(); // a(t+dt)
        kick();                    // v(t+dt) = v(t+dt/2) + (dt/2) a(t+dt)
    }
}

void TdeSim::ReadBack() {
    m_posMassCpu.resize(m_n);
    m_velCpu.resize(m_n);
    m_rhoPressCpu.resize(m_n);
    m_hCpu.resize(m_n);
    glGetNamedBufferSubData(m_posMass, 0, m_n * sizeof(glm::vec4), m_posMassCpu.data());
    glGetNamedBufferSubData(m_vel, 0, m_n * sizeof(glm::vec4), m_velCpu.data());
    glGetNamedBufferSubData(m_rhoPress, 0, m_n * sizeof(glm::vec4), m_rhoPressCpu.data());
    glGetNamedBufferSubData(m_h, 0, m_n * sizeof(float), m_hCpu.data());
}

void TdeSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();

    double kinetic = 0.0, thermal = 0.0, potential = 0.0, totalMass = 0.0;
    glm::dvec3 com(0.0), comP(0.0); // comP = mass-weighted momentum, for COM velocity
    for (int i = 0; i < m_n; ++i) {
        const double m = m_posMassCpu[i].w;
        const glm::dvec3 v(m_velCpu[i]);
        const double rho = m_rhoPressCpu[i].x;
        const double P = m_rhoPressCpu[i].y;
        totalMass += m;
        com += m * glm::dvec3(m_posMassCpu[i]);
        comP += m * v;
        kinetic += 0.5 * m * glm::dot(v, v);
        if (rho > 0.0) thermal += m * P / ((m_gamma - 1.0) * rho);
    }
    for (int i = 0; i < m_n; ++i) {
        const glm::dvec3 ri(m_posMassCpu[i]);
        for (int j = i + 1; j < m_n; ++j) {
            const glm::dvec3 rij = ri - glm::dvec3(m_posMassCpu[j]);
            const double r = std::sqrt(glm::dot(rij, rij) + m_softening * m_softening);
            potential -= m_G * m_posMassCpu[i].w * m_posMassCpu[j].w / r;
        }
    }
    double potentialBh = 0.0;
    if (m_bhType != 0) {
        for (int i = 0; i < m_n; ++i) {
            const double r = glm::length(glm::dvec3(m_posMassCpu[i]));
            const double m = m_posMassCpu[i].w;
            if (m_bhType == 1) {
                potentialBh -= m_G * m_bhMass * m / std::sqrt(r * r + m_softening * m_softening);
            } else {
                const double dr = std::max(r - m_bhRs, m_softening);
                potentialBh -= m_G * m_bhMass * m / dr;
            }
        }
    }
    const glm::dvec3 comVel = comP / totalMass;
    com /= totalMass;
    const double energy = kinetic + thermal + potential + potentialBh;
    const double virial = (potential != 0.0) ? 2.0 * kinetic / std::abs(potential) : 0.0;

    writer.WriteField("pos_mass", m_posMassCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("vel", m_velCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("rho_press", m_rhoPressCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("h", m_hCpu.data(), fw::NpyDtype::F4, m_n, 1);
    writer.WriteScalar("kinetic", kinetic);
    writer.WriteScalar("thermal", thermal);
    writer.WriteScalar("potential", potential);
    writer.WriteScalar("potential_bh", potentialBh);
    writer.WriteScalar("energy", energy);
    writer.WriteScalar("virial_2T_over_W", virial);
    writer.WriteScalar("com_x", com.x);
    writer.WriteScalar("com_y", com.y);
    writer.WriteScalar("com_z", com.z);
    writer.WriteScalar("com_speed", glm::length(comVel));
    writer.WriteScalar("bh_mass", m_bhMass);
    writer.WriteScalar("total_accreted", m_totalAccreted);
    writer.WriteScalar("n_particles", static_cast<double>(m_n));
}

fw::SimInfo TdeSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"pos_mass", "vel", "rho_press", "h"};
    info.diagnostics = m_diagNames;
    return info;
}

void TdeSim::Render(int fbWidth, int fbHeight) {
    ReadBack();

    // Accreted (mass<=0) particles are excluded from both the brightness
    // scale and the draw list -- they simply vanish once consumed, rather
    // than lingering at the horizon as dead-but-visible dots (see
    // kernels::Accretion's comment on why they keep drifting internally).
    double rhoMax = 1e-30;
    for (int i = 0; i < m_n; ++i) {
        if (m_posMassCpu[i].w <= 0.0f) continue;
        rhoMax = std::max(rhoMax, static_cast<double>(m_rhoPressCpu[i].x));
    }

    // ParticleCloud's point size is perspective-correct (a fixed world-space
    // radius shrinks on screen as the camera pulls back), which is right for
    // watching the compact relaxed star up close but makes an encounter's
    // particles vanish to sub-pixel once zoomed out far enough to see the
    // whole orbit (tens to hundreds of length units). Scaling world-space
    // size by the current camera distance (calibrated against the initial
    // distance=4 setup below, where 0.02 already looked right for the
    // compact star) keeps the on-screen size roughly constant across zoom
    // levels instead.
    const float sizeScale = m_camera.distance / 4.0f;
    const float starSize = 0.02f * sizeScale;

    std::vector<fw::ParticleInstance> particles;
    particles.reserve(static_cast<size_t>(m_n) + 1);
    for (int i = 0; i < m_n; ++i) {
        if (m_posMassCpu[i].w <= 0.0f) continue;
        fw::ParticleInstance p;
        const float t = static_cast<float>(std::pow(m_rhoPressCpu[i].x / rhoMax, 0.35)); // gamma-lift for dim outskirts
        p.position = glm::vec3(m_posMassCpu[i]);
        p.color = glm::vec4(0.3f + 0.7f * t, 0.5f * (1.0f - t) + 0.2f, 1.0f - 0.6f * t, 1.0f);
        p.size = starSize;
        particles.push_back(p);
    }
    if (m_bhType != 0) {
        fw::ParticleInstance bh;
        bh.position = glm::vec3(0.0f); // fixed at the origin, see Forces()'s black-hole comment
        bh.color = glm::vec4(1.0f, 0.95f, 0.8f, 1.0f); // bright warm white, distinct from the star's magma tones
        // Grows visibly (cube-root: mass ~ radius^3 for a fixed-density
        // scaling) as it accretes, so feeding has an on-screen payoff
        // beyond the HUD number below.
        const float growth = static_cast<float>(std::cbrt(std::max(m_bhMass, 1e-9) / std::max(m_bhMassInitial, 1e-9)));
        bh.size = 3.0f * starSize * growth;
        particles.push_back(bh);
    }
    m_particles->SetParticles(particles);

    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const float aspect = static_cast<float>(fbWidth) / static_cast<float>(std::max(fbHeight, 1));
    m_particles->Draw(m_camera.ViewMatrix(), m_camera.ProjectionMatrix(aspect), static_cast<float>(fbHeight));

    // Status HUD -- own Font/TextRenderer, see TdeSim.hpp's member comment.
    m_text->SetViewport(fbWidth, fbHeight);
    const glm::vec4 white(1.0f, 1.0f, 1.0f, 0.9f);
    const glm::vec4 dim(0.75f, 0.85f, 1.0f, 0.85f);
    char line[256];
    float y = 22.0f;
    std::snprintf(line, sizeof(line), "BH mass: %.3f  (accreted %.3f, x%.2f)", m_bhMass, m_totalAccreted,
                  m_bhMass / std::max(m_bhMassInitial, 1e-9));
    m_text->Draw(m_font, line, glm::vec2(12.0f, y), white);
    y += 20.0f;
    std::snprintf(line, sizeof(line), "particles: %d", m_n);
    m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    y += 20.0f;
    if (m_bhType != 0) {
        std::snprintf(line, sizeof(line), "spawn [N]: beta=%.2f  ecc=%.2f   (UP/DOWN: beta, [ / ]: ecc)",
                      m_pendingBeta, m_pendingEccentricity);
        m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    }
}

void TdeSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) m_camera.Orbit(static_cast<float>(in.dx), static_cast<float>(in.dy));
    if (in.scrollDelta != 0.0) m_camera.Zoom(static_cast<float>(in.scrollDelta));
}

void TdeSim::OnKey(int key, int action) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_N:
            if (action == GLFW_PRESS) SpawnStar(m_pendingBeta, m_pendingEccentricity);
            break;
        case GLFW_KEY_UP:
            m_pendingBeta = std::min(m_pendingBeta + 0.02, 2.0);
            break;
        case GLFW_KEY_DOWN:
            m_pendingBeta = std::max(m_pendingBeta - 0.02, 0.05);
            break;
        case GLFW_KEY_LEFT_BRACKET:
            m_pendingEccentricity = std::max(m_pendingEccentricity - 0.05, 0.0);
            break;
        case GLFW_KEY_RIGHT_BRACKET:
            m_pendingEccentricity = std::min(m_pendingEccentricity + 0.05, 0.9);
            break;
        default:
            break;
    }
}

} // namespace tde
