#pragma once

#include "framework/Camera.hpp"
#include "framework/ComputeShader.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Simulation.hpp"
#include "framework/Text.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tde {

// One SPH particle's persistent CPU-side state, matching the 7-float32
// (x,y,z,mass,vx,vy,vz) record written by tools/make_star_ic.py.
struct ParticleRecord {
    glm::vec3 pos;
    float mass;
    glm::vec3 vel;
};

// Newtonian self-gravitating SPH star (Tier 1 of the tidal-disruption
// project -- see README.md): equal-mass SPH particles feel mutual softened
// self-gravity plus a polytropic pressure/artificial-viscosity force,
// integrated by kick-drift-kick leapfrog, all on the GPU (see kernels.hpp).
// Initial conditions are a Lane-Emden polytrope Monte-Carlo-sampled by
// tools/make_star_ic.py; this phase has no black hole yet -- the deliverable
// is a star that sits in hydrostatic equilibrium instead of collapsing or
// flying apart, checked via the diagnostics this Snapshot()s.
class TdeSim : public fw::Simulation {
public:
    TdeSim() = default;
    ~TdeSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;
    void OnKey(int key, int action) override;

private:
    void CreateBuffers();
    void UploadInitial();
    void ComputeDensityAndForces();  // one density pass + one forces pass, with barriers
    void RunAccretion();             // Accretion() pass + CPU readback/sum onto m_bhMass
    void ReadBack();                 // GPU buffers -> m_posMass/m_vel/m_rhoPress/m_h mirrors
    void ResizeBuffersTo(int newN);  // recreate all GPU buffers at a new particle count (immutable storage)
    void SpawnStar(double beta, double eccentricity); // insert a copy of the template star on a new orbit
    void BuildSpatialGrid();         // hash+scatter particles into the grid Density()/Forces() query for SPH neighbors
    void BuildBarnesHutTree();       // bounding box + octree build + bottom-up mass pass, for Forces()'s gravity tree walk
    void SortParticlesByMorton();    // once per FRAME (not substep): Morton-sort for Forces()'s grouped gravity walk

    int m_n = 0;
    int m_nInitial = 0; // Configure()-time particle count -- Reset()'s target, independent of later spawns
    double m_dt = 1e-3;
    int m_substepsPerFrame = 1;

    // Physics parameters (see README.md's deck format for units/derivation).
    double m_G = 1.0;
    double m_softening = 0.05;
    double m_hInit = 0.1;     // initial smoothing-length guess (adapts per-particle from here)
    double m_eta = 1.2;       // adaptive-h target: h = eta*(m/rho)^(1/3), ~40-60 neighbors in 3D
    int m_hIters = 3;         // fixed-point iterations per step, warm-started from last step's h
    double m_hMin = 0.0;      // safety clamp, set from m_hInit in Configure()
    double m_hMax = 0.0;
    double m_K = 1.0;         // polytropic EOS constant, P = K rho^Gamma
    double m_gamma = 5.0 / 3.0;
    double m_viscAlpha = 1.0;
    double m_viscBeta = 2.0;
    double m_damping = 0.0;   // relaxation-only velocity damping rate, 1/time; 0 = off

    // Black hole: fixed point mass at the origin. 0 = off, 1 = Newtonian
    // point mass, 2 = Paczynski-Wiita. m_bhMass is now LIVE, not a fixed
    // deck constant -- it grows via accretion (see RunAccretion); m_bhMassInitial
    // is the Configure()-time value, restored by Reset().
    int m_bhType = 0;
    double m_bhMass = 0.0;
    double m_bhMassInitial = 0.0;
    double m_bhRs = 0.0;
    // A live particle within this radius of the BH is consumed (see
    // kernels::Accretion) -- only meaningful/enabled for PW (m_bhType==2),
    // which is the only case with an actual horizon-like length scale;
    // defaults to m_bhRs if the deck doesn't set one explicitly.
    double m_accretionRadius = 0.0;
    bool m_accretionEnabled = false;
    double m_totalAccreted = 0.0; // cumulative, diagnostic + HUD display only

    std::string m_title = "Tidal Disruption -- SPH star";
    std::vector<std::string> m_diagNames;

    // GPU state (see kernels.hpp for the buffer layout). Immutable-size
    // storage (glNamedBufferStorage) -- growing particle count (spawning a
    // new star, see SpawnStar) means recreating all five plus
    // m_accretedMass at the new size, not resizing in place.
    GLuint m_posMass = 0;
    GLuint m_vel = 0;
    GLuint m_acc = 0;
    GLuint m_rhoPress = 0;
    GLuint m_h = 0;
    GLuint m_accretedMass = 0;

    // Atomic-linked-list spatial hash grid for SPH neighbor search (see
    // kernels.hpp's header comment) -- sized off m_n like the buffers
    // above, rebuilt every substep (BuildSpatialGrid) since positions
    // change every substep. Self-gravity has no cutoff radius; this only
    // accelerates the SPH pressure/viscosity term (self-gravity is now
    // Barnes-Hut, below).
    GLuint m_cellHead = 0;
    GLuint m_nextIndex = 0;
    int m_hashTableSize = 0;
    float m_cellSize = 0.0f; // fixed at 2*m_hMax for the whole run -- see kernels.hpp

    // Barnes-Hut octree for self-gravity (see kernels.hpp's header comment
    // for the full design). Rebuilt every substep since positions move
    // every substep. m_maxCells is sized off m_n like the buffers above
    // (grows on SpawnStar via ResizeBuffersTo, same as everything else).
    GLuint m_treeChild = 0;      // int[8*maxCells]
    GLuint m_treeCellMass = 0;   // vec4[maxCells]: center-of-mass.xyz, total mass
    GLuint m_treeCellGeom = 0;   // vec4[maxCells]: center.xyz, half-size
    GLuint m_treeCellDepth = 0;  // int[maxCells]
    GLuint m_treeCellCounter = 0; // uint[1], atomic bump allocator
    GLuint m_treeBBox = 0;       // uint[6], atomic min/max reduction scratch
    // Level-by-level (BFS) build scratch -- see kernels.hpp's header
    // comment for why this replaced an earlier lock-based CAS design that
    // deadlocked on this hardware.
    GLuint m_treeParticleCell = 0; // int[N]
    GLuint m_treeSlotHead = 0;     // uint[8*maxCells]
    GLuint m_treeSlotNext = 0;     // uint[N]
    int m_maxCells = 0;
    double m_treeTheta = 0.5;      // opening angle; deck gravity.theta (0 = exact/brute-force-equivalent)
    // Both defaults measured empirically against a real n=1.5 polytrope
    // (N=4000), not guessed. First attempt (depth=24, factor=4, no leaf
    // grouping) left 1709/4000 (43%) of particles silently dropped from
    // the tree every substep -- catastrophic (kinetic energy rose ~6400x,
    // virial ratio went from ~0 to 1.7, star no longer bound). Raising
    // factor to 50 and depth to 32 got that down to 108/4000 (2.7%, a
    // float32 precision floor -- 40 gave the same 108, ruling out "just
    // needs more depth"), which was survivable but required subdividing
    // every dense cluster all the way down to single-particle leaves.
    // ResolveSlots' Ncrit=8 leaf grouping (see its own comment) fixed the
    // actual root cause instead of papering over it: a cell with <=8
    // candidates stops subdividing right there, so the real depth needed
    // collapsed to ~log8(N) (measured: depth=4 already gave 0/4000
    // unresolved for this same N=4000 profile, vs. needing 32 before).
    // These defaults keep generous headroom above that measured minimum
    // for larger/denser configurations, not the bare minimum itself.
    int m_treeMaxDepth = 16;       // deck gravity.tree_max_depth
    double m_treeMaxCellsFactor = 4.0; // deck gravity.tree_max_cells_factor -- maxCells = max(64, factor*N)

    // Bonsai-style grouped gravity walk (see kernels.hpp's header comment
    // and Forces() itself for the full design): particles are sorted by
    // Morton code once per FRAME (SortParticlesByMorton, called from the
    // top of Step() -- NOT every substep, since the O(log^2 paddedN)
    // bitonic-sort dispatch count would itself be a real per-substep cost)
    // so that kernels::kGravityGroupSize CONSECUTIVE sorted slots form a
    // genuinely spatially-tight group for Forces()'s cooperative walk.
    // m_paddedN is N rounded up to a power of 2 (bitonic sort's compare-
    // exchange network requires it) and further up to a multiple of
    // kGravityGroupSize (trivially satisfied once padded past it, since
    // both are powers of 2) -- sized off m_n like everything else above,
    // recomputed on SpawnStar via ResizeBuffersTo.
    GLuint m_treeSortKey = 0;   // uint[paddedN]
    GLuint m_treeSortIndex = 0; // int[paddedN]
    int m_paddedN = 0;

    fw::ComputeShader m_density;
    fw::ComputeShader m_forces;
    fw::ComputeShader m_kick;
    fw::ComputeShader m_drift;
    fw::ComputeShader m_accretion;
    fw::ComputeShader m_buildGrid;
    fw::ComputeShader m_computeBoundingBox;
    fw::ComputeShader m_initTreeRoot;
    fw::ComputeShader m_claimSlots;
    fw::ComputeShader m_resolveSlots;
    fw::ComputeShader m_computeTreeCellMass;
    fw::ComputeShader m_computeMortonKeys;
    fw::ComputeShader m_bitonicSortStep;

    // CPU mirrors: m_initial is the reset baseline (never mutated after
    // Configure); the other three are refreshed by ReadBack() (for Snapshot
    // and interactive Render).
    std::vector<ParticleRecord> m_initial;
    std::vector<glm::vec4> m_posMassCpu;
    std::vector<glm::vec4> m_velCpu;
    std::vector<glm::vec4> m_rhoPressCpu;
    std::vector<float> m_hCpu;
    std::vector<float> m_accretedCpu; // RunAccretion's per-step readback scratch

    // Reusable star "shape" for SpawnStar -- position/velocity relative to
    // the loaded star.ic_file's own mass-weighted COM, so a fresh copy can
    // be rigidly re-boosted onto an arbitrary NEW orbit (independent of
    // whatever orbit that file's own particles already carry). Equal-mass
    // particles throughout this project, so one scalar mass suffices.
    std::vector<glm::vec3> m_templatePos;
    std::vector<glm::vec3> m_templateVel;
    int m_templateCount = 0;
    double m_templateMassPerParticle = 0.0;
    double m_templateTotalMass = 0.0;
    double m_templateRStar = 1.0; // spawn.r_star deck key; tidal-radius R_star for SpawnStar's orbit calc

    // Interactive spawn UI state (see OnKey): the orbit parameters the next
    // 'N' press will use, adjustable live; m_spawnAngle fans successive
    // spawns out around the BH instead of stacking them at the same point.
    double m_pendingBeta = 0.47;    // matches the validated "steady feeding" circular case
    double m_pendingEccentricity = 0.0;
    double m_spawnAngle = 0.0;

    // Interactive view. fw::ParticleCloud's constructor calls OpenGL
    // functions immediately (shader compile, glGenVertexArrays/Buffers), so
    // it cannot be a plain default-constructed member: main.cpp's
    // interactive path constructs TdeSim before fw::SimApp creates the
    // window and loads GL function pointers, which made this crash with
    // an access violation (glad's pointers are still null at that point).
    // Headless mode never hit this -- it creates the GL context before
    // constructing TdeSim. Deferred construction to Configure() (which by
    // contract always runs after a valid current GL context exists, in
    // both modes) fixes it.
    fw::Camera m_camera;
    std::unique_ptr<fw::ParticleCloud> m_particles;

    // Status text (BH mass, particle count, pending spawn params) -- own
    // Font/TextRenderer rather than sharing SimApp's, since Simulation::Render
    // only gets a framebuffer size, not the host's font/text objects.
    // TextRenderer's constructor touches GL immediately (VAO/VBO/shader), so
    // like m_particles it must be deferred to Configure(), not default-constructed.
    fw::Font m_font;
    std::unique_ptr<fw::TextRenderer> m_text;
};

} // namespace tde
