#include "ngrav/gpu/GpuBarnesHut.hpp"

#include "framework/ComputeShader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

// Ported from projects/06_tidal_disruption/src/kernels.hpp + TdeSim.cpp,
// stripped to gravity only (see this module's own header comment). Binding
// numbers are kept IDENTICAL to 06's for every buffer that survives the
// port, so a reader who knows 06's pipeline can cross-reference directly;
// bindings 1 (velocity), 3-7 (SPH density/pressure/smoothing-length/spatial-
// hash) are simply never declared here.
namespace ngrav::gpu {

namespace {

constexpr int kWorkgroupSize = 256;
constexpr int kGravityGroupSize = 64;

// float<->uint bijection that preserves ordering (so atomicMin/atomicMax on
// the *bit pattern* of a float behaves like min/max on the float itself,
// including correctly ordering negative values -- GLSL has no atomic float
// min/max); octant/child-center helpers for the tree build.
constexpr const char* kTreeCommonGlsl = R"(
uint floatFlipForAtomic(uint u) {
    uint mask = (0u - (u >> 31)) | 0x80000000u;
    return u ^ mask;
}
float floatUnflipFromAtomic(uint u) {
    uint mask = ((u >> 31) - 1u) | 0x80000000u;
    return uintBitsToFloat(u ^ mask);
}
int octantOf(vec3 pos, vec3 center) {
    int oct = 0;
    if (pos.x >= center.x) oct |= 1;
    if (pos.y >= center.y) oct |= 2;
    if (pos.z >= center.z) oct |= 4;
    return oct;
}
vec3 childCenter(vec3 parentCenter, float parentHalfSize, int octant) {
    float q = 0.5 * parentHalfSize;
    return parentCenter + vec3(((octant & 1) != 0) ? q : -q,
                               ((octant & 2) != 0) ? q : -q,
                               ((octant & 4) != 0) ? q : -q);
}
)";

std::string ComputeBoundingBoxSrc() {
    return std::string(R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 13) buffer TreeBBoxBuf { uint bbox[]; };
layout(std430, binding = 14) writeonly buffer TreeParticleCellBuf { int treeParticleCell[]; };

uniform int uN;
)") + kTreeCommonGlsl + R"(
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vec3 p = posMass[i].xyz;
    uint fx = floatFlipForAtomic(floatBitsToUint(p.x));
    uint fy = floatFlipForAtomic(floatBitsToUint(p.y));
    uint fz = floatFlipForAtomic(floatBitsToUint(p.z));
    atomicMin(bbox[0], fx);
    atomicMin(bbox[1], fy);
    atomicMin(bbox[2], fz);
    atomicMax(bbox[3], fx);
    atomicMax(bbox[4], fy);
    atomicMax(bbox[5], fz);
    treeParticleCell[i] = 0; // everyone starts owned by the root this step
}
)";
}

std::string ComputeMortonKeysSrc() {
    return std::string(R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 13) readonly buffer TreeBBoxBuf { uint bbox[]; };
layout(std430, binding = 17) writeonly buffer TreeSortKeyBuf { uint sortKey[]; };
layout(std430, binding = 18) writeonly buffer TreeSortIndexBuf { int sortIndex[]; };

uniform int uN;
uniform int uPaddedN;
)") + kTreeCommonGlsl + R"(
uint expandBits10(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

void main() {
    uint slot = gl_GlobalInvocationID.x;
    if (slot >= uint(uPaddedN)) return;

    if (slot >= uint(uN)) {
        sortKey[slot] = 0xFFFFFFFFu; // padding -- always sorts last
        sortIndex[slot] = -1;
        return;
    }

    vec3 lo = vec3(floatUnflipFromAtomic(bbox[0]), floatUnflipFromAtomic(bbox[1]), floatUnflipFromAtomic(bbox[2]));
    vec3 hi = vec3(floatUnflipFromAtomic(bbox[3]), floatUnflipFromAtomic(bbox[4]), floatUnflipFromAtomic(bbox[5]));
    vec3 extent = max(hi - lo, vec3(1e-6));

    vec3 t = clamp((posMass[slot].xyz - lo) / extent, 0.0, 0.999999);
    uint xx = expandBits10(uint(t.x * 1024.0));
    uint yy = expandBits10(uint(t.y * 1024.0));
    uint zz = expandBits10(uint(t.z * 1024.0));
    sortKey[slot] = (xx << 2) | (yy << 1) | zz; // already-spread coords, bit-interleaved by the shifts
    sortIndex[slot] = int(slot);
}
)";
}

constexpr const char* kBitonicSortStepSrc = R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 17) buffer TreeSortKeyBuf { uint sortKey[]; };
layout(std430, binding = 18) buffer TreeSortIndexBuf { int sortIndex[]; };

uniform int uPaddedN;
uniform int uK;
uniform int uJ;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uPaddedN)) return;
    uint ixj = i ^ uint(uJ);
    if (ixj <= i || ixj >= uint(uPaddedN)) return;

    bool ascending = ((i & uint(uK)) == 0u);
    uint ki = sortKey[i];
    uint kj = sortKey[ixj];
    if (ascending ? (ki > kj) : (ki < kj)) {
        sortKey[i] = kj;
        sortKey[ixj] = ki;
        int vi = sortIndex[i];
        sortIndex[i] = sortIndex[ixj];
        sortIndex[ixj] = vi;
    }
}
)";

std::string InitTreeRootSrc() {
    return std::string(R"(#version 460 core
layout(local_size_x = 1) in;

layout(std430, binding = 8) writeonly buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 10) writeonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 11) writeonly buffer TreeCellDepthBuf { int treeCellDepth[]; };
layout(std430, binding = 12) writeonly buffer TreeCellCounterBuf { uint treeCellCounter[]; };
layout(std430, binding = 13) readonly buffer TreeBBoxBuf { uint bbox[]; };
)") + kTreeCommonGlsl + R"(
void main() {
    float minX = floatUnflipFromAtomic(bbox[0]);
    float minY = floatUnflipFromAtomic(bbox[1]);
    float minZ = floatUnflipFromAtomic(bbox[2]);
    float maxX = floatUnflipFromAtomic(bbox[3]);
    float maxY = floatUnflipFromAtomic(bbox[4]);
    float maxZ = floatUnflipFromAtomic(bbox[5]);
    vec3 center = 0.5 * (vec3(minX, minY, minZ) + vec3(maxX, maxY, maxZ));
    float halfSize = 0.5 * max(maxX - minX, max(maxY - minY, maxZ - minZ));
    halfSize = max(halfSize * 1.001, 1e-6);
    treeCellGeom[0] = vec4(center, halfSize);
    treeCellDepth[0] = 0;
    for (int k = 0; k < 8; ++k) treeChild[8 * 0 + k] = -1;
    treeCellCounter[0] = 1u;
}
)";
}

std::string ClaimSlotsSrc() {
    return std::string(R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 10) readonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 14) readonly buffer TreeParticleCellBuf { int treeParticleCell[]; };
layout(std430, binding = 15) buffer TreeSlotHeadBuf { uint treeSlotHead[]; };
layout(std430, binding = 16) writeonly buffer TreeSlotNextBuf { uint treeSlotNext[]; };

uniform int uN;
)") + kTreeCommonGlsl + R"(
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    int cell = treeParticleCell[i];
    if (cell < 0) return; // already resolved into a leaf in an earlier level

    vec4 geom = treeCellGeom[cell];
    int octant = octantOf(posMass[i].xyz, geom.xyz);
    uint slot = uint(8 * cell + octant);
    treeSlotNext[i] = atomicExchange(treeSlotHead[slot], i);
}
)";
}

std::string ResolveSlotsSrc() {
    return std::string(R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 8) buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 10) buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 11) buffer TreeCellDepthBuf { int treeCellDepth[]; };
layout(std430, binding = 12) buffer TreeCellCounterBuf { uint treeCellCounter[]; };
layout(std430, binding = 14) buffer TreeParticleCellBuf { int treeParticleCell[]; };
layout(std430, binding = 15) readonly buffer TreeSlotHeadBuf { uint treeSlotHead[]; };
layout(std430, binding = 16) readonly buffer TreeSlotNextBuf { uint treeSlotNext[]; };

uniform int uN;
uniform int uMaxCells;
uniform int uLevel;
)") + kTreeCommonGlsl + R"(
const uint kEmptySlot = 0xFFFFFFFFu;
const int kNcrit = 8;

void main() {
    uint slotIdx = gl_GlobalInvocationID.x;
    if (slotIdx >= uint(8 * uMaxCells)) return;
    int cell = int(slotIdx / 8u);
    int octant = int(slotIdx % 8u);
    if (uint(cell) >= treeCellCounter[0]) return;
    if (treeCellDepth[cell] != uLevel) return;

    uint head = treeSlotHead[slotIdx];
    if (head == kEmptySlot) return; // nothing claimed this slot this level

    uint next = treeSlotNext[head];
    if (next == kEmptySlot) {
        // exactly one candidate -- direct leaf
        treeChild[int(slotIdx)] = int(head);
        treeParticleCell[head] = -1; // resolved, stop participating
        return;
    }

    int members[8];
    int count = 0;
    bool overflow = false;
    uint p = head;
    while (p != kEmptySlot) {
        if (count == kNcrit) { overflow = true; break; }
        members[count] = int(p);
        count++;
        p = treeSlotNext[p];
    }

    vec4 geom = treeCellGeom[cell];
    int newCell = int(atomicAdd(treeCellCounter[0], 1u));
    if (newCell >= uMaxCells) return; // cell pool exhausted
    vec3 newCenter = childCenter(geom.xyz, geom.w, octant);
    float newHalfSize = 0.5 * geom.w;
    treeCellGeom[newCell] = vec4(newCenter, newHalfSize);
    treeCellDepth[newCell] = uLevel + 1;
    treeChild[int(slotIdx)] = uN + newCell;

    if (!overflow) {
        for (int k = 0; k < kNcrit; ++k) {
            treeChild[8 * newCell + k] = (k < count) ? members[k] : -1;
        }
        for (int k = 0; k < count; ++k) {
            treeParticleCell[members[k]] = -1;
        }
    } else {
        for (int k = 0; k < 8; ++k) treeChild[8 * newCell + k] = -1;
        uint pp = head;
        while (pp != kEmptySlot) {
            uint pNext = treeSlotNext[pp];
            treeParticleCell[pp] = newCell;
            pp = pNext;
        }
    }
}
)";
}

constexpr const char* kComputeTreeCellMassSrc = R"(#version 460 core
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 8) readonly buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 9) buffer TreeCellMassBuf { vec4 treeCellMass[]; };
layout(std430, binding = 11) readonly buffer TreeCellDepthBuf { int treeCellDepth[]; };
layout(std430, binding = 12) readonly buffer TreeCellCounterBuf { uint treeCellCounter[]; };

uniform int uN;
uniform int uLevel;

void main() {
    uint cell = gl_GlobalInvocationID.x;
    if (cell >= treeCellCounter[0]) return;
    if (treeCellDepth[cell] != uLevel) return;

    vec3 comSum = vec3(0.0);
    float massSum = 0.0;
    for (int k = 0; k < 8; ++k) {
        int child = treeChild[8 * int(cell) + k];
        if (child < 0) continue; // empty
        vec3 pos;
        float mass;
        if (child < uN) {
            pos = posMass[child].xyz;
            mass = posMass[child].w;
        } else {
            vec4 cm = treeCellMass[child - uN];
            pos = cm.xyz;
            mass = cm.w;
        }
        comSum += mass * pos;
        massSum += mass;
    }
    treeCellMass[cell] = (massSum > 0.0) ? vec4(comSum / massSum, massSum) : vec4(0.0);
}
)";

// Gravity-only Forces(): the grouped, shared-stack cooperative tree walk
// from 06_tidal_disruption's Forces(), with the SPH pressure/viscosity
// block and the central-point-mass (uBhType) term both removed -- see this
// module's own header comment. Every barrier()/shared-state discipline is
// preserved exactly (see this file's header comment for why: every branch
// gating a barrier() must evaluate identically across the whole group).
constexpr const char* kForcesSrc = R"(#version 460 core
layout(local_size_x = 64) in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 2) writeonly buffer Accel { vec4 accel[]; };
layout(std430, binding = 8) readonly buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 9) readonly buffer TreeCellMassBuf { vec4 treeCellMass[]; };
layout(std430, binding = 10) readonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 18) readonly buffer TreeSortIndexBuf { int sortIndex[]; };

uniform int uN;
uniform int uPaddedN;
uniform float uG;
uniform float uSoftening2;
uniform float uTreeTheta;

const int kTreeMaxStack = 48;
shared int sStackNode[kTreeMaxStack];
shared int sStackNextChild[kTreeMaxStack];
shared int sSp;
shared vec3 sReduceMin[64];
shared vec3 sReduceMax[64];
shared vec3 sGroupCenter;
shared float sGroupRadius;

void main() {
    uint localIdx = gl_LocalInvocationID.x;
    uint slot = gl_WorkGroupID.x * 64u + localIdx;
    int myIndex = (slot < uint(uPaddedN)) ? sortIndex[slot] : -1;
    bool valid = (myIndex >= 0);
    uint i = valid ? uint(myIndex) : 0u; // dummy index 0 for padding threads -- never written back

    vec3 ri = posMass[i].xyz;

    sReduceMin[localIdx] = valid ? ri : vec3(3.0e38);
    sReduceMax[localIdx] = valid ? ri : vec3(-3.0e38);
    barrier();
    if (localIdx == 0u) {
        vec3 mn = sReduceMin[0];
        vec3 mx = sReduceMax[0];
        for (int k = 1; k < 64; ++k) {
            mn = min(mn, sReduceMin[k]);
            mx = max(mx, sReduceMax[k]);
        }
        sGroupCenter = 0.5 * (mn + mx);
        sGroupRadius = length(0.5 * (mx - mn));
        sStackNode[0] = 0; // root
        sStackNextChild[0] = 0;
        sSp = 1;
    }
    barrier();

    vec3 aGrav = vec3(0.0);
    while (sSp > 0) {
        int frame = sSp - 1;
        int node = sStackNode[frame];
        int c = sStackNextChild[frame];

        if (c >= 8) { // exhausted this node's children -- pop
            barrier();
            if (localIdx == 0u) sSp--;
            barrier();
            continue;
        }
        barrier(); // every invocation has read frame/node/c before it's advanced
        if (localIdx == 0u) sStackNextChild[frame] = c + 1;
        barrier();

        int childVal = treeChild[8 * node + c]; // read-only global data, identical for every invocation
        if (childVal < 0) continue; // empty slot -- uniform (childVal is shared-derived)

        if (childVal < uN) {
            vec3 rij = ri - posMass[childVal].xyz;
            float r2 = dot(rij, rij);
            float invR3 = pow(r2 + uSoftening2, -1.5);
            aGrav -= uG * posMass[childVal].w * invR3 * rij;
            continue;
        }

        int cellIdx = childVal - uN;
        vec4 geom = treeCellGeom[cellIdx];
        vec4 cm = treeCellMass[cellIdx];
        float d = distance(sGroupCenter, cm.xyz);
        float rMin = max(0.0, d - sGroupRadius);
        float cellSize = 2.0 * geom.w; // full cube width, not half-size
        bool accept = (rMin > 0.0) && (cellSize < uTreeTheta * rMin);

        if (accept) {
            vec3 rij = ri - cm.xyz;
            float dist2 = dot(rij, rij);
            float invR3 = pow(dist2 + uSoftening2, -1.5);
            aGrav -= uG * cm.w * invR3 * rij;
            continue;
        }

        barrier();
        if (localIdx == 0u && sSp < kTreeMaxStack) {
            sStackNode[sSp] = cellIdx;
            sStackNextChild[sSp] = 0;
            sSp++;
        }
        barrier();
    }

    if (valid) accel[i] = vec4(aGrav, 0.0);
}
)";

GLuint MakeBuffer(std::size_t bytes) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferData(buf, static_cast<GLsizeiptr>(bytes), nullptr, GL_DYNAMIC_DRAW);
    return buf;
}

// Persistent GL state (shaders + buffers), lazily built/resized on first use
// or whenever N changes -- compiling 8 shaders and reallocating a dozen
// SSBOs every call would defeat the point of a GPU solver. Scoped to
// whatever GL context is current when it's first touched (same caveat as
// GpuDirect's cached shader).
struct State {
    bool built = false;
    int n = 0, maxCells = 0, paddedN = 0;

    fw::ComputeShader computeBoundingBox, computeMortonKeys, bitonicSortStep, initTreeRoot, claimSlots, resolveSlots,
        computeTreeCellMass, forces;

    GLuint posMass = 0, accel = 0;
    GLuint treeChild = 0, treeCellMass = 0, treeCellGeom = 0, treeCellDepth = 0, treeCellCounter = 0, treeBBox = 0;
    GLuint treeParticleCell = 0, treeSlotHead = 0, treeSlotNext = 0, sortKeyBuf = 0, sortIndexBuf = 0;

    void EnsureShaders() {
        if (built) return;
        computeBoundingBox = fw::ComputeShader::FromSource(ComputeBoundingBoxSrc());
        computeMortonKeys = fw::ComputeShader::FromSource(ComputeMortonKeysSrc());
        bitonicSortStep = fw::ComputeShader::FromSource(kBitonicSortStepSrc);
        initTreeRoot = fw::ComputeShader::FromSource(InitTreeRootSrc());
        claimSlots = fw::ComputeShader::FromSource(ClaimSlotsSrc());
        resolveSlots = fw::ComputeShader::FromSource(ResolveSlotsSrc());
        computeTreeCellMass = fw::ComputeShader::FromSource(kComputeTreeCellMassSrc);
        forces = fw::ComputeShader::FromSource(kForcesSrc);
        built = true;
    }

    void Resize(int newN, double treeMaxCellsFactor) {
        int paddedNNew = 1;
        while (paddedNNew < newN) paddedNNew *= 2;
        paddedNNew = std::max(paddedNNew, kGravityGroupSize);
        const int maxCellsNew = std::max(static_cast<int>(treeMaxCellsFactor * newN), 64);

        if (newN == n && paddedNNew == paddedN && maxCellsNew == maxCells) return; // already the right size

        n = newN;
        paddedN = paddedNNew;
        maxCells = maxCellsNew;

        auto recreate = [](GLuint& buf, std::size_t bytes) {
            if (buf) glDeleteBuffers(1, &buf);
            buf = MakeBuffer(bytes);
        };
        recreate(posMass, static_cast<std::size_t>(n) * sizeof(glm::vec4));
        recreate(accel, static_cast<std::size_t>(n) * sizeof(glm::vec4));
        recreate(treeChild, static_cast<std::size_t>(8 * maxCells) * sizeof(int32_t));
        recreate(treeCellMass, static_cast<std::size_t>(maxCells) * sizeof(glm::vec4));
        recreate(treeCellGeom, static_cast<std::size_t>(maxCells) * sizeof(glm::vec4));
        recreate(treeCellDepth, static_cast<std::size_t>(maxCells) * sizeof(int32_t));
        recreate(treeCellCounter, sizeof(uint32_t));
        recreate(treeBBox, 6 * sizeof(uint32_t));
        recreate(treeParticleCell, static_cast<std::size_t>(n) * sizeof(int32_t));
        recreate(treeSlotHead, static_cast<std::size_t>(8 * maxCells) * sizeof(uint32_t));
        recreate(treeSlotNext, static_cast<std::size_t>(n) * sizeof(uint32_t));
        recreate(sortKeyBuf, static_cast<std::size_t>(paddedN) * sizeof(uint32_t));
        recreate(sortIndexBuf, static_cast<std::size_t>(paddedN) * sizeof(int32_t));
    }
};

State& Gpu() {
    static State state;
    return state;
}

} // namespace

void ComputeAccelGpuBarnesHut(const PosMassView<3>& pts, double G, double eps2, double theta, SoA<3>& out,
                             GpuBarnesHutStats* stats, int treeMaxDepth, double treeMaxCellsFactor) {
    const int n = static_cast<int>(pts.Count());
    out.ResizeAccel(static_cast<std::size_t>(n));
    if (n == 0) return;

    State& s = Gpu();
    s.EnsureShaders();
    s.Resize(n, treeMaxCellsFactor);

    std::vector<glm::vec4> posMass(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        posMass[ii] = glm::vec4(static_cast<float>(pts.x[ii]), static_cast<float>(pts.y[ii]),
                                static_cast<float>(pts.z[ii]), static_cast<float>(pts.m[ii]));
    }
    glNamedBufferSubData(s.posMass, 0, static_cast<GLsizeiptr>(posMass.size() * sizeof(glm::vec4)), posMass.data());

    const GLuint groups = static_cast<GLuint>((n + kWorkgroupSize - 1) / kWorkgroupSize);
    const GLuint cellGroups = static_cast<GLuint>((s.maxCells + kWorkgroupSize - 1) / kWorkgroupSize);
    const GLuint slotGroups = static_cast<GLuint>((8 * s.maxCells + kWorkgroupSize - 1) / kWorkgroupSize);
    const GLuint paddedGroups = static_cast<GLuint>((s.paddedN + kWorkgroupSize - 1) / kWorkgroupSize);
    const GLuint gravityGroups = static_cast<GLuint>((s.paddedN + kGravityGroupSize - 1) / kGravityGroupSize);

    const auto t0 = std::chrono::steady_clock::now();

    // Bounding box (seeded to the flipped-uint sentinel for min/max, same
    // trick as 06 -- see kTreeCommonGlsl's floatFlipForAtomic).
    const uint32_t bboxSeed[6] = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0u, 0u, 0u};
    glNamedBufferSubData(s.treeBBox, 0, sizeof(bboxSeed), bboxSeed);
    s.computeBoundingBox.Use();
    s.computeBoundingBox.SetInt("uN", n);
    fw::ComputeShader::BindBuffer(0, s.posMass);
    fw::ComputeShader::BindBuffer(13, s.treeBBox);
    fw::ComputeShader::BindBuffer(14, s.treeParticleCell);
    s.computeBoundingBox.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    // Morton sort (amortizing 06's own per-frame-not-per-substep caveat
    // doesn't apply here -- this is one solver call, not a stepped sim).
    s.computeMortonKeys.Use();
    s.computeMortonKeys.SetInt("uN", n);
    s.computeMortonKeys.SetInt("uPaddedN", s.paddedN);
    fw::ComputeShader::BindBuffer(0, s.posMass);
    fw::ComputeShader::BindBuffer(13, s.treeBBox);
    fw::ComputeShader::BindBuffer(17, s.sortKeyBuf);
    fw::ComputeShader::BindBuffer(18, s.sortIndexBuf);
    s.computeMortonKeys.Dispatch(paddedGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    s.bitonicSortStep.Use();
    fw::ComputeShader::BindBuffer(17, s.sortKeyBuf);
    fw::ComputeShader::BindBuffer(18, s.sortIndexBuf);
    for (int k = 2; k <= s.paddedN; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            s.bitonicSortStep.SetInt("uPaddedN", s.paddedN);
            s.bitonicSortStep.SetInt("uK", k);
            s.bitonicSortStep.SetInt("uJ", j);
            s.bitonicSortStep.Dispatch(paddedGroups);
            fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        }
    }
    if (stats) glFinish(); // honest per-stage timing -- see the note by t4 below
    const auto t1 = std::chrono::steady_clock::now();

    // Tree build: bounding box was already computed above and
    // treeParticleCell already seeded to 0 by ComputeBoundingBox -- no need
    // to redo either (unlike 06, which rebuilds the tree fresh every
    // substep and so redoes both; a single solver call only needs them once).
    s.initTreeRoot.Use();
    fw::ComputeShader::BindBuffer(8, s.treeChild);
    fw::ComputeShader::BindBuffer(10, s.treeCellGeom);
    fw::ComputeShader::BindBuffer(11, s.treeCellDepth);
    fw::ComputeShader::BindBuffer(12, s.treeCellCounter);
    fw::ComputeShader::BindBuffer(13, s.treeBBox);
    s.initTreeRoot.Dispatch(1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const uint32_t slotSentinel = 0xFFFFFFFFu;
    uint32_t prevCellCount = 1u;
    int actualMaxDepth = treeMaxDepth;
    for (int level = 0; level <= treeMaxDepth; ++level) {
        glClearNamedBufferData(s.treeSlotHead, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, &slotSentinel);

        s.claimSlots.Use();
        s.claimSlots.SetInt("uN", n);
        fw::ComputeShader::BindBuffer(0, s.posMass);
        fw::ComputeShader::BindBuffer(10, s.treeCellGeom);
        fw::ComputeShader::BindBuffer(14, s.treeParticleCell);
        fw::ComputeShader::BindBuffer(15, s.treeSlotHead);
        fw::ComputeShader::BindBuffer(16, s.treeSlotNext);
        s.claimSlots.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        s.resolveSlots.Use();
        s.resolveSlots.SetInt("uN", n);
        s.resolveSlots.SetInt("uMaxCells", s.maxCells);
        s.resolveSlots.SetInt("uLevel", level);
        fw::ComputeShader::BindBuffer(8, s.treeChild);
        fw::ComputeShader::BindBuffer(10, s.treeCellGeom);
        fw::ComputeShader::BindBuffer(11, s.treeCellDepth);
        fw::ComputeShader::BindBuffer(12, s.treeCellCounter);
        fw::ComputeShader::BindBuffer(14, s.treeParticleCell);
        fw::ComputeShader::BindBuffer(15, s.treeSlotHead);
        fw::ComputeShader::BindBuffer(16, s.treeSlotNext);
        s.resolveSlots.Dispatch(slotGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        uint32_t curCellCount = 0;
        glGetNamedBufferSubData(s.treeCellCounter, 0, sizeof(uint32_t), &curCellCount);
        if (curCellCount == prevCellCount) {
            actualMaxDepth = level; // no cell exists past this depth -- stop
            break;
        }
        prevCellCount = curCellCount;
    }
    const auto t2 = std::chrono::steady_clock::now();

    s.computeTreeCellMass.Use();
    s.computeTreeCellMass.SetInt("uN", n);
    fw::ComputeShader::BindBuffer(0, s.posMass);
    fw::ComputeShader::BindBuffer(8, s.treeChild);
    fw::ComputeShader::BindBuffer(9, s.treeCellMass);
    fw::ComputeShader::BindBuffer(11, s.treeCellDepth);
    fw::ComputeShader::BindBuffer(12, s.treeCellCounter);
    for (int level = actualMaxDepth; level >= 0; --level) {
        s.computeTreeCellMass.SetInt("uLevel", level);
        s.computeTreeCellMass.Dispatch(cellGroups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    }
    if (stats) glFinish();
    const auto t3 = std::chrono::steady_clock::now();

    s.forces.Use();
    s.forces.SetInt("uN", n);
    s.forces.SetInt("uPaddedN", s.paddedN);
    s.forces.SetFloat("uG", static_cast<float>(G));
    s.forces.SetFloat("uSoftening2", static_cast<float>(eps2));
    s.forces.SetFloat("uTreeTheta", static_cast<float>(theta));
    fw::ComputeShader::BindBuffer(0, s.posMass);
    fw::ComputeShader::BindBuffer(2, s.accel);
    fw::ComputeShader::BindBuffer(8, s.treeChild);
    fw::ComputeShader::BindBuffer(9, s.treeCellMass);
    fw::ComputeShader::BindBuffer(10, s.treeCellGeom);
    fw::ComputeShader::BindBuffer(18, s.sortIndexBuf);
    s.forces.Dispatch(gravityGroups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    // glMemoryBarrier orders memory effects, it does not stall the CPU until
    // the GPU finishes -- without a real sync point here, this stage's own
    // execution time would silently get folded into the next stage's
    // readback stall (glGetNamedBufferSubData IS a real sync point) instead
    // of being attributed to Forces, the stage that actually did the work.
    // Only paid when a caller asks for stats (profiling), never on the
    // fast, no-stats path other callers use.
    if (stats) glFinish();
    const auto t4 = std::chrono::steady_clock::now();

    std::vector<glm::vec4> accel(static_cast<std::size_t>(n));
    glGetNamedBufferSubData(s.accel, 0, static_cast<GLsizeiptr>(accel.size() * sizeof(glm::vec4)), accel.data());
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        out.ax[ii] = accel[ii].x;
        out.ay[ii] = accel[ii].y;
        out.az[ii] = accel[ii].z;
    }
    const auto t5 = std::chrono::steady_clock::now();

    if (stats) {
        auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        uint32_t cellCount = 0;
        glGetNamedBufferSubData(s.treeCellCounter, 0, sizeof(uint32_t), &cellCount);
        stats->cellCount = static_cast<int>(cellCount);
        stats->levelsUsed = actualMaxDepth;
        stats->sortMs = ms(t0, t1);
        stats->treeBuildMs = ms(t1, t2);
        stats->massUpsweepMs = ms(t2, t3);
        stats->forcesMs = ms(t3, t4);
        stats->readbackMs = ms(t4, t5);
    }
}

} // namespace ngrav::gpu
