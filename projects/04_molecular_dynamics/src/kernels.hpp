#pragma once

// GPU compute-shader port of MDSystem's periodic linked-cell LJ force
// evaluation (single species, shifted-force truncated), following the
// atomic-linked-list spatial hash technique used by 06_tidal_disruption/
// 07_grhd's kernels.hpp -- see that file's header comment for why a
// counting-sort/scan-based grid build loses to this one-pass scheme.
//
// SCOPE: this is a validated building block, not a live simulation
// backend -- main.cpp's --selftest cross-checks it against MDSystem's CPU
// implementation and nothing else in this project calls it. Wiring a real
// GPU-resident time-stepping loop (thermostat/barostat state, multi-substep
// integration without a CPU round-trip every step) is future work; see the
// README's Known Simplifications.
//
// Difference from 06/07's grid: THIS grid is periodic (an open spatial hash
// with no wraparound is wrong for a periodic box -- two particles near
// opposite faces are neighbors, and an unbounded hash table has no notion
// of "opposite face"). So cells are indexed directly into a dense
// uNc*uNc*uNc array (uNc = floor(L/cellSize), the same value MDSystem.cpp's
// BuildCellListPairs computes on the CPU) with explicit modular wraparound
// on the neighbor-cell offset, instead of an open hash table -- exact as
// long as uNc>=3 (below that, +1/-1 neighbor offsets can collide onto the
// same wrapped cell more than once; MDSystem's CPU path falls back to
// brute force in exactly that regime, and this GPU kernel simply requires
// the caller to stay in the uNc>=3 regime rather than replicating that
// fallback).

#include <string>

namespace md::kernels {

inline constexpr int kWorkgroupSize = 256;

// binding 0: posBuf (vec4, xyz=position in [0,L), w unused)
// binding 1: accelEnergyBuf (vec4, xyz=acceleration, w=this particle's HALF
//            share of the pairwise potential energy sum -- summing w over
//            all particles on the CPU gives the exact total potential
//            energy with no double-counting, since each pair contributes
//            0.5*U from each of its two particles' threads)
// binding 6: cellHead (uint[uNc^3], cleared to 0xFFFFFFFFu before BuildGrid)
// binding 7: nextIndex (uint[uN])

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosBuf { vec4 pos[]; };

uniform int uN;
uniform int uNc;         // cells per axis, floor(L/cellSize), must be >= 3
uniform float uCellSize;
uniform float uL;        // box length (periodic wrap length)

int wrapAxis(int c) {
    if (c < 0) return c + uNc;
    if (c >= uNc) return c - uNc;
    return c;
}
ivec3 cellOf(vec3 p) { return ivec3(floor(p / uCellSize)); }
uint cellIndex(ivec3 c) { return uint((c.x * uNc + c.y) * uNc + c.z); }

vec3 minImage(vec3 d) { return d - uL * round(d / uL); }
)";
}

// One pass, no scan (see this file's header comment). CellHead must already
// be cleared to 0xFFFFFFFFu (a host-side glClearNamedBufferData call, not a
// shader) before this dispatch.
inline std::string BuildGrid() {
    return CommonHeader() + R"(
layout(std430, binding = 6) buffer CellHeadBuf { uint cellHead[]; };
layout(std430, binding = 7) writeonly buffer NextIndexBuf { uint nextIndex[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    uint bucket = cellIndex(cellOf(pos[i].xyz));
    // atomicExchange both sets cellHead[bucket]=i and returns its previous
    // value in one atomic op -- race-free regardless of how many particles
    // in this dispatch hash to the same bucket.
    nextIndex[i] = atomicExchange(cellHead[bucket], i);
}
)";
}

// Shifted-force-truncated LJ, single species (sigma=epsilon=1): one thread
// per particle, walks the 27 neighboring cells (3x3x3 including its own --
// exact given uCellSize=cutoff and uNc>=3, no approximation), computing the
// FULL double sum over neighbors j != i (not exploiting i<j symmetry, since
// independent GPU threads can't safely accumulate into each other's output
// without atomics) -- same pattern as the CPU reference used by
// --selftest's other cases.
inline std::string Forces() {
    return CommonHeader() + R"(
layout(std430, binding = 1) writeonly buffer AccelEnergyBuf { vec4 accelEnergy[]; };
layout(std430, binding = 6) readonly buffer CellHeadBuf { uint cellHead[]; };
layout(std430, binding = 7) readonly buffer NextIndexBuf { uint nextIndex[]; };

uniform float uCutoff;
uniform float uSigma;
uniform float uEpsilon;
uniform float uFc; // F(cutoff), precomputed host-side
uniform float uUc; // U(cutoff), precomputed host-side

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = pos[i].xyz;
    ivec3 base = cellOf(ri);
    float cutoff2 = uCutoff * uCutoff;

    vec3 accel = vec3(0.0);
    float potentialHalf = 0.0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        ivec3 c = ivec3(wrapAxis(base.x + dx), wrapAxis(base.y + dy), wrapAxis(base.z + dz));
        uint j = cellHead[cellIndex(c)];
        while (j != 0xFFFFFFFFu) {
            if (j != i) {
                vec3 rij = minImage(ri - pos[j].xyz);
                float r2 = dot(rij, rij);
                if (r2 < cutoff2 && r2 > 1e-12) {
                    float r = sqrt(r2);
                    float sr6 = pow(uSigma / r, 6.0);
                    float fmag = 24.0 * uEpsilon * (2.0 * sr6 * sr6 - sr6) / r - uFc;
                    accel += (fmag / r) * rij;
                    float u = 4.0 * uEpsilon * (sr6 * sr6 - sr6) - uUc + (r - uCutoff) * uFc;
                    potentialHalf += 0.5 * u;
                }
            }
            j = nextIndex[j];
        }
    }

    accelEnergy[i] = vec4(accel, potentialHalf);
}
)";
}

} // namespace md::kernels
