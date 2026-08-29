#pragma once

#include <string>

// GLSL compute-shader source for the self-gravitating SPH star: smoothed
// particle hydrodynamics (density -> pressure -> pressure/viscosity force)
// plus brute-force softened self-gravity, integrated by kick-drift-kick
// leapfrog. Kept as string builders rather than files, matching 05_tdse_gpu's
// inline-shader convention.
//
// Buffer layout (std430 SSBOs, one array element per particle):
//   binding 0  PosMass   vec4(x, y, z, mass)
//   binding 1  Vel       vec4(vx, vy, vz, _)
//   binding 2  Acc       vec4(ax, ay, az, _)          -- gravity + SPH, summed in one pass
//   binding 3  RhoPress  vec4(rho, pressure, soundspeed, _)
//   binding 4  H         float smoothing length, per particle (adaptive)
//   binding 5  AccretedMass  float, Accretion() pass only -- see that function
//   binding 6  CellHead  uint per hash bucket, index of the most-recently-inserted
//              particle there, or the sentinel 0xFFFFFFFFu if none (spatial grid, below)
//   binding 7  NextIndex uint per particle, the next node in its cell's linked list
//   binding 8  TreeChild     int[8*maxCells], octree child pointers (Barnes-Hut, below)
//   binding 9  TreeCellMass  vec4[maxCells], (comX,comY,comZ,totalMass), bottom-up
//   binding 10 TreeCellGeom  vec4[maxCells], (centerX,centerY,centerZ,halfSize)
//   binding 11 TreeCellDepth int[maxCells]
//   binding 12 TreeCellCounter uint[1], atomic bump allocator for the next free cell index
//   binding 13 TreeBBox      uint[6], atomic min/max reduction scratch (float-flipped, see below)
//   binding 14 TreeParticleCell int[N], which (not-yet-resolved) particle currently
//              belongs to which cell during the build, -1 once resolved as a leaf
//   binding 15 TreeSlotHead  uint[8*maxCells], per (cell,octant) slot: index of one
//              particle claiming it this level, or the sentinel 0xFFFFFFFFu
//   binding 16 TreeSlotNext  uint[N], the next particle in that slot's candidate list
//   binding 17 TreeSortKey   uint[paddedN], Morton code sort key (Bonsai-style grouped
//              gravity walk, below) -- 0xFFFFFFFFu for padding slots [N,paddedN)
//   binding 18 TreeSortIndex int[paddedN], particle index for each sorted slot, -1 for padding
//
// All kernels are one-thread-per-particle with a uN bounds check (particle
// count need not be a multiple of the workgroup size, unlike the FFT's
// power-of-two rows). Self-terms (i==j) need no special-casing: the cubic
// spline's gradient is exactly zero at r=0, and gravity's own softened
// numerator (ri-rj) is exactly zero when i==j, so both loops can safely
// include j==i.
//
// Spatial hash grid (bindings 6-7), for SPH neighbor search ONLY --
// self-gravity has no cutoff radius (every pair matters, no matter how far
// apart) and stays a brute-force O(N^2) sum; SPH only interacts within 2h,
// so restricting that search to nearby grid cells turns it from O(N^2) into
// ~O(N) instead of computing-then-discarding almost every pair (this used
// to be the biggest single cost added by spawning more particles at once
// via TdeSim::SpawnStar, on top of gravity's own unavoidable O(N^2)).
//
// Atomic-linked-list uniform grid (classic technique, e.g. NVIDIA's CUDA
// "particles" sample), rebuilt every substep since positions change every
// substep. NOT a counting-sort spatial hash (Teschner et al. 2003) --
// TWO EARLIER VERSIONS of this file used that scheme instead, and BOTH
// measured net SLOWER than plain O(N^2) brute force at every particle
// count tried (4000-24000): counting sort needs an exclusive scan of
// per-bucket counts before the actual particle scatter, and (1) doing that
// scan via a CPU readback+loop+upload forces a synchronous pipeline stall
// every single substep -- the sync point, not the transfer volume, was the
// cost; (2) doing it as a single-GPU-thread serial loop avoided the CPU
// sync but was ALSO slower -- a lone GPU thread has none of a CPU core's
// serial-execution advantages (high per-op memory latency, nothing to hide
// it behind, unlike a warp of parallel threads). BuildGrid() below replaces
// all of that with ONE single parallel pass and no scan whatsoever:
//   BuildGrid: each particle's cell = floor(pos/uCellSize), hashed into a
//   fixed-size table (NOT a dense grid over all of space -- the domain
//   here ranges from a compact star to slingshot-ejected debris thousands
//   of units out, so a dense grid sized to fit everything would need an
//   infeasible number of cells). It then atomically pushes itself onto
//   that bucket's linked list: atomicExchange(cellHead[bucket], i) both
//   sets cellHead[bucket] to i AND returns the bucket's PREVIOUS head in
//   one atomic op, which becomes nextIndex[i] -- race-free even when many
//   particles target the same bucket in the same dispatch, and needs
//   CellHead cleared to the sentinel (TdeSim, not a shader) before each
//   dispatch, nothing else.
// A query at position ri hashes its own 27 neighboring cells (3x3x3,
// including its own) via the SAME uCellSize, and for each walks that
// bucket's linked list (cellHead[bucket], then repeatedly nextIndex[j])
// until the sentinel. This is exact (no missed neighbors), not
// approximate: uCellSize is fixed at 2*hMax (Configure()'s
// sph.h_max_factor clamp) for the whole run, and h is clamped to <=hMax by
// construction, so any pair within a particle's own 2h support radius is
// guaranteed to fall in its own cell or an immediately adjacent one. A
// hash COLLISION between one of the 27 searched cells and some OTHER,
// unrelated cell is harmless (adds extra candidates, filtered by the
// existing r<2h check, never drops a real neighbor). A collision BETWEEN
// TWO OF THE 27 SEARCHED CELLS THEMSELVES is a different matter and is NOT
// harmless: both offsets resolve to the same bucket, so that bucket's
// particles get visited (and their density/force contribution summed)
// twice. Caught empirically in the earlier counting-sort version, not just
// reasoned about: a version without a dedup check gave up to 6 such
// collisions among the 27 offsets in a 4000-bucket table for a single
// query (measured -- see git history), even though random collision alone
// predicts under 0.1 -- a plain multiply-and-XOR hash apparently does not
// scatter nearby small-magnitude cell coordinates as well as it scatters
// generic ones. gatherDensity/Forces() below track which buckets they've
// already visited (a small fixed-size local array, at most 27 entries) and
// skip a repeat -- this is what actually makes the 27-cell search exact,
// not the hash function's distribution quality; still required with the
// linked-list scheme for the exact same reason.
//
// Adaptive smoothing length. A single fixed h under-resolves the star's
// outer envelope (where particles are sparse) relative to its dense core --
// the free-surface density excess documented in README.md's first
// validation pass. Standard fix: solve h_i per particle from
// h_i = eta*(m_i/rho_i)^(1/3) (Springel & Hernquist 2002's ansatz: h such
// that the kernel's ~(4/3)pi(2h)^3 support volume holds a roughly constant
// effective neighbor count, eta~1.2 -> ~40-60 neighbors in 3D for the cubic
// spline). rho_i(h_i) itself depends on h_i, so Density() iterates a small
// fixed-point loop per particle (warm-started from last step's h_i, which
// changes slowly frame to frame) entirely inside one shader invocation --
// each particle's solve is independent, ideal for one-thread-per-particle.
// A density DEPENDS on h "gather"-style (rho_i uses only h_i) as usual for
// this iteration; but Forces() uses the symmetrized pair length
// h_ij = 0.5*(h_i+h_j) for both the kernel gradient AND artificial
// viscosity, so the same W(r,h_ij) is used for both directions of a pair
// and F_ij = -F_ji exactly (momentum conservation to machine precision, as
// before) despite h_i != h_j in general.
//
// Barnes-Hut octree for self-gravity (bindings 8-16). Unlike the SPH grid
// above, gravity has no cutoff radius -- every particle pulls on every
// other one, however far away -- so it can't be sped up by only visiting
// "nearby" cells the way SPH was. The Barnes-Hut approximation instead
// exploits that a distant, compact CLUSTER of mass looks, to good
// approximation, just like a single point mass at that cluster's center of
// mass: a body only needs to interact directly with individual nearby
// particles, while a distant group can be summarized as one (mass, center
// of mass) pair. An octree recursively partitions space into cubes so
// "how compact is this group, relative to how far away it is" (the
// opening-angle criterion below) can be tested cheaply level by level.
// This turns the force pass from O(N^2) into ~O(N log N).
//
// Level-by-level (breadth-first) construction, NOT the more commonly
// cited Burtscher & Pingali lock-free-CAS-insertion GPU technique (2011,
// "An Efficient CUDA Implementation of the Tree-Based Barnes Hut n-Body
// Algorithm") that this was originally built against and had to be
// abandoned: that design has every thread walk all the way down from the
// root in one dispatch, using atomicCompSwap to claim child slots and a
// "lock sentinel + spin-retry" pattern so a losing thread waits for
// whichever thread is subdividing a contested slot to publish the result.
// That spin-wait implicitly assumes every thread in the same warp/
// wavefront gets independent forward progress -- true on modern NVIDIA
// hardware (Volta+'s "independent thread scheduling"), NOT guaranteed by
// the GLSL/SPIR-V compute model in general, and this was caught for real,
// not just reasoned about in the abstract: an early version of that
// design hung indefinitely on a tiny 37-particle selftest case, almost
// certainly exactly this class of cross-lane deadlock (a winner's lane
// unable to make progress while a loser's lane in the same warp spins
// waiting for it). The level-by-level design below uses the SAME
// atomic-linked-list technique already proven safe for the SPH grid above
// (a plain atomicExchange, never a spin/retry loop) instead of any lock:
//
//   1. ComputeBoundingBox: parallel min/max reduction over all particle
//      positions -> a single bounding CUBE (equal-size octree cells need a
//      cube, not a general box: it uses the LARGEST of the three axis
//      extents for all three dimensions). Floats can't be atomicMin/Max'd
//      directly in GLSL, so this uses the standard "flip" trick
//      (floatFlipForAtomic below): reinterpret each float's bits as a
//      uint, then XOR-transform it so unsigned integer ordering of the
//      transformed bits exactly matches the floats' own numeric ordering
//      -- then plain atomicMin/atomicMax on those transformed uints
//      correctly finds the float min/max, and floatUnflipFromAtomic
//      inverts the transform when reading the result back. Also seeds
//      TreeParticleCell[i]=0 for every particle (everyone starts owned by
//      the root).
//   2. InitTreeRoot: a single-invocation pass that reads the now-final
//      bounding box and writes cell 0 (the root)'s center/half-size/depth,
//      resets all 8 of its children to "empty", and resets the cell
//      bump-allocator (TreeCellCounter) to 1 (root is index 0; the next
//      *new* cell allocated gets index 1).
//   3. A level loop, from depth 0 up to uTreeMaxDepth, each level being
//      TWO passes separated by a barrier (TdeSim also clears
//      TreeSlotHead to the empty sentinel before each level's first pass):
//        a. ClaimSlot: one thread per particle that hasn't been resolved
//           into a leaf yet (TreeParticleCell[i]<0 means "already done,
//           skip"). Each still-active particle computes which of its
//           current cell's 8 octants it falls in and does a plain
//           atomicExchange(TreeSlotHead[slot], i) -- this can NEVER block
//           or require another thread's cooperation: every thread's
//           atomicExchange completes independently and immediately,
//           leaving a valid (if arbitrarily ordered) singly-linked list of
//           every particle that wanted that slot this level, exactly like
//           BuildGrid's cellHead/nextIndex above.
//        b. ResolveSlot: one thread per POTENTIAL (cell, octant) slot
//           (dispatched over 8*maxCells, skipping cells that don't exist
//           yet or aren't at the current depth). If nothing claimed a
//           slot this level, it's left alone (stays "empty"). If exactly
//           one particle claimed it, that particle becomes a direct leaf
//           of this slot -- a plain, uncontested write is safe here
//           because ResolveSlot has exactly one thread per slot index, so
//           two threads can never target the same TreeChild entry. If two
//           or more particles claimed it, this thread allocates a new
//           cell (still via atomicAdd -- allocating a globally unique
//           index is the one place an atomic op is still needed, but
//           there's nothing to spin/retry: the returned index is
//           immediately and unconditionally valid, no CAS involved) and
//           publishes it into the parent's TreeChild slot. Up to kNcrit=8
//           candidates (ResolveSlots' own header comment has the full
//           rationale) get packed directly into the new cell's 8 child
//           slots as a leaf group, fully resolved, right here -- no
//           further level needed. Only when there are MORE than 8 does
//           this thread instead reassign every candidate to belong to the
//           new cell (TreeParticleCell[p]=newCell) so they compete again
//           at the NEXT level -- a plain sequential walk of a list this
//           ONE thread already owns, not a wait on any other thread.
//      Unambiguous child-slot encoding, same as before: -1 = empty,
//      [0,N) = a body index, [N, N+maxCells) = a cell index (subtract N
//      to get the actual cell array index) -- body and cell arrays share
//      one integer ID space by construction, not by accident.
//   4. ComputeTreeCellMass: bottom-up pass computing each internal cell's
//      total mass and center of mass from its (up to 8) children. Cells
//      are only safe to sum once ALL their children (whether bodies,
//      trivially always "ready", or other cells, which must already have
//      been summed) are finalized. Rather than track per-cell "children
//      ready" counts with more atomics, this exploits that step 3 already
//      recorded each cell's own depth: dispatching once per depth level,
//      from the deepest level actually reached down to the root (0), and
//      having each invocation act only on cells at that exact depth,
//      guarantees every cell's children are already finalized by the time
//      that cell's own level is processed -- deeper cells are always
//      handled in an earlier pass. Most level-passes (in BOTH step 3 and
//      step 4) are cheap near-no-ops once the tree's real (typically much
//      shallower) depth is exhausted, since every thread still checks its
//      own depth/state and returns immediately if there's nothing to do.
//
// The force pass itself (in Forces()) walks this tree to compute
// self-gravity. A first version had each PARTICLE walk independently with
// its own per-thread stack -- correct (validated against brute force by
// --selftest), but measured net SLOWER overall than the old brute-force
// O(N^2) loop it replaced (24000 particles: ~55s vs ~31s for the same
// benchmark). Splitting the timing showed why: tree build was cheap
// (~10ms/substep) but the walk itself was the dominant cost, and going
// from uTreeTheta=0.5 to 0 (exhaustive, mathematically brute-force-
// equivalent) only made it 2-5x worse rather than the many-orders-of-
// magnitude gap the O(N log N) vs O(N^2) complexity gap would suggest --
// meaning the approximation was doing real algorithmic work, but each
// individual node visit was expensive: a 48-entry stack resident in every
// one of thousands of threads' registers/local memory (crushing GPU
// occupancy), every thread independently re-fetching the exact same node
// data as its neighbors, and warp/subgroup-divergent branching wherever
// nearby threads happened to disagree on accept-vs-open for the same
// node. This is a well documented failure mode for naive one-thread-per-
// particle GPU tree codes (see e.g. Burtscher & Pingali 2011; production
// codes like Bonsai, Bedorf/Gaburov/Portegies Zwart 2012, fix it with
// GROUPED traversal instead: nearby particles share ONE walk).
//
// Grouped (Bonsai-style) walk, what's actually implemented below:
//   0. Once per FRAME (not every substep -- TdeSim::SortParticlesByMorton,
//      called from the top of Step(), before its substep loop), particles
//      are sorted by a Morton (Z-order) code of their position (quantized
//      to 10 bits/axis within this frame's bounding box, then bit-
//      interleaved -- ComputeMortonKeys()) via a standard GPU bitonic sort
//      (BitonicSortStep(), one compare-exchange dispatch per (k,j) network
//      stage, O(log^2 paddedN) dispatches -- paddedN is N rounded up to a
//      power of 2, padding slots keyed 0xFFFFFFFFu so they sort last and
//      are never treated as real particles). This is why it's once per
//      FRAME, not per substep: O(log^2 N) sort dispatches carries real
//      fixed cost the same way the tree build's own per-level dispatches
//      did (see ResolveSlots' and BuildBarnesHutTree's comments) -- paying
//      it once and amortizing over a frame's substeps (positions barely
//      move substep to substep at this integrator's timesteps, so a
//      slightly stale ordering only loosens group bounding boxes, never
//      breaks correctness) is far cheaper than paying it every substep.
//      Sorting doesn't touch the tree build above AT ALL -- that still
//      operates on plain particle index order, unaffected.
//   1. Forces() dispatches in GROUPS of kGravityGroupSize (64) CONSECUTIVE
//      sorted slots, not one thread per raw particle index -- sorted
//      consecutive slots are spatially compact (that's the whole point of
//      the Morton sort), so a group's own bounding box is tight. Grouping
//      by raw, unsorted particle index instead would span the whole star
//      (Monte-Carlo IC generation has no spatial locality in array order),
//      making the group opening-angle test below fail almost every time
//      and defeating the entire purpose.
//   2. Every invocation in a group cooperatively computes ONE shared group
//      bounding box (via workgroup `shared` arrays + a single-leader
//      reduction) and then walks the SAME tree path together using ONE
//      SHARED stack (kTreeMaxStack entries in `shared` memory, not one
//      per thread) instead of each thread maintaining its own. At each
//      shared-stack frame, the opening-angle accept/reject decision uses
//      the GROUP's bounding sphere, not any individual particle's exact
//      position: rMin = max(0, distance(groupCenter, cellCOM) -
//      groupRadius) is a PROVABLY CONSERVATIVE (never wrong, only
//      possibly overcautious) lower bound on every actual member's true
//      distance to that cell, by the triangle inequality -- accepting
//      when cellSize < uTreeTheta*rMin is therefore always at least as
//      safe as the old per-particle test, just occasionally more
//      conservative (opens a node a tight per-particle test could have
//      skipped) in exchange for every member of the group sharing that
//      one decision and that one global-memory fetch of the node's data,
//      instead of each of 64 threads separately deciding and fetching.
//      uTreeTheta=0 still forces every cellSize<0 comparison to fail
//      unconditionally, so exhaustive/exact mode is unaffected by any of
//      this and --selftest still validates against brute force exactly.
//   3. Correctness of the shared-memory/barrier() pattern: GLSL requires
//      barrier() to execute under UNIFORM control flow -- every invocation
//      in the workgroup must reach the same barrier() calls, or the
//      result is undefined (a stricter, spec-mandated version of the
//      cross-lane assumption that doomed the CAS-lock tree-build design
//      above, not an implicit hardware assumption this time). This is
//      satisfied by construction: every branch that gates a barrier()
//      call (stack empty? this node's children exhausted? accept or
//      open?) is computed from `shared` or read-only-global data that is
//      BIT-IDENTICAL for every invocation in the group (same inputs, same
//      floating-point ops -> same IEEE-754 result on every lane), never
//      from a single thread's own particle position -- so the whole group
//      always agrees and always takes the same path. The ONLY per-thread
//      divergence is in DATA, never control flow reaching a barrier():
//      whether a slot holds a real particle (gates its bounding-box
//      contribution and its final acc[] write, both un-barriered), and
//      each thread's own accumulated force value.
//
// Robustness: a particle that hasn't resolved into a leaf by uTreeMaxDepth
// (step 3's level loop only runs that many iterations) simply never gets
// inserted into the tree this substep. Two DISTINCT particles at truly
// identical (to float precision) positions can never be separated by
// further subdivision, so this cap is a real, reachable case, not just a
// theoretical one -- and this codebase has already demonstrated genuine
// gravitational clustering driving particles very close together (see
// README's Physics section on the binary-feeding decks). Missing the cap
// simply drops that one body's contribution to the tree for this single
// substep (it still gets integrated normally by Kick/Drift, and still
// contributes next substep once the fixed-point loop's warm-started state
// has moved slightly) -- a rare, bounded, self-correcting approximation,
// not a hang or a corrupted tree.

namespace tde::kernels {

inline constexpr int kWorkgroupSize = 256;

// Forces()' own workgroup size -- deliberately NOT kWorkgroupSize. See
// this file's header comment on the Bonsai-style grouped gravity walk:
// this must be small enough that kGravityGroupSize CONSECUTIVE
// Morton-sorted particles form a genuinely spatially tight group (a
// bounding box loose enough to span much of the star would make the
// group opening-angle test fail almost always, same failure mode as not
// sorting at all).
inline constexpr int kGravityGroupSize = 64;

// Cubic spline kernel (Monaghan & Lattanzio 1985), 3D normalization 1/pi.
// W(q) sigma/h^3 * f(q), dW/dr = sigma/h^4 * f'(q), q = r/h.
inline const char* SphKernelGlsl = R"(
float kernelW(float r, float h) {
    float q = r / h;
    float sigma = 1.0 / (3.14159265358979 * h * h * h);
    if (q < 1.0) return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    if (q < 2.0) { float t = 2.0 - q; return sigma * 0.25 * t * t * t; }
    return 0.0;
}
// dW/dr (signed; note this is <= 0 everywhere since W is non-increasing in r).
float kernelDwDr(float r, float h) {
    float q = r / h;
    float sigma = 1.0 / (3.14159265358979 * h * h * h);
    if (q < 1.0) return (sigma / h) * (-3.0 * q + 2.25 * q * q);
    if (q < 2.0) { float t = 2.0 - q; return (sigma / h) * (-0.75 * t * t); }
    return 0.0;
}
)";

// Shared spatial-hash helpers -- see this file's header comment for the
// scheme. uCellSize/uHashTableSize are the same for every kernel that
// touches the grid (built and queried against the identical partition).
inline const char* SpatialHashGlsl = R"(
uniform float uCellSize;
uniform int uHashTableSize;

ivec3 cellOf(vec3 pos) { return ivec3(floor(pos / uCellSize)); }

// Teschner et al. 2003 -- large odd-ish primes, XORed after per-axis
// multiplication. Cast to uint wraps negative cell coordinates via two's
// complement, which is fine: this only needs to be a well-mixed bucket
// index, not an order-preserving one.
uint spatialHash(ivec3 c) {
    uint h = (uint(c.x) * 73856093u) ^ (uint(c.y) * 19349663u) ^ (uint(c.z) * 83492791u);
    return h % uint(uHashTableSize);
}
)";

inline std::string CommonHeader() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) buffer Vel { vec4 vel[]; };
layout(std430, binding = 2) buffer Acc { vec4 acc[]; };
layout(std430, binding = 3) buffer RhoPress { vec4 rhoPress[]; };
layout(std430, binding = 4) buffer HBuf { float hArr[]; };

uniform int uN;
)";
}

// Grid build: one parallel pass, no scan (see this file's header comment
// for why -- two earlier counting-sort attempts were both net slower than
// brute force). CellHead must be cleared to the sentinel 0xFFFFFFFFu
// (TdeSim, not a shader) before this dispatch.
inline std::string BuildGrid() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 6) buffer CellHeadBuf { uint cellHead[]; };
layout(std430, binding = 7) writeonly buffer NextIndexBuf { uint nextIndex[]; };

uniform int uN;
)" + std::string(SpatialHashGlsl) + R"(
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    uint bucket = spatialHash(cellOf(posMass[i].xyz));
    // atomicExchange both sets cellHead[bucket]=i AND returns its previous
    // value in one atomic op -- race-free no matter how many particles in
    // this dispatch target the same bucket simultaneously.
    nextIndex[i] = atomicExchange(cellHead[bucket], i);
}
)";
}

// Shared by every Barnes-Hut pass: the float<->uint bit-transform that
// makes atomicMin/atomicMax (uint-only in GLSL) correctly order floats,
// and the octant-index helper used both while building the tree and while
// walking it. See this file's header comment for the transform's
// derivation/correctness argument.
inline const char* TreeCommonGlsl = R"(
uint floatFlipForAtomic(uint u) {
    uint mask = (0u - (u >> 31)) | 0x80000000u;
    return u ^ mask;
}
float floatUnflipFromAtomic(uint u) {
    uint mask = ((u >> 31) - 1u) | 0x80000000u;
    return uintBitsToFloat(u ^ mask);
}
// Which of a cube's 8 octants `pos` falls in, relative to `center` --
// bit 0 = +x half, bit 1 = +y half, bit 2 = +z half.
int octantOf(vec3 pos, vec3 center) {
    int oct = 0;
    if (pos.x >= center.x) oct |= 1;
    if (pos.y >= center.y) oct |= 2;
    if (pos.z >= center.z) oct |= 4;
    return oct;
}
// The center of the sub-cube for a given octant of a cube with the given
// parent center/half-size.
vec3 childCenter(vec3 parentCenter, float parentHalfSize, int octant) {
    float q = 0.5 * parentHalfSize;
    return parentCenter + vec3(((octant & 1) != 0) ? q : -q,
                               ((octant & 2) != 0) ? q : -q,
                               ((octant & 4) != 0) ? q : -q);
}
)";

// Bonsai-style grouped gravity walk, stage 0a: Morton (Z-order) sort key
// per particle -- see this file's header comment for why. Quantizes each
// axis to 10 bits within THIS frame's bounding box (read from TreeBBox,
// left over from whichever tree build most recently ran -- TdeSim
// refreshes it with a dedicated ComputeBoundingBox() dispatch right
// before this, since this runs once per FRAME rather than reusing a
// possibly-stale prior substep's box), then bit-interleaves the three
// 10-bit axis codes into one 30-bit key via the standard "spread bits"
// technique (three shift/and/multiply passes per axis -- the same
// technique used by, among many others, Karras 2012's reference LBVH
// construction). Precision only needs to be good enough for GROUPING
// (not exact geometry -- the tree itself already handles exact geometry),
// so 10 bits/axis (1024^3 cells across the whole bounding box) is ample.
// Slots [N, paddedN) are padding needed to round up to the power-of-two
// size BitonicSortStep's compare-exchange network requires -- keyed
// 0xFFFFFFFFu (larger than any real 30-bit key) so they always sort to
// the very end, and marked with sortIndex=-1 so Forces() can recognize
// and skip them.
inline std::string ComputeMortonKeys() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 13) readonly buffer TreeBBoxBuf { uint bbox[]; };
layout(std430, binding = 17) writeonly buffer TreeSortKeyBuf { uint sortKey[]; };
layout(std430, binding = 18) writeonly buffer TreeSortIndexBuf { int sortIndex[]; };

uniform int uN;
uniform int uPaddedN;
)" + std::string(TreeCommonGlsl) + R"(
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
    sortKey[slot] = (xx << 2) | (yy << 1) | zz;
    sortIndex[slot] = int(slot);
}
)";
}

// Bonsai-style grouped gravity walk, stage 0b: one compare-exchange stage
// of a standard GPU bitonic sort network, sorting (sortKey, sortIndex)
// pairs together by key. TdeSim drives the full sort as a CPU-side double
// loop over (k,j) calling this once per stage -- log2(paddedN)*
// (log2(paddedN)+1)/2 dispatches total, each a barrier-separated pass
// exactly like the octree build's own level loop, but with NO shared
// state between invocations WITHIN one dispatch (every invocation only
// ever touches its own (i, i^j) pair), so unlike the octree build this
// needs no atomics at all -- only the ixj>i guard below to ensure each
// pair is swapped by exactly one of its two invocations, never both.
inline std::string BitonicSortStep() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

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
}

// Pass 1/4 (Barnes-Hut): parallel reduction of every particle's position
// into a single global bounding box, encoded as flipped uints so plain
// atomicMin/atomicMax work (see TreeCommonGlsl). uBBox layout: [0,1,2] =
// flipped min x,y,z; [3,4,5] = flipped max x,y,z. TdeSim seeds [0,1,2] to
// 0xFFFFFFFFu and [3,4,5] to 0u before dispatch (the identity elements for
// atomicMin/atomicMax respectively).
inline std::string ComputeBoundingBox() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 13) buffer TreeBBoxBuf { uint bbox[]; };
layout(std430, binding = 14) writeonly buffer TreeParticleCellBuf { int treeParticleCell[]; };

uniform int uN;
)" + std::string(TreeCommonGlsl) + R"(
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
    treeParticleCell[i] = 0; // everyone starts owned by the root this substep
}
)";
}

// Pass 2/4 (Barnes-Hut): single-invocation pass (TdeSim dispatches this
// with a group count of 1x1x1 and the shader only acts on invocation 0)
// that turns the now-final bounding box into the root cell (index 0)'s
// center/half-size, clears its 8 children to "empty", sets its depth to
// 0, and resets the cell bump-allocator to 1 (index 0 is already taken by
// the root).
inline std::string InitTreeRoot() {
    return R"(#version 460 core
layout(local_size_x = 1) in;

layout(std430, binding = 8) writeonly buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 10) writeonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 11) writeonly buffer TreeCellDepthBuf { int treeCellDepth[]; };
layout(std430, binding = 12) writeonly buffer TreeCellCounterBuf { uint treeCellCounter[]; };
layout(std430, binding = 13) readonly buffer TreeBBoxBuf { uint bbox[]; };
)" + std::string(TreeCommonGlsl) + R"(
void main() {
    float minX = floatUnflipFromAtomic(bbox[0]);
    float minY = floatUnflipFromAtomic(bbox[1]);
    float minZ = floatUnflipFromAtomic(bbox[2]);
    float maxX = floatUnflipFromAtomic(bbox[3]);
    float maxY = floatUnflipFromAtomic(bbox[4]);
    float maxZ = floatUnflipFromAtomic(bbox[5]);
    vec3 center = 0.5 * (vec3(minX, minY, minZ) + vec3(maxX, maxY, maxZ));
    float halfSize = 0.5 * max(maxX - minX, max(maxY - minY, maxZ - minZ));
    // A tiny padding factor keeps every particle strictly INSIDE the root
    // cube (not exactly on its boundary), and guards against a degenerate
    // zero-size box if every particle coincided exactly (halfSize would
    // otherwise be 0, making octantOf's comparisons meaningless).
    halfSize = max(halfSize * 1.001, 1e-6);
    treeCellGeom[0] = vec4(center, halfSize);
    treeCellDepth[0] = 0;
    for (int k = 0; k < 8; ++k) treeChild[8 * 0 + k] = -1;
    treeCellCounter[0] = 1u;
}
)";
}

// Pass 3a/4 (Barnes-Hut level loop, first half): one thread per particle
// not yet resolved into a leaf (TreeParticleCell<0 means "done, skip").
// Each active particle computes which of its CURRENT cell's 8 octants it
// falls in and does a single, unconditional atomicExchange to claim that
// (cell,octant) slot -- see this file's header comment for why this can
// never block on another thread, unlike the CAS-lock design this
// replaced. TdeSim clears TreeSlotHead to the empty sentinel before every
// level's dispatch of this pass.
inline std::string ClaimSlots() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 10) readonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 14) readonly buffer TreeParticleCellBuf { int treeParticleCell[]; };
layout(std430, binding = 15) buffer TreeSlotHeadBuf { uint treeSlotHead[]; };
layout(std430, binding = 16) writeonly buffer TreeSlotNextBuf { uint treeSlotNext[]; };

uniform int uN;
)" + std::string(TreeCommonGlsl) + R"(
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

// Pass 3b/4 (Barnes-Hut level loop, second half): one thread per
// POTENTIAL (cell, octant) slot -- dispatched over 8*maxCells, skipping
// slots whose cell doesn't exist yet or isn't at the depth being expanded
// this level. Reads the (now-finalized-for-this-level, since this runs
// after a barrier following ClaimSlots) candidate list ClaimSlots just
// built and resolves it: nothing claimed -> leave empty; exactly one
// candidate -> that particle becomes a direct leaf (a PLAIN write is safe
// -- this pass has exactly one thread per slot index, so two threads
// never target the same TreeChild entry); two or more candidates -> this
// thread allocates a new cell (the one remaining atomic op: a bump
// allocator with no CAS/retry, its result immediately valid) and
// reassigns every candidate in the list to that new cell for the next
// level's ClaimSlots pass to pick up -- a plain sequential walk of a list
// this one thread already exclusively owns, not a wait on any other
// thread's progress.
inline std::string ResolveSlots() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

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
)" + std::string(TreeCommonGlsl) + R"(
const uint kEmptySlot = 0xFFFFFFFFu;
// Ncrit leaf grouping (Barnes 1990's "8 or fewer particles" bucket variant --
// see this file's header comment): a slot with 2..kNcrit candidates stops
// subdividing right here and packs them directly into the new cell's own 8
// child slots as plain bodies, instead of recursing into another level.
// kNcrit==8 is not an arbitrary tuning choice -- it's exactly this array's
// existing per-cell width, so a leaf-group cell needs no new buffer and no
// change to ComputeTreeCellMass or the Forces() walk: both already treat
// any child<uN as "direct body, use its own mass/position" regardless of
// whether it got there via octant subdivision or leaf-group packing.
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

    // 2+ candidates: walk up to kNcrit+1 of them (bounded -- cheap
    // regardless of how long the full list actually is) to decide whether
    // this is a leaf group or needs real subdivision.
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
    if (newCell >= uMaxCells) return; // cell pool exhausted -- see TdeSim::CreateBuffers's
                                      // comment: these candidates simply stay unresolved and
                                      // silently drop out of the tree once the level loop ends,
                                      // same bounded/rare-case spirit as the max-depth cutoff
    vec3 newCenter = childCenter(geom.xyz, geom.w, octant);
    float newHalfSize = 0.5 * geom.w;
    treeCellGeom[newCell] = vec4(newCenter, newHalfSize);
    treeCellDepth[newCell] = uLevel + 1;
    treeChild[int(slotIdx)] = uN + newCell;

    if (!overflow) {
        // <= kNcrit candidates -- leaf group, done: pack them directly as
        // this cell's children and mark every one of them resolved.
        for (int k = 0; k < kNcrit; ++k) {
            treeChild[8 * newCell + k] = (k < count) ? members[k] : -1;
        }
        for (int k = 0; k < count; ++k) {
            treeParticleCell[members[k]] = -1;
        }
    } else {
        // more than kNcrit -- genuinely must subdivide further next level
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

// Pass 4/4 (Barnes-Hut): bottom-up mass/center-of-mass accumulation.
// TdeSim dispatches this once per depth level, from the deepest level the
// build actually reached down to 0 -- see this file's header comment for
// why a depth-driven dispatch order (rather than per-cell "children
// ready" tracking) is sufficient and correct: a cell's children always
// have a strictly greater depth, so by the time this runs for depth D,
// every depth->D+1 cell is already finalized.
inline std::string ComputeTreeCellMass() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

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
}

// Pass 1: adaptive smoothing length + density (kernel-weighted mass sum,
// "gather" form: rho_i uses only h_i) + EOS pressure + sound speed. h_i is
// solved by a short fixed-point loop, warm-started from last step's value
// (see kernels.hpp's header comment for the ansatz and why Forces() uses a
// different, symmetrized h for the pair interaction). Polytropic EOS
// P = K*rho^Gamma matches the Lane-Emden profile the initial conditions were
// sampled from (tools/make_star_ic.py), so a correctly-sampled star starts
// (approximately) in hydrostatic equilibrium.
inline std::string Density() {
    return CommonHeader() + SphKernelGlsl + std::string(SpatialHashGlsl) + R"(
layout(std430, binding = 6) readonly buffer CellHeadBuf { uint cellHead[]; };
layout(std430, binding = 7) readonly buffer NextIndexBuf { uint nextIndex[]; };

uniform float uK;
uniform float uGamma;
uniform float uEta;
uniform int uHIters;
uniform float uHMin;
uniform float uHMax;

// Includes j==i: standard SPH summation density is rho_i = sum_j m_j
// W(r_ij, h_i) over ALL particles including i itself (Monaghan 1992; Price
// 2012 review, eq. 5) -- W(0,h) is nonzero and its self-contribution is
// what keeps the estimate well-behaved as the local neighbor count drops
// (e.g. near the star's free surface). An earlier version of this shader
// excluded j==i after observing a 25-30% high bulk-density bias with it
// included; that bias was a fixed-point *convergence* artifact of only
// iterating uHIters=3 times (self-inclusion makes the rho->h->rho loop
// stiffer near the core), not a flaw in the self-term itself -- excluding
// it "fixed" that symptom but introduced the opposite, a systematic
// 10-30% *under*-estimate everywhere the neighbor count is modest, because
// every particle was missing its own legitimate mass contribution. Fixed
// for real by keeping the self-term and instead iterating enough times to
// converge (see sph.h_iters in the deck).
// Grid-accelerated: only visits the 27 cells (3x3x3, including its own)
// around ri, via the spatial hash built this step by
// TdeSim::BuildSpatialGrid -- see this file's header comment for why this
// is exact (no missed neighbors), not an approximation, as long as
// uCellSize (fixed at 2*hMax for the whole run) is used consistently here
// and when the grid was built. The r^2-before-sqrt check is still worth
// keeping even with the grid narrowing candidates down: a 3x3x3 cell block
// is a cube circumscribing (not inscribed in) the actual spherical 2h
// support, so corner cells still contain some pairs beyond the real cutoff.
float gatherDensity(vec3 ri, float h) {
    float rho = 0.0;
    float support2 = (2.0 * h) * (2.0 * h);
    ivec3 base = cellOf(ri);
    uint visited[27];
    int nVisited = 0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        uint bucket = spatialHash(base + ivec3(dx, dy, dz));
        bool dup = false;
        for (int v = 0; v < nVisited; ++v) {
            if (visited[v] == bucket) { dup = true; break; }
        }
        if (dup) continue;
        visited[nVisited++] = bucket;

        uint j = cellHead[bucket];
        while (j != 0xFFFFFFFFu) {
            vec3 rij = ri - posMass[j].xyz;
            float r2 = dot(rij, rij);
            if (r2 < support2) rho += posMass[j].w * kernelW(sqrt(r2), h);
            j = nextIndex[j];
        }
    }
    return rho;
}

// A particle in the far-stretched, thinned-out tail of a tidal debris
// stream can end up with literally zero neighbors within its kernel
// support even at h=uHMax (measured: ~0.3% of particles, rho exactly 0.0,
// ~30t after pericenter once the stream has stretched over 100+ length
// units) -- rho=0 then makes csound=sqrt(Gamma*pressure/rho) a literal
// 0/0 = NaN, which poisons every OTHER particle's force the very next
// step (Forces() loops over all N, so one NaN position contaminates the
// whole buffer within one frame). Flooring rho alone is not enough to fix
// this: P/rho^2 ~ rho^(Gamma-2) actually *diverges* as rho->0 for
// Gamma<2 (true here, Gamma=5/3), so a naive floor silently replaces a
// crash with a large spurious pressure kick instead. The physically
// correct treatment is that a particle with no real neighbors should feel
// no SPH pressure force at all (it still feels gravity, which doesn't
// depend on rho) -- so pressure/soundspeed are set to exactly 0 below
// that floor rather than computed from a floored rho, and Forces() must
// skip the P/rho^2 term entirely below the same floor (next function).
const float kRhoFloor = 1e-8;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;

    vec3 ri = posMass[i].xyz;
    float mi = posMass[i].w;
    float h = hArr[i];
    float rho = 0.0;
    for (int it = 0; it < uHIters; ++it) {
        rho = gatherDensity(ri, h);
        h = clamp(uEta * pow(mi / rho, 1.0 / 3.0), uHMin, uHMax); // rho==0 -> mi/rho=+inf -> clamps to uHMax, no NaN
    }
    rho = gatherDensity(ri, h); // final density consistent with the converged h

    hArr[i] = h;
    float pressure = (rho > kRhoFloor) ? uK * pow(rho, uGamma) : 0.0;
    float csound = (rho > kRhoFloor) ? sqrt(uGamma * pressure / rho) : 0.0;
    rhoPress[i] = vec4(rho, pressure, csound, 0.0);
}
)";
}

// Pass 2: total acceleration = self-gravity (Barnes-Hut tree walk) + SPH
// pressure gradient + Monaghan artificial viscosity. Self-gravity has no
// cutoff radius (every one of the N particles matters, however far away),
// so it can't be restricted to nearby space the way SPH's search was --
// instead it walks the octree built this substep by BuildTree/
// ComputeTreeCellMass, approximating distant compact clusters as a single
// point mass at their center of mass (see this file's header comment for
// the full design). The SPH term is unrelated and unchanged: it still
// only interacts within 2h, using the same spatial-hash grid Density()
// just built, visiting only the ~27-cells'-worth of actually-nearby
// candidates. Uses the symmetrized pair length h_ij = 0.5*(h_i+h_j) (same
// value regardless of which particle is "i"), so the kernel gradient --
// and hence the force -- is exactly antisymmetric under i<->j despite
// h_i != h_j in general.
inline std::string Forces() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kGravityGroupSize) + R"() in;

layout(std430, binding = 0) readonly buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) readonly buffer Vel { vec4 vel[]; };
layout(std430, binding = 2) writeonly buffer Acc { vec4 acc[]; };
layout(std430, binding = 3) readonly buffer RhoPress { vec4 rhoPress[]; };
layout(std430, binding = 4) readonly buffer HBuf { float hArr[]; };
layout(std430, binding = 6) readonly buffer CellHeadBuf { uint cellHead[]; };
layout(std430, binding = 7) readonly buffer NextIndexBuf { uint nextIndex[]; };
layout(std430, binding = 8) readonly buffer TreeChildBuf { int treeChild[]; };
layout(std430, binding = 9) readonly buffer TreeCellMassBuf { vec4 treeCellMass[]; };
layout(std430, binding = 10) readonly buffer TreeCellGeomBuf { vec4 treeCellGeom[]; };
layout(std430, binding = 18) readonly buffer TreeSortIndexBuf { int sortIndex[]; };

uniform int uN;
uniform int uPaddedN;
)" + SphKernelGlsl + std::string(SpatialHashGlsl) + R"(
uniform float uG;
uniform float uSoftening2;   // softening length squared
uniform float uViscAlpha;
uniform float uViscBeta;
uniform float uTreeTheta;      // Barnes-Hut opening angle; 0 = exhaustive (exact), ~0.5 = standard

// Shared (one per WORKGROUP, not one per thread) tree-walk stack -- see
// this file's header comment for the Bonsai-style grouped-walk design and
// why a shared stack, shared bounding box, and barrier()-synchronized
// traversal replace the old per-thread walk. Same depth margin as before
// (kTreeMaxStack over uTreeMaxDepth); only WHERE it lives changed.
const int kTreeMaxStack = 48;
shared int sStackNode[kTreeMaxStack];
shared int sStackNextChild[kTreeMaxStack];
shared int sSp;
shared vec3 sReduceMin[)" + std::to_string(kGravityGroupSize) + R"(];
shared vec3 sReduceMax[)" + std::to_string(kGravityGroupSize) + R"(];
shared vec3 sGroupCenter;
shared float sGroupRadius;

// Black hole: a fixed point mass at the origin (M_BH >> M_star, so its
// recoil from the star's gravity is negligible to leading order in the
// mass ratio -- not integrated). uBhType: 0 off, 1 Newtonian point mass,
// 2 Paczynski-Wiita (a = -G*M/(r-r_s)^2, reproduces the true ISCO at 6M and
// diverges at the Schwarzschild radius r_s -- a pseudo-relativistic
// stand-in for pericenter precession/plunge without a real metric). Both
// forms are floored at uSoftening (already sized for the star's own
// resolution) below r_s so a particle passing through the horizon gets a
// large but finite kick instead of a NaN.
uniform int uBhType;
uniform float uBhMass;
uniform float uBhRs;

// See Density()'s kRhoFloor comment: a particle with no real SPH neighbors
// (rho==0, measured in the far-stretched tail of a tidal debris stream)
// must contribute exactly zero pressure/viscosity force -- P/rho^2 diverges
// as rho->0 for Gamma<2 (true here), so skipping the term below the floor
// is required for correctness, not just a NaN-avoidance patch.
const float kRhoFloor = 1e-8;

void main() {
    uint localIdx = gl_LocalInvocationID.x;
    uint slot = gl_WorkGroupID.x * )" + std::to_string(kGravityGroupSize) + R"(u + localIdx;
    int myIndex = (slot < uint(uPaddedN)) ? sortIndex[slot] : -1;
    bool valid = (myIndex >= 0);
    uint i = valid ? uint(myIndex) : 0u; // dummy index 0 for padding threads -- never written back

    vec3 ri = posMass[i].xyz;

    // Group bounding box: every invocation contributes its own particle's
    // position (padding threads contribute a neutral value that can never
    // win the min/max), then invocation 0 alone reduces the group's 64
    // entries serially (cheap -- kGravityGroupSize is small) and derives
    // a bounding SPHERE (center, radius) from it. This -- not any single
    // thread's own position -- is what every accept/open decision below
    // is based on, which is what keeps the whole walk loop's control flow
    // uniform across the group (see this file's header comment).
    sReduceMin[localIdx] = valid ? ri : vec3(3.0e38);
    sReduceMax[localIdx] = valid ? ri : vec3(-3.0e38);
    barrier();
    if (localIdx == 0u) {
        vec3 mn = sReduceMin[0];
        vec3 mx = sReduceMax[0];
        for (int k = 1; k < )" + std::to_string(kGravityGroupSize) + R"(; ++k) {
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

    // Self-gravity: cooperative Barnes-Hut tree walk, whole group in
    // lockstep. Each invocation accumulates its OWN force independently
    // (aGrav) whenever the shared walk accepts or reaches a leaf body --
    // that accumulation, and it alone, is genuinely per-thread; every
    // stack push/pop/frame-advance below is done ONCE by invocation 0 and
    // read back by everyone via `shared` memory.
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
            // a leaf body -- direct per-thread contribution (i==childVal
            // is safe and needs no special-casing: rij=0 makes the
            // numerator vanish, same convention as the rest of this file)
            vec3 rij = ri - posMass[childVal].xyz;
            float r2 = dot(rij, rij);
            float invR3 = pow(r2 + uSoftening2, -1.5);
            aGrav -= uG * posMass[childVal].w * invR3 * rij;
            continue;
        }

        int cellIdx = childVal - uN;
        vec4 geom = treeCellGeom[cellIdx];
        vec4 cm = treeCellMass[cellIdx];
        // Group opening-angle criterion: rMin is a PROVABLY CONSERVATIVE
        // (by the triangle inequality) lower bound on every actual group
        // member's true distance to this cell's center of mass, using
        // only the group's own bounding sphere -- see this file's header
        // comment for the full argument. Every invocation computes this
        // from purely shared/uniform inputs, so every invocation gets the
        // bit-identical answer and the group always agrees.
        float d = distance(sGroupCenter, cm.xyz);
        float rMin = max(0.0, d - sGroupRadius);
        float cellSize = 2.0 * geom.w; // full cube width, not half-size
        bool accept = (rMin > 0.0) && (cellSize < uTreeTheta * rMin);
        // uTreeTheta=0 makes "cellSize < 0" false unconditionally, forcing
        // exhaustive descent to every leaf -- --selftest still validates
        // this exact (not approximated) mode against brute force.

        if (accept) {
            vec3 rij = ri - cm.xyz;
            float dist2 = dot(rij, rij);
            float invR3 = pow(dist2 + uSoftening2, -1.5);
            aGrav -= uG * cm.w * invR3 * rij;
            continue;
        }

        // must open -- invocation 0 alone pushes this cell's children
        barrier();
        if (localIdx == 0u && sSp < kTreeMaxStack) {
            sStackNode[sSp] = cellIdx;
            sStackNextChild[sSp] = 0;
            sSp++;
        }
        barrier();
        // sSp==kTreeMaxStack: stack exhausted (tree deeper than this fixed
        // budget allows) -- drop this cell's contribution for this one
        // group this substep rather than overflow the array. Exceedingly
        // rare given kTreeMaxStack's margin over uTreeMaxDepth; same
        // accepted-rare-approximation spirit as BuildBarnesHutTree's own
        // max-depth cutoff (see header comment).
    }

    vec3 a = aGrav;
    if (uBhType == 1) {
        float r2 = dot(ri, ri);
        a -= uG * uBhMass * pow(r2 + uSoftening2, -1.5) * ri;
    } else if (uBhType == 2) {
        float r = length(ri);
        float dr = max(r - uBhRs, sqrt(uSoftening2));
        a -= uG * uBhMass / (dr * dr) * (ri / max(r, 1e-8));
    }

    // SPH pressure gradient + artificial viscosity, symmetrized h --
    // grid-accelerated (see this file's header comment and Density()'s
    // gatherDensity for the identical 27-cell scheme). Purely per-thread,
    // using this invocation's own REAL particle index -- unaffected by
    // any of the grouped-gravity machinery above, since SPH's own search
    // never depends on array/traversal order.
    vec3 vi = vel[i].xyz;
    float hi = hArr[i];
    float rhoi = rhoPress[i].x;
    float Pi = rhoPress[i].y;
    float ci = rhoPress[i].z;

    ivec3 base = cellOf(ri);
    uint visitedF[27];
    int nVisitedF = 0;
    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        uint bucket = spatialHash(base + ivec3(dx, dy, dz));
        bool dupF = false;
        for (int v = 0; v < nVisitedF; ++v) {
            if (visitedF[v] == bucket) { dupF = true; break; }
        }
        if (dupF) continue;
        visitedF[nVisitedF++] = bucket;

        // A linked-list walk, not a for-loop over a contiguous range --
        // every early-out below must advance j to nextIndex[j] BEFORE
        // "continue" (a plain "continue" here would re-check the same j
        // forever: unlike a for-loop's implicit ++k, a while loop has no
        // separate increment step for continue to fall into).
        uint j = cellHead[bucket];
        while (j != 0xFFFFFFFFu) {
            uint jNext = nextIndex[j];
            vec3 rij = ri - posMass[j].xyz;
            float r2 = dot(rij, rij);
            float mj = posMass[j].w;
            float hij = 0.5 * (hi + hArr[j]);
            float support2 = (2.0 * hij) * (2.0 * hij);
            if (r2 >= support2) { j = jNext; continue; }

            float r = sqrt(r2);
            float dWdr = kernelDwDr(r, hij);
            if (dWdr == 0.0) { j = jNext; continue; }
            vec3 gradW = (r > 0.0) ? (dWdr / r) * rij : vec3(0.0);
            float rhoj = rhoPress[j].x;
            float Pj = rhoPress[j].y;

            float rhobar = 0.5 * (rhoi + rhoj);
            float piVisc = 0.0;
            if (rhobar > kRhoFloor) {
                vec3 vij = vi - vel[j].xyz;
                float vijDotRij = dot(vij, rij);
                if (vijDotRij < 0.0) {
                    float cj = rhoPress[j].z;
                    float mu = hij * vijDotRij / (r2 + 0.01 * hij * hij);
                    float cbar = 0.5 * (ci + cj);
                    piVisc = (-uViscAlpha * cbar * mu + uViscBeta * mu * mu) / rhobar;
                }
            }

            float PiOverRho2 = (rhoi > kRhoFloor) ? Pi / (rhoi * rhoi) : 0.0;
            float PjOverRho2 = (rhoj > kRhoFloor) ? Pj / (rhoj * rhoj) : 0.0;
            a -= mj * (PiOverRho2 + PjOverRho2 + piVisc) * gradW;
            j = jNext;
        }
    }

    if (valid) acc[i] = vec4(a, 0.0);
}
)";
}

// Pass: accretion check. A live particle (mass>0) that has fallen within
// uAccretionRadius of the black hole is "consumed" -- its mass is recorded
// into AccretedMass (for the CPU to sum and add onto the BH's own growing
// mass, see TdeSim::RunAccretion) and zeroed in PosMass, which makes it
// inert everywhere else for free: Density()'s and Forces()'s mass-weighted
// sums already give a zero-mass particle exactly zero contribution (no
// separate "is it dead" branch needed there), and Render() separately
// skips mass<=0 particles so it simply vanishes rather than continuing to
// sit at the horizon. It keeps drifting under gravity from everything else
// after death (a mass-independent test-particle trajectory) since nothing
// stops its own Kick/Drift -- harmless (it influences nothing) and cheap
// to just ignore rather than special-case out of the integrator.
// Needs its own read-write PosMass declaration (CommonHeader's is
// read-only, shared by Density/Forces) -- same reason Drift() has one.
inline std::string Accretion() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 5) writeonly buffer AccretedMass { float accreted[]; };

uniform int uN;
uniform float uAccretionRadius;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    float m = posMass[i].w;
    float r = length(posMass[i].xyz);
    if (m > 0.0 && r < uAccretionRadius) {
        accreted[i] = m;
        posMass[i].w = 0.0;
    } else {
        accreted[i] = 0.0;
    }
}
)";
}

// Leapfrog half-kick: vel += halfDt * acc, then an optional exponential
// velocity damping (uDampingFactor = exp(-damping*halfDt), so 1.0 = off).
// Used only during relaxation runs (decks/star_relax_*.toml) to let a
// Monte-Carlo-sampled star settle into numerical equilibrium -- artificial
// viscosity alone damps convergent/shock flows, not the smooth global
// radial breathing mode a freshly-sampled star tends to ring at, which
// otherwise oscillates indefinitely without growing or decaying. Off
// (uDampingFactor=1) once the star is actually being disrupted, since real
// tidal dynamics must not be damped.
inline std::string Kick() {
    return CommonHeader() + R"(
uniform float uHalfDt;
uniform float uDampingFactor;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    vel[i].xyz = (vel[i].xyz + uHalfDt * acc[i].xyz) * uDampingFactor;
}
)";
}

// Leapfrog drift: pos += dt * vel. PosMass is read-write here (unlike the
// Density/Forces passes), so this kernel declares its own binding-0 buffer.
inline std::string Drift() {
    return std::string(R"(#version 460 core
layout(local_size_x = )") + std::to_string(kWorkgroupSize) + R"() in;

layout(std430, binding = 0) buffer PosMass { vec4 posMass[]; };
layout(std430, binding = 1) readonly buffer Vel { vec4 vel[]; };

uniform int uN;
uniform float uDt;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= uint(uN)) return;
    posMass[i].xyz += uDt * vel[i].xyz;
}
)";
}

} // namespace tde::kernels
