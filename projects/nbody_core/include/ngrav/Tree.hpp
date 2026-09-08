#pragma once

#include "State.hpp"
#include "Vec.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace ngrav {

// Flat, structure-of-arrays adaptive tree (octree for D==3, quadtree for
// D==2). Replaces 02's AdaptiveOctree and 03's AdaptiveQuadtree with one
// implementation:
//
//  - Node scalars/vectors live in parallel arrays (center, halfSize, com,
//    mass, maxRadius, parent), children in a flat `kChildren * numNodes`
//    int array (-1 = empty octant), so nothing is pointer-chased and the
//    whole tree is one contiguous block that a GPU port can upload directly
//    (this is exactly 06_tidal_disruption's treeChild[8*maxCells] layout).
//  - Leaves own a *contiguous range* [firstParticle, firstParticle +
//    particleCount) into an internal `order` permutation array -- no
//    std::vector<int> per node, no per-leaf heap allocation, and leaf
//    particle data is streamed in order during the walk.
//  - The "child node index > its parent's index" invariant of the old trees
//    is preserved (insertion order), so the FMM's L2L pass is one
//    increasing-index sweep and the quadrupole/multipole upsweep is one
//    decreasing-index sweep -- no recursion, no explicit BFS.
//
// `ncrit` is the leaf bucket size (a leaf splits once it would hold more than
// ncrit particles); default 8 matches 06 and bounds the near-field pair cost.
// `maxDepth` is a safety valve for (near-)coincident positions only.
template <int D>
class AdaptiveTree {
public:
    static constexpr int kNC = kChildren<D>;

    AdaptiveTree(const PosMassView<D>& pts, int ncrit = 8, int maxDepth = 40);

    int NumNodes() const { return static_cast<int>(mass.size()); }
    int Root() const { return 0; }
    bool IsLeaf(int n) const { return childCount[static_cast<std::size_t>(n)] == 0; }

    // --- node SoA (index by node id) ---
    std::vector<Vec<D>> center;   // geometric box center
    std::vector<double> halfSize; // half the box edge length
    std::vector<Vec<D>> com;      // center of mass
    std::vector<double> mass;     // total mass under the node
    std::vector<double> maxRadius; // farthest member particle from `com` (0 for a 1-particle leaf)
    std::vector<std::array<int, kNC>> children; // -1 = empty
    std::vector<int8_t> childCount;             // 0 => leaf
    std::vector<int> parent;                    // -1 for the root
    std::vector<int> firstParticle;             // offset into `order` (leaves only, else -1)
    std::vector<int> particleCount;             // leaves only, else 0

    // Particle permutation: leaf n owns order[firstParticle[n] ..
    // firstParticle[n] + particleCount[n]). Original particle index = order[k].
    std::vector<int> order;

    // Traceless Cartesian quadrupole about each node's `com` (D == 3 only;
    // Qzz recovered as -Qxx-Qyy). Empty until ComputeQuadrupoles() runs;
    // Barnes-Hut's monopole walk never needs it.
    std::vector<double> Qxx, Qyy, Qxy, Qxz, Qyz;
    void ComputeQuadrupoles(const PosMassView<D>& pts);
    bool HasQuadrupoles() const { return !Qxx.empty(); }

private:
    void BuildRoot(const PosMassView<D>& pts);
    void Insert(const PosMassView<D>& pts, int node, int p, int depth);
    void InsertIntoChild(const PosMassView<D>& pts, int node, int p, int depth);
    int NewChild(int parentNode, int octant);
    void Finalize(const PosMassView<D>& pts); // build `order`, maxRadius

    int m_ncrit;
    int m_maxDepth;
    // Transient per-node particle lists, used only during build, cleared in Finalize.
    std::vector<std::vector<int>> m_buildParticles;
};

} // namespace ngrav
