#pragma once

#include "Mac.hpp"
#include "Softening.hpp"
#include "State.hpp"
#include "Tree.hpp"
#include "Vec.hpp"

namespace ngrav {

enum class Solver {
    Direct,       // O(N^2) all-pairs reference
    BarnesHut,    // adaptive tree, opening-angle MAC
    Fmm,          // mutual dual-tree FMM, monopole+quadrupole (3D) -- Phase 6; retired
                  // the one-directional AdaptiveFmm half-measure it replaced
    SphericalFmm, // arbitrary-order solid-harmonic FMM (3D) -- accuracy reference
    ComplexFmm,   // 2D complex-Laurent FMM
};

// Everything a single force evaluation needs. One value type threaded through
// System::Step and the standalone benchmark path.
struct StepParams {
    double G = 1.0;
    Softening soft = Softening::Plummer(0.05);
    double dt = 1e-3;
    MacParams mac{};
    Solver solver = Solver::BarnesHut;

    bool adaptive = false;  // adaptive global timestep (P3)
    double eta = 0.03;      // dt = eta * min sqrt(eps / |a|)

    int fmmOrder = 3;       // Cartesian-Taylor order (P6)
    int sphericalOrder = 5; // solid-harmonic expansion order (P7)
};

// ---- Direct O(N^2) ----------------------------------------------------------
// Writes acceleration (G-included) into out.ax/ay/az. `out` is resized/zeroed.
template <int D>
void ComputeAccelDirect(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out);

// ---- Barnes-Hut -----------------------------------------------------------
// Monopole walk for P1 (+ quadrupole and relative MAC in P5). `aOld` (per
// particle previous |a|) is used only when sp.mac.kind == Relative; pass an
// empty span for the geometric path.
template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                           std::span<const double> aOld, SoA<D>& out);

// Convenience: build the tree internally.
template <int D>
void ComputeAccelBarnesHut(const PosMassView<D>& pts, const StepParams& sp, SoA<D>& out);

} // namespace ngrav
