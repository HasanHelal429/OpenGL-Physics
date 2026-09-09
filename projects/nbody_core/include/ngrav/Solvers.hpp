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

    bool block = false;     // block / power-of-2 rung timesteps (P15; Direct/Barnes-Hut only,
                            // else falls back to `adaptive`). `dt` is the coarsest (rung-0) step;
                            // a particle whose eta-criterion wants a smaller step drops to a finer
                            // rung dt/2^k. One System::Step advances by `dt` regardless of scheme.
    int blockMaxRung = 8;   // finest rung is dt / 2^blockMaxRung

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

// The pre-Phase-8 per-particle walk (one tree descent per particle), kept
// as an explicit, separately-callable entry point: it's what `Relative` MAC
// still uses internally (needs each particle's own previous |a|, which has
// no shared group value), and it's the correctness/speed baseline the
// Phase-8 selftest measures the new grouped walk against.
template <int D>
void ComputeAccelBarnesHutPerParticle(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                                      std::span<const double> aOld, SoA<D>& out);

// ---- target-subset force evaluation (P15 block timesteps) ------------------
// Compute acceleration for ONLY the particles named in `targets` (all N
// particles are still sources, at their current positions). Writes only
// out.a{x,y,z}[targets[k]]; other entries of `out` are left untouched.
// `out` must already be sized to N (ResizeAccel). This is what makes rung/
// block timestepping actually cheaper -- an inactive particle's force isn't
// recomputed, but the force *from* it stays current because it drifted.
// The mutual FMM has no target-subset form (its M2L is symmetric by
// construction), so block timestepping is Direct/Barnes-Hut only; see
// Integrator.hpp.
template <int D>
void ComputeAccelDirectTargets(const PosMassView<D>& pts, const StepParams& sp, std::span<const int> targets,
                               SoA<D>& out);
template <int D>
void ComputeAccelBarnesHutTargets(const PosMassView<D>& pts, const StepParams& sp, const AdaptiveTree<D>& tree,
                                  std::span<const int> targets, SoA<D>& out);

} // namespace ngrav
