#pragma once

#include "Solvers.hpp"
#include "State.hpp"

#include <functional>

namespace ngrav {

// Force-evaluation callback: fills out.ax/ay/az (G included) for the given
// positions/masses. System supplies this, closing over the solver dispatch
// (Direct / Barnes-Hut natively, FMM via a project-registered adapter).
template <int D>
using AccelFn = std::function<void(const SoA<D>& in, const StepParams& sp, SoA<D>& out)>;

// Kick-drift-kick leapfrog (velocity Verlet), symplectic for a fixed dt.
// P1 ships the fixed-dt path only; ChooseAdaptiveDt (P3) and the block/rung
// scheme (P15) layer on here.
//
// Precondition: state.ax/ay/az already hold a(t) (System::Prime, or the
// previous Step's trailing half-kick left them current).
// Postcondition: state advanced to t+dt, ax/ay/az hold a(t+dt).
// Returns the dt actually taken (== dt for the fixed path).
template <int D>
double LeapfrogStep(SoA<D>& state, const StepParams& sp, const AccelFn<D>& accel, SoA<D>& scratch);

// dt = eta * min_i sqrt(eps / |a_i|), clamped to [dtMin, dtMax]. Symmetric
// (uses the current a) so the leapfrog stays approximately time-reversible.
template <int D>
double ChooseAdaptiveDt(const SoA<D>& state, double eta, double eps, double dtFallback);

} // namespace ngrav
