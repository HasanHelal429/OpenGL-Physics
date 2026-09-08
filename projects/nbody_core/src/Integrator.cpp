#include "ngrav/Integrator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ngrav {

template <int D>
double LeapfrogStep(SoA<D>& s, const StepParams& sp, const AccelFn<D>& accel, SoA<D>& scratch) {
    const std::size_t n = s.Count();
    const double dt = sp.dt;
    const double hdt = 0.5 * dt;

    // half kick
    for (std::size_t i = 0; i < n; ++i) {
        s.vx[i] += hdt * s.ax[i];
        s.vy[i] += hdt * s.ay[i];
        if constexpr (D == 3) s.vz[i] += hdt * s.az[i];
    }
    // drift
    for (std::size_t i = 0; i < n; ++i) {
        s.x[i] += dt * s.vx[i];
        s.y[i] += dt * s.vy[i];
        if constexpr (D == 3) s.z[i] += dt * s.vz[i];
    }
    // recompute a(t+dt)
    accel(s, sp, scratch);
    s.ax = scratch.ax;
    s.ay = scratch.ay;
    if constexpr (D == 3) s.az = scratch.az;
    // half kick
    for (std::size_t i = 0; i < n; ++i) {
        s.vx[i] += hdt * s.ax[i];
        s.vy[i] += hdt * s.ay[i];
        if constexpr (D == 3) s.vz[i] += hdt * s.az[i];
    }
    return dt;
}

template <int D>
double ChooseAdaptiveDt(const SoA<D>& s, double eta, double eps, double dtFallback) {
    const std::size_t n = s.Count();
    if (n == 0 || eps <= 0.0) return dtFallback;
    double minTau = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < n; ++i) {
        const double ax = s.ax[i], ay = s.ay[i];
        const double az = (D == 3) ? s.az[i] : 0.0;
        const double a = std::sqrt(ax * ax + ay * ay + az * az);
        if (a > 0.0) minTau = std::min(minTau, std::sqrt(eps / a));
    }
    if (minTau == std::numeric_limits<double>::max()) return dtFallback;
    const double dt = eta * minTau;
    // Clamp within 4x of the deck's nominal dt so a single close pass can't
    // stall the whole run to a crawl (block timesteps in P15 remove this cap).
    return std::clamp(dt, 0.25 * dtFallback, 4.0 * dtFallback);
}

template double LeapfrogStep<2>(SoA<2>&, const StepParams&, const AccelFn<2>&, SoA<2>&);
template double LeapfrogStep<3>(SoA<3>&, const StepParams&, const AccelFn<3>&, SoA<3>&);
template double ChooseAdaptiveDt<2>(const SoA<2>&, double, double, double);
template double ChooseAdaptiveDt<3>(const SoA<3>&, double, double, double);

} // namespace ngrav
