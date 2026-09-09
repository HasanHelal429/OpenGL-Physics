#pragma once

#include "Integrator.hpp"
#include "MutualFmm.hpp"
#include "Solvers.hpp"
#include "State.hpp"
#include "Vec.hpp"

#include <cmath>
#include <functional>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ngrav {

// The one object both consumers (the ImGui NBodyApp and the deck-driven
// NBodySim) talk to. Owns the SoA particle state, the leapfrog integrator,
// and the solver dispatch. Direct and Barnes-Hut are handled natively; the
// FMM solver types are handled through adapters the project registers
// (SetAuxSolver) -- in P1 those still run the legacy in-project FMM code
// against a rebuilt AoS view; P6/P7 move them into nbody_core.
template <int D>
class System {
public:
    System() = default;

    void SetParticles(SoA<D> s) {
        m_state = std::move(s);
        const std::size_t n = m_state.Count();
        m_state.ZeroAcc();
        m_aOldMag.assign(n, 0.0);
        m_primed = false;
    }

    // Register a force evaluator for an FMM solver type.
    void SetAuxSolver(Solver which, AccelFn<D> fn) { m_aux[static_cast<int>(which)] = std::move(fn); }

    // Fill acceleration for `solver` over `in`, writing into `out` (G included).
    void ComputeAccel(Solver solver, const SoA<D>& in, const StepParams& sp, SoA<D>& out) const {
        ++m_forceEvals;
        const PosMassView<D> v = ViewOf(in);
        if constexpr (D == 3) {
            if (solver == Solver::Fmm) {
                ComputeAccelMutualFmm(v, sp, out);
                return;
            }
        }
        switch (solver) {
            case Solver::Direct:
                ComputeAccelDirect<D>(v, sp, out);
                return;
            case Solver::BarnesHut: {
                AdaptiveTree<D> tree(v);
                if constexpr (D == 3) tree.ComputeQuadrupoles(v);
                const std::span<const double> aOld =
                    (sp.mac.kind == MacKind::Relative && m_aOldMag.size() == in.Count())
                        ? std::span<const double>(m_aOldMag)
                        : std::span<const double>();
                ComputeAccelBarnesHut<D>(v, sp, tree, aOld, out);
                return;
            }
            default: {
                auto it = m_aux.find(static_cast<int>(solver));
                if (it != m_aux.end()) {
                    it->second(in, sp, out);
                } else {
                    // Unregistered FMM -> fall back to Barnes-Hut so the app never wedges.
                    ComputeAccelBarnesHut<D>(v, sp, out);
                }
                return;
            }
        }
    }

    // Prime a(t) so the first leapfrog half-kick is valid. Call after
    // SetParticles and whenever G/softening/solver/theta change while paused.
    void Prime(const StepParams& sp) {
        RecomputeAccel(sp);
        m_primed = true;
    }

    // One leapfrog step. Returns the dt actually taken (== sp.dt unless
    // sp.adaptive). Requires Prime() to have run.
    double Step(const StepParams& sp) {
        if (m_state.Count() == 0) return sp.dt;
        if (!m_primed) Prime(sp);

        StepParams eff = sp;
        if (sp.adaptive) eff.dt = ChooseAdaptiveDt<D>(m_state, sp.eta, sp.soft.eps, sp.dt);

        const double taken = LeapfrogStep<D>(
            m_state, eff, [this](const SoA<D>& in, const StepParams& p, SoA<D>& o) { ComputeAccel(p.solver, in, p, o); },
            m_scratch);
        CaptureAOld();
        return taken;
    }

    // ---- diagnostics (O(N^2) potential; call sparingly / off-thread) ----
    double TotalEnergy(const StepParams& sp) const {
        const SoA<D>& s = m_state;
        const long n = static_cast<long>(s.Count());
        double kinetic = 0.0;
        for (long i = 0; i < n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double v2 = s.vx[ii] * s.vx[ii] + s.vy[ii] * s.vy[ii] + ((D == 3) ? s.vz[ii] * s.vz[ii] : 0.0);
            kinetic += 0.5 * s.m[ii] * v2;
        }
        const Softening soft = sp.soft;
        const double G = sp.G;
        const bool spline = (soft.kind == SofteningKind::Spline);
        double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 32) if (n > 256)
        for (long i = 0; i < n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            double local = 0.0;
            for (long j = i + 1; j < n; ++j) {
                const std::size_t jj = static_cast<std::size_t>(j);
                const double dx = s.x[jj] - s.x[ii], dy = s.y[jj] - s.y[ii];
                const double dz = (D == 3) ? (s.z[jj] - s.z[ii]) : 0.0;
                const double mij = s.m[ii] * s.m[jj];
                if (spline) {
                    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    const double q = (r <= 1e-300) ? 0.0 : r / soft.eps;
                    if constexpr (D == 3) {
                        // phi(q)*h/(Gm) -> phi = G*m*[that]/h; potential energy term = G*mi*mj*phi(q)/h.
                        local += G * mij * detail::SplinePotential3D(q) / soft.eps;
                    } else {
                        // phi_actual(r) = ln(h) + phi(q); the table gives phi(q) directly
                        // (-> ln(q) for q>=2, matching the unsoftened ln(r) branch exactly).
                        local += G * mij * (std::log(soft.eps) + detail::Spline2DTable()(q));
                    }
                } else {
                    const double r2 = dx * dx + dy * dy + dz * dz + soft.eps2;
                    if constexpr (D == 3) {
                        local -= G * mij / std::sqrt(r2);
                    } else {
                        local += G * mij * std::log(std::sqrt(r2));
                    }
                }
            }
            potential += local;
        }
        return kinetic + potential;
    }

    Vec<D> CenterOfMass() const {
        const SoA<D>& s = m_state;
        double tm = 0.0;
        Vec<D> c(0.0);
        for (std::size_t i = 0; i < s.Count(); ++i) {
            c += s.m[i] * s.Pos(i);
            tm += s.m[i];
        }
        return tm > 0.0 ? c / tm : c;
    }

    AngMom<D> AngularMomentum() const {
        const SoA<D>& s = m_state;
        const Vec<D> com = CenterOfMass();
        if constexpr (D == 3) {
            glm::dvec3 L(0.0);
            for (std::size_t i = 0; i < s.Count(); ++i)
                L += s.m[i] * glm::cross(s.Pos(i) - com, s.Vel(i));
            return L;
        } else {
            double L = 0.0;
            for (std::size_t i = 0; i < s.Count(); ++i) {
                const Vec<2> r = s.Pos(i) - com;
                L += s.m[i] * (r.x * s.vy[i] - r.y * s.vx[i]);
            }
            return L;
        }
    }

    std::size_t Count() const { return m_state.Count(); }
    const SoA<D>& State() const { return m_state; }
    SoA<D>& MutableState() { return m_state; }
    std::span<const double> AOldMag() const { return m_aOldMag; }

    long ForceEvals() const { return m_forceEvals; }
    void ResetForceEvals() { m_forceEvals = 0; }

private:
    void RecomputeAccel(const StepParams& sp) {
        ComputeAccel(sp.solver, m_state, sp, m_state);
        CaptureAOld();
    }
    void CaptureAOld() {
        const std::size_t n = m_state.Count();
        if (m_aOldMag.size() != n) m_aOldMag.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            const double ax = m_state.ax[i], ay = m_state.ay[i];
            const double az = (D == 3) ? m_state.az[i] : 0.0;
            m_aOldMag[i] = std::sqrt(ax * ax + ay * ay + az * az);
        }
    }

    SoA<D> m_state;
    SoA<D> m_scratch;
    std::vector<double> m_aOldMag;
    std::unordered_map<int, AccelFn<D>> m_aux;
    bool m_primed = false;
    mutable long m_forceEvals = 0;
};

} // namespace ngrav
