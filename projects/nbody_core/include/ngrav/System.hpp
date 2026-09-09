#pragma once

#include "Integrator.hpp"
#include "MutualFmm.hpp"
#include "Solvers.hpp"
#include "State.hpp"
#include "Vec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
        m_particleEvals += static_cast<long>(in.Count());
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

        // Block / power-of-2 rung timesteps (P15): Direct and Barnes-Hut
        // only (they support a target subset; the mutual FMM's M2L is
        // symmetric by construction and has no target-subset form). For any
        // other solver, block silently degrades to the global scheme.
        const bool blockOk = sp.block && (sp.solver == Solver::Direct || sp.solver == Solver::BarnesHut);
        if (blockOk) return BlockStep(sp);

        StepParams eff = sp;
        if (sp.adaptive) eff.dt = ChooseAdaptiveDt<D>(m_state, sp.eta, sp.soft.eps, sp.dt);

        const double taken = LeapfrogStep<D>(
            m_state, eff, [this](const SoA<D>& in, const StepParams& p, SoA<D>& o) { ComputeAccel(p.solver, in, p, o); },
            m_scratch);
        CaptureAOld();
        return taken;
    }

    // Per-particle acceleration evaluations since the last ResetForceEvals:
    // one per particle per global-scheme force pass, or (block scheme) the
    // running sum of |active set| over every finest substep. This is the
    // apples-to-apples cost metric for the P15 gate -- ForceEvals() counts
    // whole passes, which isn't comparable across schemes.
    long ParticleEvals() const { return m_particleEvals; }
    void ResetParticleEvals() { m_particleEvals = 0; }

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

    // Per-particle power-of-2 rungs from the last block Step (empty / all-zero
    // for the global scheme). rung k => that particle stepped at dt/2^k.
    std::span<const int> Rungs() const { return m_rung; }

private:
    void RecomputeAccel(const StepParams& sp) {
        ComputeAccel(sp.solver, m_state, sp, m_state);
        CaptureAOld();
    }

    // Fill m_state.a{x,y,z} for ONLY the particles in `targets` (all N are
    // sources). Direct / Barnes-Hut only -- the caller (BlockStep) guarantees
    // that. For Barnes-Hut the tree is rebuilt each call: the fine particles
    // move enough within a coarse step that a stale tree would degrade the
    // MAC; this keeps the block scheme's accuracy honest at the cost of the
    // repeated (O(N)) builds -- the walk savings still dominate at the N
    // where trees matter (see the P15 gate discussion).
    void ComputeAccelTargets(Solver solver, const StepParams& sp, const std::vector<int>& targets) {
        if (targets.empty()) return;
        m_particleEvals += static_cast<long>(targets.size());
        const PosMassView<D> v = ViewOf(m_state);
        if (solver == Solver::Direct) {
            ComputeAccelDirectTargets<D>(v, sp, targets, m_state);
        } else {
            AdaptiveTree<D> tree(v);
            if constexpr (D == 3) tree.ComputeQuadrupoles(v);
            ComputeAccelBarnesHutTargets<D>(v, sp, tree, targets, m_state);
        }
    }

    // One coarse (rung-0) step of length sp.dt, subdivided into power-of-2
    // rungs. KDK per rung: opening half-kick at the rung's step start, a
    // global drift every finest substep, closing half-kick + force refresh
    // at the rung's step end. Every coarse step ends with a full force
    // evaluation (the finest substep's closing set is all N), so a(t+dt) is
    // current for the next call and for diagnostics.
    double BlockStep(const StepParams& sp) {
        const int n = static_cast<int>(m_state.Count());
        const double base = sp.dt;
        const int maxRung = std::max(0, sp.blockMaxRung);
        const double eps = sp.soft.eps;

        if (static_cast<int>(m_rung.size()) != n) m_rung.assign(static_cast<std::size_t>(n), 0);

        // Assign each particle's rung from its current |a|: it wants
        // tau_i = eta*sqrt(eps/|a_i|); k_i is the smallest power-of-2
        // subdivision of `base` that is <= tau_i, clamped to [0, maxRung].
        int rMax = 0;
        for (int i = 0; i < n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double az = (D == 3) ? m_state.az[ii] : 0.0;
            const double amag = std::sqrt(m_state.ax[ii] * m_state.ax[ii] + m_state.ay[ii] * m_state.ay[ii] + az * az);
            int k = 0;
            if (amag > 0.0 && eps > 0.0) {
                const double tau = sp.eta * std::sqrt(eps / amag);
                if (tau < base) k = std::min(maxRung, static_cast<int>(std::ceil(std::log2(base / tau))));
            }
            m_rung[ii] = k;
            rMax = std::max(rMax, k);
        }

        const int nSub = 1 << rMax;
        const double dtMin = base / nSub;

        std::vector<int> active;
        active.reserve(static_cast<std::size_t>(n));
        for (int sub = 0; sub < nSub; ++sub) {
            // opening half-kick: particles whose step starts at `sub`
            for (int i = 0; i < n; ++i) {
                const int stepUnits = 1 << (rMax - m_rung[static_cast<std::size_t>(i)]);
                if (sub % stepUnits == 0) HalfKick(i, 0.5 * (base / (1 << m_rung[static_cast<std::size_t>(i)])));
            }
            // drift everyone by the finest substep
            for (int i = 0; i < n; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                m_state.x[ii] += dtMin * m_state.vx[ii];
                m_state.y[ii] += dtMin * m_state.vy[ii];
                if constexpr (D == 3) m_state.z[ii] += dtMin * m_state.vz[ii];
            }
            // closing set: particles whose step ends at sub+1
            active.clear();
            for (int i = 0; i < n; ++i) {
                const int stepUnits = 1 << (rMax - m_rung[static_cast<std::size_t>(i)]);
                if ((sub + 1) % stepUnits == 0) active.push_back(i);
            }
            ComputeAccelTargets(sp.solver, sp, active);
            for (int i : active) HalfKick(i, 0.5 * (base / (1 << m_rung[static_cast<std::size_t>(i)])));
        }

        CaptureAOld();
        return base;
    }

    void HalfKick(int i, double h) {
        const std::size_t ii = static_cast<std::size_t>(i);
        m_state.vx[ii] += h * m_state.ax[ii];
        m_state.vy[ii] += h * m_state.ay[ii];
        if constexpr (D == 3) m_state.vz[ii] += h * m_state.az[ii];
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
    std::vector<int> m_rung; // P15 block scheme: per-particle power-of-2 rung
    std::unordered_map<int, AccelFn<D>> m_aux;
    bool m_primed = false;
    mutable long m_forceEvals = 0;
    mutable long m_particleEvals = 0;
};

} // namespace ngrav
