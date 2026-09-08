#include "NBodySystem.hpp"

#include "AdaptiveFmm.hpp"
#include "Octree.hpp"
#include "SphericalFmm.hpp"

#include <cmath>

namespace nbody {

namespace {

ngrav::Solver ToCore(SolverType s) {
    switch (s) {
        case SolverType::Direct: return ngrav::Solver::Direct;
        case SolverType::BarnesHut: return ngrav::Solver::BarnesHut;
        case SolverType::AdaptiveFmm: return ngrav::Solver::AdaptiveFmm;
        case SolverType::SphericalFmm: return ngrav::Solver::SphericalFmm;
    }
    return ngrav::Solver::BarnesHut;
}

// Build an AoS position array from an ngrav SoA view -- the two FMM solvers
// still take std::vector<glm::dvec3>. One alloc per FMM force eval; these are
// O(N log N)+ so it's noise. Removed in P6/P7 when they move into nbody_core.
std::vector<glm::dvec3> AoSPositions(const ngrav::SoA<3>& s) {
    std::vector<glm::dvec3> p(s.Count());
    for (std::size_t i = 0; i < s.Count(); ++i) p[i] = glm::dvec3(s.x[i], s.y[i], s.z[i]);
    return p;
}

void WriteAccel(const std::vector<glm::dvec3>& a, ngrav::SoA<3>& out) {
    out.ResizeAccel(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out.ax[i] = a[i].x;
        out.ay[i] = a[i].y;
        out.az[i] = a[i].z;
    }
}

} // namespace

void ComputeAccel(SolverType solver, const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec3>& accelOut) {
    const std::size_t n = pos.size();
    if (solver == SolverType::AdaptiveFmm) {
        ComputeAccelAdaptiveFmm(pos, mass, G, softening, theta, accelOut);
        return;
    }
    if (solver == SolverType::SphericalFmm) {
        ComputeAccelSphericalFmm(pos, mass, G, softening, theta, accelOut);
        return;
    }
    // Direct / Barnes-Hut via nbody_core.
    ngrav::SoA<3> in;
    in.Resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        in.x[i] = pos[i].x;
        in.y[i] = pos[i].y;
        in.z[i] = pos[i].z;
        in.m[i] = mass[i];
    }
    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(softening);
    sp.mac.theta = theta;
    ngrav::SoA<3> out;
    const ngrav::PosMassView<3> v = ngrav::ViewOf(in);
    if (solver == SolverType::Direct)
        ngrav::ComputeAccelDirect<3>(v, sp, out);
    else
        ngrav::ComputeAccelBarnesHut<3>(v, sp, out);
    accelOut.resize(n);
    for (std::size_t i = 0; i < n; ++i) accelOut[i] = glm::dvec3(out.ax[i], out.ay[i], out.az[i]);
}

void RegisterFmmAdaptersOn(ngrav::System<3>& sys) {
    sys.SetAuxSolver(ngrav::Solver::AdaptiveFmm,
                     [](const ngrav::SoA<3>& in, const ngrav::StepParams& sp, ngrav::SoA<3>& out) {
                         const std::vector<glm::dvec3> pos = AoSPositions(in);
                         std::vector<glm::dvec3> a;
                         ComputeAccelAdaptiveFmm(pos, in.m, sp.G, sp.soft.eps, sp.mac.theta, a);
                         WriteAccel(a, out);
                     });
    sys.SetAuxSolver(ngrav::Solver::SphericalFmm,
                     [](const ngrav::SoA<3>& in, const ngrav::StepParams& sp, ngrav::SoA<3>& out) {
                         const std::vector<glm::dvec3> pos = AoSPositions(in);
                         std::vector<glm::dvec3> a;
                         ComputeAccelSphericalFmm(pos, in.m, sp.G, sp.soft.eps, sp.mac.theta, a);
                         WriteAccel(a, out);
                     });
}

NBodySystem::NBodySystem() { RegisterFmmAdapters(); }

void NBodySystem::RegisterFmmAdapters() { RegisterFmmAdaptersOn(m_core); }

ngrav::StepParams NBodySystem::MakeParams(double G, double softening, SolverType solver, double theta) const {
    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(softening);
    sp.mac.theta = theta;
    sp.solver = ToCore(solver);
    return sp;
}

void NBodySystem::SetParticles(std::vector<glm::dvec3> positions, std::vector<glm::dvec3> velocities,
                                std::vector<double> masses) {
    m_pos = std::move(positions);
    m_vel = std::move(velocities);
    m_mass = std::move(masses);

    ngrav::SoA<3> s;
    s.Resize(m_pos.size());
    for (std::size_t i = 0; i < m_pos.size(); ++i) {
        s.SetPos(i, m_pos[i]);
        s.SetVel(i, m_vel[i]);
        s.m[i] = m_mass[i];
    }
    m_core.SetParticles(std::move(s));
}

void NBodySystem::PrimeAccelerations(double G, double softening, SolverType solver, double theta) {
    m_core.Prime(MakeParams(G, softening, solver, theta));
}

void NBodySystem::SyncMirrorFromCore() {
    const ngrav::SoA<3>& s = m_core.State();
    m_pos.resize(s.Count());
    m_vel.resize(s.Count());
    for (std::size_t i = 0; i < s.Count(); ++i) {
        m_pos[i] = glm::dvec3(s.x[i], s.y[i], s.z[i]);
        m_vel[i] = glm::dvec3(s.vx[i], s.vy[i], s.vz[i]);
    }
}

void NBodySystem::Step(double dt, double G, double softening, SolverType solver, double theta) {
    if (m_pos.empty()) return;
    ngrav::StepParams sp = MakeParams(G, softening, solver, theta);
    sp.dt = dt;
    m_core.Step(sp);
    SyncMirrorFromCore();
}

double NBodySystem::TotalEnergy(double G, double softening) const {
    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(softening);
    return m_core.TotalEnergy(sp);
}

glm::dvec3 NBodySystem::AngularMomentum() const { return m_core.AngularMomentum(); }
glm::dvec3 NBodySystem::CenterOfMass() const { return m_core.CenterOfMass(); }

} // namespace nbody
