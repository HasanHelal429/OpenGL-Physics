#include "NBodySystem.hpp"

#include "ComplexFmmTree.hpp"
#include "Quadtree.hpp"

namespace nbody2d {

namespace {

ngrav::Solver ToCore(SolverType s) {
    switch (s) {
        case SolverType::Direct: return ngrav::Solver::Direct;
        case SolverType::BarnesHut: return ngrav::Solver::BarnesHut;
        case SolverType::ComplexFmm: return ngrav::Solver::ComplexFmm;
    }
    return ngrav::Solver::BarnesHut;
}

std::vector<glm::dvec2> AoSPositions(const ngrav::SoA<2>& s) {
    std::vector<glm::dvec2> p(s.Count());
    for (std::size_t i = 0; i < s.Count(); ++i) p[i] = glm::dvec2(s.x[i], s.y[i]);
    return p;
}

void WriteAccel(const std::vector<glm::dvec2>& a, ngrav::SoA<2>& out) {
    out.ResizeAccel(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out.ax[i] = a[i].x;
        out.ay[i] = a[i].y;
    }
}

} // namespace

void ComputeAccel(SolverType solver, const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec2>& accelOut) {
    const std::size_t n = pos.size();
    if (solver == SolverType::ComplexFmm) {
        ComputeAccelComplexFmm(pos, mass, G, softening, theta, accelOut);
        return;
    }
    ngrav::SoA<2> in;
    in.Resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        in.x[i] = pos[i].x;
        in.y[i] = pos[i].y;
        in.m[i] = mass[i];
    }
    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(softening);
    sp.mac.theta = theta;
    ngrav::SoA<2> out;
    const ngrav::PosMassView<2> v = ngrav::ViewOf(in);
    if (solver == SolverType::Direct)
        ngrav::ComputeAccelDirect<2>(v, sp, out);
    else
        ngrav::ComputeAccelBarnesHut<2>(v, sp, out);
    accelOut.resize(n);
    for (std::size_t i = 0; i < n; ++i) accelOut[i] = glm::dvec2(out.ax[i], out.ay[i]);
}

void RegisterFmmAdaptersOn(ngrav::System<2>& sys) {
    sys.SetAuxSolver(ngrav::Solver::ComplexFmm,
                     [](const ngrav::SoA<2>& in, const ngrav::StepParams& sp, ngrav::SoA<2>& out) {
                         const std::vector<glm::dvec2> pos = AoSPositions(in);
                         std::vector<glm::dvec2> a;
                         ComputeAccelComplexFmm(pos, in.m, sp.G, sp.soft.eps, sp.mac.theta, a);
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

void NBodySystem::SetParticles(std::vector<glm::dvec2> positions, std::vector<glm::dvec2> velocities,
                                std::vector<double> masses) {
    m_pos = std::move(positions);
    m_vel = std::move(velocities);
    m_mass = std::move(masses);

    ngrav::SoA<2> s;
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
    const ngrav::SoA<2>& s = m_core.State();
    m_pos.resize(s.Count());
    m_vel.resize(s.Count());
    for (std::size_t i = 0; i < s.Count(); ++i) {
        m_pos[i] = glm::dvec2(s.x[i], s.y[i]);
        m_vel[i] = glm::dvec2(s.vx[i], s.vy[i]);
    }
}

double NBodySystem::Step(double dt, double G, double softening, SolverType solver, double theta, bool adaptive,
                          double eta) {
    if (m_pos.empty()) return dt;
    ngrav::StepParams sp = MakeParams(G, softening, solver, theta);
    sp.dt = dt;
    sp.adaptive = adaptive;
    sp.eta = eta;
    const double taken = m_core.Step(sp);
    SyncMirrorFromCore();
    return taken;
}

double NBodySystem::TotalEnergy(double G, double softening) const {
    ngrav::StepParams sp;
    sp.G = G;
    sp.soft = ngrav::Softening::Plummer(softening);
    return m_core.TotalEnergy(sp);
}

double NBodySystem::AngularMomentum() const { return m_core.AngularMomentum(); }
glm::dvec2 NBodySystem::CenterOfMass() const { return m_core.CenterOfMass(); }

} // namespace nbody2d
