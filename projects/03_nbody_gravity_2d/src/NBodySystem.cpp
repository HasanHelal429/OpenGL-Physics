#include "NBodySystem.hpp"

#include "ComplexFmmTree.hpp"
#include "Quadtree.hpp"

#include <cmath>

namespace nbody2d {

namespace {

void ComputeAccelDirect(const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                         double softening, std::vector<glm::dvec2>& accel) {
    const int n = static_cast<int>(pos.size());
    const double eps2 = softening * softening;
    accel.assign(static_cast<size_t>(n), glm::dvec2(0.0));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        glm::dvec2 acc(0.0);
        const glm::dvec2 pi = pos[static_cast<size_t>(i)];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const glm::dvec2 d = pos[static_cast<size_t>(j)] - pi;
            const double dist2 = glm::dot(d, d) + eps2;
            acc += G * mass[static_cast<size_t>(j)] * d / dist2; // 2D force: ~1/r, so d/r^2 (not 3D's d/r^3)
        }
        accel[static_cast<size_t>(i)] = acc;
    }
}

} // namespace

void ComputeAccel(SolverType solver, const std::vector<glm::dvec2>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec2>& accelOut) {
    switch (solver) {
        case SolverType::Direct: ComputeAccelDirect(pos, mass, G, softening, accelOut); return;
        case SolverType::BarnesHut: ComputeAccelBarnesHut(pos, mass, G, softening, theta, accelOut); return;
        case SolverType::ComplexFmm: ComputeAccelComplexFmm(pos, mass, G, softening, theta, accelOut); return;
    }
}

void NBodySystem::SetParticles(std::vector<glm::dvec2> positions, std::vector<glm::dvec2> velocities,
                                std::vector<double> masses) {
    m_pos = std::move(positions);
    m_vel = std::move(velocities);
    m_mass = std::move(masses);
    m_accel.assign(m_pos.size(), glm::dvec2(0.0));
}

void NBodySystem::RecomputeAccel(double G, double softening, SolverType solver, double theta) {
    ComputeAccel(solver, m_pos, m_mass, G, softening, theta, m_accel);
}

void NBodySystem::PrimeAccelerations(double G, double softening, SolverType solver, double theta) {
    RecomputeAccel(G, softening, solver, theta);
}

void NBodySystem::Step(double dt, double G, double softening, SolverType solver, double theta) {
    const size_t n = m_pos.size();
    if (n == 0) return;

    for (size_t i = 0; i < n; ++i) m_vel[i] += 0.5 * dt * m_accel[i];
    for (size_t i = 0; i < n; ++i) m_pos[i] += dt * m_vel[i];

    RecomputeAccel(G, softening, solver, theta);

    for (size_t i = 0; i < n; ++i) m_vel[i] += 0.5 * dt * m_accel[i];
}

double NBodySystem::TotalEnergy(double G, double softening) const {
    const long n = static_cast<long>(m_pos.size());

    double kinetic = 0.0;
    for (long i = 0; i < n; ++i) {
        const glm::dvec2& v = m_vel[static_cast<size_t>(i)];
        kinetic += 0.5 * m_mass[static_cast<size_t>(i)] * glm::dot(v, v);
    }

    // 2D potential energy: pairwise -G*m_i*m_j*ln(r) (the ln(r) fundamental
    // solution, not 3D's -1/r), softened the same way as the force law.
    const double eps2 = softening * softening;
    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 32)
    for (long i = 0; i < n; ++i) {
        double local = 0.0;
        for (long j = i + 1; j < n; ++j) {
            const glm::dvec2 d = m_pos[static_cast<size_t>(j)] - m_pos[static_cast<size_t>(i)];
            const double dist = std::sqrt(glm::dot(d, d) + eps2);
            local += G * m_mass[static_cast<size_t>(i)] * m_mass[static_cast<size_t>(j)] * std::log(dist);
        }
        potential += local;
    }

    return kinetic + potential;
}

glm::dvec2 NBodySystem::CenterOfMass() const {
    double totalMass = 0.0;
    glm::dvec2 com(0.0);
    for (size_t i = 0; i < m_pos.size(); ++i) {
        com += m_mass[i] * m_pos[i];
        totalMass += m_mass[i];
    }
    return totalMass > 0.0 ? com / totalMass : com;
}

double NBodySystem::AngularMomentum() const {
    const glm::dvec2 com = CenterOfMass();
    double L = 0.0;
    for (size_t i = 0; i < m_pos.size(); ++i) {
        const glm::dvec2 r = m_pos[i] - com;
        L += m_mass[i] * (r.x * m_vel[i].y - r.y * m_vel[i].x); // z-component of r x v
    }
    return L;
}

} // namespace nbody2d
