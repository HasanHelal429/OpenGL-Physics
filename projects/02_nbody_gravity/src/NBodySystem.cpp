#include "NBodySystem.hpp"

#include "AdaptiveFmm.hpp"
#include "Octree.hpp"

#include <cmath>

namespace nbody {

namespace {

void ComputeAccelDirect(const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                         double softening, std::vector<glm::dvec3>& accel) {
    const int n = static_cast<int>(pos.size());
    const double eps2 = softening * softening;
    accel.assign(static_cast<size_t>(n), glm::dvec3(0.0));

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        glm::dvec3 acc(0.0);
        const glm::dvec3 pi = pos[static_cast<size_t>(i)];
        for (int j = 0; j < n; ++j) {
            if (j == i) continue;
            const glm::dvec3 d = pos[static_cast<size_t>(j)] - pi;
            const double dist2 = glm::dot(d, d) + eps2;
            const double invDist = 1.0 / std::sqrt(dist2);
            const double invDist3 = invDist * invDist * invDist;
            acc += G * mass[static_cast<size_t>(j)] * invDist3 * d;
        }
        accel[static_cast<size_t>(i)] = acc;
    }
}

} // namespace

void ComputeAccel(SolverType solver, const std::vector<glm::dvec3>& pos, const std::vector<double>& mass, double G,
                   double softening, double theta, std::vector<glm::dvec3>& accelOut) {
    switch (solver) {
        case SolverType::Direct: ComputeAccelDirect(pos, mass, G, softening, accelOut); return;
        case SolverType::BarnesHut: ComputeAccelBarnesHut(pos, mass, G, softening, theta, accelOut); return;
        case SolverType::AdaptiveFmm: ComputeAccelAdaptiveFmm(pos, mass, G, softening, theta, accelOut); return;
    }
}

void NBodySystem::SetParticles(std::vector<glm::dvec3> positions, std::vector<glm::dvec3> velocities,
                                std::vector<double> masses) {
    m_pos = std::move(positions);
    m_vel = std::move(velocities);
    m_mass = std::move(masses);
    m_accel.assign(m_pos.size(), glm::dvec3(0.0));
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
        const glm::dvec3& v = m_vel[static_cast<size_t>(i)];
        kinetic += 0.5 * m_mass[static_cast<size_t>(i)] * glm::dot(v, v);
    }

    const double eps2 = softening * softening;
    double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 32)
    for (long i = 0; i < n; ++i) {
        double local = 0.0;
        for (long j = i + 1; j < n; ++j) {
            const glm::dvec3 d = m_pos[static_cast<size_t>(j)] - m_pos[static_cast<size_t>(i)];
            const double dist = std::sqrt(glm::dot(d, d) + eps2);
            local -= G * m_mass[static_cast<size_t>(i)] * m_mass[static_cast<size_t>(j)] / dist;
        }
        potential += local;
    }

    return kinetic + potential;
}

glm::dvec3 NBodySystem::CenterOfMass() const {
    double totalMass = 0.0;
    glm::dvec3 com(0.0);
    for (size_t i = 0; i < m_pos.size(); ++i) {
        com += m_mass[i] * m_pos[i];
        totalMass += m_mass[i];
    }
    return totalMass > 0.0 ? com / totalMass : com;
}

glm::dvec3 NBodySystem::AngularMomentum() const {
    const glm::dvec3 com = CenterOfMass();
    glm::dvec3 L(0.0);
    for (size_t i = 0; i < m_pos.size(); ++i) {
        L += m_mass[i] * glm::cross(m_pos[i] - com, m_vel[i]);
    }
    return L;
}

} // namespace nbody
