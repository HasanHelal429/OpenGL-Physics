#include "DiagnosticsWorker.hpp"

#include <cmath>

namespace nbody {

DiagnosticsWorker::~DiagnosticsWorker() {
    if (m_thread.joinable()) m_thread.join();
}

void DiagnosticsWorker::RequestCompute(double simTime, std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel,
                                        std::vector<double> mass, double G, double softening) {
    if (m_busy.load(std::memory_order_acquire)) return; // previous run still in flight -- drop this tick

    if (m_thread.joinable()) m_thread.join(); // previous run finished; reap it before starting another
    m_busy.store(true, std::memory_order_release);

    m_thread = std::thread([this, simTime, pos = std::move(pos), vel = std::move(vel), mass = std::move(mass), G,
                             softening]() mutable {
        const long n = static_cast<long>(pos.size());

        double kinetic = 0.0;
        for (long i = 0; i < n; ++i) {
            const glm::dvec3& v = vel[static_cast<size_t>(i)];
            kinetic += 0.5 * mass[static_cast<size_t>(i)] * glm::dot(v, v);
        }

        const double eps2 = softening * softening;
        double potential = 0.0;
#pragma omp parallel for reduction(+ : potential) schedule(dynamic, 32)
        for (long i = 0; i < n; ++i) {
            double local = 0.0;
            for (long j = i + 1; j < n; ++j) {
                const glm::dvec3 d = pos[static_cast<size_t>(j)] - pos[static_cast<size_t>(i)];
                const double dist = std::sqrt(glm::dot(d, d) + eps2);
                local -= G * mass[static_cast<size_t>(i)] * mass[static_cast<size_t>(j)] / dist;
            }
            potential += local;
        }

        glm::dvec3 com(0.0);
        double totalMass = 0.0;
        for (long i = 0; i < n; ++i) {
            com += mass[static_cast<size_t>(i)] * pos[static_cast<size_t>(i)];
            totalMass += mass[static_cast<size_t>(i)];
        }
        if (totalMass > 0.0) com /= totalMass;

        glm::dvec3 L(0.0);
        for (long i = 0; i < n; ++i) {
            L += mass[static_cast<size_t>(i)] * glm::cross(pos[static_cast<size_t>(i)] - com, vel[static_cast<size_t>(i)]);
        }

        {
            std::lock_guard<std::mutex> lock(m_resultMutex);
            m_result = Result{simTime, kinetic + potential, L};
            m_hasResult = true;
        }
        m_busy.store(false, std::memory_order_release);
    });
}

bool DiagnosticsWorker::PollResult(Result& result) {
    std::lock_guard<std::mutex> lock(m_resultMutex);
    if (!m_hasResult) return false;
    result = m_result;
    m_hasResult = false;
    return true;
}

} // namespace nbody
