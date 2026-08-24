#pragma once

#include <glm/glm.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace nbody {

// Total energy is an O(N^2) diagnostic (angular momentum is O(N) and rides
// along essentially for free). Computing it synchronously in the physics
// loop is fine at a few thousand particles but starves both the solver and
// the renderer at N in the tens of thousands -- so it runs on a background
// thread instead, mirroring 01_hartree_fock's ScfWorker: the caller hands
// over a state snapshot, and polls for a finished result each tick.
// RequestCompute silently drops the request if a previous one is still in
// flight, so the caller can call it every tick without ever blocking or
// piling up work.
class DiagnosticsWorker {
public:
    ~DiagnosticsWorker();

    void RequestCompute(double simTime, std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel,
                         std::vector<double> mass, double G, double softening);

    bool Busy() const { return m_busy.load(std::memory_order_acquire); }

    struct Result {
        double simTime = 0.0;
        double energy = 0.0;
        glm::dvec3 angularMomentum{0.0};
    };
    // Returns true (and fills result) exactly once per completed computation.
    bool PollResult(Result& result);

private:
    std::thread m_thread;
    std::atomic<bool> m_busy{false};

    std::mutex m_resultMutex;
    Result m_result;
    bool m_hasResult = false;
};

} // namespace nbody
