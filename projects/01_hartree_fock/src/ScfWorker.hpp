#pragma once

#include "ScfSolver.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace hf {

// Runs RunScf on a background thread and publishes a thread-safe "latest
// snapshot" for a render thread to poll each frame, so the SCF loop's
// progress can be visualized live without blocking the UI.
class ScfWorker {
public:
    enum class State { Idle, Running, Converged, NotConverged, Cancelled, Failed };

    ScfWorker() = default;
    ~ScfWorker();

    ScfWorker(const ScfWorker&) = delete;
    ScfWorker& operator=(const ScfWorker&) = delete;

    // Cancels any in-flight run (blocking join) and starts a new one.
    void Start(const ScfParams& params);
    // Requests cancellation without blocking; does not join.
    void RequestCancel();

    State GetState() const { return m_state.load(std::memory_order_acquire); }
    std::string StatusMessage() const;

    // Cheap: just copies a shared_ptr under a lock. Null if no iteration
    // has completed yet.
    std::shared_ptr<const ScfSnapshot> LatestSnapshot() const;

    // Only meaningful once GetState() is a terminal state (Converged/
    // NotConverged/Cancelled/Failed).
    const ScfResult* Result() const;

private:
    std::jthread m_thread;
    mutable std::mutex m_snapshotMutex;
    std::shared_ptr<const ScfSnapshot> m_latestSnapshot;
    std::atomic<State> m_state{State::Idle};
    mutable std::mutex m_statusMutex;
    std::string m_statusMessage;
    std::optional<ScfResult> m_result;
};

} // namespace hf
