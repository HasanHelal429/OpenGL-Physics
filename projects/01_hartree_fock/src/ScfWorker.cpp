#include "ScfWorker.hpp"

namespace hf {

ScfWorker::~ScfWorker() {
    if (m_thread.joinable()) {
        m_thread.request_stop();
        m_thread.join();
    }
}

void ScfWorker::Start(const ScfParams& params) {
    if (m_thread.joinable()) {
        m_thread.request_stop();
        m_thread.join();
    }

    {
        std::lock_guard<std::mutex> lock(m_snapshotMutex);
        m_latestSnapshot.reset();
    }
    {
        std::lock_guard<std::mutex> lock(m_statusMutex);
        m_result.reset();
        m_statusMessage = "Running SCF loop...";
    }
    m_state.store(State::Running, std::memory_order_release);

    m_thread = std::jthread([this, params](std::stop_token stopToken) {
        auto onSnapshot = [this](const ScfSnapshot& snapshot) {
            auto shared = std::make_shared<ScfSnapshot>(snapshot);
            std::lock_guard<std::mutex> lock(m_snapshotMutex);
            m_latestSnapshot = std::move(shared);
        };

        try {
            ScfResult result = RunScf(params, onSnapshot, stopToken);
            const bool cancelled = result.cancelled;
            const bool converged = result.converged;
            {
                std::lock_guard<std::mutex> lock(m_statusMutex);
                m_result = std::move(result);
                m_statusMessage = cancelled ? "Cancelled" : (converged ? "Converged" : "Stopped: did not converge");
            }
            m_state.store(cancelled ? State::Cancelled : (converged ? State::Converged : State::NotConverged),
                          std::memory_order_release);
        } catch (const std::exception& ex) {
            {
                std::lock_guard<std::mutex> lock(m_statusMutex);
                m_statusMessage = std::string("Failed: ") + ex.what();
            }
            m_state.store(State::Failed, std::memory_order_release);
        }
    });
}

void ScfWorker::RequestCancel() {
    if (m_thread.joinable()) {
        m_thread.request_stop();
    }
}

std::string ScfWorker::StatusMessage() const {
    std::lock_guard<std::mutex> lock(m_statusMutex);
    return m_statusMessage;
}

std::shared_ptr<const ScfSnapshot> ScfWorker::LatestSnapshot() const {
    std::lock_guard<std::mutex> lock(m_snapshotMutex);
    return m_latestSnapshot;
}

const ScfResult* ScfWorker::Result() const {
    std::lock_guard<std::mutex> lock(m_statusMutex);
    return m_result ? &*m_result : nullptr;
}

} // namespace hf
