#include "ScalingSweepWorker.hpp"

#include "NBodySystem.hpp" // ComputeAccel (Direct dispatch)

#include <chrono>

namespace nbody {

namespace {

// Geometric-ish progression spanning the app's particle-count slider range
// (10..60000): dense enough at the low end to see Direct's O(N^2) curve
// clearly, and reaching the slider's actual maximum at the high end where
// Barnes-Hut/FMM's asymptotic advantage should be most visible.
constexpr int kSweepNs[] = {500, 1000, 2000, 4000, 8000, 16000, 32000, 60000};

// Repeated per N to average out scheduling/measurement noise (see the
// wide run-to-run variance this project's own profiling has hit before);
// kept modest since this whole sweep already runs many N values back to
// back on a single background thread.
constexpr int kReps = 3;

// Above this, an O(N^2) Direct reference call is both too slow to include
// in an interactive sweep and not something you'd run at this N anyway.
constexpr int kDirectMaxN = 6000;

} // namespace

ScalingSweepWorker::~ScalingSweepWorker() {
    if (m_thread.joinable()) m_thread.join();
}

void ScalingSweepWorker::Start(ScenarioType scenarioType, double diskRotationFraction, unsigned seed) {
    if (m_busy.load(std::memory_order_acquire)) return;
    if (m_thread.joinable()) m_thread.join();

    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_points.clear();
    }
    m_busy.store(true, std::memory_order_release);

    m_thread = std::thread([this, scenarioType, diskRotationFraction, seed]() {
        for (int n : kSweepNs) {
            ScenarioParams params;
            params.n = n;
            params.diskRotationFraction = diskRotationFraction;
            params.seed = seed;
            const ScenarioResult scenario = BuildScenario(scenarioType, params);

            ScalingSweepPoint point;
            point.n = n;
            std::vector<glm::dvec3> accel;

            if (n <= kDirectMaxN) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < kReps; ++r) {
                    ComputeAccel(SolverType::Direct, scenario.pos, scenario.mass, scenario.G, scenario.softening,
                                 scenario.suggestedTheta, accel);
                }
                const auto t1 = std::chrono::steady_clock::now();
                point.directMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
            }

            {
                BarnesHutStats stats;
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < kReps; ++r) {
                    ComputeAccelBarnesHut(scenario.pos, scenario.mass, scenario.G, scenario.softening,
                                          scenario.suggestedTheta, accel, &stats);
                }
                const auto t1 = std::chrono::steady_clock::now();
                point.bhMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
                point.bhStats = stats;
            }

            {
                ngrav::SoA<3> in;
                in.Resize(scenario.pos.size());
                for (std::size_t i = 0; i < scenario.pos.size(); ++i) {
                    in.SetPos(i, scenario.pos[i]);
                    in.m[i] = scenario.mass[i];
                }
                ngrav::StepParams sp;
                sp.G = scenario.G;
                sp.soft = ngrav::Softening::Plummer(scenario.softening);
                sp.mac.theta = scenario.suggestedTheta;
                ngrav::SoA<3> out;
                ngrav::MutualFmmStats stats;
                const ngrav::PosMassView<3> v = ngrav::ViewOf(in);
                const auto t0 = std::chrono::steady_clock::now();
                for (int r = 0; r < kReps; ++r) {
                    ngrav::ComputeAccelMutualFmm(v, sp, out, &stats);
                }
                const auto t1 = std::chrono::steady_clock::now();
                point.fmmMs = std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
                point.fmmStats = stats;
            }

            std::lock_guard<std::mutex> lock(m_mutex);
            m_points.push_back(point);
        }

        m_busy.store(false, std::memory_order_release);
    });
}

std::vector<ScalingSweepPoint> ScalingSweepWorker::LatestPoints() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_points;
}

} // namespace nbody
