#pragma once

#include "AdaptiveFmm.hpp"
#include "Octree.hpp"
#include "Scenarios.hpp"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

namespace nbody {

// One particle count's result: average wall-clock cost of Direct (when
// affordable), Barnes-Hut, and adaptive FMM at that N, plus each solver's
// internal phase breakdown -- the raw data a scaling study needs to fit a
// power-law exponent and see which phase actually drives it.
struct ScalingSweepPoint {
    int n = 0;
    double directMs = -1.0; // -1 = skipped, N too large for an O(N^2) reference to be affordable
    double bhMs = 0.0;
    double fmmMs = 0.0;
    BarnesHutStats bhStats;
    FmmStats fmmStats;
};

// Runs a full N-sweep (see ScalingSweepWorker.cpp for the exact N list) on a
// background thread, publishing each point as it completes so the UI can
// show a live-filling table instead of freezing for the sweep's full
// duration (tens of seconds at the largest N) -- the same background-
// worker-plus-poll pattern as DiagnosticsWorker, for the same reason.
class ScalingSweepWorker {
public:
    ~ScalingSweepWorker();

    // No-ops if a run is already in flight.
    void Start(ScenarioType scenarioType, double diskRotationFraction, unsigned seed);

    bool Busy() const { return m_busy.load(std::memory_order_acquire); }

    // Cheap-ish (copies a small vector of small structs); call once/frame
    // while the results window is open.
    std::vector<ScalingSweepPoint> LatestPoints() const;

private:
    std::thread m_thread;
    std::atomic<bool> m_busy{false};

    mutable std::mutex m_mutex;
    std::vector<ScalingSweepPoint> m_points;
};

} // namespace nbody
