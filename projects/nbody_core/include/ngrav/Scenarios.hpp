#pragma once

#include "Softening.hpp"
#include "State.hpp"
#include "Vec.hpp"

#include <cstdint>
#include <string>

namespace ngrav {

enum class ScenarioType {
    TwoBodyKepler,
    LagrangeTriangle,
    Cluster,        // roughly virialised uniform sphere/disk
    RotatingDisk,   // partial rotational support -> collapse
    PlummerSphere,  // isotropic Plummer equilibrium (P2)
    ColdCollapse,   // Plummer positions, zero velocity
};

struct ScenarioParams {
    int n = 2000;
    double diskRotationFraction = 0.7;
    std::uint32_t seed = 1;
};

template <int D>
struct Scenario {
    SoA<D> state;
    double G = 1.0;
    Softening soft = Softening::Plummer(0.05);
    double suggestedDt = 1e-3;
    double suggestedTheta = 0.5;
    double cameraScale = 3.5; // 3D: orbit distance; 2D: half-height
};

const char* ScenarioName(ScenarioType t);
bool ScenarioNeedsN(ScenarioType t);
ScenarioType ScenarioFromString(const std::string& s, ScenarioType fallback);

template <int D>
Scenario<D> BuildScenario(ScenarioType type, const ScenarioParams& params);

} // namespace ngrav
