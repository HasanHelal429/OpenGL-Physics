#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody {

enum class ScenarioType { TwoBodyKepler, LagrangeTriangle, Cluster, RotatingDisk };

struct ScenarioParams {
    int n = 2000;                      // Cluster / RotatingDisk particle count
    double diskRotationFraction = 0.7; // f: 0 = cold collapse .. 1 = full circular support (RotatingDisk only)
    unsigned seed = 1;
};

struct ScenarioResult {
    std::vector<glm::dvec3> pos;
    std::vector<glm::dvec3> vel;
    std::vector<double> mass;

    double G = 1.0;
    double softening = 0.05;
    double suggestedDt = 1e-3;
    double suggestedTheta = 0.5;
    float cameraDistance = 4.0f;
};

const char* ScenarioName(ScenarioType type);
ScenarioResult BuildScenario(ScenarioType type, const ScenarioParams& params);

} // namespace nbody
