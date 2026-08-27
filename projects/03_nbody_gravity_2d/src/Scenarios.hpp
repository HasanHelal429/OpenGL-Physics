#pragma once

#include <glm/glm.hpp>
#include <vector>

namespace nbody2d {

enum class ScenarioType { TwoBodyKepler, LagrangeTriangle, Cluster, RotatingDisk };

struct ScenarioParams {
    int n = 2000;                      // Cluster / RotatingDisk particle count
    double diskRotationFraction = 0.7; // f: 0 = cold collapse .. 1 = full circular support (RotatingDisk only)
    unsigned seed = 1;
};

struct ScenarioResult {
    std::vector<glm::dvec2> pos;
    std::vector<glm::dvec2> vel;
    std::vector<double> mass;

    double G = 1.0;
    double softening = 0.05;
    double suggestedDt = 1e-3;
    double suggestedTheta = 0.5;
    float cameraExtent = 2.0f; // half-width of the initial view, world units
};

const char* ScenarioName(ScenarioType type);
ScenarioResult BuildScenario(ScenarioType type, const ScenarioParams& params);

} // namespace nbody2d
