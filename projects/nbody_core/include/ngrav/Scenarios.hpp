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
    HernquistSphere, // Hernquist (1990) profile, Jeans velocities (P13; 3D only, 2D falls back to Plummer)
    KingSphere,      // King (1966) lowered-isothermal sphere (P13; 3D only)
    DiskGalaxy,      // exponential disk + Hernquist bulge + NFW halo (P13; 3D only)
    IcFile,          // raw float32 particle dump (06_tidal_disruption's format; P13)
};

struct ScenarioParams {
    int n = 2000;
    double diskRotationFraction = 0.7;
    double eccentricity = 0.36; // TwoBodyKepler (3D only); 0.36 == the original fixed orbit
    double w0 = 6.0;            // KingSphere central dimensionless potential
    std::string icFile;         // IcFile scenario: path to the raw dump
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
