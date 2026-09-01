#include "Scenarios.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace md {

const char* PresetName(Preset preset) {
    switch (preset) {
        case Preset::Gas: return "Gas";
        case Preset::Liquid: return "Liquid";
        case Preset::Solid: return "Solid (FCC)";
        case Preset::Custom: return "Custom";
    }
    return "Custom";
}

bool PresetValues(Preset preset, double& outDensity, double& outTemperature) {
    switch (preset) {
        case Preset::Gas: outDensity = 0.05; outTemperature = 2.0; return true;
        case Preset::Liquid: outDensity = 0.70; outTemperature = 1.00; return true;
        case Preset::Solid: outDensity = 1.00; outTemperature = 0.30; return true;
        case Preset::Custom: return false;
    }
    return false;
}

ScenarioResult BuildFccLattice(int targetN, double density, double temperature, unsigned seed) {
    ScenarioResult result;

    const int cellsPerAxis = std::max(1, static_cast<int>(std::round(std::cbrt(std::max(targetN, 1) / 4.0))));
    const int n = 4 * cellsPerAxis * cellsPerAxis * cellsPerAxis;

    const double volume = n / std::max(density, 1e-6);
    const double L = std::cbrt(volume);
    const double a = L / cellsPerAxis; // conventional-cell edge length

    static const glm::dvec3 kBasis[4] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5},
    };

    result.pos.reserve(static_cast<size_t>(n));
    for (int ix = 0; ix < cellsPerAxis; ++ix) {
        for (int iy = 0; iy < cellsPerAxis; ++iy) {
            for (int iz = 0; iz < cellsPerAxis; ++iz) {
                for (const glm::dvec3& b : kBasis) {
                    result.pos.emplace_back(a * (ix + b.x), a * (iy + b.y), a * (iz + b.z));
                }
            }
        }
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, std::sqrt(std::max(temperature, 0.0)));
    result.vel.resize(static_cast<size_t>(n));
    for (glm::dvec3& v : result.vel) v = glm::dvec3(gauss(rng), gauss(rng), gauss(rng));

    glm::dvec3 com(0.0);
    for (const glm::dvec3& v : result.vel) com += v;
    com /= static_cast<double>(n);
    for (glm::dvec3& v : result.vel) v -= com;

    // Rescale so the sampled instantaneous temperature lands exactly on the
    // target rather than merely near it (finite-N MB sampling has O(1/sqrt(N))
    // scatter around the mean otherwise). Matches MDSystem::Temperature's
    // 3*(N-1)-dof convention (COM already zeroed above).
    if (n >= 2 && temperature > 0.0) {
        double ke = 0.0;
        for (const glm::dvec3& v : result.vel) ke += 0.5 * glm::dot(v, v);
        const double tSampled = 2.0 * ke / (3.0 * (n - 1));
        if (tSampled > 1e-12) {
            const double lambda = std::sqrt(temperature / tSampled);
            for (glm::dvec3& v : result.vel) v *= lambda;
        }
    }

    result.boxLength = L;
    return result;
}

ScenarioResult BuildFccLatticeBinary(int targetN, double density, double temperature, double fractionA,
                                      unsigned seed) {
    ScenarioResult result;

    const int cellsPerAxis = std::max(1, static_cast<int>(std::round(std::cbrt(std::max(targetN, 1) / 4.0))));
    const int n = 4 * cellsPerAxis * cellsPerAxis * cellsPerAxis;

    const double volume = n / std::max(density, 1e-6);
    const double L = std::cbrt(volume);
    const double a = L / cellsPerAxis;

    static const glm::dvec3 kBasis[4] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5},
    };

    result.pos.reserve(static_cast<size_t>(n));
    for (int ix = 0; ix < cellsPerAxis; ++ix) {
        for (int iy = 0; iy < cellsPerAxis; ++iy) {
            for (int iz = 0; iz < cellsPerAxis; ++iz) {
                for (const glm::dvec3& b : kBasis) {
                    result.pos.emplace_back(a * (ix + b.x), a * (iy + b.y), a * (iz + b.z));
                }
            }
        }
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, std::sqrt(std::max(temperature, 0.0)));
    result.vel.resize(static_cast<size_t>(n));
    for (glm::dvec3& v : result.vel) v = glm::dvec3(gauss(rng), gauss(rng), gauss(rng));

    glm::dvec3 com(0.0);
    for (const glm::dvec3& v : result.vel) com += v;
    com /= static_cast<double>(n);
    for (glm::dvec3& v : result.vel) v -= com;

    if (n >= 2 && temperature > 0.0) {
        double ke = 0.0;
        for (const glm::dvec3& v : result.vel) ke += 0.5 * glm::dot(v, v);
        const double tSampled = 2.0 * ke / (3.0 * (n - 1));
        if (tSampled > 1e-12) {
            const double lambda = std::sqrt(temperature / tSampled);
            for (glm::dvec3& v : result.vel) v *= lambda;
        }
    }

    // Independent per-site coin flip (species A with probability fractionA)
    // -- same seeded RNG stream as the velocities, so the whole IC stays
    // deterministic given `seed`.
    std::uniform_real_distribution<double> coin(0.0, 1.0);
    result.species.resize(static_cast<size_t>(n));
    for (int& s : result.species) s = (coin(rng) < fractionA) ? 0 : 1;

    result.boxLength = L;
    return result;
}

ScenarioResult BuildSlab(int targetN, double density, double temperature, double slabFraction, unsigned seed) {
    ScenarioResult result;

    slabFraction = std::clamp(slabFraction, 0.05, 0.95);

    // Same FCC math as BuildFccLattice (4 atoms/conventional cell, density
    // = 4/a^3), but with independent cell counts along z (the slab's
    // thickness) vs. x/y (the box's full width): N = 4*cellsXY^2*cellsZ,
    // chosen so cellsZ/cellsXY ~= slabFraction -- the slab's z-extent ends
    // up slabFraction of the box's x/y extent, which becomes the cubic
    // box's overall side length (the slab spans x/y fully).
    const int cellsXY = std::max(
        1, static_cast<int>(std::round(std::cbrt(std::max(targetN, 1) / (4.0 * slabFraction)))));
    const int cellsZ = std::max(1, static_cast<int>(std::round(slabFraction * cellsXY)));
    const int n = 4 * cellsXY * cellsXY * cellsZ;

    const double a = std::cbrt(4.0 / std::max(density, 1e-6)); // conventional-cell edge at the slab's bulk density
    const double L = cellsXY * a;      // cubic box side (slab spans this fully in x and y)
    const double slabZ1 = cellsZ * a;  // < L: [slabZ1, L) is vacuum, no particles placed there

    static const glm::dvec3 kBasis[4] = {
        {0.0, 0.0, 0.0}, {0.5, 0.5, 0.0}, {0.5, 0.0, 0.5}, {0.0, 0.5, 0.5},
    };

    result.pos.reserve(static_cast<size_t>(n));
    for (int ix = 0; ix < cellsXY; ++ix) {
        for (int iy = 0; iy < cellsXY; ++iy) {
            for (int iz = 0; iz < cellsZ; ++iz) {
                for (const glm::dvec3& b : kBasis) {
                    result.pos.emplace_back(a * (ix + b.x), a * (iy + b.y), a * (iz + b.z));
                }
            }
        }
    }

    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, std::sqrt(std::max(temperature, 0.0)));
    result.vel.resize(static_cast<size_t>(n));
    for (glm::dvec3& v : result.vel) v = glm::dvec3(gauss(rng), gauss(rng), gauss(rng));

    glm::dvec3 com(0.0);
    for (const glm::dvec3& v : result.vel) com += v;
    com /= static_cast<double>(n);
    for (glm::dvec3& v : result.vel) v -= com;

    if (n >= 2 && temperature > 0.0) {
        double ke = 0.0;
        for (const glm::dvec3& v : result.vel) ke += 0.5 * glm::dot(v, v);
        const double tSampled = 2.0 * ke / (3.0 * (n - 1));
        if (tSampled > 1e-12) {
            const double lambda = std::sqrt(temperature / tSampled);
            for (glm::dvec3& v : result.vel) v *= lambda;
        }
    }

    result.boxLength = L;
    result.slabZ1 = slabZ1;
    return result;
}

} // namespace md
