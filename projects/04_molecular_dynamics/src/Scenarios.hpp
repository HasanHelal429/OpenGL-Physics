#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace md {

// Illustrative starting points on the reduced-unit LJ phase diagram (not
// precision phase-boundary values -- just density/temperature pairs picked
// to visibly land in the gas, liquid, and FCC-solid regions so the preset
// dropdown gives a one-click "what does each phase look like" demo).
enum class Preset { Gas, Liquid, Solid, Custom };

const char* PresetName(Preset preset);
// Returns false (density/temperature left untouched) for Preset::Custom.
bool PresetValues(Preset preset, double& outDensity, double& outTemperature);

struct ScenarioResult {
    std::vector<glm::dvec3> pos;
    std::vector<glm::dvec3> vel;
    double boxLength = 1.0;
    // Empty for a single-species scenario (MDSystem::SetParticles pads an
    // empty vector to all-species-A) -- only populated by
    // BuildFccLatticeBinary below.
    std::vector<int> species;
};

// Places particles on an FCC lattice (4-atom conventional cell) sized to
// hit `density` (= N/V in reduced units) as closely as an integer number of
// unit cells allows -- the actual particle count (4 * cellsPerAxis^3) is
// usually not exactly `targetN`, and is what the caller should treat as the
// live N from here on. Velocities are drawn per-component from a
// Maxwell-Boltzmann distribution at `temperature`, then the center-of-mass
// velocity is subtracted (zero net momentum, matching MDSystem::Temperature
// 's degrees-of-freedom convention) and the whole set rescaled so the
// sampled instantaneous temperature lands exactly on `temperature` rather
// than merely near it.
//
// Starting every density/temperature combination from the same ordered
// lattice (rather than only using it for the "Solid" preset) is standard
// MD practice: a lattice start never has two atoms accidentally overlapping
// (which an unstructured random placement risks, and which the repulsive
// LJ core would turn into an force blow-up on the very first step) and, at
// the temperatures/densities in the Gas or Liquid presets, the LJ dynamics
// itself melts the order away within the first few hundred steps.
ScenarioResult BuildFccLattice(int targetN, double density, double temperature, unsigned seed);

// Same FCC lattice placement as BuildFccLattice, but each site is randomly
// labeled species A or B (independent per-site coin flip at `fractionA`,
// not a fixed exact count -- fine for the few-hundred/thousand-particle
// systems this project runs, where the sampling noise around the target
// fraction is a small effect). Velocities are drawn the same way regardless
// of species since a real binary mixture normally used with this (e.g.
// Kob-Andersen) has equal mass for both species, so equipartition gives
// both species the same velocity distribution at a given temperature.
ScenarioResult BuildFccLatticeBinary(int targetN, double density, double temperature, double fractionA,
                                      unsigned seed);

} // namespace md
