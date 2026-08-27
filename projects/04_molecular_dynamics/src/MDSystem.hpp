#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace md {

// Reduced Lennard-Jones units throughout (mass = epsilon = sigma = kB = 1),
// the standard convention for a demo LJ fluid -- lets density, temperature,
// and pressure be quoted as the dimensionless rho*, T*, P* used across the
// LJ phase-diagram literature, with no unit conversion anywhere in the code.

enum class Thermostat { None, Berendsen, VelocityRescale };

struct MDParams {
    double cutoff = 2.5;             // LJ force cutoff, in sigma
    double skin = 0.3;                // Verlet-list skin beyond cutoff, in sigma
    int neighborRebuildEvery = 5;     // physics steps between neighbor-list rebuilds
    double dt = 0.002;

    Thermostat thermostat = Thermostat::Berendsen;
    double targetT = 1.0;
    double berendsenTau = 1.0;        // Berendsen relaxation time, in time units
    int rescaleEvery = 20;            // VelocityRescale: steps between hard rescales
};

// A periodic cubic box of N Lennard-Jones particles integrated with
// velocity-Verlet. Force evaluation uses a linked-cell neighbor list (O(N)
// at fixed density, not O(N^2)) with a Verlet skin so the list only needs
// rebuilding every few steps; the LJ force is shifted-force-truncated at
// the cutoff so energy stays conserved (an unshifted hard cutoff makes the
// force jump discontinuously at the boundary, which shows up as steady
// energy drift -- see Physics Simulations/Thermodynamics/Van_der_Waals_Gas
// /vdw_gas.py's lj_force docstring for the empirical numbers that motivated
// this, which this port carries over unchanged).
class MDSystem {
public:
    // boxLength: cubic periodic box side, in sigma.
    void SetParticles(std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel, double boxLength);

    // Must be called once after SetParticles (and again if `params` changes
    // materially while paused) so the first velocity-Verlet half-step has a
    // valid acceleration/energy snapshot to use.
    void PrimeForces(const MDParams& params);

    void Step(const MDParams& params);

    // All O(N) byproducts of the (already O(N)) force evaluation inside
    // Step -- no separate O(N^2) diagnostic pass needed, unlike an
    // unstructured N-body force law.
    double KineticEnergy() const;
    double PotentialEnergy() const { return m_potentialEnergy; }
    double TotalEnergy() const { return KineticEnergy() + m_potentialEnergy; }
    // Instantaneous temperature from equipartition, T = 2*KE / (3*(N-1)):
    // N-1 because particles are initialized with zero net momentum and LJ's
    // internal forces conserve it exactly, removing 3 translational degrees
    // of freedom from the N*3 total.
    double Temperature() const;
    // Virial pressure: P = rho*T + virial / (3*V), virial = sum_pairs r_ij . F_ij.
    double Pressure() const;

    // Radial distribution function g(r) over nBins bins spanning [0, rMax).
    // O(N) via the same linked-cell approach as the force neighbor list
    // (rMax is typically larger than the force cutoff, so this rebuilds its
    // own candidate list rather than reusing the live one).
    std::vector<double> ComputeRDF(int nBins, double rMax) const;

    size_t Count() const { return m_pos.size(); }
    double BoxLength() const { return m_L; }
    const std::vector<glm::dvec3>& Positions() const { return m_pos; }
    const std::vector<glm::dvec3>& Velocities() const { return m_vel; }

private:
    void RebuildNeighborList(double listCutoff);
    void ComputeForces(double cutoff); // fills m_accel, m_potentialEnergy, m_virial
    void ApplyThermostat(const MDParams& params);

    std::vector<glm::dvec3> m_pos, m_vel, m_accel;
    double m_L = 1.0;

    std::vector<std::pair<int, int>> m_neighborPairs; // candidates within cutoff+skin, rebuilt periodically
    int m_stepsSinceRebuild = 0;
    int m_stepsSinceRescale = 0;

    double m_potentialEnergy = 0.0;
    double m_virial = 0.0;
};

} // namespace md
