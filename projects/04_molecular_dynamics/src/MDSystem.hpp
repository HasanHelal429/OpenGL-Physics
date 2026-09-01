#pragma once

#include <glm/glm.hpp>

#include <vector>

namespace md {

// Reduced Lennard-Jones units throughout (mass = epsilon = sigma = kB = 1),
// the standard convention for a demo LJ fluid -- lets density, temperature,
// and pressure be quoted as the dimensionless rho*, T*, P* used across the
// LJ phase-diagram literature, with no unit conversion anywhere in the code.

// NoseHoover is a deterministic extended-Lagrangian thermostat that samples
// the canonical ensemble correctly (Berendsen/VelocityRescale don't -- both
// suppress KE fluctuations rather than reproducing the true NVT variance).
// Berendsen still exists for fast, robust equilibration; Nose-Hoover is what
// production/fluctuation-sensitive runs should use once equilibrated.
enum class Thermostat { None, Berendsen, VelocityRescale, NoseHoover };

// Isotropic Berendsen pressure coupling: rescales the box (and every
// particle position with it) toward a target pressure. Weak coupling, not a
// rigorous NPT ensemble (no fluctuation-dissipation guarantee, same caveat
// as the Berendsen thermostat) -- good enough to trace out P-V-T state
// points without hand-picking box sizes.
enum class Barostat { None, Berendsen };

struct MDParams {
    double cutoff = 2.5;             // LJ force cutoff, in sigma
    double skin = 0.3;                // Verlet-list skin beyond cutoff, in sigma
    int neighborRebuildEvery = 5;     // physics steps between neighbor-list rebuilds
    double dt = 0.002;

    Thermostat thermostat = Thermostat::Berendsen;
    double targetT = 1.0;
    double berendsenTau = 1.0;        // Berendsen relaxation time, in time units
    int rescaleEvery = 20;            // VelocityRescale: steps between hard rescales
    double noseHooverTau = 1.0;       // NoseHoover relaxation time; thermostat "mass" Q = dof*targetT*tau^2

    Barostat barostat = Barostat::None;
    double targetP = 0.0;
    double berendsenTauP = 1.0;       // barostat relaxation time, in time units
    double compressibility = 1.0;     // isothermal compressibility estimate, reduced units (order-1 for an LJ liquid)
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
    // boxLength: cubic periodic box side, in sigma_AA. `species` (0=A, 1=B)
    // may be shorter than `pos` (padded with 0/species-A) or omitted
    // entirely -- an empty/short species vector means "everyone is species
    // A", which combined with the default sigma/epsilon table below (all
    // 1.0) reproduces the original single-species behavior exactly.
    void SetParticles(std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel, double boxLength,
                       std::vector<int> species = {});

    // Per-species-pair sigma/epsilon (species A vs. B); AA is always
    // (1.0, 1.0) -- the reference units everything else (including
    // `MDParams::cutoff`) is quoted in, matching the Kob-Andersen
    // convention of expressing lengths/energies in sigma_AA/epsilon_AA.
    // Defaults (all 1.0) make every species behave identically, so calling
    // this is only necessary for an actual mixture.
    void SetSpeciesLJParams(double sigmaBB, double epsilonBB, double sigmaAB, double epsilonAB);

    const std::vector<int>& Species() const { return m_species; }
    // Partial RDF over only (speciesA, speciesB) pairs (order doesn't
    // matter -- AB and BA are the same cross term), normalized by the
    // actual number of such pairs (not total N). Returns all-zero if
    // either species has zero members present. Same O(N) linked-cell
    // approach as ComputeRDF.
    std::vector<double> ComputePartialRDF(int nBins, double rMax, int speciesA, int speciesB) const;

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

    // Analytic mean-field corrections for everything the cutoff throws away
    // (assumes g(r)=1 beyond `cutoff`, the standard approximation -- see
    // Allen & Tildesley eq. 2.98/eq. 2.120). These do NOT feed back into the
    // dynamics (forces stay cutoff-only, so NVE energy conservation checks
    // must use the raw PotentialEnergy()/Pressure() above, not these) --
    // they only correct the *reported* thermodynamics for comparison
    // against literature equations of state, which are quoted for the full
    // (untruncated) LJ potential.
    double TailEnergyCorrection(double cutoff) const;
    double TailPressureCorrection(double cutoff) const;

    // Nose-Hoover friction variable (0 if that thermostat isn't active).
    double ThermostatXi() const { return m_xi; }
    // KE + PE + 0.5*Q*xi^2 + dof*targetT*integral(xi dt): the quantity the
    // Nose-Hoover extended Lagrangian actually conserves (drifts far less
    // than raw KE+PE, which the thermostat is deliberately pumping/damping).
    // Reduces to TotalEnergy() when Nose-Hoover isn't active (xi stays 0).
    double NoseHooverInvariant(const MDParams& params) const;

    // Radial distribution function g(r) over nBins bins spanning [0, rMax).
    // O(N) via the same linked-cell approach as the force neighbor list
    // (rMax is typically larger than the force cutoff, so this rebuilds its
    // own candidate list rather than reusing the live one).
    std::vector<double> ComputeRDF(int nBins, double rMax) const;

    size_t Count() const { return m_pos.size(); }
    double BoxLength() const { return m_L; }
    const std::vector<glm::dvec3>& Positions() const { return m_pos; }
    const std::vector<glm::dvec3>& Velocities() const { return m_vel; }
    // Per-particle acceleration (mass=1 reduced units, so this is also the
    // force) from the last ComputeForces() call -- exposed for --selftest's
    // independent-reference cross-check, not used by production code.
    const std::vector<glm::dvec3>& Accelerations() const { return m_accel; }

private:
    void RebuildNeighborList(double listCutoff);
    void ComputeForces(double cutoff); // fills m_accel, m_potentialEnergy, m_virial
    void ApplyThermostat(const MDParams& params);
    void ApplyBarostat(const MDParams& params);
    void StepVelocityVerlet(const MDParams& params); // None / Berendsen / VelocityRescale
    void StepNoseHoover(const MDParams& params);      // extended-Lagrangian leapfrog, see .cpp

    std::vector<glm::dvec3> m_pos, m_vel, m_accel;
    std::vector<int> m_species; // 0=A, 1=B; size() == m_pos.size()
    // m_sigma[i][j]/m_epsilon[i][j]: LJ parameters for a (species i, species
    // j) pair. Defaults reproduce plain single-species LJ regardless of
    // m_species' contents. Set via SetSpeciesLJParams.
    double m_sigma[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    double m_epsilon[2][2] = {{1.0, 1.0}, {1.0, 1.0}};
    double m_L = 1.0;

    std::vector<std::pair<int, int>> m_neighborPairs; // candidates within cutoff+skin, rebuilt periodically
    int m_stepsSinceRebuild = 0;
    int m_stepsSinceRescale = 0;

    double m_potentialEnergy = 0.0;
    double m_virial = 0.0;

    double m_xi = 0.0;         // Nose-Hoover friction variable
    double m_xiIntegral = 0.0; // running integral(xi dt), for NoseHooverInvariant
};

} // namespace md
