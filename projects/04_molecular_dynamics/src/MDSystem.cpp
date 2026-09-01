#include "MDSystem.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <utility>

namespace md {

namespace {

constexpr double kSigma = 1.0;
constexpr double kEpsilon = 1.0;
constexpr double kPi = 3.14159265358979323846;

// Minimum-image displacement r_i - r_j, wrapped to the nearest periodic image.
inline glm::dvec3 MinImage(const glm::dvec3& d, double L) {
    glm::dvec3 out = d;
    out.x -= L * std::round(out.x / L);
    out.y -= L * std::round(out.y / L);
    out.z -= L * std::round(out.z / L);
    return out;
}

inline glm::dvec3 WrapPosition(glm::dvec3 r, double L) {
    r.x -= L * std::floor(r.x / L);
    r.y -= L * std::floor(r.y / L);
    r.z -= L * std::floor(r.z / L);
    return r;
}

// Candidate (i,j) index pairs (i<j) within `cutoff` of each other under
// periodic boundaries, found via a uniform linked-cell list: cell size =
// cutoff, so any two particles closer than cutoff must share a cell or one
// of its 26 neighbors -- turns the force/RDF pair search from O(N^2) to
// roughly O(N) at fixed density. Falls back to brute-force all-pairs
// (still filtered by `cutoff` via minimum image) when the box is too
// small to fit at least 3 cells per axis: below that, the periodic wrap of
// neighbor offsets (-1,0,+1) can collide onto the same cell, and that's
// exactly the tiny-N regime where brute force is cheap anyway.
std::vector<std::pair<int, int>> BuildCellListPairs(const std::vector<glm::dvec3>& pos, double L, double cutoff) {
    const int n = static_cast<int>(pos.size());
    std::vector<std::pair<int, int>> pairs;

    const int nc = static_cast<int>(std::floor(L / cutoff));
    if (nc < 3) {
        pairs.reserve(static_cast<size_t>(n) * 8);
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const glm::dvec3 d = MinImage(pos[static_cast<size_t>(i)] - pos[static_cast<size_t>(j)], L);
                if (glm::dot(d, d) <= cutoff * cutoff) pairs.emplace_back(i, j);
            }
        }
        return pairs;
    }

    const double cellSize = L / nc;
    auto cellIndex = [&](double coord) {
        int c = static_cast<int>(std::floor(coord / cellSize));
        c %= nc;
        if (c < 0) c += nc;
        return c;
    };

    std::vector<std::vector<int>> cells(static_cast<size_t>(nc) * static_cast<size_t>(nc) * static_cast<size_t>(nc));
    std::vector<glm::ivec3> cellOf(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const glm::dvec3 w = WrapPosition(pos[static_cast<size_t>(i)], L);
        const glm::ivec3 c(cellIndex(w.x), cellIndex(w.y), cellIndex(w.z));
        cellOf[static_cast<size_t>(i)] = c;
        cells[(static_cast<size_t>(c.x) * nc + static_cast<size_t>(c.y)) * nc + static_cast<size_t>(c.z)].push_back(i);
    }

    const double cutoff2 = cutoff * cutoff;
    pairs.reserve(static_cast<size_t>(n) * 20);
    for (int i = 0; i < n; ++i) {
        const glm::ivec3 c = cellOf[static_cast<size_t>(i)];
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    const int nx = (c.x + dx + nc) % nc;
                    const int ny = (c.y + dy + nc) % nc;
                    const int nz = (c.z + dz + nc) % nc;
                    const std::vector<int>& bucket = cells[(static_cast<size_t>(nx) * nc + static_cast<size_t>(ny)) * nc + static_cast<size_t>(nz)];
                    for (int j : bucket) {
                        if (j <= i) continue; // dedupe: each unordered pair kept exactly once, from i's own expansion
                        const glm::dvec3 d = MinImage(pos[static_cast<size_t>(i)] - pos[static_cast<size_t>(j)], L);
                        if (glm::dot(d, d) <= cutoff2) pairs.emplace_back(i, j);
                    }
                }
            }
        }
    }
    return pairs;
}

// LJ radial force magnitude F(r) = -dU/dr = 24*epsilon*(2*(sigma/r)^12 - (sigma/r)^6)/r.
// Positive (repulsive) for r < 2^(1/6)*sigma, negative (attractive) beyond.
inline double LjForceMagnitude(double r, double epsilon, double sigma) {
    const double sr6 = std::pow(sigma / r, 6);
    return 24.0 * epsilon * (2.0 * sr6 * sr6 - sr6) / r;
}

inline double LjPotential(double r, double epsilon, double sigma) {
    const double sr6 = std::pow(sigma / r, 6);
    return 4.0 * epsilon * (sr6 * sr6 - sr6);
}

} // namespace

void MDSystem::SetParticles(std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel, double boxLength,
                             std::vector<int> species) {
    m_pos = std::move(pos);
    m_vel = std::move(vel);
    m_L = boxLength;
    m_species = std::move(species);
    m_species.resize(m_pos.size(), 0); // short/empty -> pad with species A
    m_accel.assign(m_pos.size(), glm::dvec3(0.0));
    m_neighborPairs.clear();
    m_stepsSinceRebuild = 0;
    m_stepsSinceRescale = 0;
    m_potentialEnergy = 0.0;
    m_virial = 0.0;
    m_xi = 0.0;
    m_xiIntegral = 0.0;
}

void MDSystem::SetSpeciesLJParams(double sigmaBB, double epsilonBB, double sigmaAB, double epsilonAB) {
    m_sigma[1][1] = sigmaBB;
    m_epsilon[1][1] = epsilonBB;
    m_sigma[0][1] = m_sigma[1][0] = sigmaAB;
    m_epsilon[0][1] = m_epsilon[1][0] = epsilonAB;
    // m_sigma[0][0]/m_epsilon[0][0] stay (1.0, 1.0) -- the reference units.
}

void MDSystem::RebuildNeighborList(double listCutoff) {
    m_neighborPairs = BuildCellListPairs(m_pos, m_L, std::min(listCutoff, 0.49 * m_L));
    m_stepsSinceRebuild = 0;
}

void MDSystem::PrimeForces(const MDParams& params) {
    RebuildNeighborList(params.cutoff + params.skin);
    ComputeForces(std::min(params.cutoff, 0.49 * m_L));
}

void MDSystem::ComputeForces(double cutoff) {
    const int n = static_cast<int>(m_pos.size());
    m_accel.assign(static_cast<size_t>(n), glm::dvec3(0.0));
    if (n == 0) {
        m_potentialEnergy = 0.0;
        m_virial = 0.0;
        return;
    }

    const double cutoff2 = cutoff * cutoff;
    // Shifted-force cutoff: subtract F(cutoff)/U(cutoff) so force and
    // energy both go smoothly to zero exactly at r=cutoff instead of
    // jumping there. An unshifted hard cutoff isn't just less accurate --
    // it breaks energy conservation outright (the discontinuity injects/
    // removes energy every time a pair crosses the cutoff), which is why
    // this port keeps the same shift the Python vdw_gas.py prototype
    // needed (see its lj_force docstring for the measured drift numbers).
    //
    // One global cutoff distance (in sigma_AA units) for every species
    // pair, rather than the original Kob-Andersen convention of a separate
    // 2.5*sigma_alphabeta cutoff per pair type -- simpler (the linked-cell
    // neighbor search stays completely species-agnostic), and the
    // difference is only a modest change in how much attractive tail is
    // included for the smaller-sigma B-containing pairs, not a qualitative
    // behavior change. Fc/Uc precomputed once per (species i, species j)
    // pair here, outside the parallel loop.
    double Fc[2][2], Uc[2][2];
    for (int si = 0; si < 2; ++si) {
        for (int sj = 0; sj < 2; ++sj) {
            Fc[si][sj] = LjForceMagnitude(cutoff, m_epsilon[si][sj], m_sigma[si][sj]);
            Uc[si][sj] = LjPotential(cutoff, m_epsilon[si][sj], m_sigma[si][sj]);
        }
    }

    const int nThreads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<glm::dvec3>> threadAccel(
        static_cast<size_t>(nThreads), std::vector<glm::dvec3>(static_cast<size_t>(n), glm::dvec3(0.0)));

    double potentialSum = 0.0;
    double virialSum = 0.0;
    const int pairCount = static_cast<int>(m_neighborPairs.size());

#pragma omp parallel reduction(+ : potentialSum, virialSum)
    {
        const int tid = omp_get_thread_num();
        std::vector<glm::dvec3>& acc = threadAccel[static_cast<size_t>(tid)];
#pragma omp for schedule(dynamic, 128)
        for (int p = 0; p < pairCount; ++p) {
            const int i = m_neighborPairs[static_cast<size_t>(p)].first;
            const int j = m_neighborPairs[static_cast<size_t>(p)].second;
            const glm::dvec3 rij = MinImage(m_pos[static_cast<size_t>(i)] - m_pos[static_cast<size_t>(j)], m_L);
            const double r2 = glm::dot(rij, rij);
            if (r2 > cutoff2) continue; // skin candidate that's outside the real cutoff this step

            const int si = m_species[static_cast<size_t>(i)];
            const int sj = m_species[static_cast<size_t>(j)];
            const double sigma = m_sigma[si][sj];
            const double epsilon = m_epsilon[si][sj];

            const double r = std::sqrt(r2);
            const double Fmag = LjForceMagnitude(r, epsilon, sigma) - Fc[si][sj];
            const glm::dvec3 f = (Fmag / r) * rij;

            acc[static_cast<size_t>(i)] += f;
            acc[static_cast<size_t>(j)] -= f;

            potentialSum += LjPotential(r, epsilon, sigma) - Uc[si][sj] + (r - cutoff) * Fc[si][sj];
            virialSum += glm::dot(rij, f);
        }
    }

    for (int t = 0; t < nThreads; ++t) {
        const std::vector<glm::dvec3>& acc = threadAccel[static_cast<size_t>(t)];
        for (int i = 0; i < n; ++i) m_accel[static_cast<size_t>(i)] += acc[static_cast<size_t>(i)];
    }

    m_potentialEnergy = potentialSum;
    m_virial = virialSum;
}

void MDSystem::ApplyThermostat(const MDParams& params) {
    if (m_vel.size() < 2) return;

    if (params.thermostat == Thermostat::Berendsen) {
        const double T = Temperature();
        if (T > 1e-12) {
            const double lambda2 = 1.0 + (params.dt / params.berendsenTau) * (params.targetT / T - 1.0);
            const double lambda = std::sqrt(std::max(lambda2, 0.0));
            for (glm::dvec3& v : m_vel) v *= lambda;
        }
    } else if (params.thermostat == Thermostat::VelocityRescale) {
        ++m_stepsSinceRescale;
        if (m_stepsSinceRescale >= params.rescaleEvery) {
            m_stepsSinceRescale = 0;
            const double T = Temperature();
            if (T > 1e-12) {
                const double lambda = std::sqrt(params.targetT / T);
                for (glm::dvec3& v : m_vel) v *= lambda;
            }
        }
    }
}

// Isotropic Berendsen barostat: mu^3 = 1 - (dt/tauP)*compressibility*(P0-P),
// applied as a uniform rescale of the box and every particle position.
// Weak-coupling scheme (same caveat as the Berendsen thermostat: it doesn't
// reproduce the true NPT volume-fluctuation variance) -- adequate for
// equilibrating to a target pressure, not for fluctuation statistics.
// mu is clamped per step (0.1 floor on mu^3) purely as a numerical safety
// rail against a transient pressure spike driving the box to zero/negative
// volume in one step; it should never bind in a well-behaved run.
void MDSystem::ApplyBarostat(const MDParams& params) {
    if (params.barostat != Barostat::Berendsen || m_pos.empty()) return;
    const double mu3 = 1.0 - (params.dt / params.berendsenTauP) * params.compressibility * (params.targetP - Pressure());
    const double mu = std::cbrt(std::max(mu3, 0.1));
    if (mu == 1.0) return;
    for (glm::dvec3& r : m_pos) r *= mu;
    m_L *= mu;
}

// Uses the species-A (sigma=epsilon=1) reference parameters regardless of
// mixture composition -- a real simplification for a binary system (the
// correct mean-field tail is a composition-weighted sum over all three
// pair types), left as-is since this correction is reporting-only and
// mixtures in this project are run dense enough (rho*=1.2) that the tail
// beyond the cutoff is a small correction either way.
double MDSystem::TailEnergyCorrection(double cutoff) const {
    const int n = static_cast<int>(m_pos.size());
    if (n == 0) return 0.0;
    const double rho = n / (m_L * m_L * m_L);
    const double rc = std::min(cutoff, 0.49 * m_L);
    const double sr3 = std::pow(kSigma / rc, 3);
    const double sr9 = sr3 * sr3 * sr3;
    return (8.0 / 3.0) * kPi * n * rho * kEpsilon * kSigma * kSigma * kSigma * (sr9 / 3.0 - sr3);
}

double MDSystem::TailPressureCorrection(double cutoff) const {
    const int n = static_cast<int>(m_pos.size());
    if (n == 0) return 0.0;
    const double rho = n / (m_L * m_L * m_L);
    const double rc = std::min(cutoff, 0.49 * m_L);
    const double sr3 = std::pow(kSigma / rc, 3);
    const double sr9 = sr3 * sr3 * sr3;
    return (16.0 / 3.0) * kPi * rho * rho * kEpsilon * kSigma * kSigma * kSigma * (2.0 * sr9 / 3.0 - sr3);
}

double MDSystem::NoseHooverInvariant(const MDParams& params) const {
    const int n = static_cast<int>(m_pos.size());
    const double dof = 3.0 * std::max(n - 1, 0);
    const double Q = std::max(dof * params.targetT * params.noseHooverTau * params.noseHooverTau, 1e-12);
    return KineticEnergy() + m_potentialEnergy + 0.5 * Q * m_xi * m_xi + dof * params.targetT * m_xiIntegral;
}

void MDSystem::StepVelocityVerlet(const MDParams& params) {
    const int n = static_cast<int>(m_pos.size());
    const double dt = params.dt;

    for (int i = 0; i < n; ++i) {
        glm::dvec3& r = m_pos[static_cast<size_t>(i)];
        r += m_vel[static_cast<size_t>(i)] * dt + 0.5 * m_accel[static_cast<size_t>(i)] * dt * dt;
        r = WrapPosition(r, m_L);
    }

    ++m_stepsSinceRebuild;
    if (m_stepsSinceRebuild >= params.neighborRebuildEvery) {
        RebuildNeighborList(params.cutoff + params.skin);
    }

    const std::vector<glm::dvec3> accelOld = m_accel;
    ComputeForces(std::min(params.cutoff, 0.49 * m_L));

    for (int i = 0; i < n; ++i) {
        m_vel[static_cast<size_t>(i)] += 0.5 * (accelOld[static_cast<size_t>(i)] + m_accel[static_cast<size_t>(i)]) * dt;
    }

    ApplyThermostat(params);
}

// Nose-Hoover thermostatted velocity-Verlet (single thermostat variable, no
// chain): the friction term -xi*v is folded directly into the equations of
// motion instead of a post-hoc rescale, which is what actually samples the
// canonical ensemble correctly. Half-step splitting (e.g. Frenkel & Smit,
// "Understanding Molecular Simulation", Sec. 6.1.2):
//
//   v(t+dt/2) = v(t) + dt/2*(a(t) - xi(t)*v(t))
//   x(t+dt)   = x(t) + dt*v(t+dt/2)
//   xi(t+dt/2)= xi(t) + dt/(2Q)*(2*KE(t+dt/2) - dof*T0)
//   a(t+dt)   = F(x(t+dt))/m
//   v(t+dt)   = (v(t+dt/2) + dt/2*a(t+dt)) / (1 + dt/2*xi(t+dt/2))   -- solves the implicit -xi*v(t+dt) term
//   xi(t+dt)  = xi(t+dt/2) + dt/(2Q)*(2*KE(t+dt) - dof*T0)
//
// Q = dof*T0*tau^2 (deck-configurable tau); dof = 3*(N-1), same convention
// as Temperature(). m_xiIntegral accumulates xi(t+dt/2)*dt for
// NoseHooverInvariant's conserved-quantity check.
void MDSystem::StepNoseHoover(const MDParams& params) {
    const int n = static_cast<int>(m_pos.size());
    const double dt = params.dt;
    const double dof = 3.0 * std::max(n - 1, 0);
    const double Q = std::max(dof * params.targetT * params.noseHooverTau * params.noseHooverTau, 1e-12);

    for (int i = 0; i < n; ++i) {
        m_vel[static_cast<size_t>(i)] += 0.5 * dt * (m_accel[static_cast<size_t>(i)] - m_xi * m_vel[static_cast<size_t>(i)]);
    }
    for (int i = 0; i < n; ++i) {
        glm::dvec3& r = m_pos[static_cast<size_t>(i)];
        r += dt * m_vel[static_cast<size_t>(i)];
        r = WrapPosition(r, m_L);
    }

    ++m_stepsSinceRebuild;
    if (m_stepsSinceRebuild >= params.neighborRebuildEvery) {
        RebuildNeighborList(params.cutoff + params.skin);
    }

    const double keHalf = KineticEnergy();
    const double xiHalf = m_xi + (0.5 * dt / Q) * (2.0 * keHalf - dof * params.targetT);

    ComputeForces(std::min(params.cutoff, 0.49 * m_L));

    const double denom = 1.0 + 0.5 * dt * xiHalf;
    for (int i = 0; i < n; ++i) {
        m_vel[static_cast<size_t>(i)] = (m_vel[static_cast<size_t>(i)] + 0.5 * dt * m_accel[static_cast<size_t>(i)]) / denom;
    }

    const double keNew = KineticEnergy();
    m_xi = xiHalf + (0.5 * dt / Q) * (2.0 * keNew - dof * params.targetT);
    m_xiIntegral += dt * xiHalf;
}

void MDSystem::Step(const MDParams& params) {
    if (m_pos.empty()) return;
    if (params.thermostat == Thermostat::NoseHoover) {
        StepNoseHoover(params);
    } else {
        StepVelocityVerlet(params);
    }
    ApplyBarostat(params);
}

double MDSystem::KineticEnergy() const {
    double ke = 0.0;
    for (const glm::dvec3& v : m_vel) ke += 0.5 * glm::dot(v, v);
    return ke;
}

double MDSystem::Temperature() const {
    const int n = static_cast<int>(m_vel.size());
    if (n < 2) return 0.0;
    // 3*(N-1) degrees of freedom: particles are initialized with zero net
    // momentum, and LJ's pairwise internal forces conserve it exactly, so
    // 3 translational dof never carry thermal energy.
    return 2.0 * KineticEnergy() / (3.0 * (n - 1));
}

double MDSystem::Pressure() const {
    const int n = static_cast<int>(m_pos.size());
    if (n == 0) return 0.0;
    const double V = m_L * m_L * m_L;
    const double rho = n / V;
    return rho * Temperature() + m_virial / (3.0 * V);
}

std::vector<double> MDSystem::ComputeRDF(int nBins, double rMax) const {
    std::vector<double> g(static_cast<size_t>(std::max(nBins, 0)), 0.0);
    const int n = static_cast<int>(m_pos.size());
    if (n < 2 || nBins <= 0 || rMax <= 0.0) return g;

    rMax = std::min(rMax, 0.49 * m_L);
    const std::vector<std::pair<int, int>> pairs = BuildCellListPairs(m_pos, m_L, rMax);

    std::vector<double> hist(static_cast<size_t>(nBins), 0.0);
    const double dr = rMax / nBins;
    for (const std::pair<int, int>& pr : pairs) {
        const glm::dvec3 d = MinImage(m_pos[static_cast<size_t>(pr.first)] - m_pos[static_cast<size_t>(pr.second)], m_L);
        const double r = std::sqrt(glm::dot(d, d));
        const int bin = static_cast<int>(r / dr);
        if (bin >= 0 && bin < nBins) hist[static_cast<size_t>(bin)] += 1.0; // one increment per unordered pair, no double count
    }

    const double V = m_L * m_L * m_L;
    const double totalPairs = 0.5 * n * (n - 1);
    for (int b = 0; b < nBins; ++b) {
        const double rLo = b * dr;
        const double rHi = rLo + dr;
        const double shellVol = (4.0 / 3.0) * kPi * (rHi * rHi * rHi - rLo * rLo * rLo);
        const double expected = totalPairs * (shellVol / V);
        g[static_cast<size_t>(b)] = (expected > 1e-12) ? hist[static_cast<size_t>(b)] / expected : 0.0;
    }
    return g;
}

std::vector<double> MDSystem::ComputePartialRDF(int nBins, double rMax, int speciesA, int speciesB) const {
    std::vector<double> g(static_cast<size_t>(std::max(nBins, 0)), 0.0);
    const int n = static_cast<int>(m_pos.size());
    if (n < 2 || nBins <= 0 || rMax <= 0.0) return g;

    int nA = 0, nB = 0;
    for (int s : m_species) {
        if (s == speciesA) ++nA;
        if (s == speciesB) ++nB;
    }
    if (nA == 0 || nB == 0) return g; // e.g. speciesB absent in a single-species run

    // Same-species: 0.5*nA*(nA-1) distinct unordered pairs (standard g(r)
    // convention). Cross-species: nA*nB -- every A-B pair is distinct (no /2,
    // A and B are disjoint sets, so there's no double-counting to remove).
    const double totalPairs = (speciesA == speciesB) ? 0.5 * nA * (nA - 1) : static_cast<double>(nA) * nB;

    rMax = std::min(rMax, 0.49 * m_L);
    const std::vector<std::pair<int, int>> pairs = BuildCellListPairs(m_pos, m_L, rMax);

    std::vector<double> hist(static_cast<size_t>(nBins), 0.0);
    const double dr = rMax / nBins;
    for (const std::pair<int, int>& pr : pairs) {
        const int si = m_species[static_cast<size_t>(pr.first)];
        const int sj = m_species[static_cast<size_t>(pr.second)];
        const bool matches = (si == speciesA && sj == speciesB) || (si == speciesB && sj == speciesA);
        if (!matches) continue;
        const glm::dvec3 d = MinImage(m_pos[static_cast<size_t>(pr.first)] - m_pos[static_cast<size_t>(pr.second)], m_L);
        const double r = std::sqrt(glm::dot(d, d));
        const int bin = static_cast<int>(r / dr);
        if (bin >= 0 && bin < nBins) hist[static_cast<size_t>(bin)] += 1.0;
    }

    const double V = m_L * m_L * m_L;
    for (int b = 0; b < nBins; ++b) {
        const double rLo = b * dr;
        const double rHi = rLo + dr;
        const double shellVol = (4.0 / 3.0) * kPi * (rHi * rHi * rHi - rLo * rLo * rLo);
        const double expected = totalPairs * (shellVol / V);
        g[static_cast<size_t>(b)] = (expected > 1e-12) ? hist[static_cast<size_t>(b)] / expected : 0.0;
    }
    return g;
}

} // namespace md
