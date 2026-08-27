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

void MDSystem::SetParticles(std::vector<glm::dvec3> pos, std::vector<glm::dvec3> vel, double boxLength) {
    m_pos = std::move(pos);
    m_vel = std::move(vel);
    m_L = boxLength;
    m_accel.assign(m_pos.size(), glm::dvec3(0.0));
    m_neighborPairs.clear();
    m_stepsSinceRebuild = 0;
    m_stepsSinceRescale = 0;
    m_potentialEnergy = 0.0;
    m_virial = 0.0;
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
    const double Fc = LjForceMagnitude(cutoff, kEpsilon, kSigma);
    const double Uc = LjPotential(cutoff, kEpsilon, kSigma);

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

            const double r = std::sqrt(r2);
            const double Fmag = LjForceMagnitude(r, kEpsilon, kSigma) - Fc;
            const glm::dvec3 f = (Fmag / r) * rij;

            acc[static_cast<size_t>(i)] += f;
            acc[static_cast<size_t>(j)] -= f;

            potentialSum += LjPotential(r, kEpsilon, kSigma) - Uc + (r - cutoff) * Fc;
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

void MDSystem::Step(const MDParams& params) {
    const int n = static_cast<int>(m_pos.size());
    if (n == 0) return;
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

} // namespace md
