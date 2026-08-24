#include "ScfSolver.hpp"

#include "Quadrature.hpp"
#include "RadialEigensolver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>

namespace hf {

namespace {

constexpr double kPi = std::numbers::pi;

// Crude seed density: an exponential profile normalized to N=Z electrons.
std::vector<double> InitialDensity(const std::vector<double>& r, int Z) {
    std::vector<double> rho(r.size());
    std::vector<double> integrand(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        rho[i] = std::exp(-2.0 * Z * r[i]);
        integrand[i] = 4.0 * kPi * r[i] * r[i] * rho[i];
    }
    const double N = Trapezoid(integrand, r);
    const double scale = Z / N;
    for (double& value : rho) value *= scale;
    return rho;
}

// rho(r) = (1/(4*pi*r^2)) * sum_nl N_nl*u_nl(r)^2.
std::vector<double> DensityFromOccupied(const std::vector<double>& r, const std::vector<OccupiedOrbital>& occupied) {
    std::vector<double> total(r.size(), 0.0);
    for (const auto& orb : occupied) {
        for (size_t i = 0; i < r.size(); ++i) {
            total[i] += orb.N * orb.u[i] * orb.u[i];
        }
    }
    std::vector<double> rho(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        rho[i] = total[i] / (4.0 * kPi * r[i] * r[i]);
    }
    return rho;
}

} // namespace

double ComputeTotalEnergy(const std::vector<double>& r, const std::vector<double>& rho, const std::vector<double>& VH,
                           const std::vector<double>& Vx, const std::vector<OccupiedOrbital>& occupied,
                           const std::vector<double>* epsC, const std::vector<double>* Vc) {
    double sumEps = 0.0;
    for (const auto& orb : occupied) sumEps += orb.N * orb.eps;

    const size_t n = r.size();
    std::vector<double> hIntegrand(n), xIntegrand(n);
    for (size_t i = 0; i < n; ++i) {
        const double weight = rho[i] * 4.0 * kPi * r[i] * r[i];
        hIntegrand[i] = VH[i] * weight;
        xIntegrand[i] = Vx[i] * weight;
    }
    const double EH = 0.5 * Trapezoid(hIntegrand, r);
    const double Ex = 0.75 * Trapezoid(xIntegrand, r);

    double eTotal = sumEps - EH - Ex / 3.0;

    if (epsC && Vc) {
        std::vector<double> cIntegrand(n), cvIntegrand(n);
        for (size_t i = 0; i < n; ++i) {
            const double weight = rho[i] * 4.0 * kPi * r[i] * r[i];
            cIntegrand[i] = (*epsC)[i] * weight;
            cvIntegrand[i] = (*Vc)[i] * weight;
        }
        const double Ec = Trapezoid(cIntegrand, r);
        eTotal += Ec - Trapezoid(cvIntegrand, r);
    }

    return eTotal;
}

ScfResult RunScf(const ScfParams& params, const SnapshotCallback& onSnapshot, std::stop_token stopToken) {
    const LogGrid grid = params.grid ? *params.grid : DefaultGrid(params.Z);
    const std::vector<double>& r = grid.r;
    const double h = grid.h;

    const Configuration config = GroundStateConfiguration(params.Z);

    std::map<int, std::map<int, int>> byL; // l -> (n -> N_nl)
    for (const auto& [nl, occ] : config) {
        byL[nl.second][nl.first] = occ;
    }
    std::map<int, int> nStatesPerL;
    for (const auto& [l, ns] : byL) {
        int maxN = 0;
        for (const auto& [n, occ] : ns) maxN = std::max(maxN, n);
        nStatesPerL[l] = maxN - l;
    }

    std::vector<double> rho = InitialDensity(r, params.Z);
    std::optional<double> ePrev;
    int convergedStreak = 0;

    ScfResult result;
    result.Z = params.Z;
    result.method = params.method;
    result.grid = grid;
    result.config = config;

    std::vector<OccupiedOrbital> occupied;
    std::vector<double> V, VH, Vx;
    double eTotal = 0.0;
    int iteration = 0;

    for (iteration = 1; iteration <= params.maxIter; ++iteration) {
        if (stopToken.stop_requested()) {
            result.cancelled = true;
            break;
        }

        VH = HartreePotential(r, rho);

        std::vector<double> Vxc;
        std::optional<std::vector<double>> epsC, Vc;
        if (params.method == Method::Lda) {
            auto [epsCVal, VcVal] = Pz81Correlation(rho);
            Vx = SlaterExchangePotential(rho, kAlphaLda);
            Vxc.resize(r.size());
            for (size_t i = 0; i < r.size(); ++i) Vxc[i] = Vx[i] + VcVal[i];
            epsC = std::move(epsCVal);
            Vc = std::move(VcVal);
        } else {
            Vx = SlaterExchangePotential(rho, params.alpha);
            Vxc = Vx;
        }

        V.resize(r.size());
        for (size_t i = 0; i < r.size(); ++i) {
            V[i] = -static_cast<double>(params.Z) / r[i] + VH[i] + Vxc[i];
        }

        occupied.clear();
        for (const auto& [l, ns] : byL) {
            const RadialSolution sol = SolveRadialChannel(r, l, V, h, nStatesPerL[l]);
            for (const auto& [n, N_nl] : ns) {
                const int idx = n - l - 1;
                occupied.push_back({n, l, N_nl, sol.energies[static_cast<size_t>(idx)], sol.u[static_cast<size_t>(idx)]});
            }
        }

        const std::vector<double> rhoNew = DensityFromOccupied(r, occupied);
        eTotal = ComputeTotalEnergy(r, rho, VH, Vx, occupied, epsC ? &*epsC : nullptr, Vc ? &*Vc : nullptr);

        std::vector<double> dRho(r.size());
        for (size_t i = 0; i < r.size(); ++i) dRho[i] = std::abs(rhoNew[i] - rho[i]) * 4.0 * kPi * r[i] * r[i];
        const double dn = Trapezoid(dRho, r);
        const double dE = ePrev ? std::abs(eTotal - *ePrev) : std::numeric_limits<double>::infinity();
        result.history.push_back(eTotal);

        if (onSnapshot) {
            ScfSnapshot snapshot;
            snapshot.iteration = iteration;
            snapshot.rho = rho;
            snapshot.V = V;
            snapshot.r = r;
            for (const auto& orb : occupied) snapshot.orbitalEnergies[{orb.n, orb.l}] = orb.eps;
            snapshot.eTotal = eTotal;
            snapshot.dE = dE;
            snapshot.dn = dn;
            onSnapshot(snapshot);
        }

        if (dE < params.tolE && dn < params.tolN) {
            if (++convergedStreak >= 2) {
                rho = rhoNew;
                result.converged = true;
                break;
            }
        } else {
            convergedStreak = 0;
        }

        for (size_t i = 0; i < r.size(); ++i) rho[i] = (1.0 - params.mixBeta) * rho[i] + params.mixBeta * rhoNew[i];
        ePrev = eTotal;
    }

    result.rho = rho;
    result.V = V;
    result.VH = VH;
    result.Vx = Vx;
    result.eTotal = eTotal;
    result.iterations = iteration;
    for (const auto& orb : occupied) {
        std::vector<double> R(r.size());
        for (size_t i = 0; i < r.size(); ++i) R[i] = orb.u[i] / r[i];
        result.orbitals[{orb.n, orb.l}] = std::move(R);
        result.orbitalEnergies[{orb.n, orb.l}] = orb.eps;
    }

    return result;
}

} // namespace hf
