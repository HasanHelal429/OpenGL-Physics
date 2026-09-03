#include "FieldSolver.hpp"

#include <algorithm>
#include <cmath>

namespace mag {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

void MakeEdgeDirichlet(const Grid& g,
                       std::vector<unsigned char>& fixedMask,
                       std::vector<double>& fixedValues) {
    fixedMask.assign(g.count(), 0);
    fixedValues.assign(g.count(), 0.0);
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            if (g.onBoundary(i, j)) fixedMask[g.idx(i, j)] = 1;
        }
    }
}

SolveResult SolvePoisson(const Grid& g,
                         const std::vector<double>& rhsIn,
                         const std::vector<unsigned char>& fixedMask,
                         const std::vector<double>& fixedValues,
                         std::vector<double>& u,
                         const SolveOptions& opt) {
    const int nx = g.nx, ny = g.ny;
    const double idx2 = 1.0 / (g.dx() * g.dx());
    const double idy2 = 1.0 / (g.dy() * g.dy());
    const bool neumann = opt.edgeBC == BC::Neumann;

    // Under all-Neumann edges the Poisson problem is singular: the rhs must be
    // mean-zero and the solution is fixed only up to an additive constant.
    std::vector<double> rhs = rhsIn;
    if (neumann) {
        double mean = 0.0;
        for (double r : rhs) mean += r;
        mean /= rhs.size();
        for (double& r : rhs) r -= mean;
    }

    // Clamp the fixed cells to their prescribed values up front so the
    // stencil reads the right boundary data during relaxation.
    for (int k = 0; k < g.count(); ++k) {
        if (fixedMask[k]) u[k] = fixedValues[k];
    }

    double omega = opt.omega;
    if (omega <= 0.0) {
        // Optimal SOR factor for the Poisson problem on this grid.
        const double s = std::sin(kPi / std::max(nx, ny));
        omega = 2.0 / (1.0 + s);
    }

    double rhsScale = 0.0;
    for (double r : rhs) rhsScale = std::max(rhsScale, std::abs(r));
    if (rhsScale < 1e-300) rhsScale = 1.0;

    // Loop bounds: Neumann relaxes the edge cells too; Dirichlet keeps them
    // fixed, so the interior [1, n-2] range suffices.
    const int i0 = neumann ? 0 : 1, i1 = neumann ? nx : nx - 1;
    const int j0 = neumann ? 0 : 1, j1 = neumann ? ny : ny - 1;

    // GS update for one cell, mirroring any missing neighbour (Neumann edge:
    // ghost == centre, so that face drops out of both the sum and the diagonal).
    auto relax = [&](int i, int j) {
        const std::size_t p = g.idx(i, j);
        if (fixedMask[p]) return;
        double accum = 0.0, diag = 0.0;
        if (i > 0)      { accum += u[p - 1] * idx2;  diag += idx2; }
        if (i < nx - 1) { accum += u[p + 1] * idx2;  diag += idx2; }
        if (j > 0)      { accum += u[p - nx] * idy2; diag += idy2; }
        if (j < ny - 1) { accum += u[p + nx] * idy2; diag += idy2; }
        const double gs = (accum - rhs[p]) / diag;
        u[p] += omega * (gs - u[p]);
    };

    auto residualAt = [&](int i, int j) {
        const std::size_t p = g.idx(i, j);
        double lap = 0.0;
        if (i > 0)      lap += (u[p - 1] - u[p]) * idx2;
        if (i < nx - 1) lap += (u[p + 1] - u[p]) * idx2;
        if (j > 0)      lap += (u[p - nx] - u[p]) * idy2;
        if (j < ny - 1) lap += (u[p + nx] - u[p]) * idy2;
        return rhs[p] - lap;
    };

    SolveResult res;
    for (int iter = 1; iter <= opt.maxIterations; ++iter) {
        for (int colour = 0; colour < 2; ++colour) {
            for (int j = j0; j < j1; ++j) {
                for (int i = i0; i < i1; ++i) {
                    if (((i + j) & 1) != colour) continue;
                    relax(i, j);
                }
            }
        }

        // Pin the Neumann null space: shift to zero mean over the free cells.
        if (neumann && iter % 4 == 0) {
            double mean = 0.0;
            int cnt = 0;
            for (int k = 0; k < g.count(); ++k)
                if (!fixedMask[k]) { mean += u[k]; ++cnt; }
            mean /= cnt;
            for (int k = 0; k < g.count(); ++k)
                if (!fixedMask[k]) u[k] -= mean;
        }

        if (iter % opt.checkEvery == 0 || iter == opt.maxIterations) {
            double rmax = 0.0;
            for (int j = j0; j < j1; ++j)
                for (int i = i0; i < i1; ++i) {
                    if (fixedMask[g.idx(i, j)]) continue;
                    rmax = std::max(rmax, std::abs(residualAt(i, j)));
                }
            res.iterations = iter;
            res.residual = rmax;
            if (rmax / rhsScale < opt.tol) {
                res.converged = true;
                break;
            }
        }
    }
    return res;
}

void CurlZ(const Grid& g, const std::vector<double>& Az,
           std::vector<double>& Bx, std::vector<double>& By) {
    const int nx = g.nx, ny = g.ny;
    Bx.assign(g.count(), 0.0);
    By.assign(g.count(), 0.0);
    const double dx = g.dx(), dy = g.dy();

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const std::size_t p = g.idx(i, j);
            double dAz_dy, dAz_dx;
            if (j == 0)            dAz_dy = (Az[p + nx] - Az[p]) / dy;
            else if (j == ny - 1) dAz_dy = (Az[p] - Az[p - nx]) / dy;
            else                  dAz_dy = (Az[p + nx] - Az[p - nx]) / (2.0 * dy);
            if (i == 0)            dAz_dx = (Az[p + 1] - Az[p]) / dx;
            else if (i == nx - 1) dAz_dx = (Az[p] - Az[p - 1]) / dx;
            else                  dAz_dx = (Az[p + 1] - Az[p - 1]) / (2.0 * dx);
            Bx[p] = dAz_dy;
            By[p] = -dAz_dx;
        }
    }
}

} // namespace mag
