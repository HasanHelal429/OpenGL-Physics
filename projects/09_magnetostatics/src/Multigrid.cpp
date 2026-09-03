#include "Multigrid.hpp"

#include <algorithm>
#include <cmath>

namespace mag {

namespace {

// One grid in the hierarchy. Level 0 is the finest.
//
// Every level uses the same boundary treatment: homogeneous Dirichlet on the
// domain edge (the face, not the cell centre), imposed by a reflective ghost
// u_ghost = -u_edge, and every cell [0, n-1] is relaxed. Keeping the boundary
// identical across levels is what makes the coarse-grid correction consistent
// -- a cell-centred hierarchy where only the fine level clamps its edge cells
// loses the near-boundary residual on every coarsening and the V-cycle decays
// to the smoother's rate. Magnetostatics on a large box has A_z -> 0 at the
// edge anyway, so the homogeneous condition is the physical one.
struct Level {
    int nx = 0, ny = 0;
    double idx2 = 0.0, idy2 = 0.0;   // 1/dx^2, 1/dy^2
    std::vector<double> u, rhs, res;

    std::size_t at(int i, int j) const {
        return static_cast<std::size_t>(j) * nx + i;
    }
};

// GS/SOR update at one cell, with a reflective Dirichlet-face ghost for any
// missing neighbour: present neighbour contributes (value/h^2) to the sum and
// (1/h^2) to the diagonal; a ghost contributes 0 to the sum and (2/h^2) to the
// diagonal (from u_ghost = -u_p).
void Smooth(Level& L, int sweeps, double omega) {
    for (int s = 0; s < sweeps; ++s) {
        for (int colour = 0; colour < 2; ++colour) {
            for (int j = 0; j < L.ny; ++j) {
                for (int i = 0; i < L.nx; ++i) {
                    if (((i + j) & 1) != colour) continue;
                    const std::size_t p = L.at(i, j);
                    double accum = 0.0, diag = 0.0;
                    if (i > 0)        { accum += L.u[p - 1] * L.idx2;    diag += L.idx2; }
                    else              {                                 diag += 2.0 * L.idx2; }
                    if (i < L.nx - 1) { accum += L.u[p + 1] * L.idx2;    diag += L.idx2; }
                    else              {                                 diag += 2.0 * L.idx2; }
                    if (j > 0)        { accum += L.u[p - L.nx] * L.idy2; diag += L.idy2; }
                    else              {                                 diag += 2.0 * L.idy2; }
                    if (j < L.ny - 1) { accum += L.u[p + L.nx] * L.idy2; diag += L.idy2; }
                    else              {                                 diag += 2.0 * L.idy2; }
                    const double gs = (accum - L.rhs[p]) / diag;
                    L.u[p] += omega * (gs - L.u[p]);
                }
            }
        }
    }
}

// res = rhs - laplacian(u), same ghost rule; returns the L-inf residual.
double Residual(Level& L) {
    double rmax = 0.0;
    std::fill(L.res.begin(), L.res.end(), 0.0);
    for (int j = 0; j < L.ny; ++j) {
        for (int i = 0; i < L.nx; ++i) {
            const std::size_t p = L.at(i, j);
            double lap = 0.0;
            lap += ((i > 0        ? L.u[p - 1]    : -L.u[p]) - L.u[p]) * L.idx2;
            lap += ((i < L.nx - 1 ? L.u[p + 1]    : -L.u[p]) - L.u[p]) * L.idx2;
            lap += ((j > 0        ? L.u[p - L.nx] : -L.u[p]) - L.u[p]) * L.idy2;
            lap += ((j < L.ny - 1 ? L.u[p + L.nx] : -L.u[p]) - L.u[p]) * L.idy2;
            L.res[p] = L.rhs[p] - lap;
            rmax = std::max(rmax, std::abs(L.res[p]));
        }
    }
    return rmax;
}

// The bilinear interpolation stencil for one fine index: the two coarse
// indices it reads and their weights. Cell-centred mapping: fine centre i
// sits at coarse coordinate i/2 - 1/4.
struct Stencil1D { int c0, c1; double w0, w1; };
Stencil1D Interp1D(int fineIdx, int coarseN) {
    const int m = fineIdx / 2;
    Stencil1D s;
    if ((fineIdx & 1) == 0) { s.c0 = m - 1; s.c1 = m;     s.w0 = 0.25; s.w1 = 0.75; }
    else                    { s.c0 = m;     s.c1 = m + 1; s.w0 = 0.75; s.w1 = 0.25; }
    s.c0 = std::clamp(s.c0, 0, coarseN - 1);
    s.c1 = std::clamp(s.c1, 0, coarseN - 1);
    return s;
}

// Restriction as the scaled transpose of the bilinear prolongation
// (R = P^T / 2^dim): each fine residual is scattered into the coarse rhs with
// the same weights prolongation would use to read it back. Transpose
// compatibility is what gives the V-cycle its grid-independent rate.
void RestrictResidual(const Level& fine, Level& coarse) {
    std::fill(coarse.u.begin(), coarse.u.end(), 0.0);
    std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
    for (int j = 0; j < fine.ny; ++j) {
        const Stencil1D sy = Interp1D(j, coarse.ny);
        for (int i = 0; i < fine.nx; ++i) {
            const Stencil1D sx = Interp1D(i, coarse.nx);
            const double r = 0.25 * fine.res[fine.at(i, j)];
            coarse.rhs[coarse.at(sx.c0, sy.c0)] += sx.w0 * sy.w0 * r;
            coarse.rhs[coarse.at(sx.c1, sy.c0)] += sx.w1 * sy.w0 * r;
            coarse.rhs[coarse.at(sx.c0, sy.c1)] += sx.w0 * sy.w1 * r;
            coarse.rhs[coarse.at(sx.c1, sy.c1)] += sx.w1 * sy.w1 * r;
        }
    }
}

// Bilinear interpolation of the coarse correction, added to the fine cells
// that the smoother relaxes.
void ProlongAdd(const Level& coarse, Level& fine) {
    for (int j = 0; j < fine.ny; ++j) {
        const Stencil1D sy = Interp1D(j, coarse.ny);
        for (int i = 0; i < fine.nx; ++i) {
            const Stencil1D sx = Interp1D(i, coarse.nx);
            fine.u[fine.at(i, j)] +=
                sx.w0 * sy.w0 * coarse.u[coarse.at(sx.c0, sy.c0)] +
                sx.w1 * sy.w0 * coarse.u[coarse.at(sx.c1, sy.c0)] +
                sx.w0 * sy.w1 * coarse.u[coarse.at(sx.c0, sy.c1)] +
                sx.w1 * sy.w1 * coarse.u[coarse.at(sx.c1, sy.c1)];
        }
    }
}

// gamma = 1 is a V-cycle, gamma = 2 a W-cycle. The W-cycle spends more work on
// the coarse levels and flattens out the mild grid-size dependence a V-cycle
// with rediscretised (non-Galerkin) coarse operators shows here.
void MuCycle(std::vector<Level>& lv, int l, const MultigridOptions& opt) {
    if (l == static_cast<int>(lv.size()) - 1) {
        Smooth(lv[l], opt.nuCoarse, opt.omega);
        return;
    }
    Smooth(lv[l], opt.nu1, opt.omega);
    Residual(lv[l]);
    RestrictResidual(lv[l], lv[l + 1]);
    for (int k = 0; k < opt.gamma; ++k) MuCycle(lv, l + 1, opt);
    ProlongAdd(lv[l + 1], lv[l]);
    Smooth(lv[l], opt.nu2, opt.omega);
}

} // namespace

MultigridResult SolvePoissonMultigrid(const Grid& g,
                                      const std::vector<double>& rhs,
                                      const std::vector<double>& fixedValues,
                                      std::vector<double>& u,
                                      const MultigridOptions& opt) {
    std::vector<Level> lv;
    {
        int nx = g.nx, ny = g.ny;
        double dx = g.dx(), dy = g.dy();
        while (true) {
            Level L;
            L.nx = nx;
            L.ny = ny;
            L.idx2 = 1.0 / (dx * dx);
            L.idy2 = 1.0 / (dy * dy);
            L.u.assign(static_cast<std::size_t>(nx) * ny, 0.0);
            L.rhs.assign(L.u.size(), 0.0);
            L.res.assign(L.u.size(), 0.0);
            lv.push_back(std::move(L));
            if ((nx % 2) || (ny % 2) || nx / 2 < opt.minCoarse ||
                ny / 2 < opt.minCoarse || nx / 2 < 4 || ny / 2 < 4)
                break;
            nx /= 2;
            ny /= 2;
            dx *= 2.0;
            dy *= 2.0;
        }
    }

    (void)fixedValues;   // homogeneous edge BC on every level -- see Level's note
    Level& fine = lv[0];
    fine.u = u;
    fine.rhs = rhs;

    double rhsScale = 0.0;
    for (double r : rhs) rhsScale = std::max(rhsScale, std::abs(r));
    if (rhsScale < 1e-300) rhsScale = 1.0;

    MultigridResult res;
    res.levels = static_cast<int>(lv.size());
    for (int cycle = 1; cycle <= opt.maxCycles; ++cycle) {
        MuCycle(lv, 0, opt);
        const double rmax = Residual(lv[0]);
        res.cycles = cycle;
        res.residual = rmax;
        if (rmax / rhsScale < opt.tol) {
            res.converged = true;
            break;
        }
    }

    u = lv[0].u;
    return res;
}

} // namespace mag
