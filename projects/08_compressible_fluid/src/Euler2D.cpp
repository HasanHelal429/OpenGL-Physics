#include "Euler2D.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>

namespace cf {

Prim2D ToPrim2D(const Cons2D& c, double gamma) {
    Prim2D p;
    p.rho = c.rho;
    p.u = c.momX / c.rho;
    p.v = c.momY / c.rho;
    const double kinetic = 0.5 * (c.momX * c.momX + c.momY * c.momY) / c.rho;
    p.p = (gamma - 1.0) * (c.energy - kinetic);
    p.tracer = c.rhoTracer / c.rho;
    return p;
}

Cons2D ToCons2D(const Prim2D& p, double gamma) {
    Cons2D c;
    c.rho = p.rho;
    c.momX = p.rho * p.u;
    c.momY = p.rho * p.v;
    c.energy = p.p / (gamma - 1.0) + 0.5 * p.rho * (p.u * p.u + p.v * p.v);
    c.rhoTracer = p.rho * p.tracer;
    return c;
}

Cons2D FluxX2D(const Prim2D& p, double gamma) {
    const Cons2D c = ToCons2D(p, gamma);
    Cons2D f;
    f.rho = c.momX;
    f.momX = c.momX * p.u + p.p;
    f.momY = c.momY * p.u;
    f.energy = p.u * (c.energy + p.p);
    f.rhoTracer = c.rhoTracer * p.u; // pure advection -- no pressure term, the tracer exerts no force
    return f;
}

Cons2D FluxY2D(const Prim2D& p, double gamma) {
    const Cons2D c = ToCons2D(p, gamma);
    Cons2D f;
    f.rho = c.momY;
    f.momX = c.momX * p.v;
    f.momY = c.momY * p.v + p.p;
    f.energy = p.v * (c.energy + p.p);
    f.rhoTracer = c.rhoTracer * p.v;
    return f;
}

Cons2D HllcFluxX(const Prim2D& left, const Prim2D& right, double gamma) {
    const double cL = std::sqrt(gamma * left.p / left.rho);
    const double cR = std::sqrt(gamma * right.p / right.rho);
    const double sL = std::min(left.u - cL, right.u - cR);
    const double sR = std::max(left.u + cL, right.u + cR);

    const Cons2D fL = FluxX2D(left, gamma);
    if (sL >= 0.0) return fL;
    const Cons2D fR = FluxX2D(right, gamma);
    if (sR <= 0.0) return fR;

    const Cons2D uL = ToCons2D(left, gamma), uR = ToCons2D(right, gamma);
    const double sStar = (right.p - left.p + left.rho * left.u * (sL - left.u) - right.rho * right.u * (sR - right.u)) /
                          (left.rho * (sL - left.u) - right.rho * (sR - right.u));

    if (sStar >= 0.0) {
        const double coef = left.rho * (sL - left.u) / (sL - sStar);
        Cons2D uStar;
        uStar.rho = coef;
        uStar.momX = coef * sStar;
        uStar.momY = coef * left.v; // transverse momentum: carried through unchanged (Toro sec. 10.4)
        uStar.energy = coef * (uL.energy / left.rho + (sStar - left.u) * (sStar + left.p / (left.rho * (sL - left.u))));
        uStar.rhoTracer = coef * left.tracer; // tracer: equally passive, same treatment
        Cons2D f;
        f.rho = fL.rho + sL * (uStar.rho - uL.rho);
        f.momX = fL.momX + sL * (uStar.momX - uL.momX);
        f.momY = fL.momY + sL * (uStar.momY - uL.momY);
        f.energy = fL.energy + sL * (uStar.energy - uL.energy);
        f.rhoTracer = fL.rhoTracer + sL * (uStar.rhoTracer - uL.rhoTracer);
        return f;
    }
    const double coef = right.rho * (sR - right.u) / (sR - sStar);
    Cons2D uStar;
    uStar.rho = coef;
    uStar.momX = coef * sStar;
    uStar.momY = coef * right.v;
    uStar.energy = coef * (uR.energy / right.rho + (sStar - right.u) * (sStar + right.p / (right.rho * (sR - right.u))));
    uStar.rhoTracer = coef * right.tracer;
    Cons2D f;
    f.rho = fR.rho + sR * (uStar.rho - uR.rho);
    f.momX = fR.momX + sR * (uStar.momX - uR.momX);
    f.momY = fR.momY + sR * (uStar.momY - uR.momY);
    f.energy = fR.energy + sR * (uStar.energy - uR.energy);
    f.rhoTracer = fR.rhoTracer + sR * (uStar.rhoTracer - uR.rhoTracer);
    return f;
}

Cons2D HllcFluxY(const Prim2D& left, const Prim2D& right, double gamma) {
    const double cL = std::sqrt(gamma * left.p / left.rho);
    const double cR = std::sqrt(gamma * right.p / right.rho);
    const double sL = std::min(left.v - cL, right.v - cR);
    const double sR = std::max(left.v + cL, right.v + cR);

    const Cons2D fL = FluxY2D(left, gamma);
    if (sL >= 0.0) return fL;
    const Cons2D fR = FluxY2D(right, gamma);
    if (sR <= 0.0) return fR;

    const Cons2D uL = ToCons2D(left, gamma), uR = ToCons2D(right, gamma);
    const double sStar = (right.p - left.p + left.rho * left.v * (sL - left.v) - right.rho * right.v * (sR - right.v)) /
                          (left.rho * (sL - left.v) - right.rho * (sR - right.v));

    if (sStar >= 0.0) {
        const double coef = left.rho * (sL - left.v) / (sL - sStar);
        Cons2D uStar;
        uStar.rho = coef;
        uStar.momY = coef * sStar;
        uStar.momX = coef * left.u; // transverse momentum
        uStar.energy = coef * (uL.energy / left.rho + (sStar - left.v) * (sStar + left.p / (left.rho * (sL - left.v))));
        uStar.rhoTracer = coef * left.tracer;
        Cons2D f;
        f.rho = fL.rho + sL * (uStar.rho - uL.rho);
        f.momX = fL.momX + sL * (uStar.momX - uL.momX);
        f.momY = fL.momY + sL * (uStar.momY - uL.momY);
        f.energy = fL.energy + sL * (uStar.energy - uL.energy);
        f.rhoTracer = fL.rhoTracer + sL * (uStar.rhoTracer - uL.rhoTracer);
        return f;
    }
    const double coef = right.rho * (sR - right.v) / (sR - sStar);
    Cons2D uStar;
    uStar.rho = coef;
    uStar.momY = coef * sStar;
    uStar.momX = coef * right.u;
    uStar.energy = coef * (uR.energy / right.rho + (sStar - right.v) * (sStar + right.p / (right.rho * (sR - right.v))));
    uStar.rhoTracer = coef * right.tracer;
    Cons2D f;
    f.rho = fR.rho + sR * (uStar.rho - uR.rho);
    f.momX = fR.momX + sR * (uStar.momX - uR.momX);
    f.momY = fR.momY + sR * (uStar.momY - uR.momY);
    f.energy = fR.energy + sR * (uStar.energy - uR.energy);
    f.rhoTracer = fR.rhoTracer + sR * (uStar.rhoTracer - uR.rhoTracer);
    return f;
}

namespace {

double Minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

constexpr int kGhost = 2;

// Ghost-cell boundary fill for one padded line, independent conditions on
// each side. Outflow reproduces Euler1D::ApplyBoundary's zero-gradient
// convention exactly. Periodic wraps to the interior cell the same
// distance in from the FAR edge (so the line reads as one period of an
// infinite periodic sequence) -- meaningful only when set on BOTH sides
// together (see Euler2D.hpp's WallBC comment). NoSlipReflective mirrors
// the interior cell the same distance in from THIS side's own (near) edge,
// negating BOTH momentum components, so velocity linearly extrapolates to
// exactly zero at the wall face -- density/energy are copied unchanged,
// since energy's kinetic term is invariant under a velocity sign flip.
// FreeSlipReflective does the same mirror but negates only the component
// `axis` selects (a symmetry wall: no penetration, but no tangential drag
// either). Inflow ignores the interior entirely, fixing the ghost to
// `inflowCons` every call.
void ApplyBoundaryLine(std::vector<Cons2D>& u, WallBC bcLeft, WallBC bcRight, const Cons2D& inflowCons,
                        BoundaryAxis axis) {
    const int total = static_cast<int>(u.size());
    const int n = total - 2 * kGhost;
    for (int g = 0; g < kGhost; ++g) {
        const int leftGhost = g;
        const int rightGhost = total - kGhost + g;

        switch (bcLeft) {
        case WallBC::Outflow:
            u[static_cast<size_t>(leftGhost)] = u[static_cast<size_t>(kGhost)];
            break;
        case WallBC::Periodic:
            u[static_cast<size_t>(leftGhost)] = u[static_cast<size_t>(kGhost + n - (kGhost - g))];
            break;
        case WallBC::NoSlipReflective: {
            Cons2D lm = u[static_cast<size_t>(kGhost + (kGhost - 1 - g))];
            lm.momX = -lm.momX;
            lm.momY = -lm.momY;
            u[static_cast<size_t>(leftGhost)] = lm;
            break;
        }
        case WallBC::FreeSlipReflective: {
            Cons2D lm = u[static_cast<size_t>(kGhost + (kGhost - 1 - g))];
            if (axis == BoundaryAxis::X) lm.momX = -lm.momX; else lm.momY = -lm.momY;
            u[static_cast<size_t>(leftGhost)] = lm;
            break;
        }
        case WallBC::Inflow:
            u[static_cast<size_t>(leftGhost)] = inflowCons;
            break;
        }

        switch (bcRight) {
        case WallBC::Outflow:
            u[static_cast<size_t>(rightGhost)] = u[static_cast<size_t>(kGhost + n - 1)];
            break;
        case WallBC::Periodic:
            u[static_cast<size_t>(rightGhost)] = u[static_cast<size_t>(kGhost + g)];
            break;
        case WallBC::NoSlipReflective: {
            Cons2D rm = u[static_cast<size_t>(kGhost + n - 1 - g)];
            rm.momX = -rm.momX;
            rm.momY = -rm.momY;
            u[static_cast<size_t>(rightGhost)] = rm;
            break;
        }
        case WallBC::FreeSlipReflective: {
            Cons2D rm = u[static_cast<size_t>(kGhost + n - 1 - g)];
            if (axis == BoundaryAxis::X) rm.momX = -rm.momX; else rm.momY = -rm.momY;
            u[static_cast<size_t>(rightGhost)] = rm;
            break;
        }
        case WallBC::Inflow:
            u[static_cast<size_t>(rightGhost)] = inflowCons;
            break;
        }
    }
}

} // namespace

// -dF/dx (or -dF/dy) per interior cell of a ghost-padded line -- identical
// structure to Euler1D::Rhs, generalized over Cons2D and a direction-specific
// HLLC flux. Writes into ws.prim/faceL/faceR and dudtOut (a Workspace field
// selected by the caller, e.g. ws.k1 or ws.k2) instead of allocating fresh
// vectors every call -- see Euler2D.hpp's Workspace comment for why.
void Euler2D::LineRhs(const std::vector<Cons2D>& u, double dx, HllcFluxFn hllc, Workspace& ws,
                       std::vector<Cons2D>& dudtOut) const {
    const int total = static_cast<int>(u.size());
    const int n = total - 2 * kGhost;
    std::vector<Prim2D>& prim = ws.prim;
    for (int i = 0; i < total; ++i) prim[static_cast<size_t>(i)] = ToPrim2D(u[static_cast<size_t>(i)], m_gamma);

    std::vector<Prim2D>& faceL = ws.faceL;
    std::vector<Prim2D>& faceR = ws.faceR;
    for (int i = 0; i < total; ++i) faceL[static_cast<size_t>(i)] = faceR[static_cast<size_t>(i)] = prim[static_cast<size_t>(i)];
    for (int i = 1; i < total - 1; ++i) {
        const Prim2D& pm = prim[static_cast<size_t>(i - 1)];
        const Prim2D& p0 = prim[static_cast<size_t>(i)];
        const Prim2D& pp = prim[static_cast<size_t>(i + 1)];
        const double dRho = Minmod(p0.rho - pm.rho, pp.rho - p0.rho);
        const double dU = Minmod(p0.u - pm.u, pp.u - p0.u);
        const double dV = Minmod(p0.v - pm.v, pp.v - p0.v);
        const double dP = Minmod(p0.p - pm.p, pp.p - p0.p);
        const double dTracer = Minmod(p0.tracer - pm.tracer, pp.tracer - p0.tracer);
        faceL[static_cast<size_t>(i)] = Prim2D{p0.rho - 0.5 * dRho, p0.u - 0.5 * dU, p0.v - 0.5 * dV,
                                                p0.p - 0.5 * dP, p0.tracer - 0.5 * dTracer};
        faceR[static_cast<size_t>(i)] = Prim2D{p0.rho + 0.5 * dRho, p0.u + 0.5 * dU, p0.v + 0.5 * dV,
                                                p0.p + 0.5 * dP, p0.tracer + 0.5 * dTracer};
    }

    dudtOut.resize(static_cast<size_t>(n)); // no-op after the first call: Workspace::EnsureSize already sized this
    Cons2D fluxPrev = hllc(faceR[static_cast<size_t>(kGhost - 1)], faceL[static_cast<size_t>(kGhost)], m_gamma);
    for (int i = 0; i < n; ++i) {
        const int gi = i + kGhost;
        const Cons2D fluxNext = hllc(faceR[static_cast<size_t>(gi)], faceL[static_cast<size_t>(gi + 1)], m_gamma);
        Cons2D& d = dudtOut[static_cast<size_t>(i)];
        d.rho = -(fluxNext.rho - fluxPrev.rho) / dx;
        d.momX = -(fluxNext.momX - fluxPrev.momX) / dx;
        d.momY = -(fluxNext.momY - fluxPrev.momY) / dx;
        d.energy = -(fluxNext.energy - fluxPrev.energy) / dx;
        d.rhoTracer = -(fluxNext.rhoTracer - fluxPrev.rhoTracer) / dx;
        fluxPrev = fluxNext;
    }
}

// RK2 (Heun) update of one interior line (a grid row or column) over a
// fractional step dt, given its own ghost-padded flux operator -- the same
// algorithm as Euler1D::Step, generalized over direction (HllcFluxX/Y).
// Writes the result into resultOut (a Workspace field, e.g. ws.result)
// instead of returning a freshly-allocated vector.
void Euler2D::AdvanceLineRK2(const std::vector<Cons2D>& interior, double dt, double dx, HllcFluxFn hllc,
                              WallBC bcLeft, WallBC bcRight, const Cons2D& inflowCons, BoundaryAxis axis,
                              Workspace& ws, std::vector<Cons2D>& resultOut) const {
    const int n = static_cast<int>(interior.size());
    std::vector<Cons2D>& padded = ws.padded;
    for (int i = 0; i < n; ++i) padded[static_cast<size_t>(i + kGhost)] = interior[static_cast<size_t>(i)];
    ApplyBoundaryLine(padded, bcLeft, bcRight, inflowCons, axis);

    LineRhs(padded, dx, hllc, ws, ws.k1);

    std::vector<Cons2D>& stage1 = ws.stage1;
    stage1 = padded; // vector assignment into an already-correctly-sized buffer: copies elements, no reallocation
    for (int i = 0; i < n; ++i) {
        Cons2D& c = stage1[static_cast<size_t>(i + kGhost)];
        const Cons2D& d = ws.k1[static_cast<size_t>(i)];
        c.rho += dt * d.rho;
        c.momX += dt * d.momX;
        c.momY += dt * d.momY;
        c.energy += dt * d.energy;
        c.rhoTracer += dt * d.rhoTracer;
    }
    ApplyBoundaryLine(stage1, bcLeft, bcRight, inflowCons, axis);
    LineRhs(stage1, dx, hllc, ws, ws.k2);

    resultOut.resize(static_cast<size_t>(n)); // no-op after the first call
    for (int i = 0; i < n; ++i) {
        const Cons2D& a = padded[static_cast<size_t>(i + kGhost)];
        const Cons2D& b = stage1[static_cast<size_t>(i + kGhost)];
        const Cons2D& d2 = ws.k2[static_cast<size_t>(i)];
        Cons2D& out = resultOut[static_cast<size_t>(i)];
        out.rho = 0.5 * (a.rho + b.rho + dt * d2.rho);
        out.momX = 0.5 * (a.momX + b.momX + dt * d2.momX);
        out.momY = 0.5 * (a.momY + b.momY + dt * d2.momY);
        out.energy = 0.5 * (a.energy + b.energy + dt * d2.energy);
        out.rhoTracer = 0.5 * (a.rhoTracer + b.rhoTracer + dt * d2.rhoTracer);
    }
}

// Central-difference Laplacian diffusion increment for one line (row or
// column), shared by DiffuseX/DiffuseY. Adds mu*d2(u)/dx_line^2 to momX and
// mu*d2(v)/dx_line^2 to momY -- summed over both DiffuseX and DiffuseY, this
// gives exactly mu*Laplacian(u) and mu*Laplacian(v), the correct
// incompressible-limit reduction of the full Newtonian viscous stress
// divergence (see Euler2D.hpp's class comment): for divergence-free flow,
// d(tau_xx)/dx+d(tau_xy)/dy = 2*mu*d^2u/dx^2 + mu*d^2u/dy^2 + mu*d^2v/dxdy
// collapses to mu*Laplacian(u) exactly, using du/dx=-dv/dy (incompressibility)
// to rewrite the mixed term as -mu*d^2u/dx^2, which cancels one of the two
// factors of 2*mu*d^2u/dx^2. Implementing that cancellation directly (a
// uniform coefficient mu in every direction, dropping the mixed term
// entirely) reproduces the same total instead of computing the identity
// via two separate uncancelled pieces (this project's first attempt at
// this function skipped that cancellation and used a coefficient of 2*mu
// on the normal derivative alone -- exactly correct for a flow with no
// dependence in the swept direction, i.e. Poiseuille, but silently 1.5x
// too strong for a genuinely 2D velocity field, i.e. Taylor-Green -- see
// docs/SIMULATION.md's Phase 4 postmortem for how the bug was caught).
//
// deltaOut (ws.delta) is a REUSED buffer -- unlike a fresh allocation, it is
// not implicitly zero-initialized, so every field of every element must be
// explicitly assigned each call (in particular .rho, which nothing upstream
// of Euler2D actually reads, and .rhoTracer, which DOES get read by
// DiffuseX/DiffuseY whenever m_tracerDiffusivity<=0 -- the default). Getting
// this wrong would silently leak a previous call's diffusion increment into
// the current one.
void Euler2D::LineDiffuse(const std::vector<Cons2D>& padded, double dx, Workspace& ws,
                           std::vector<Cons2D>& deltaOut) const {
    const int total = static_cast<int>(padded.size());
    const int n = total - 2 * kGhost;
    std::vector<Prim2D>& prim = ws.prim;
    for (int i = 0; i < total; ++i) prim[static_cast<size_t>(i)] = ToPrim2D(padded[static_cast<size_t>(i)], m_gamma);

    const double invDx2 = 1.0 / (dx * dx);
    deltaOut.resize(static_cast<size_t>(n)); // no-op after the first call
    for (int i = 0; i < n; ++i) {
        const int gi = i + kGhost;
        const Prim2D& pm = prim[static_cast<size_t>(gi - 1)];
        const Prim2D& p0 = prim[static_cast<size_t>(gi)];
        const Prim2D& pp = prim[static_cast<size_t>(gi + 1)];
        const double d2u = (pp.u - 2.0 * p0.u + pm.u) * invDx2;
        const double d2v = (pp.v - 2.0 * p0.v + pm.v) * invDx2;
        const double tM = pm.p / pm.rho, t0 = p0.p / p0.rho, tP = pp.p / pp.rho;
        const double d2T = (tP - 2.0 * t0 + tM) * invDx2;
        Cons2D& d = deltaOut[static_cast<size_t>(i)];
        d.rho = 0.0;
        d.momX = m_mu * d2u;
        d.momY = m_mu * d2v;
        d.energy = m_conductivity * d2T;
        // Tracer diffusion (Fick's law, constant-density-weighted form
        // mu*Laplacian(tracer)): zero by default, since a dye-visualization
        // tracer usually wants sharp interfaces limited only by the
        // scheme's own numerical dissipation, not an explicit smoothing on
        // top of it -- SetTracerDiffusivity opts into a nonzero value.
        d.rhoTracer = 0.0;
        if (m_tracerDiffusivity > 0.0) {
            const double d2Tracer = (pp.tracer - 2.0 * p0.tracer + pm.tracer) * invDx2;
            d.rhoTracer = m_tracerDiffusivity * d2Tracer;
        }
    }
}

void Euler2D::Workspace::EnsureSize(int n) {
    const int total = n + 2 * kGhost;
    interior.resize(static_cast<size_t>(n));
    padded.resize(static_cast<size_t>(total));
    stage1.resize(static_cast<size_t>(total));
    prim.resize(static_cast<size_t>(total));
    faceL.resize(static_cast<size_t>(total));
    faceR.resize(static_cast<size_t>(total));
    k1.resize(static_cast<size_t>(n));
    k2.resize(static_cast<size_t>(n));
    result.resize(static_cast<size_t>(n));
    delta.resize(static_cast<size_t>(n));
}

void Euler2D::Init(int nx, int ny, double xMin, double xMax, double yMin, double yMax, double gamma) {
    m_nx = nx;
    m_ny = ny;
    m_xMin = xMin;
    m_yMin = yMin;
    m_dx = (xMax - xMin) / static_cast<double>(nx);
    m_dy = (yMax - yMin) / static_cast<double>(ny);
    m_gamma = gamma;
    m_u.assign(static_cast<size_t>(nx * ny), Cons2D{});
    // One Workspace per OpenMP thread, sized to the upper bound
    // omp_get_max_threads() reports right now -- SweepX/SweepY/DiffuseX/
    // DiffuseY index into this by omp_get_thread_num() inside a
    // #pragma omp parallel for, which is always < the parallel region's
    // actual team size, itself always <= omp_get_max_threads() as long as
    // nothing later calls omp_set_num_threads() with a larger value (this
    // project never does). Each thread's buffers are therefore fully
    // private -- no shared mutable state between threads, so no data races
    // and no allocator contention from allocating inside a parallel loop.
    const int numThreads = omp_get_max_threads();
    m_workspacesX.assign(static_cast<size_t>(numThreads), Workspace{});
    for (Workspace& ws : m_workspacesX) ws.EnsureSize(nx);
    m_workspacesY.assign(static_cast<size_t>(numThreads), Workspace{});
    for (Workspace& ws : m_workspacesY) ws.EnsureSize(ny);
}

void Euler2D::SetInitialCondition(const std::function<Prim2D(double x, double y)>& f) {
    for (int j = 0; j < m_ny; ++j) {
        const double y = m_yMin + (static_cast<double>(j) + 0.5) * m_dy;
        for (int i = 0; i < m_nx; ++i) {
            const double x = m_xMin + (static_cast<double>(i) + 0.5) * m_dx;
            m_u[static_cast<size_t>(j * m_nx + i)] = ToCons2D(f(x, y), m_gamma);
        }
    }
}

// Each row is fully independent of every other row within one SweepX call
// (no cross-row data dependency: HllcFluxX/reconstruction/RK2 only ever
// read/write within the single line being processed), so parallelizing the
// j loop is safe as long as each thread has its own Workspace (m_workspacesX
// is sized to omp_get_max_threads() in Init(), see its comment there) --
// no shared mutable state between threads.
void Euler2D::SweepX(std::vector<Cons2D>& grid, double dt) const {
    #pragma omp parallel for
    for (int j = 0; j < m_ny; ++j) {
        Workspace& ws = m_workspacesX[static_cast<size_t>(omp_get_thread_num())];
        std::vector<Cons2D>& row = ws.interior;
        const double y = m_yMin + (static_cast<double>(j) + 0.5) * m_dy;
        const Cons2D inflowCons = ToCons2D(InflowStateAt(y), m_gamma);
        for (int i = 0; i < m_nx; ++i) row[static_cast<size_t>(i)] = grid[static_cast<size_t>(j * m_nx + i)];
        AdvanceLineRK2(row, dt, m_dx, HllcFluxX, m_bcLeft, m_bcRight, inflowCons, BoundaryAxis::X, ws, ws.result);
        for (int i = 0; i < m_nx; ++i) grid[static_cast<size_t>(j * m_nx + i)] = ws.result[static_cast<size_t>(i)];
    }
}

// Same independence argument as SweepX, over columns instead of rows.
void Euler2D::SweepY(std::vector<Cons2D>& grid, double dt) const {
    #pragma omp parallel for
    for (int i = 0; i < m_nx; ++i) {
        Workspace& ws = m_workspacesY[static_cast<size_t>(omp_get_thread_num())];
        std::vector<Cons2D>& col = ws.interior;
        const double x = m_xMin + (static_cast<double>(i) + 0.5) * m_dx;
        const Cons2D inflowCons = ToCons2D(InflowStateAt(x), m_gamma);
        for (int j = 0; j < m_ny; ++j) col[static_cast<size_t>(j)] = grid[static_cast<size_t>(j * m_nx + i)];
        AdvanceLineRK2(col, dt, m_dy, HllcFluxY, m_bcBottom, m_bcTop, inflowCons, BoundaryAxis::Y, ws, ws.result);
        for (int j = 0; j < m_ny; ++j) grid[static_cast<size_t>(j * m_nx + i)] = ws.result[static_cast<size_t>(j)];
    }
}

void Euler2D::DiffuseX(std::vector<Cons2D>& grid, double dt) const {
    #pragma omp parallel for
    for (int j = 0; j < m_ny; ++j) {
        Workspace& ws = m_workspacesX[static_cast<size_t>(omp_get_thread_num())];
        std::vector<Cons2D>& padded = ws.padded;
        const double y = m_yMin + (static_cast<double>(j) + 0.5) * m_dy;
        const Cons2D inflowCons = ToCons2D(InflowStateAt(y), m_gamma);
        for (int i = 0; i < m_nx; ++i) padded[static_cast<size_t>(i + kGhost)] = grid[static_cast<size_t>(j * m_nx + i)];
        ApplyBoundaryLine(padded, m_bcLeft, m_bcRight, inflowCons, BoundaryAxis::X);
        LineDiffuse(padded, m_dx, ws, ws.delta);
        for (int i = 0; i < m_nx; ++i) {
            Cons2D& c = grid[static_cast<size_t>(j * m_nx + i)];
            c.momX += dt * ws.delta[static_cast<size_t>(i)].momX;
            c.momY += dt * ws.delta[static_cast<size_t>(i)].momY;
            c.energy += dt * ws.delta[static_cast<size_t>(i)].energy;
            c.rhoTracer += dt * ws.delta[static_cast<size_t>(i)].rhoTracer;
        }
    }
}

void Euler2D::DiffuseY(std::vector<Cons2D>& grid, double dt) const {
    #pragma omp parallel for
    for (int i = 0; i < m_nx; ++i) {
        Workspace& ws = m_workspacesY[static_cast<size_t>(omp_get_thread_num())];
        std::vector<Cons2D>& padded = ws.padded;
        const double x = m_xMin + (static_cast<double>(i) + 0.5) * m_dx;
        const Cons2D inflowCons = ToCons2D(InflowStateAt(x), m_gamma);
        for (int j = 0; j < m_ny; ++j) padded[static_cast<size_t>(j + kGhost)] = grid[static_cast<size_t>(j * m_nx + i)];
        ApplyBoundaryLine(padded, m_bcBottom, m_bcTop, inflowCons, BoundaryAxis::Y);
        LineDiffuse(padded, m_dy, ws, ws.delta);
        for (int j = 0; j < m_ny; ++j) {
            Cons2D& c = grid[static_cast<size_t>(j * m_nx + i)];
            c.momX += dt * ws.delta[static_cast<size_t>(j)].momX;
            c.momY += dt * ws.delta[static_cast<size_t>(j)].momY;
            c.energy += dt * ws.delta[static_cast<size_t>(j)].energy;
            c.rhoTracer += dt * ws.delta[static_cast<size_t>(j)].rhoTracer;
        }
    }
}

void Euler2D::ApplyBodyForce(double dt) {
    if (m_bodyForceX == 0.0) return;
    const int total = static_cast<int>(m_u.size());
    #pragma omp parallel for
    for (int idx = 0; idx < total; ++idx) {
        Cons2D& c = m_u[static_cast<size_t>(idx)];
        const double u = c.momX / c.rho;
        c.energy += m_bodyForceX * u * dt; // work done by the force (explicit, using pre-update u)
        c.momX += m_bodyForceX * dt;
    }
}

void Euler2D::SetObstacleMask(const std::function<bool(double x, double y)>& isSolid) {
    m_obstacleMask.assign(static_cast<size_t>(m_nx * m_ny), 0);
    for (int j = 0; j < m_ny; ++j) {
        const double y = m_yMin + (static_cast<double>(j) + 0.5) * m_dy;
        for (int i = 0; i < m_nx; ++i) {
            const double x = m_xMin + (static_cast<double>(i) + 0.5) * m_dx;
            m_obstacleMask[static_cast<size_t>(j * m_nx + i)] = isSolid(x, y) ? 1 : 0;
        }
    }
}

// Immersed-boundary obstacle: the mu->infinity (equivalently, penalization
// time->0) limit of Brinkman penalization -- a standard, simple technique
// for imposing a no-slip solid region on a Cartesian grid without changing
// the flux/Riemann-solver machinery at all (see Euler2D.hpp's SetObstacleMask
// comment). Forcing velocity to exactly zero every step, rather than
// relaxing it over a finite penalization time, is the hard/instantaneous
// limit of that technique -- simpler to implement and tune (no extra
// timescale to choose), at the cost of a less smooth transition at the
// obstacle's (already jagged, Cartesian-staircase) boundary. Density is
// left alone (mass isn't created or destroyed by this); energy is reduced
// by exactly the kinetic energy being removed, so pressure -- and hence
// the force the obstacle exerts on the surrounding fluid via ordinary
// pressure-gradient physics -- is preserved rather than spiking from a
// sudden kinetic-to-internal-energy conversion this isn't meant to model.
void Euler2D::ApplyObstacleMask() {
    if (m_obstacleMask.empty()) return;
    const int total = static_cast<int>(m_u.size());
    #pragma omp parallel for
    for (int idx = 0; idx < total; ++idx) {
        if (!m_obstacleMask[static_cast<size_t>(idx)]) continue;
        Cons2D& c = m_u[static_cast<size_t>(idx)];
        const double kinetic = 0.5 * (c.momX * c.momX + c.momY * c.momY) / c.rho;
        c.energy -= kinetic;
        c.momX = 0.0;
        c.momY = 0.0;
        c.rhoTracer = 0.0; // no fluid (hence no dye) actually occupies a solid cell
    }
}

void Euler2D::Step(double dt) {
    // Strang splitting: symmetric X-half / Y-full / X-half sequence, 2nd
    // order in time for a 2nd-order-accurate 1D sub-integrator (see
    // Euler2D.hpp's class comment).
    SweepX(m_u, 0.5 * dt);
    SweepY(m_u, dt);
    SweepX(m_u, 0.5 * dt);

    if (m_mu > 0.0 || m_conductivity > 0.0 || m_tracerDiffusivity > 0.0) {
        // Explicit diffusion has a parabolic stability limit (dt <~
        // dx^2/(2*coeff)), generally tighter than the hyperbolic CFL limit
        // dt was chosen from (see CompressibleSim2D.cpp) once viscosity is
        // resolved on a fine grid -- sub-cycle so correctness never depends
        // on the caller's dt happening to already satisfy it.
        const double minDx2 = std::min(m_dx * m_dx, m_dy * m_dy);
        const double diffCoeff = std::max({m_mu, m_conductivity, m_tracerDiffusivity});
        const double dtDiffMax = 0.4 * minDx2 / diffCoeff;
        const int nSub = std::max(1, static_cast<int>(std::ceil(dt / dtDiffMax)));
        const double subDt = dt / nSub;
        for (int s = 0; s < nSub; ++s) {
            DiffuseX(m_u, subDt);
            DiffuseY(m_u, subDt);
        }
    }
    ApplyBodyForce(dt);
    ApplyObstacleMask();
}

double Euler2D::MaxWaveSpeedX() const {
    double maxSpeed = 0.0;
    for (const Cons2D& c : m_u) {
        const Prim2D p = ToPrim2D(c, m_gamma);
        const double cs = std::sqrt(m_gamma * p.p / p.rho);
        maxSpeed = std::max(maxSpeed, std::abs(p.u) + cs);
    }
    return maxSpeed;
}

double Euler2D::MaxWaveSpeedY() const {
    double maxSpeed = 0.0;
    for (const Cons2D& c : m_u) {
        const Prim2D p = ToPrim2D(c, m_gamma);
        const double cs = std::sqrt(m_gamma * p.p / p.rho);
        maxSpeed = std::max(maxSpeed, std::abs(p.v) + cs);
    }
    return maxSpeed;
}

Prim2D Euler2D::PrimAt(int i, int j) const {
    return ToPrim2D(m_u[static_cast<size_t>(j * m_nx + i)], m_gamma);
}

double Euler2D::TotalMass() const {
    double sum = 0.0;
    for (const Cons2D& c : m_u) sum += c.rho;
    return sum * m_dx * m_dy;
}

double Euler2D::TotalEnergy() const {
    double sum = 0.0;
    for (const Cons2D& c : m_u) sum += c.energy;
    return sum * m_dx * m_dy;
}

} // namespace cf
