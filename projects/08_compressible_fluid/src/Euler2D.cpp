#include "Euler2D.hpp"

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
    return p;
}

Cons2D ToCons2D(const Prim2D& p, double gamma) {
    Cons2D c;
    c.rho = p.rho;
    c.momX = p.rho * p.u;
    c.momY = p.rho * p.v;
    c.energy = p.p / (gamma - 1.0) + 0.5 * p.rho * (p.u * p.u + p.v * p.v);
    return c;
}

Cons2D FluxX2D(const Prim2D& p, double gamma) {
    const Cons2D c = ToCons2D(p, gamma);
    Cons2D f;
    f.rho = c.momX;
    f.momX = c.momX * p.u + p.p;
    f.momY = c.momY * p.u;
    f.energy = p.u * (c.energy + p.p);
    return f;
}

Cons2D FluxY2D(const Prim2D& p, double gamma) {
    const Cons2D c = ToCons2D(p, gamma);
    Cons2D f;
    f.rho = c.momY;
    f.momX = c.momX * p.v;
    f.momY = c.momY * p.v + p.p;
    f.energy = p.v * (c.energy + p.p);
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
        Cons2D f;
        f.rho = fL.rho + sL * (uStar.rho - uL.rho);
        f.momX = fL.momX + sL * (uStar.momX - uL.momX);
        f.momY = fL.momY + sL * (uStar.momY - uL.momY);
        f.energy = fL.energy + sL * (uStar.energy - uL.energy);
        return f;
    }
    const double coef = right.rho * (sR - right.u) / (sR - sStar);
    Cons2D uStar;
    uStar.rho = coef;
    uStar.momX = coef * sStar;
    uStar.momY = coef * right.v;
    uStar.energy = coef * (uR.energy / right.rho + (sStar - right.u) * (sStar + right.p / (right.rho * (sR - right.u))));
    Cons2D f;
    f.rho = fR.rho + sR * (uStar.rho - uR.rho);
    f.momX = fR.momX + sR * (uStar.momX - uR.momX);
    f.momY = fR.momY + sR * (uStar.momY - uR.momY);
    f.energy = fR.energy + sR * (uStar.energy - uR.energy);
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
        Cons2D f;
        f.rho = fL.rho + sL * (uStar.rho - uL.rho);
        f.momX = fL.momX + sL * (uStar.momX - uL.momX);
        f.momY = fL.momY + sL * (uStar.momY - uL.momY);
        f.energy = fL.energy + sL * (uStar.energy - uL.energy);
        return f;
    }
    const double coef = right.rho * (sR - right.v) / (sR - sStar);
    Cons2D uStar;
    uStar.rho = coef;
    uStar.momY = coef * sStar;
    uStar.momX = coef * right.u;
    uStar.energy = coef * (uR.energy / right.rho + (sStar - right.v) * (sStar + right.p / (right.rho * (sR - right.v))));
    Cons2D f;
    f.rho = fR.rho + sR * (uStar.rho - uR.rho);
    f.momX = fR.momX + sR * (uStar.momX - uR.momX);
    f.momY = fR.momY + sR * (uStar.momY - uR.momY);
    f.energy = fR.energy + sR * (uStar.energy - uR.energy);
    return f;
}

namespace {

double Minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}

constexpr int kGhost = 2;
using HllcFluxFn = Cons2D (*)(const Prim2D&, const Prim2D&, double);
using ToPrimFn = Prim2D (*)(const Cons2D&, double);

// Zero-gradient (transmissive/outflow) ghost cells, same convention as
// Euler1D::ApplyBoundary.
void ApplyBoundaryLine(std::vector<Cons2D>& u) {
    const int total = static_cast<int>(u.size());
    for (int g = 0; g < kGhost; ++g) {
        u[static_cast<size_t>(g)] = u[static_cast<size_t>(kGhost)];
        u[static_cast<size_t>(total - 1 - g)] = u[static_cast<size_t>(total - 1 - kGhost)];
    }
}

// -dF/dx (or -dF/dy) per interior cell of a ghost-padded line -- identical
// structure to Euler1D::Rhs, generalized over Cons2D and a direction-specific
// HLLC flux.
std::vector<Cons2D> LineRhs(const std::vector<Cons2D>& u, double gamma, double dx, HllcFluxFn hllc, ToPrimFn toPrim) {
    const int total = static_cast<int>(u.size());
    const int n = total - 2 * kGhost;
    std::vector<Prim2D> prim(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) prim[static_cast<size_t>(i)] = toPrim(u[static_cast<size_t>(i)], gamma);

    std::vector<Prim2D> faceL(static_cast<size_t>(total)), faceR(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) faceL[static_cast<size_t>(i)] = faceR[static_cast<size_t>(i)] = prim[static_cast<size_t>(i)];
    for (int i = 1; i < total - 1; ++i) {
        const Prim2D& pm = prim[static_cast<size_t>(i - 1)];
        const Prim2D& p0 = prim[static_cast<size_t>(i)];
        const Prim2D& pp = prim[static_cast<size_t>(i + 1)];
        const double dRho = Minmod(p0.rho - pm.rho, pp.rho - p0.rho);
        const double dU = Minmod(p0.u - pm.u, pp.u - p0.u);
        const double dV = Minmod(p0.v - pm.v, pp.v - p0.v);
        const double dP = Minmod(p0.p - pm.p, pp.p - p0.p);
        faceL[static_cast<size_t>(i)] = Prim2D{p0.rho - 0.5 * dRho, p0.u - 0.5 * dU, p0.v - 0.5 * dV, p0.p - 0.5 * dP};
        faceR[static_cast<size_t>(i)] = Prim2D{p0.rho + 0.5 * dRho, p0.u + 0.5 * dU, p0.v + 0.5 * dV, p0.p + 0.5 * dP};
    }

    std::vector<Cons2D> dudt(static_cast<size_t>(n));
    Cons2D fluxPrev = hllc(faceR[static_cast<size_t>(kGhost - 1)], faceL[static_cast<size_t>(kGhost)], gamma);
    for (int i = 0; i < n; ++i) {
        const int gi = i + kGhost;
        const Cons2D fluxNext = hllc(faceR[static_cast<size_t>(gi)], faceL[static_cast<size_t>(gi + 1)], gamma);
        Cons2D& d = dudt[static_cast<size_t>(i)];
        d.rho = -(fluxNext.rho - fluxPrev.rho) / dx;
        d.momX = -(fluxNext.momX - fluxPrev.momX) / dx;
        d.momY = -(fluxNext.momY - fluxPrev.momY) / dx;
        d.energy = -(fluxNext.energy - fluxPrev.energy) / dx;
        fluxPrev = fluxNext;
    }
    return dudt;
}

// RK2 (Heun) update of one interior line (a grid row or column) over a
// fractional step dt, given its own ghost-padded flux operator -- the same
// algorithm as Euler1D::Step, generalized over direction (HllcFluxX/Y).
std::vector<Cons2D> AdvanceLineRK2(const std::vector<Cons2D>& interior, double dt, double dx, double gamma,
                                    HllcFluxFn hllc) {
    const int n = static_cast<int>(interior.size());
    std::vector<Cons2D> padded(static_cast<size_t>(n + 2 * kGhost));
    for (int i = 0; i < n; ++i) padded[static_cast<size_t>(i + kGhost)] = interior[static_cast<size_t>(i)];
    ApplyBoundaryLine(padded);

    const std::vector<Cons2D> k1 = LineRhs(padded, gamma, dx, hllc, ToPrim2D);

    std::vector<Cons2D> stage1 = padded;
    for (int i = 0; i < n; ++i) {
        Cons2D& c = stage1[static_cast<size_t>(i + kGhost)];
        const Cons2D& d = k1[static_cast<size_t>(i)];
        c.rho += dt * d.rho;
        c.momX += dt * d.momX;
        c.momY += dt * d.momY;
        c.energy += dt * d.energy;
    }
    ApplyBoundaryLine(stage1);
    const std::vector<Cons2D> k2 = LineRhs(stage1, gamma, dx, hllc, ToPrim2D);

    std::vector<Cons2D> result(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const Cons2D& a = padded[static_cast<size_t>(i + kGhost)];
        const Cons2D& b = stage1[static_cast<size_t>(i + kGhost)];
        const Cons2D& d2 = k2[static_cast<size_t>(i)];
        Cons2D& out = result[static_cast<size_t>(i)];
        out.rho = 0.5 * (a.rho + b.rho + dt * d2.rho);
        out.momX = 0.5 * (a.momX + b.momX + dt * d2.momX);
        out.momY = 0.5 * (a.momY + b.momY + dt * d2.momY);
        out.energy = 0.5 * (a.energy + b.energy + dt * d2.energy);
    }
    return result;
}

} // namespace

void Euler2D::Init(int nx, int ny, double xMin, double xMax, double yMin, double yMax, double gamma) {
    m_nx = nx;
    m_ny = ny;
    m_xMin = xMin;
    m_yMin = yMin;
    m_dx = (xMax - xMin) / static_cast<double>(nx);
    m_dy = (yMax - yMin) / static_cast<double>(ny);
    m_gamma = gamma;
    m_u.assign(static_cast<size_t>(nx * ny), Cons2D{});
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

void Euler2D::SweepX(std::vector<Cons2D>& grid, double dt) const {
    std::vector<Cons2D> row(static_cast<size_t>(m_nx));
    for (int j = 0; j < m_ny; ++j) {
        for (int i = 0; i < m_nx; ++i) row[static_cast<size_t>(i)] = grid[static_cast<size_t>(j * m_nx + i)];
        const std::vector<Cons2D> updated = AdvanceLineRK2(row, dt, m_dx, m_gamma, HllcFluxX);
        for (int i = 0; i < m_nx; ++i) grid[static_cast<size_t>(j * m_nx + i)] = updated[static_cast<size_t>(i)];
    }
}

void Euler2D::SweepY(std::vector<Cons2D>& grid, double dt) const {
    std::vector<Cons2D> col(static_cast<size_t>(m_ny));
    for (int i = 0; i < m_nx; ++i) {
        for (int j = 0; j < m_ny; ++j) col[static_cast<size_t>(j)] = grid[static_cast<size_t>(j * m_nx + i)];
        const std::vector<Cons2D> updated = AdvanceLineRK2(col, dt, m_dy, m_gamma, HllcFluxY);
        for (int j = 0; j < m_ny; ++j) grid[static_cast<size_t>(j * m_nx + i)] = updated[static_cast<size_t>(j)];
    }
}

void Euler2D::Step(double dt) {
    // Strang splitting: symmetric X-half / Y-full / X-half sequence, 2nd
    // order in time for a 2nd-order-accurate 1D sub-integrator (see
    // Euler2D.hpp's class comment).
    SweepX(m_u, 0.5 * dt);
    SweepY(m_u, dt);
    SweepX(m_u, 0.5 * dt);
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
