#include "Euler1D.hpp"

#include <algorithm>
#include <cmath>

namespace cf {

Prim ToPrim(const Cons& c, double gamma) {
    Prim p;
    p.rho = c.rho;
    p.u = c.mom / c.rho;
    const double kinetic = 0.5 * c.mom * c.mom / c.rho;
    p.p = (gamma - 1.0) * (c.energy - kinetic);
    return p;
}

Cons ToCons(const Prim& p, double gamma) {
    Cons c;
    c.rho = p.rho;
    c.mom = p.rho * p.u;
    c.energy = p.p / (gamma - 1.0) + 0.5 * p.rho * p.u * p.u;
    return c;
}

Cons Flux(const Prim& p, double gamma) {
    const Cons c = ToCons(p, gamma);
    Cons f;
    f.rho = c.mom;
    f.mom = c.mom * p.u + p.p;
    f.energy = p.u * (c.energy + p.p);
    return f;
}

namespace {
Cons HllcStarFlux(const Prim& k, const Cons& uk, const Cons& fk, double sk, double sStar) {
    const double coef = k.rho * (sk - k.u) / (sk - sStar);
    Cons uStar;
    uStar.rho = coef;
    uStar.mom = coef * sStar;
    uStar.energy = coef * (uk.energy / k.rho + (sStar - k.u) * (sStar + k.p / (k.rho * (sk - k.u))));
    Cons f;
    f.rho = fk.rho + sk * (uStar.rho - uk.rho);
    f.mom = fk.mom + sk * (uStar.mom - uk.mom);
    f.energy = fk.energy + sk * (uStar.energy - uk.energy);
    return f;
}
} // namespace

Cons HllcFlux(const Prim& left, const Prim& right, double gamma) {
    const double cL = std::sqrt(gamma * left.p / left.rho);
    const double cR = std::sqrt(gamma * right.p / right.rho);
    const double sL = std::min(left.u - cL, right.u - cR);
    const double sR = std::max(left.u + cL, right.u + cR);

    const Cons fL = Flux(left, gamma);
    if (sL >= 0.0) return fL;
    const Cons fR = Flux(right, gamma);
    if (sR <= 0.0) return fR;

    const double sStar = (right.p - left.p + left.rho * left.u * (sL - left.u) - right.rho * right.u * (sR - right.u)) /
                          (left.rho * (sL - left.u) - right.rho * (sR - right.u));

    if (sStar >= 0.0) return HllcStarFlux(left, ToCons(left, gamma), fL, sL, sStar);
    return HllcStarFlux(right, ToCons(right, gamma), fR, sR, sStar);
}

void Euler1D::Init(int n, double xMin, double xMax, double gamma) {
    m_n = n;
    m_xMin = xMin;
    m_xMax = xMax;
    m_gamma = gamma;
    m_dx = (xMax - xMin) / static_cast<double>(n);
    m_u.assign(static_cast<size_t>(n + 2 * kGhost), Cons{});
}

void Euler1D::SetRiemannIC(double x0, const Prim& left, const Prim& right) {
    for (int i = 0; i < m_n; ++i) {
        const double x = m_xMin + (static_cast<double>(i) + 0.5) * m_dx;
        m_u[static_cast<size_t>(i + kGhost)] = ToCons(x < x0 ? left : right, m_gamma);
    }
}

void Euler1D::ApplyBoundary(std::vector<Cons>& u) const {
    const int total = static_cast<int>(u.size());
    for (int g = 0; g < kGhost; ++g) {
        u[static_cast<size_t>(g)] = u[static_cast<size_t>(kGhost)];
        u[static_cast<size_t>(total - 1 - g)] = u[static_cast<size_t>(total - 1 - kGhost)];
    }
}

namespace {
double Minmod(double a, double b) {
    if (a * b <= 0.0) return 0.0;
    return (std::abs(a) < std::abs(b)) ? a : b;
}
} // namespace

std::vector<Cons> Euler1D::Rhs(const std::vector<Cons>& u) const {
    const int total = static_cast<int>(u.size());
    std::vector<Prim> prim(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) prim[static_cast<size_t>(i)] = ToPrim(u[static_cast<size_t>(i)], m_gamma);

    // Left/right cell-edge extrapolated primitive states, MinMod-limited.
    // Cells 0 and total-1 have no far neighbor and are never referenced
    // (see the ng=2 comment in Euler1D.hpp), so they're left at their
    // (unused) default first-order value.
    std::vector<Prim> faceL(static_cast<size_t>(total)), faceR(static_cast<size_t>(total));
    for (int i = 0; i < total; ++i) faceL[static_cast<size_t>(i)] = faceR[static_cast<size_t>(i)] = prim[static_cast<size_t>(i)];
    for (int i = 1; i < total - 1; ++i) {
        const Prim& pm = prim[static_cast<size_t>(i - 1)];
        const Prim& p0 = prim[static_cast<size_t>(i)];
        const Prim& pp = prim[static_cast<size_t>(i + 1)];
        const double dRho = Minmod(p0.rho - pm.rho, pp.rho - p0.rho);
        const double dU = Minmod(p0.u - pm.u, pp.u - p0.u);
        const double dP = Minmod(p0.p - pm.p, pp.p - p0.p);
        faceL[static_cast<size_t>(i)] = Prim{p0.rho - 0.5 * dRho, p0.u - 0.5 * dU, p0.p - 0.5 * dP};
        faceR[static_cast<size_t>(i)] = Prim{p0.rho + 0.5 * dRho, p0.u + 0.5 * dU, p0.p + 0.5 * dP};
    }

    std::vector<Cons> dudt(static_cast<size_t>(m_n));
    Cons fluxPrev = HllcFlux(faceR[static_cast<size_t>(kGhost - 1)], faceL[static_cast<size_t>(kGhost)], m_gamma);
    for (int i = 0; i < m_n; ++i) {
        const int gi = i + kGhost;
        const Cons fluxNext = HllcFlux(faceR[static_cast<size_t>(gi)], faceL[static_cast<size_t>(gi + 1)], m_gamma);
        Cons& d = dudt[static_cast<size_t>(i)];
        d.rho = -(fluxNext.rho - fluxPrev.rho) / m_dx;
        d.mom = -(fluxNext.mom - fluxPrev.mom) / m_dx;
        d.energy = -(fluxNext.energy - fluxPrev.energy) / m_dx;
        fluxPrev = fluxNext;
    }
    return dudt;
}

void Euler1D::Step(double dt) {
    const std::vector<Cons> u0 = m_u; // pre-step interior state, kept for Heun's averaging below

    ApplyBoundary(m_u);
    const std::vector<Cons> k1 = Rhs(m_u);

    std::vector<Cons> u1 = m_u;
    for (int i = 0; i < m_n; ++i) {
        Cons& c = u1[static_cast<size_t>(i + kGhost)];
        c.rho += dt * k1[static_cast<size_t>(i)].rho;
        c.mom += dt * k1[static_cast<size_t>(i)].mom;
        c.energy += dt * k1[static_cast<size_t>(i)].energy;
    }
    ApplyBoundary(u1);
    const std::vector<Cons> k2 = Rhs(u1);

    for (int i = 0; i < m_n; ++i) {
        const Cons& a = u0[static_cast<size_t>(i + kGhost)];
        const Cons& b = u1[static_cast<size_t>(i + kGhost)];
        Cons& out = m_u[static_cast<size_t>(i + kGhost)];
        out.rho = 0.5 * (a.rho + b.rho + dt * k2[static_cast<size_t>(i)].rho);
        out.mom = 0.5 * (a.mom + b.mom + dt * k2[static_cast<size_t>(i)].mom);
        out.energy = 0.5 * (a.energy + b.energy + dt * k2[static_cast<size_t>(i)].energy);
    }
}

double Euler1D::MaxWaveSpeed() const {
    double maxSpeed = 0.0;
    for (int i = 0; i < m_n; ++i) {
        const Prim p = ToPrim(m_u[static_cast<size_t>(i + kGhost)], m_gamma);
        const double c = std::sqrt(m_gamma * p.p / p.rho);
        maxSpeed = std::max(maxSpeed, std::abs(p.u) + c);
    }
    return maxSpeed;
}

Prim Euler1D::PrimAt(int i) const {
    return ToPrim(m_u[static_cast<size_t>(i + kGhost)], m_gamma);
}

double Euler1D::TotalMass() const {
    double sum = 0.0;
    for (int i = 0; i < m_n; ++i) sum += m_u[static_cast<size_t>(i + kGhost)].rho * m_dx;
    return sum;
}

double Euler1D::TotalMomentum() const {
    double sum = 0.0;
    for (int i = 0; i < m_n; ++i) sum += m_u[static_cast<size_t>(i + kGhost)].mom * m_dx;
    return sum;
}

double Euler1D::TotalEnergy() const {
    double sum = 0.0;
    for (int i = 0; i < m_n; ++i) sum += m_u[static_cast<size_t>(i + kGhost)].energy * m_dx;
    return sum;
}

} // namespace cf
