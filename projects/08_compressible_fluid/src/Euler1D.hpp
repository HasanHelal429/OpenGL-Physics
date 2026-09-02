#pragma once

#include <vector>

namespace cf {

// Conserved variables: mass density, momentum density, total energy density.
struct Cons {
    double rho = 0.0;
    double mom = 0.0;
    double energy = 0.0;
};

// Primitive variables: density, velocity, pressure.
struct Prim {
    double rho = 0.0;
    double u = 0.0;
    double p = 0.0;
};

Prim ToPrim(const Cons& c, double gamma);
Cons ToCons(const Prim& p, double gamma);
Cons Flux(const Prim& p, double gamma);
// HLLC (Toro ch. 10) approximate Riemann solver flux across the interface
// between state L (left) and R (right). Davis wave-speed estimates for
// SL/SR -- simple and robust, not the tightest bound, but standard and
// sufficient once HLLC's resolved contact wave is already doing the work
// HLLE can't (a sharp, non-smeared contact discontinuity).
Cons HllcFlux(const Prim& left, const Prim& right, double gamma);

// 1D compressible Euler equations, ideal Gamma-law gas: conservative
// finite-volume, MinMod-limited piecewise-linear reconstruction, HLLC flux,
// RK2 (Heun) method-of-lines time integration -- the flat-spacetime,
// non-relativistic limit of 07_grhd's Valencia formulation (see this
// project's docs/SIMULATION.md): same reconstruction/flux/time-integration
// structure, but with a closed-form primitive recovery (no con2prim Newton
// iteration -- the ideal-gas EOS inverts directly once there's no Lorentz
// factor to solve for) and no geometric source terms (flat metric).
class Euler1D {
public:
    void Init(int n, double xMin, double xMax, double gamma);
    void SetRiemannIC(double x0, const Prim& left, const Prim& right);

    void Step(double dt);

    // max(|u|+c) over the interior -- used once at Configure() to pick a
    // fixed dt via CFL; see CompressibleSim.cpp for why fixed rather than
    // recomputed every step.
    double MaxWaveSpeed() const;

    int Count() const { return m_n; }
    double Dx() const { return m_dx; }
    double Gamma() const { return m_gamma; }
    Prim PrimAt(int i) const;

    double TotalMass() const;
    double TotalMomentum() const;
    double TotalEnergy() const;

private:
    // Zero-gradient (transmissive/outflow) ghost cells -- correct boundary
    // treatment for a Riemann problem whose waves never reach the domain
    // edge within the validated run time.
    void ApplyBoundary(std::vector<Cons>& u) const;
    // -dF/dx per interior cell, from a fully ghost-padded state array.
    std::vector<Cons> Rhs(const std::vector<Cons>& u) const;

    int m_n = 0;
    static constexpr int kGhost = 2; // 2 layers/side: enough for MinMod slopes at the outermost interior face
    double m_xMin = 0.0, m_xMax = 1.0, m_dx = 1.0, m_gamma = 1.4;
    std::vector<Cons> m_u; // size m_n + 2*kGhost
};

} // namespace cf
