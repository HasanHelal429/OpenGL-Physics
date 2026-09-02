#pragma once

#include <functional>
#include <vector>

namespace cf {

// Conserved variables: mass density, x/y momentum density, total energy density.
struct Cons2D {
    double rho = 0.0;
    double momX = 0.0;
    double momY = 0.0;
    double energy = 0.0;
};

// Primitive variables: density, x/y velocity, pressure.
struct Prim2D {
    double rho = 0.0;
    double u = 0.0;
    double v = 0.0;
    double p = 0.0;
};

Prim2D ToPrim2D(const Cons2D& c, double gamma);
Cons2D ToCons2D(const Prim2D& p, double gamma);
Cons2D FluxX2D(const Prim2D& p, double gamma);
Cons2D FluxY2D(const Prim2D& p, double gamma);
// Direction-specific HLLC flux, generalizing Euler1D's HllcFlux with a
// passively-advected transverse momentum component (Toro sec. 10.4's
// "additional/passive variable" extension: the transverse velocity is
// carried unchanged into whichever star state -- left or right of the
// contact -- the sampled point falls in, since only the normal-direction
// fields jump across the two non-linear (shock/rarefaction) waves).
Cons2D HllcFluxX(const Prim2D& left, const Prim2D& right, double gamma);
Cons2D HllcFluxY(const Prim2D& left, const Prim2D& right, double gamma);

// Ghost-cell boundary treatment for one line (row or column). Outflow is
// the zero-gradient condition used by Phases 1-3 (default, unchanged
// behavior); Periodic and NoSlipReflective are new in Phase 4, needed for
// channel (Poiseuille) flow: periodic in the streamwise direction,
// reflective (u=v=0 at the wall, enforced by negating ghost momentum --
// density/energy are copied unchanged, since energy's kinetic term is
// invariant under a velocity sign flip) cross-channel.
enum class WallBC { Outflow, Periodic, NoSlipReflective };

// 2D compressible Navier-Stokes via Strang dimensional splitting: each
// timestep is [X half-step][Y full-step][X half-step] (LeVeque, "Finite
// Volume Methods for Hyperbolic Problems", ch. 17) of the inviscid (Euler)
// flux, where each fractional step reuses Euler1D's exact MinMod+HLLC+RK2
// line update (AdvanceLineRK2 in Euler2D.cpp) applied row-by-row (X) or
// column-by-column (Y). When viscosity is set (SetViscosity), an explicit
// diffusion sub-step follows, using a SIMPLIFIED Newtonian stress tensor
// (shear terms only, tau_xx=2*mu*du/dx, tau_yy=2*mu*dv/dy,
// tau_xy=mu*(du/dy+dv/dx) -- Stokes' hypothesis' bulk-viscosity correction
// dropped) further approximated by keeping only the "same-line" derivative
// of each stress term (e.g. d(tau_xy)/dy's du/dy part, not its dv/dx part)
// so every viscous term stays a row-local or column-local Laplacian, exactly
// like the inviscid sweeps. Both omissions are exact (introduce zero error)
// for a flow with no streamwise variation and no cross-channel velocity --
// precisely this project's Poiseuille validation (see docs/SIMULATION.md) --
// but would need the full tensor for a general viscous flow.
class Euler2D {
public:
    void Init(int nx, int ny, double xMin, double xMax, double yMin, double yMax, double gamma);
    void SetInitialCondition(const std::function<Prim2D(double x, double y)>& f);
    void SetBoundaryConditions(WallBC bcX, WallBC bcY) { m_bcX = bcX; m_bcY = bcY; }
    // mu: dynamic viscosity. conductivity: Fourier heat-conduction
    // coefficient (energy equation gets +conductivity*Laplacian(T),
    // T=p/rho -- viscous heating is not modeled, see the class comment).
    void SetViscosity(double mu, double conductivity) {
        m_mu = mu;
        m_conductivity = conductivity;
    }
    // Constant force per unit volume added to the x-momentum (and, as
    // f*u*dt, to the energy) each step -- the standard way to drive a
    // periodic channel flow without needing an actual streamwise pressure
    // drop (which a periodic domain can't otherwise sustain).
    void SetBodyForceX(double forcePerVolume) { m_bodyForceX = forcePerVolume; }

    void Step(double dt);

    double MaxWaveSpeedX() const; // max(|u|+c) over the grid
    double MaxWaveSpeedY() const; // max(|v|+c) over the grid

    int Nx() const { return m_nx; }
    int Ny() const { return m_ny; }
    double Dx() const { return m_dx; }
    double Dy() const { return m_dy; }
    double Gamma() const { return m_gamma; }
    Prim2D PrimAt(int i, int j) const;

    double TotalMass() const;
    double TotalEnergy() const;

private:
    void SweepX(std::vector<Cons2D>& grid, double dt) const;
    void SweepY(std::vector<Cons2D>& grid, double dt) const;
    void DiffuseX(std::vector<Cons2D>& grid, double dt) const;
    void DiffuseY(std::vector<Cons2D>& grid, double dt) const;
    void ApplyBodyForce(double dt);

    int m_nx = 0, m_ny = 0;
    double m_xMin = 0.0, m_yMin = 0.0, m_dx = 1.0, m_dy = 1.0, m_gamma = 1.4;
    WallBC m_bcX = WallBC::Outflow, m_bcY = WallBC::Outflow;
    double m_mu = 0.0, m_conductivity = 0.0, m_bodyForceX = 0.0;
    std::vector<Cons2D> m_u; // row-major, size nx*ny, index = j*nx + i
};

} // namespace cf
