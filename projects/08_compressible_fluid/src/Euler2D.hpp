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

// 2D compressible Euler via Strang dimensional splitting: each timestep is
// [X half-step][Y full-step][X half-step] (LeVeque, "Finite Volume Methods
// for Hyperbolic Problems", ch. 17), where each fractional step reuses
// Euler1D's exact MinMod+HLLC+RK2 line update (AdvanceLineRK2 below) applied
// row-by-row (X) or column-by-column (Y) -- so a genuinely 1D flow
// (velocity aligned with a grid axis) reduces to running Euler1D on every
// line, and the only new code here is bookkeeping the two sweep directions
// and the transverse-momentum HLLC extension above.
class Euler2D {
public:
    void Init(int nx, int ny, double xMin, double xMax, double yMin, double yMax, double gamma);
    void SetInitialCondition(const std::function<Prim2D(double x, double y)>& f);

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

    int m_nx = 0, m_ny = 0;
    double m_xMin = 0.0, m_yMin = 0.0, m_dx = 1.0, m_dy = 1.0, m_gamma = 1.4;
    std::vector<Cons2D> m_u; // row-major, size nx*ny, index = j*nx + i
};

} // namespace cf
