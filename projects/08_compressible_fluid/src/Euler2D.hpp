#pragma once

#include <cstdint>
#include <functional>
#include <vector>

namespace cf {

// Conserved variables: mass density, x/y momentum density, total energy
// density, and (rho*tracer) -- a passive scalar's conserved form, see
// Prim2D::tracer. Named `rhoTracer` (not the shorter `c`) specifically to
// avoid colliding with the sound-speed local variable named `c` throughout
// Euler2D.cpp's wave-speed code.
struct Cons2D {
    double rho = 0.0;
    double momX = 0.0;
    double momY = 0.0;
    double energy = 0.0;
    double rhoTracer = 0.0;
};

// Primitive variables: density, x/y velocity, pressure, and a passive
// scalar tracer concentration (dimensionless, dye-like -- carried by the
// flow, contributes no pressure/force of its own; see HllcFluxX/Y's
// comment for how it's advected). Defaults to 0 so every existing
// aggregate-init call site (`Prim2D{rho, u, v, p}`) keeps compiling
// unchanged, with tracer=0 everywhere unless a sim explicitly sets it.
struct Prim2D {
    double rho = 0.0;
    double u = 0.0;
    double v = 0.0;
    double p = 0.0;
    double tracer = 0.0;
};

Prim2D ToPrim2D(const Cons2D& c, double gamma);
Cons2D ToCons2D(const Prim2D& p, double gamma);
Cons2D FluxX2D(const Prim2D& p, double gamma);
Cons2D FluxY2D(const Prim2D& p, double gamma);
// Direction-specific HLLC flux, generalizing Euler1D's HllcFlux with two
// passively-advected quantities (Toro sec. 10.4's "additional/passive
// variable" extension: each is carried unchanged into whichever star
// state -- left or right of the contact -- the sampled point falls in,
// since only the normal-direction fields jump across the two non-linear
// (shock/rarefaction) waves): the transverse momentum component, and the
// scalar tracer concentration (Prim2D::tracer) -- a tracer is exactly as
// "passive" as transverse velocity in this sense, just carried as a
// concentration rather than a velocity component.
Cons2D HllcFluxX(const Prim2D& left, const Prim2D& right, double gamma);
Cons2D HllcFluxY(const Prim2D& left, const Prim2D& right, double gamma);

// Ghost-cell boundary treatment for one line (row or column). Outflow is
// the zero-gradient condition used by Phases 1-3 (default, unchanged
// behavior); Periodic and NoSlipReflective were added in Phase 4 for
// channel (Poiseuille) flow: periodic in the streamwise direction,
// reflective (u=v=0 at the wall, enforced by negating ghost momentum --
// density/energy are copied unchanged, since energy's kinetic term is
// invariant under a velocity sign flip) cross-channel. FreeSlipReflective
// and Inflow are new for external flow past an obstacle (cylinder vortex
// shedding): FreeSlipReflective negates only the velocity component
// NORMAL to the boundary (a symmetry/no-penetration wall -- used at the
// domain's top/bottom truncation, which isn't a physical wall, just where
// the domain was cut off); Inflow fixes the ghost cells to a caller-
// supplied constant state regardless of the interior (used at the domain
// inlet, where the upstream condition is prescribed, not derived).
enum class WallBC { Outflow, Periodic, NoSlipReflective, FreeSlipReflective, Inflow };

// Which velocity component a FreeSlipReflective boundary should treat as
// "normal" (negate) vs "tangential" (keep) -- X for a left/right (vertical)
// boundary, Y for a top/bottom (horizontal) one.
enum class BoundaryAxis { X, Y };

// 2D compressible Navier-Stokes via Strang dimensional splitting: each
// timestep is [X half-step][Y full-step][X half-step] (LeVeque, "Finite
// Volume Methods for Hyperbolic Problems", ch. 17) of the inviscid (Euler)
// flux, where each fractional step reuses Euler1D's exact MinMod+HLLC+RK2
// line update (AdvanceLineRK2 in Euler2D.cpp) applied row-by-row (X) or
// column-by-column (Y). When viscosity is set (SetViscosity), an explicit
// diffusion sub-step follows, adding mu*Laplacian(u) to x-momentum and
// mu*Laplacian(v) to y-momentum (each Laplacian assembled from one
// row-local d^2/dx^2 pass plus one column-local d^2/dy^2 pass, so every
// viscous term stays a 1D operation, exactly like the inviscid sweeps).
// This is the exact incompressible-limit reduction of the full Newtonian
// viscous stress divergence (dropping only Stokes' hypothesis' bulk-
// viscosity correction and the mixed d^2/dxdy cross term, which cancel a
// compensating factor of 2 in the full tensor -- see LineDiffuse's comment
// in Euler2D.cpp for the derivation, including a real bug this project hit
// by getting that cancellation wrong on the first attempt) -- exact for a
// genuinely divergence-free flow, and a good approximation at the low Mach
// numbers this project's validation targets use.
class Euler2D {
public:
    void Init(int nx, int ny, double xMin, double xMax, double yMin, double yMax, double gamma);
    void SetInitialCondition(const std::function<Prim2D(double x, double y)>& f);
    // Independent per-side boundary conditions. Periodic must be set on
    // BOTH sides of an axis together (it wraps to the far interior cell of
    // the same line, so a one-sided Periodic is meaningless) -- every other
    // combination (e.g. Inflow left / Outflow right) is the point of
    // splitting this from the old single-WallBC-per-axis API.
    void SetBoundaryConditions(WallBC left, WallBC right, WallBC bottom, WallBC top) {
        m_bcLeft = left;
        m_bcRight = right;
        m_bcBottom = bottom;
        m_bcTop = top;
    }
    // The fixed upstream state used wherever a side's WallBC is Inflow,
    // unless SetInflowProfile overrides it.
    void SetInflowState(const Prim2D& state) { m_inflowState = state; }
    // Position-dependent inflow, for e.g. an alternating dye-stripe
    // tracer pattern fed continuously at the inlet (a one-time initial-
    // condition pulse washes downstream and out of the domain long before
    // a run like cylinder shedding's is over -- a real, persistent tracer
    // visualization needs the inflow itself to keep supplying pattern).
    // `coord` is y for a left/right (X-sweep) boundary, x for a top/bottom
    // (Y-sweep) one -- whichever this side actually varies along. Once
    // set, this REPLACES the fixed state from SetInflowState wherever
    // Inflow is used, on every side (not just the one the profile was
    // designed for) -- pass an empty function (the default) to go back to
    // the fixed state.
    void SetInflowProfile(const std::function<Prim2D(double coord)>& profile) { m_inflowProfile = profile; }
    // mu: dynamic viscosity. conductivity: Fourier heat-conduction
    // coefficient (energy equation gets +conductivity*Laplacian(T),
    // T=p/rho -- viscous heating is not modeled, see the class comment).
    void SetViscosity(double mu, double conductivity) {
        m_mu = mu;
        m_conductivity = conductivity;
    }
    // Explicit tracer diffusion coefficient, mu*Laplacian(tracer) added to
    // rhoTracer -- default 0 (pure advection, sharp dye interfaces limited
    // only by the scheme's own numerical dissipation, the usual choice for
    // a flow-visualization tracer).
    void SetTracerDiffusivity(double diffusivity) { m_tracerDiffusivity = diffusivity; }
    // Constant force per unit volume added to the x-momentum (and, as
    // f*u*dt, to the energy) each step -- the standard way to drive a
    // periodic channel flow without needing an actual streamwise pressure
    // drop (which a periodic domain can't otherwise sustain).
    void SetBodyForceX(double forcePerVolume) { m_bodyForceX = forcePerVolume; }
    // Immersed-boundary obstacle via cell masking (Brinkman-penalization
    // limit, see ApplyObstacleMask's comment in Euler2D.cpp): cells where
    // isSolid(x,y) is true have their velocity forced to zero every step,
    // letting the existing viscous machinery diffuse the resulting no-slip
    // condition into the surrounding fluid -- no changes needed to the
    // Riemann solver or line-sweep structure.
    void SetObstacleMask(const std::function<bool(double x, double y)>& isSolid);

    void Step(double dt);

    double MaxWaveSpeedX() const; // max(|u|+c) over the grid
    double MaxWaveSpeedY() const; // max(|v|+c) over the grid

    int Nx() const { return m_nx; }
    int Ny() const { return m_ny; }
    double Dx() const { return m_dx; }
    double Dy() const { return m_dy; }
    double Gamma() const { return m_gamma; }
    Prim2D PrimAt(int i, int j) const;
    bool IsSolidAt(int i, int j) const { return !m_obstacleMask.empty() && m_obstacleMask[static_cast<size_t>(j * m_nx + i)] != 0; }

    double TotalMass() const;
    double TotalEnergy() const;

private:
    void SweepX(std::vector<Cons2D>& grid, double dt) const;
    void SweepY(std::vector<Cons2D>& grid, double dt) const;
    void DiffuseX(std::vector<Cons2D>& grid, double dt) const;
    void DiffuseY(std::vector<Cons2D>& grid, double dt) const;
    void ApplyBodyForce(double dt);
    void ApplyObstacleMask();
    // SetInflowState's fixed value, or SetInflowProfile's function
    // evaluated at `coord` if one was set -- see SetInflowProfile's
    // comment for what `coord` means on each side.
    Prim2D InflowStateAt(double coord) const { return m_inflowProfile ? m_inflowProfile(coord) : m_inflowState; }

    using HllcFluxFn = Cons2D (*)(const Prim2D&, const Prim2D&, double);

    // Reused per-line scratch buffers, threaded through LineRhs/AdvanceLineRK2/
    // LineDiffuse so a full sweep (every row in SweepX, every column in
    // SweepY) makes zero heap allocations after the first call. Before this,
    // each of these was a fresh std::vector allocated and freed on every
    // single row/column of every single Step() call -- roughly 11
    // allocations per line; for a typical grid and a multi-hundred-time-unit
    // run, on the order of 10^8 allocations total (measured: ~169 million
    // for the cylinder-shedding deck's validated run). See
    // docs/SIMULATION.md's efficiency section for the full derivation.
    // One Workspace per OpenMP thread once the sweeps are parallelized (not
    // yet -- currently always index 0 in m_workspacesX/m_workspacesY).
    struct Workspace {
        std::vector<Cons2D> interior; // extracted row/column, before ghost padding
        std::vector<Cons2D> padded, stage1, result, k1, k2, delta;
        std::vector<Prim2D> prim, faceL, faceR;
        void EnsureSize(int n); // n = interior line length (nx for X-direction, ny for Y)
    };
    // gamma/mu/conductivity/tracerDiffusivity come from the object's own
    // members (m_gamma etc.) rather than parameters, now that these are
    // member functions instead of free functions -- one fewer thing to keep
    // in sync at each call site.
    void LineRhs(const std::vector<Cons2D>& u, double dx, HllcFluxFn hllc, Workspace& ws,
                 std::vector<Cons2D>& dudtOut) const;
    void AdvanceLineRK2(const std::vector<Cons2D>& interior, double dt, double dx, HllcFluxFn hllc, WallBC bcLeft,
                         WallBC bcRight, const Cons2D& inflowCons, BoundaryAxis axis, Workspace& ws,
                         std::vector<Cons2D>& resultOut) const;
    void LineDiffuse(const std::vector<Cons2D>& padded, double dx, Workspace& ws,
                      std::vector<Cons2D>& deltaOut) const;

    int m_nx = 0, m_ny = 0;
    double m_xMin = 0.0, m_yMin = 0.0, m_dx = 1.0, m_dy = 1.0, m_gamma = 1.4;
    WallBC m_bcLeft = WallBC::Outflow, m_bcRight = WallBC::Outflow;
    WallBC m_bcBottom = WallBC::Outflow, m_bcTop = WallBC::Outflow;
    Prim2D m_inflowState;
    std::function<Prim2D(double)> m_inflowProfile;
    double m_mu = 0.0, m_conductivity = 0.0, m_tracerDiffusivity = 0.0, m_bodyForceX = 0.0;
    std::vector<Cons2D> m_u; // row-major, size nx*ny, index = j*nx + i
    std::vector<uint8_t> m_obstacleMask; // empty if no obstacle set; else size nx*ny, row-major like m_u
    // mutable: SweepX/SweepY/DiffuseX/DiffuseY are logically const (they
    // never modify the object's own physical state, only the caller-owned
    // `grid` argument) but need to write into scratch space -- exactly the
    // textbook case for `mutable`.
    mutable std::vector<Workspace> m_workspacesX, m_workspacesY;
};

} // namespace cf
