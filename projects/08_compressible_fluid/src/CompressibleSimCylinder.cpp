#include "CompressibleSimCylinder.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <algorithm>
#include <cmath>

namespace cf {

void CompressibleSimCylinder::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);

    m_diameter = deck.GetDouble("cylinder.diameter", 1.0);
    const double reynolds = deck.GetDouble("cylinder.reynolds", 100.0);
    const double uInf = deck.GetDouble("cylinder.u_inf", 1.0);
    const double blockage = deck.GetDouble("cylinder.blockage", 0.125); // D/H
    const double upstreamD = deck.GetDouble("cylinder.upstream_d", 5.0);
    const double downstreamD = deck.GetDouble("cylinder.downstream_d", 20.0);
    const double cellsPerD = deck.GetDouble("cylinder.cells_per_d", 8.0);
    const double mach = deck.GetDouble("physics.mach", 0.2);
    const double gamma = deck.GetDouble("physics.gamma", 1.4);
    const double rho0 = deck.GetDouble("physics.rho0", 1.0);

    const double height = m_diameter / blockage;
    const double length = (upstreamD + downstreamD) * m_diameter;
    const double dx = m_diameter / cellsPerD;
    const int nx = static_cast<int>(std::round(length / dx));
    const int ny = static_cast<int>(std::round(height / dx));

    const double soundSpeed = uInf / mach;
    const double p0 = rho0 * soundSpeed * soundSpeed / gamma;
    const double mu = rho0 * uInf * m_diameter / reynolds; // nu=U*D/Re, mu=rho0*nu

    m_inflow = Prim2D{rho0, uInf, 0.0, p0};

    m_solver.Init(nx, ny, 0.0, length, 0.0, height, gamma);
    // Left: prescribed upstream state. Right: zero-gradient (the flow
    // leaves the domain far enough downstream -- 20D -- that the wake's
    // periodic structure should have mostly decayed before reaching it,
    // though some outflow-boundary reflection is an inherent limitation of
    // this simple BC, not eliminated here). Top/bottom: free-slip -- these
    // aren't physical walls, just where the domain was truncated, so only
    // no-penetration (not no-slip) is physically correct there.
    m_solver.SetBoundaryConditions(WallBC::Inflow, WallBC::Outflow, WallBC::FreeSlipReflective,
                                    WallBC::FreeSlipReflective);
    m_solver.SetInflowState(m_inflow);
    m_solver.SetViscosity(mu, 0.0);

    // A passive dye tracer, fed continuously as alternating bands at the
    // inflow (not just a one-time initial-condition pulse -- that would
    // wash downstream and out of the domain long before a run this long is
    // over, see Reset()'s comment on the same issue for the symmetry-
    // breaking perturbation). Density/velocity/pressure stay the uniform
    // inflow state; only the tracer concentration varies with y, striping
    // the incoming flow so the shed vortices' roll-up and mixing become
    // directly visible downstream (tools/make_movie.py renders this
    // alongside vorticity). stripe_width_d<=0 disables the tracer entirely
    // (an inflow profile is only installed if stripes are requested).
    const double stripeWidthD = deck.GetDouble("cylinder.tracer_stripe_width_d", 1.0);
    if (stripeWidthD > 0.0) {
        const Prim2D inflowBase = m_inflow;
        const double stripeWidth = stripeWidthD * m_diameter;
        m_solver.SetInflowProfile([=](double y) {
            Prim2D p = inflowBase;
            const int band = static_cast<int>(std::floor(y / stripeWidth));
            p.tracer = (band % 2 == 0) ? 1.0 : 0.0;
            return p;
        });
    }

    m_cylX = upstreamD * m_diameter;
    // A small permanent offset off the exact centerline, not just a
    // transient initial-condition perturbation (see Reset()'s comment for
    // why a one-time IC blip alone proved insufficient): this domain,
    // mask, and BC set are otherwise symmetric to floating-point
    // precision, so without a persistent geometric asymmetry there is
    // nothing to keep seeding the (linearly unstable, but only unstable
    // relative to a non-symmetric perturbation) shedding instability once
    // any transient disturbance has convected away downstream.
    const double yOffsetD = deck.GetDouble("cylinder.y_offset_d", 0.02);
    m_cylY = height / 2.0 + yOffsetD * m_diameter;
    const double cylRadius = m_diameter / 2.0;
    const double cylX = m_cylX, cylY = m_cylY;
    m_solver.SetObstacleMask([=](double x, double y) {
        const double dxp = x - cylX, dyp = y - cylY;
        return dxp * dxp + dyp * dyp <= cylRadius * cylRadius;
    });

    m_perturbAmplitude = deck.GetDouble("cylinder.perturb_amplitude", 0.02);
    Reset();

    // Probe a few diameters downstream on the centerline -- v(t) there
    // oscillates at the shedding frequency because vortices shed
    // alternately from the top/bottom of the cylinder, deflecting the wake
    // first one way then the other (same diagnostic MAC_Grid_Solver's
    // notebook uses).
    const double probeDistD = deck.GetDouble("cylinder.probe_distance_d", 4.0);
    const double probeX = cylX + probeDistD * m_diameter;
    m_probeI = std::clamp(static_cast<int>((probeX - 0.0) / dx), 0, nx - 1);
    m_probeJ = ny / 2;

    m_cfl = deck.GetDouble("time.cfl", 0.3);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 20);
    // Fixed dt from the INITIAL uniform-inflow wave speed, with a
    // deliberately lower CFL number (0.3, vs. 0.4 elsewhere in this
    // project) as a safety margin: unlike the Riemann-problem/Poiseuille/
    // Taylor-Green targets, this flow's peak velocity isn't provably
    // bounded by the initial condition (potential-flow theory puts the
    // cylinder's peak surface speed at up to 2x U_inf). At this Mach
    // number the SOUND speed dominates the max-wave-speed estimate by a
    // wide margin (soundSpeed/uInf = 1/mach = 5 at the default mach=0.2),
    // so even a 2x velocity excursion changes the true max wave speed by
    // only ~15% -- comfortably inside the margin the lower CFL number
    // buys back.
    m_dt = m_cfl / (m_solver.MaxWaveSpeedX() / m_solver.Dx() + m_solver.MaxWaveSpeedY() / m_solver.Dy());
}

void CompressibleSimCylinder::Reset() {
    const Prim2D inflow = m_inflow;
    const double cylX = m_cylX, cylY = m_cylY, diameter = m_diameter, amplitude = m_perturbAmplitude;
    // This one-time antisymmetric velocity bump (Gaussian, radius ~1D,
    // amplitude a small fraction of U_inf, opposite sign above/below the
    // offset centerline) is a SECOND, belt-and-suspenders symmetry-breaker
    // on top of Configure()'s permanent cylinder y-offset -- a real flow
    // never starts from perfect rest either, so this seeds the instability
    // faster than waiting for it to grow from the offset's much smaller
    // steady-state asymmetry alone. Confirmed necessary in practice: a
    // first attempt at this deck used NEITHER (perfectly symmetric
    // geometry, uniform IC) and the wake stayed exactly symmetric for the
    // entire simulated run, never shedding at all -- a perfectly
    // symmetric simulation is the artificial case; a real cylinder always
    // sheds because real flows always carry some asymmetric disturbance
    // (free-stream turbulence, structural vibration).
    m_solver.SetInitialCondition([=](double x, double y) {
        const double dxp = x - cylX, dyp = y - cylY;
        const double r2 = dxp * dxp + dyp * dyp;
        const double sigma = diameter;
        const double sign = dyp >= 0.0 ? 1.0 : -1.0;
        const double vPerturb = amplitude * inflow.u * sign * std::exp(-r2 / (2.0 * sigma * sigma));
        return Prim2D{inflow.rho, inflow.u, vPerturb, inflow.p};
    });
}

void CompressibleSimCylinder::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSimCylinder::Snapshot(fw::OutputWriter& writer) {
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    std::vector<float> rho(static_cast<size_t>(nx * ny)), u(static_cast<size_t>(nx * ny)),
        v(static_cast<size_t>(nx * ny)), p(static_cast<size_t>(nx * ny)), tracer(static_cast<size_t>(nx * ny));
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Prim2D pr = m_solver.PrimAt(i, j);
            const size_t idx = static_cast<size_t>(j * nx + i);
            rho[idx] = static_cast<float>(pr.rho);
            u[idx] = static_cast<float>(pr.u);
            v[idx] = static_cast<float>(pr.v);
            p[idx] = static_cast<float>(pr.p);
            tracer[idx] = static_cast<float>(pr.tracer);
        }
    }
    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("u", u.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("tracer", tracer.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteScalar("mass_total", m_solver.TotalMass());
    writer.WriteScalar("probe_v", m_solver.PrimAt(m_probeI, m_probeJ).v);
}

fw::SimInfo CompressibleSimCylinder::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Nx();
    info.gridNy = m_solver.Ny();
    info.lx = m_solver.Nx() * m_solver.Dx();
    info.ly = m_solver.Ny() * m_solver.Dy();
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "u", "v", "p", "tracer"};
    info.diagnostics = {"mass_total", "probe_v"};
    return info;
}

} // namespace cf
