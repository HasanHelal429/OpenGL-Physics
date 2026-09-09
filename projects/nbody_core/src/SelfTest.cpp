#include "ngrav/SelfTest.hpp"

#include "ngrav/Solvers.hpp"
#include "ngrav/System.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace ngrav {

namespace {

void Check(bool& ok, const char* name, double value, double tol) {
    const bool good = std::isfinite(value) && std::abs(value) <= tol;
    std::printf("  %-40s %12.4e  (tol %.1e)  %s\n", name, value, tol, good ? "ok" : "WRONG");
    if (!good) ok = false;
}

template <int D>
System<D> MakeSystem(const std::vector<Vec<D>>& pos, const std::vector<Vec<D>>& vel, const std::vector<double>& mass) {
    SoA<D> s;
    s.Resize(pos.size());
    for (std::size_t i = 0; i < pos.size(); ++i) {
        s.SetPos(i, pos[i]);
        s.SetVel(i, vel[i]);
        s.m[i] = mass[i];
    }
    System<D> sys;
    sys.SetParticles(std::move(s));
    return sys;
}

template <int D>
void BHvsDirect(bool& ok, const char* label) {
    std::printf("[%s]\n", label);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    const int n = 1000;
    SoA<D> in;
    in.Resize(n);
    for (int i = 0; i < n; ++i) {
        in.x[i] = u(rng);
        in.y[i] = u(rng);
        if constexpr (D == 3) in.z[i] = u(rng);
        in.m[i] = 0.5 + 0.5 * (u(rng) + 1.0);
    }
    StepParams sp;
    sp.G = 1.0;
    sp.soft = Softening::Plummer(0.02);
    sp.mac.theta = 0.0;

    SoA<D> aD, aBH;
    const PosMassView<D> v = ViewOf(in);
    ComputeAccelDirect<D>(v, sp, aD);
    ComputeAccelBarnesHut<D>(v, sp, aBH);
    double maxRel = 0.0;
    for (int i = 0; i < n; ++i) {
        const double rx = aD.ax[i], ry = aD.ay[i], rz = (D == 3) ? aD.az[i] : 0.0;
        const double ex = aBH.ax[i] - rx, ey = aBH.ay[i] - ry, ez = (D == 3) ? (aBH.az[i] - rz) : 0.0;
        maxRel = std::max(maxRel, std::sqrt(ex * ex + ey * ey + ez * ez) /
                                      std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30));
    }
    Check(ok, "max per-particle |da|/|a| (theta=0)", maxRel, 1e-12);
}

} // namespace

bool CoreSelfTest3D() {
    bool ok = true;
    const double G = 1.0, m = 1.0;

    // Two-body Kepler.
    {
        std::printf("[kepler 3D]\n");
        const double sep = 1.0;
        const double vRel = 0.8 * std::sqrt(G * 2.0 * m / sep);
        const double period = 2.0 * M_PI * std::sqrt(sep * sep * sep / (G * 2.0 * m));
        System<3> sys =
            MakeSystem<3>({{-sep / 2, 0, 0}, {sep / 2, 0, 0}}, {{0, -vRel / 2, 0}, {0, vRel / 2, 0}}, {m, m});
        StepParams sp;
        sp.G = G;
        sp.soft = Softening::Plummer(1e-6);
        sp.solver = Solver::Direct;
        sp.dt = period / 2000.0;
        sys.Prime(sp);
        const double E0 = sys.TotalEnergy(sp);
        const double L0 = glm::length(sys.AngularMomentum());
        double rMin = 1e30, rMax = 0, Emin = E0, Emax = E0, Lmin = L0, Lmax = L0;
        for (int k = 0; k < 4 * 2000; ++k) {
            sys.Step(sp);
            const double r = glm::length(sys.State().Pos(1) - sys.State().Pos(0));
            rMin = std::min(rMin, r);
            rMax = std::max(rMax, r);
            if (k % 50 == 0) {
                const double E = sys.TotalEnergy(sp), L = glm::length(sys.AngularMomentum());
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
                Lmin = std::min(Lmin, L);
                Lmax = std::max(Lmax, L);
            }
        }
        const double a = 1.0 / (1.0 + 0.36);
        Check(ok, "periapsis rel err", (rMin - a * (1.0 - 0.36)) / (a * (1.0 - 0.36)), 1e-4);
        Check(ok, "apoapsis rel err", (rMax - 1.0), 1e-4);
        Check(ok, "energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
        Check(ok, "|L| drift", (Lmax - Lmin) / (0.5 * (Lmax + Lmin)), 1e-6);
    }

    // Lagrange equilateral triangle (side L = 1, Omega^2 = 3Gm/L^3).
    {
        std::printf("[lagrange triangle 3D]\n");
        const double L = 1.0, omega = std::sqrt(3.0 * G * m / (L * L * L)), R = L / std::sqrt(3.0);
        std::vector<Vec<3>> pos(3), vel(3);
        for (int k = 0; k < 3; ++k) {
            const double ang = 2.0 * M_PI * k / 3.0;
            pos[k] = {R * std::cos(ang), R * std::sin(ang), 0};
            vel[k] = {-omega * R * std::sin(ang), omega * R * std::cos(ang), 0};
        }
        System<3> sys = MakeSystem<3>(pos, vel, {m, m, m});
        StepParams sp;
        sp.G = G;
        sp.soft = Softening::Plummer(1e-6);
        sp.solver = Solver::Direct;
        sp.dt = (2.0 * M_PI / omega) / 2000.0;
        sys.Prime(sp);
        const double E0 = sys.TotalEnergy(sp);
        double shapeErr = 0, Emin = E0, Emax = E0;
        for (int k = 0; k < 5 * 2000; ++k) {
            sys.Step(sp);
            const auto& st = sys.State();
            for (double d : {glm::length(st.Pos(1) - st.Pos(0)), glm::length(st.Pos(2) - st.Pos(1)),
                             glm::length(st.Pos(0) - st.Pos(2))})
                shapeErr = std::max(shapeErr, std::abs(d - L) / L);
            if (k % 50 == 0) {
                const double E = sys.TotalEnergy(sp);
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
            }
        }
        Check(ok, "triangle shape deviation", shapeErr, 1e-5);
        Check(ok, "energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 2e-8);
    }

    BHvsDirect<3>(ok, "barnes-hut vs direct 3D");

    // --- compact-support spline softening (3D) ---------------------------
    {
        std::printf("[spline softening 3D]\n");
        const double h = 0.1;
        // (a) force-exactness: for r >= H = 2h, spline softening must match
        // unsoftened Newtonian gravity to machine precision.
        double maxRel = 0.0;
        std::mt19937 rng(777);
        std::uniform_real_distribution<double> ur(2.0 * h, 5.0 * h);
        for (int t = 0; t < 200; ++t) {
            const double r = ur(rng);
            SoA<3> in;
            in.Resize(2);
            in.x[1] = r;
            in.m[0] = in.m[1] = 1.0;
            StepParams sp;
            sp.G = 1.0;
            sp.soft = Softening::Spline(h);
            SoA<3> out;
            ComputeAccelDirect<3>(ViewOf(in), sp, out);
            const double exact = 1.0 / (r * r); // G=m=1, attractive: a on particle 0 points toward +x
            const double got = out.ax[0];
            maxRel = std::max(maxRel, std::abs(got - exact) / exact);
        }
        Check(ok, "spline force == Newtonian for r>=2h (max rel err)", maxRel, 1e-12);

        // (b) energy conservation: an eccentric two-body orbit with spline
        // softening instead of Plummer -- if AccumPairSpline (the force) and
        // TotalEnergy's SplinePotential3D (the potential) were inconsistent,
        // this would drift; if consistent, it conserves just as well as the
        // Plummer Kepler test.
        const double sep = 1.0, e = 0.5;
        const double vRel = std::sqrt(2.0 * (2.0 / sep - (1.0 + e) / sep));
        const double a = sep / (1.0 + e);
        const double period = 2.0 * M_PI * std::sqrt(a * a * a / 2.0);
        System<3> orbit = MakeSystem<3>({{-sep / 2, 0, 0}, {sep / 2, 0, 0}}, {{0, -vRel / 2, 0}, {0, vRel / 2, 0}},
                                        {1.0, 1.0});
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Spline(0.02);
        sp.solver = Solver::Direct;
        sp.dt = period / 3000.0;
        orbit.Prime(sp);
        const double E0 = orbit.TotalEnergy(sp);
        double Emin = E0, Emax = E0;
        for (int k = 0; k < 4 * 3000; ++k) {
            orbit.Step(sp);
            if (k % 50 == 0) {
                const double E = orbit.TotalEnergy(sp);
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
            }
        }
        Check(ok, "spline-softened orbit energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
    }

    // --- quadrupole Barnes-Hut walk (3D only) -----------------------------
    {
        std::printf("[quadrupole barnes-hut 3D]\n");
        std::mt19937 rng(2024);
        std::uniform_real_distribution<double> uni(0.0, 1.0), gau(-1.0, 1.0);
        const int n = 4000;
        SoA<3> in;
        in.Resize(n);
        for (int i = 0; i < n; ++i) {
            const double r = std::cbrt(uni(rng));
            const double cosT = 2.0 * uni(rng) - 1.0, sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
            const double phi = 2.0 * M_PI * uni(rng);
            in.x[static_cast<std::size_t>(i)] = r * sinT * std::cos(phi);
            in.y[static_cast<std::size_t>(i)] = r * sinT * std::sin(phi);
            in.z[static_cast<std::size_t>(i)] = r * cosT;
            in.m[static_cast<std::size_t>(i)] = 1.0 / n;
        }
        const PosMassView<3> v = ViewOf(in);
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Plummer(0.01);
        sp.mac.theta = 0.5;

        SoA<3> aDirect;
        ComputeAccelDirect<3>(v, sp, aDirect);

        AdaptiveTree<3> treeMono(v); // no ComputeQuadrupoles() -> monopole-only walk
        SoA<3> aMono;
        ComputeAccelBarnesHut<3>(v, sp, treeMono, {}, aMono);

        AdaptiveTree<3> treeQuad(v);
        treeQuad.ComputeQuadrupoles(v);
        SoA<3> aQuad;
        ComputeAccelBarnesHut<3>(v, sp, treeQuad, {}, aQuad);

        auto maxRelErr = [&](const SoA<3>& test) {
            double m = 0.0;
            for (int i = 0; i < n; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                const double rx = aDirect.ax[ii], ry = aDirect.ay[ii], rz = aDirect.az[ii];
                const double ex = test.ax[ii] - rx, ey = test.ay[ii] - ry, ez = test.az[ii] - rz;
                m = std::max(m, std::sqrt(ex * ex + ey * ey + ez * ez) /
                                    std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30));
            }
            return m;
        };
        const double maxMono = maxRelErr(aMono);
        const double maxQuad = maxRelErr(aQuad);
        std::printf("  theta=0.5: monopole-only max err=%.3e   mono+quad max err=%.3e   improvement=%.2fx\n", maxMono,
                    maxQuad, maxMono / std::max(maxQuad, 1e-30));
        // max(0, 3 - ratio) <= 0 encodes "ratio >= 3" via Check's abs<=tol form.
        Check(ok, "quadrupole gives >=3x lower max force err at theta=0.5", std::max(0.0, 3.0 - maxMono / maxQuad),
              0.0);
    }

    return ok;
}

bool CoreSelfTest2D() {
    bool ok = true;
    const double G = 1.0, m = 1.0;

    // 2D two-body: precessing orbit -> assert conservation only.
    {
        std::printf("[kepler 2D]\n");
        const double sep = 1.0, vCirc = std::sqrt(G * 2.0 * m), vRel = 0.8 * vCirc;
        System<2> sys = MakeSystem<2>({{-sep / 2, 0}, {sep / 2, 0}}, {{0, -vRel / 2}, {0, vRel / 2}}, {m, m});
        StepParams sp;
        sp.G = G;
        sp.soft = Softening::Plummer(1e-6);
        sp.solver = Solver::Direct;
        sp.dt = (2.0 * M_PI * sep / vCirc) / 2000.0;
        sys.Prime(sp);
        const double E0 = sys.TotalEnergy(sp), L0 = sys.AngularMomentum();
        double Emin = E0, Emax = E0, Lmin = L0, Lmax = L0;
        for (int k = 0; k < 4 * 2000; ++k) {
            sys.Step(sp);
            if (k % 50 == 0) {
                const double E = sys.TotalEnergy(sp), Lz = sys.AngularMomentum();
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
                Lmin = std::min(Lmin, Lz);
                Lmax = std::max(Lmax, Lz);
            }
        }
        Check(ok, "energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
        Check(ok, "L drift", (Lmax - Lmin) / std::abs(0.5 * (Lmax + Lmin)), 1e-6);
    }

    // 2D Lagrange (Omega = sqrt(Gm)/L, vertices at radius L in 2D convention).
    {
        std::printf("[lagrange triangle 2D]\n");
        const double L = 1.0, omega = std::sqrt(G * m) / L;
        std::vector<Vec<2>> pos(3), vel(3);
        for (int k = 0; k < 3; ++k) {
            const double ang = 2.0 * M_PI * k / 3.0;
            pos[k] = {L * std::cos(ang), L * std::sin(ang)};
            vel[k] = {-omega * L * std::sin(ang), omega * L * std::cos(ang)};
        }
        System<2> sys = MakeSystem<2>(pos, vel, {m, m, m});
        StepParams sp;
        sp.G = G;
        sp.soft = Softening::Plummer(1e-6);
        sp.solver = Solver::Direct;
        sp.dt = (2.0 * M_PI / omega) / 4000.0;
        sys.Prime(sp);
        const double E0 = sys.TotalEnergy(sp);
        double Emin = E0, Emax = E0;
        for (int k = 0; k < 3 * 4000; ++k) {
            sys.Step(sp);
            if (k % 50 == 0) {
                const double E = sys.TotalEnergy(sp);
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
            }
        }
        Check(ok, "energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
    }

    BHvsDirect<2>(ok, "barnes-hut vs direct 2D");

    // --- compact-support spline softening (2D) ---------------------------
    {
        std::printf("[spline softening 2D]\n");
        const double h = 0.1;
        double maxRel = 0.0;
        std::mt19937 rng(778);
        std::uniform_real_distribution<double> ur(2.0 * h, 5.0 * h);
        for (int t = 0; t < 200; ++t) {
            const double r = ur(rng);
            SoA<2> in;
            in.Resize(2);
            in.x[1] = r;
            in.m[0] = in.m[1] = 1.0;
            StepParams sp;
            sp.G = 1.0;
            sp.soft = Softening::Spline(h);
            SoA<2> out;
            ComputeAccelDirect<2>(ViewOf(in), sp, out);
            const double exact = 1.0 / r; // 2D: a ~ Gm/r, attractive toward +x
            maxRel = std::max(maxRel, std::abs(out.ax[0] - exact) / exact);
        }
        Check(ok, "spline force == Newtonian for r>=2h (max rel err)", maxRel, 1e-12);

        // Energy conservation for a spline-softened 2D orbit (precessing,
        // as for the plain Kepler2D test -- assert conservation only).
        const double vCirc = std::sqrt(2.0), vRel = 0.8 * vCirc;
        System<2> orbit = MakeSystem<2>({{-0.5, 0}, {0.5, 0}}, {{0, -vRel / 2}, {0, vRel / 2}}, {1.0, 1.0});
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Spline(0.02);
        sp.solver = Solver::Direct;
        sp.dt = (2.0 * M_PI / vCirc) / 3000.0;
        orbit.Prime(sp);
        const double E0 = orbit.TotalEnergy(sp);
        double Emin = E0, Emax = E0;
        for (int k = 0; k < 4 * 3000; ++k) {
            orbit.Step(sp);
            if (k % 50 == 0) {
                const double E = orbit.TotalEnergy(sp);
                Emin = std::min(Emin, E);
                Emax = std::max(Emax, E);
            }
        }
        Check(ok, "spline-softened 2D orbit energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
    }

    return ok;
}

namespace {

// Two equal masses, apoapsis r0 = 1, eccentricity e. v_rel from vis-viva.
struct KeplerRun {
    double energyDrift = 0.0;
    long forceEvals = 0;
    bool finite = true;
};

KeplerRun RunEccentric(double e, double dt, bool adaptive, double eta, int periods) {
    const double G = 1.0, m = 1.0, r0 = 1.0;
    const double a = r0 / (1.0 + e);
    const double vRel = std::sqrt(G * 2.0 * m * (2.0 / r0 - 1.0 / a));
    const double period = 2.0 * M_PI * std::sqrt(a * a * a / (G * 2.0 * m));

    System<3> sys =
        MakeSystem<3>({{-r0 / 2, 0, 0}, {r0 / 2, 0, 0}}, {{0, -vRel / 2, 0}, {0, vRel / 2, 0}}, {m, m});
    StepParams sp;
    sp.G = G;
    sp.soft = Softening::Plummer(1e-4);
    sp.solver = Solver::Direct;
    sp.dt = dt;
    sp.adaptive = adaptive;
    sp.eta = eta;
    sys.Prime(sp);
    sys.ResetForceEvals();

    const double E0 = sys.TotalEnergy(sp);
    double Emin = E0, Emax = E0;
    double t = 0.0;
    const double tEnd = periods * period;
    // TotalEnergy() (and, internally, Direct) each open an OpenMP parallel
    // region -- negligible per call, but at up to ~6e5 steps for the finest
    // fixed-dt calibration run, sampling every step turns that overhead into
    // the dominant cost. Every other step is plenty to catch the true
    // min/max of a smoothly-oscillating orbital energy.
    constexpr int kSampleEvery = 32;
    int k = 0;
    int guard = 0;
    while (t < tEnd && guard++ < 20000000) {
        t += sys.Step(sp);
        if ((k++ % kSampleEvery) == 0) {
            const double E = sys.TotalEnergy(sp);
            if (!std::isfinite(E)) return {1e30, sys.ForceEvals(), false};
            Emin = std::min(Emin, E);
            Emax = std::max(Emax, E);
        }
    }
    KeplerRun r;
    r.energyDrift = (Emax - Emin) / std::abs(0.5 * (Emax + Emin));
    r.forceEvals = sys.ForceEvals();
    return r;
}

} // namespace

bool CoreDtSelfTest() {
    bool ok = true;
    std::printf("[adaptive global dt]\n");

    // Fixed-dt path unchanged: a mild orbit reproduces the CoreSelfTest3D number.
    const KeplerRun fixedMild = RunEccentric(0.36, 2.0 * M_PI * std::sqrt(std::pow(1.0 / 1.36, 3) / 2.0) / 2000.0,
                                             false, 0.0, 4);
    Check(ok, "fixed-dt mild-orbit energy drift", fixedMild.energyDrift, 1e-4);

    // Highly eccentric (e = 0.9): global dt must resolve the periapsis, so a
    // uniform step that's cheap away from periapsis is wasteful there, and
    // one fine enough for periapsis is wasteful everywhere else -- exactly
    // what adaptive dt = eta*min sqrt(eps/|a|) is for.
    const double aEcc = 1.0 / 1.9;
    const double Tecc = 2.0 * M_PI * std::sqrt(aEcc * aEcc * aEcc / 2.0);
    const KeplerRun adaptiveRun = RunEccentric(0.9, Tecc / 600.0, true, 0.015, 3);
    std::printf("  adaptive(eta=0.015): drift=%.3e  evals=%ld\n", adaptiveRun.energyDrift, adaptiveRun.forceEvals);
    Check(ok, "adaptive e=0.9 energy drift", adaptiveRun.energyDrift, 1e-3);

    // How many evals a FIXED dt needs to match that accuracy: this leapfrog
    // is 2nd order, so drift ~ dt^2 (confirmed empirically -- see the plan
    // doc's Phase 3 entry) once dt is fine enough to be in the asymptotic
    // regime. Calibrate the exponent from two runs, extrapolate the fixed
    // div needed to reach adaptiveRun's drift, then verify with one run at
    // that div rather than trusting the extrapolation blindly.
    const KeplerRun cal1 = RunEccentric(0.9, Tecc / 4800.0, false, 0.0, 3);
    const KeplerRun cal2 = RunEccentric(0.9, Tecc / 9600.0, false, 0.0, 3);
    const double p = std::log(cal1.energyDrift / cal2.energyDrift) / std::log(2.0); // dt halves -> drift / 2^p
    const double divNeeded = 9600.0 * std::pow(cal2.energyDrift / adaptiveRun.energyDrift, 1.0 / p);
    const KeplerRun fixedMatched = RunEccentric(0.9, Tecc / divNeeded, false, 0.0, 3);

    std::printf("  fixed dt calibration: order p=%.2f, extrapolated div=%.0f -> verified drift=%.3e  evals=%ld\n", p,
                divNeeded, fixedMatched.energyDrift, fixedMatched.forceEvals);

    const double evalRatio = static_cast<double>(fixedMatched.forceEvals) / std::max(1L, adaptiveRun.forceEvals);
    std::printf("  fixed/adaptive force-eval ratio at matched accuracy = %.2f\n", evalRatio);
    Check(ok, "extrapolation sanity: verified drift within 50% of target", fixedMatched.energyDrift / adaptiveRun.energyDrift - 1.0,
          0.5);
    // evalRatio >= 2 required; encode as max(0, 2-evalRatio) <= 0 so Check's
    // abs(value)<=tol form reads correctly (a shortfall shows as a positive,
    // failing value; met-or-exceeded collapses to exactly 0).
    Check(ok, "adaptive uses >=2x fewer evals at matched accuracy", std::max(0.0, 2.0 - evalRatio), 0.0);

    return ok;
}

} // namespace ngrav
