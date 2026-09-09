#include "ngrav/SelfTest.hpp"

#include "ngrav/Integrator.hpp"
#include "ngrav/MutualFmm.hpp"
#include "ngrav/Solvers.hpp"
#include "ngrav/System.hpp"
#include "ngrav/ic/Hernquist.hpp"
#include "ngrav/ic/King.hpp"
#include "ngrav/ic/Plummer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <utility>
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

bool CoreFmmSelfTest() {
    bool ok = true;
    std::printf("[mutual dual-tree FMM 3D]\n");

    // theta=0 exactness: forces every pair to resolve via exact near-field
    // (see MutualFmm.cpp -- a leaf that fails the MAC and can't split
    // further loops the full cross-product of its members against the
    // partner leaf's), so this should match Direct to machine precision.
    {
        std::mt19937 rng(555);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        const int n = 300;
        SoA<3> in;
        in.Resize(n);
        for (int i = 0; i < n; ++i) {
            in.x[i] = u(rng);
            in.y[i] = u(rng);
            in.z[i] = u(rng);
            in.m[i] = 0.5 + 0.5 * (u(rng) + 1.0);
        }
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Plummer(0.02);
        sp.mac.theta = 0.0;
        SoA<3> aD, aF;
        const PosMassView<3> v = ViewOf(in);
        ComputeAccelDirect<3>(v, sp, aD);
        ComputeAccelMutualFmm(v, sp, aF);
        double maxRel = 0.0;
        for (int i = 0; i < n; ++i) {
            const double rx = aD.ax[i], ry = aD.ay[i], rz = aD.az[i];
            const double ex = aF.ax[i] - rx, ey = aF.ay[i] - ry, ez = aF.az[i] - rz;
            maxRel = std::max(maxRel, std::sqrt(ex * ex + ey * ey + ez * ez) /
                                          std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30));
        }
        Check(ok, "theta=0 max rel err vs Direct", maxRel, 1e-11);
    }

    // Momentum conservation: the headline result. A pair-interaction energy
    // that depends only on the separation vector makes F_T = -F_S exact by
    // construction (see MutualFmm.cpp's header comment) -- verified at
    // several theta, not just one, since the M2L fraction (and hence
    // anything that COULD leak momentum) grows with theta.
    {
        ic::PlummerParams pp;
        pp.n = 3000;
        pp.seed = 42;
        SoA<3> s = ic::Plummer<3>(pp);
        const PosMassView<3> v = ViewOf(s);
        double worst = 0.0;
        for (double theta : {0.2, 0.4, 0.6, 0.8}) {
            StepParams sp;
            sp.G = 1.0;
            sp.soft = Softening::Plummer(0.02);
            sp.mac.theta = theta;
            SoA<3> a;
            ComputeAccelMutualFmm(v, sp, a);
            double px = 0, py = 0, pz = 0, wsum = 0;
            for (int i = 0; i < pp.n; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                px += s.m[ii] * a.ax[ii];
                py += s.m[ii] * a.ay[ii];
                pz += s.m[ii] * a.az[ii];
                wsum += s.m[ii] * std::sqrt(a.ax[ii] * a.ax[ii] + a.ay[ii] * a.ay[ii] + a.az[ii] * a.az[ii]);
            }
            worst = std::max(worst, std::sqrt(px * px + py * py + pz * pz) / wsum);
        }
        std::printf("  momentum |sum m*a|/sum m|a|, worst over theta in {0.2,0.4,0.6,0.8}: %.3e\n", worst);
        Check(ok, "momentum conservation (worst-case theta)", worst, 1e-10);
    }

    // Accuracy vs Direct, and the honest characterization of how it depends
    // on theta: this order-2 (monopole+quadrupole source, linear-shift
    // local) FMM needs a noticeably tighter theta than Barnes-Hut to reach
    // comparable accuracy -- a known, structural property of cluster-
    // cluster methods at this local-expansion order (BH evaluates the exact
    // field at each query particle's own position; FMM evaluates once per
    // *cluster* and Taylor-shifts, which costs accuracy for cheaper scaling
    // and, crucially, exact momentum conservation). Measured on a 5000-body
    // Plummer sphere: mean rel err is ~2e-4 by theta=0.2, and effectively
    // exact (near-field-dominated) by theta<=0.15; at BH's typical
    // theta=0.5 operating point FMM's mean error (~5e-2) is worse than BH's
    // own (~5e-4) -- a real, documented tradeoff, not a numeric target to
    // paper over. This check only asserts the *achievable* regime.
    {
        ic::PlummerParams pp;
        pp.n = 4000;
        pp.seed = 7;
        SoA<3> s = ic::Plummer<3>(pp);
        const PosMassView<3> v = ViewOf(s);
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Plummer(0.02);
        SoA<3> aDirect;
        ComputeAccelDirect<3>(v, sp, aDirect);
        sp.mac.theta = 0.2;
        SoA<3> aFmm;
        ComputeAccelMutualFmm(v, sp, aFmm);
        double sumRel = 0.0;
        for (int i = 0; i < pp.n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double rx = aDirect.ax[ii], ry = aDirect.ay[ii], rz = aDirect.az[ii];
            const double ex = aFmm.ax[ii] - rx, ey = aFmm.ay[ii] - ry, ez = aFmm.az[ii] - rz;
            sumRel += std::sqrt(ex * ex + ey * ey + ez * ez) / std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30);
        }
        Check(ok, "theta=0.2 mean rel err vs Direct", sumRel / pp.n, 1e-3);
    }

    // Phase 8: OpenMP-task traversal, verified against the still-serial
    // oracle (forceSerial=true) at N above the parallel threshold (4096).
    // Not just "close enough" -- the two paths must visit the exact same
    // set of M2L/near pairs (order can differ across threads, the count
    // can't), and the resulting forces should agree to floating-point
    // reassociation error, several orders tighter than any physics
    // tolerance elsewhere in this file.
    {
        ic::PlummerParams pp;
        pp.n = 6000; // > kFmmParallelThreshold, so this exercises the new path
        pp.seed = 99;
        SoA<3> s = ic::Plummer<3>(pp);
        const PosMassView<3> v = ViewOf(s);
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Plummer(0.02);
        sp.mac.theta = 0.5;
        SoA<3> aSerial, aParallel;
        MutualFmmStats statsSerial, statsParallel;
        ComputeAccelMutualFmm(v, sp, aSerial, &statsSerial, /*forceSerial=*/true);
        ComputeAccelMutualFmm(v, sp, aParallel, &statsParallel, /*forceSerial=*/false);
        double maxRel = 0.0;
        for (int i = 0; i < pp.n; ++i) {
            const std::size_t ii = static_cast<std::size_t>(i);
            const double rx = aSerial.ax[ii], ry = aSerial.ay[ii], rz = aSerial.az[ii];
            const double ex = aParallel.ax[ii] - rx, ey = aParallel.ay[ii] - ry, ez = aParallel.az[ii] - rz;
            maxRel =
                std::max(maxRel, std::sqrt(ex * ex + ey * ey + ez * ez) / std::max(std::sqrt(rx * rx + ry * ry + rz * rz), 1e-30));
        }
        Check(ok, "parallel vs serial traversal max rel diff", maxRel, 1e-9);
        const bool samePairCounts =
            (statsSerial.m2lPairs == statsParallel.m2lPairs) && (statsSerial.nearPairs == statsParallel.nearPairs);
        std::printf("  parallel/serial m2lPairs %ld/%ld  nearPairs %ld/%ld  %s\n", statsSerial.m2lPairs,
                    statsParallel.m2lPairs, statsSerial.nearPairs, statsParallel.nearPairs,
                    samePairCounts ? "match" : "MISMATCH");
        Check(ok, "parallel traversal visits identical pair set", samePairCounts ? 0.0 : 1.0, 0.0);
    }

    // O(N) scaling: fitted log-log exponent over a modest range (kept small
    // so the selftest stays fast). N=32000 now exceeds the Phase-8 parallel
    // threshold, so this exercises the OpenMP-task traversal, not just the
    // single-threaded prototype.
    {
        std::vector<std::pair<double, double>> pts;
        for (int n : {2000, 8000, 32000}) {
            ic::PlummerParams pp;
            pp.n = n;
            pp.seed = 1;
            SoA<3> s = ic::Plummer<3>(pp);
            const PosMassView<3> v = ViewOf(s);
            StepParams sp;
            sp.G = 1.0;
            sp.soft = Softening::Plummer(0.02);
            sp.mac.theta = 0.5;
            SoA<3> out;
            const auto t0 = std::chrono::steady_clock::now();
            ComputeAccelMutualFmm(v, sp, out);
            const auto t1 = std::chrono::steady_clock::now();
            pts.emplace_back(static_cast<double>(n), std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        const double x0 = std::log(pts[0].first), x1 = std::log(pts[2].first);
        const double y0 = std::log(pts[0].second), y1 = std::log(pts[2].second);
        const double exponent = (y1 - y0) / (x1 - x0);
        std::printf("  scaling exponent (N=2000..32000, theta=0.5): %.2f  (O(N)->1.0; N=32000 uses the Phase-8\n"
                    "    OpenMP-task traversal, N=2000 the serial path -- the fit mixes both)\n",
                    exponent);
        Check(ok, "scaling exponent is sub-quadratic (sanity, not O(N) itself)", exponent, 1.8);
    }

    return ok;
}

namespace {

// median of a copy (small helper -- these vectors are a few thousand long)
double Median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// 2T/|W| for a system, computed exactly (O(N^2)) -- only used on the
// modest-N IC-validation clouds here.
double VirialRatio(const SoA<3>& s, double G, double eps2) {
    const std::size_t n = s.Count();
    double T = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        T += 0.5 * s.m[i] * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    double W = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j) {
            const double dx = s.x[j] - s.x[i], dy = s.y[j] - s.y[i], dz = s.z[j] - s.z[i];
            W -= G * s.m[i] * s.m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
    return (W != 0.0) ? (2.0 * T / std::abs(W)) : 0.0;
}

double HalfMassRadius(const SoA<3>& s) {
    std::vector<double> r(s.Count());
    for (std::size_t i = 0; i < s.Count(); ++i)
        r[i] = std::sqrt(s.x[i] * s.x[i] + s.y[i] * s.y[i] + s.z[i] * s.z[i]);
    std::sort(r.begin(), r.end());
    return r[r.size() / 2]; // equal-mass particles -> half-mass == median radius
}

// median fractional error of a binned rho(r) vs an analytic rho, over
// [0.1, 2] r_half. `analyticRho` takes r and returns the model density.
template <class F>
double DensityProfileMedianError(const SoA<3>& s, double rHalf, F analyticRho) {
    const int nBins = 12;
    const double rLo = 0.1 * rHalf, rHi = 2.0 * rHalf;
    std::vector<double> lo(nBins), hi(nBins), massInBin(nBins, 0.0);
    for (int b = 0; b < nBins; ++b) {
        lo[static_cast<std::size_t>(b)] = rLo * std::pow(rHi / rLo, double(b) / nBins);
        hi[static_cast<std::size_t>(b)] = rLo * std::pow(rHi / rLo, double(b + 1) / nBins);
    }
    for (std::size_t i = 0; i < s.Count(); ++i) {
        const double rr = std::sqrt(s.x[i] * s.x[i] + s.y[i] * s.y[i] + s.z[i] * s.z[i]);
        for (int b = 0; b < nBins; ++b)
            if (rr >= lo[static_cast<std::size_t>(b)] && rr < hi[static_cast<std::size_t>(b)]) {
                massInBin[static_cast<std::size_t>(b)] += s.m[i];
                break;
            }
    }
    std::vector<double> relErr;
    for (int b = 0; b < nBins; ++b) {
        const double rb0 = lo[static_cast<std::size_t>(b)], rb1 = hi[static_cast<std::size_t>(b)];
        const double shellVol = (4.0 / 3.0) * M_PI * (rb1 * rb1 * rb1 - rb0 * rb0 * rb0);
        const double rhoMeasured = massInBin[static_cast<std::size_t>(b)] / shellVol;
        const double rhoAnalytic = analyticRho(0.5 * (rb0 + rb1));
        if (rhoAnalytic > 0.0) relErr.push_back(std::abs(rhoMeasured - rhoAnalytic) / rhoAnalytic);
    }
    return Median(relErr);
}

} // namespace

bool CoreIcSelfTest() {
    bool ok = true;
    std::printf("[realistic IC generators]\n");

    const double G = 1.0;
    const int n = 20000;

    // --- Plummer: density profile + virial ratio + dynamical hold ---------
    {
        ic::PlummerParams pp;
        pp.n = n;
        pp.seed = 3;
        SoA<3> s = ic::Plummer<3>(pp);
        const double rHalf = HalfMassRadius(s);
        // Plummer rho(r) = (3/4pi) (1+r^2)^-5/2 in G=M=a=1 units.
        const double err = DensityProfileMedianError(s, rHalf, [](double r) {
            return (3.0 / (4.0 * M_PI)) * std::pow(1.0 + r * r, -2.5);
        });
        std::printf("  Plummer  r_half=%.3f (analytic ~1.305)  rho(r) median rel err=%.3f\n", rHalf, err);
        Check(ok, "Plummer density profile median rel err", err, 0.10);

        const double vr = VirialRatio(s, G, 0.0025);
        std::printf("  Plummer  2T/|W| = %.3f\n", vr);
        Check(ok, "Plummer virial ratio |2T/|W| - 1|", vr - 1.0, 0.12);

        // Dynamical hold: integrate a smaller cloud a few crossing times with
        // Direct, check the half-mass radius doesn't run away. t_cross ~ 2pi.
        ic::PlummerParams pp2;
        pp2.n = 2000;
        pp2.seed = 5;
        SoA<3> sd = ic::Plummer<3>(pp2);
        System<3> sys;
        sys.SetParticles(SoA<3>(sd));
        StepParams sp;
        sp.G = G;
        sp.soft = Softening::Plummer(0.05);
        sp.dt = (2.0 * M_PI) / 400.0;
        sp.solver = Solver::BarnesHut;
        sp.mac.theta = 0.5;
        const double rHalf0 = HalfMassRadius(sys.State());
        sys.Prime(sp);
        const int steps = static_cast<int>(6.0 * (2.0 * M_PI) / sp.dt); // ~6 crossing times
        for (int i = 0; i < steps; ++i) sys.Step(sp);
        const double rHalf1 = HalfMassRadius(sys.State());
        const double drift = std::abs(rHalf1 - rHalf0) / rHalf0;
        std::printf("  Plummer  half-mass radius: %.4f -> %.4f over ~6 t_cross (drift %.1f%%)\n", rHalf0, rHalf1,
                    100.0 * drift);
        Check(ok, "Plummer half-mass radius drift over ~6 t_cross", drift, 0.15);
    }

    // --- Hernquist: density profile + virial ratio -----------------------
    {
        ic::HernquistParams hp;
        hp.n = n;
        hp.seed = 7;
        SoA<3> s = ic::Hernquist<3>(hp);
        const double rHalf = HalfMassRadius(s);
        // Hernquist rho(r) = 1/(2pi) 1/(r (1+r)^3) in G=M=a=1.
        const double err = DensityProfileMedianError(
            s, rHalf, [](double r) { return (1.0 / (2.0 * M_PI)) / (r * std::pow(1.0 + r, 3.0)); });
        std::printf("  Hernquist  r_half=%.3f (analytic ~2.414)  rho(r) median rel err=%.3f\n", rHalf, err);
        Check(ok, "Hernquist density profile median rel err", err, 0.10);

        const double vr = VirialRatio(s, G, 0.0025);
        std::printf("  Hernquist  2T/|W| = %.3f\n", vr);
        Check(ok, "Hernquist virial ratio |2T/|W| - 1|", vr - 1.0, 0.20);
    }

    // --- King: W0 round-trips, compact support, virial ratio -------------
    {
        ic::KingParams kp;
        kp.n = n;
        kp.seed = 11;
        kp.w0 = 6.0;
        SoA<3> s = ic::King<3>(kp);
        const double rHalf = HalfMassRadius(s);
        double rMax = 0.0;
        for (std::size_t i = 0; i < s.Count(); ++i)
            rMax = std::max(rMax, std::sqrt(s.x[i] * s.x[i] + s.y[i] * s.y[i] + s.z[i] * s.z[i]));
        // King has compact support: the outermost particle sits at the tidal
        // radius, which for W0=6, r0=1 is ~ 8-9 core radii.
        std::printf("  King(W0=6)  r_half=%.3f  r_max=%.3f (tidal radius, finite -- compact support)\n", rHalf, rMax);
        Check(ok, "King has finite tidal radius (r_max < 20 r0)", rMax - 10.0, 10.0);

        const double vr = VirialRatio(s, G, 0.0025);
        std::printf("  King(W0=6)  2T/|W| = %.3f\n", vr);
        Check(ok, "King virial ratio |2T/|W| - 1|", vr - 1.0, 0.25);
    }

    return ok;
}

bool CoreRungSelfTest() {
    bool ok = true;
    std::printf("[block / rung timesteps]\n");

    // A slow Plummer cloud (the bulk -- all on rung 0) plus one tight, fast
    // binary bolted on far away (needs a deep rung). Global-adaptive drives
    // EVERY particle at the tight pair's tiny step; the block scheme lets
    // the ~200-particle bulk stay coarse. Direct solver.
    {
        auto build = []() {
            ic::PlummerParams pp;
            pp.n = 200;
            pp.seed = 12;
            SoA<3> cloud = ic::Plummer<3>(pp);
            SoA<3> s;
            s.Resize(202);
            for (int i = 0; i < 200; ++i) {
                const std::size_t ii = static_cast<std::size_t>(i);
                s.x[ii] = cloud.x[ii] - 8.0; // shift the cloud away from the binary
                s.y[ii] = cloud.y[ii];
                s.z[ii] = cloud.z[ii];
                s.vx[ii] = cloud.vx[ii];
                s.vy[ii] = cloud.vy[ii];
                s.vz[ii] = cloud.vz[ii];
                s.m[ii] = cloud.m[ii];
            }
            // tight fast pair near +8, separation 0.08
            s.SetPos(200, Vec<3>(7.96, 0, 0));
            s.SetPos(201, Vec<3>(8.04, 0, 0));
            const double vt = 0.5 * std::sqrt(1.0 * 2.0 / 0.08);
            s.SetVel(200, Vec<3>(0, -vt, 0));
            s.SetVel(201, Vec<3>(0, vt, 0));
            s.m[200] = s.m[201] = 1.0;
            return s;
        };
        const double T = 2.0; // total physical time -- both schemes integrate to here
        auto run = [&](bool block, bool adaptive, long& pevals, double& drift, double& elapsed) {
            System<3> sys;
            sys.SetParticles(build());
            StepParams sp;
            sp.G = 1.0;
            sp.soft = Softening::Plummer(0.01);
            sp.solver = Solver::Direct;
            sp.dt = 5e-3; // fine for the wide pair, way too coarse for the tight one
            sp.eta = 0.02;
            sp.block = block;
            sp.adaptive = adaptive;
            sp.blockMaxRung = 8;
            sys.Prime(sp);
            const double e0 = sys.TotalEnergy(sp);
            sys.ResetParticleEvals();
            double t = 0.0;
            while (t < T) t += sys.Step(sp); // block/fixed advance by sp.dt; adaptive by its chosen dt
            pevals = sys.ParticleEvals();
            elapsed = t;
            drift = std::abs((sys.TotalEnergy(sp) - e0) / e0);
        };

        long peBlock = 0, peAdaptive = 0, peGlobalFixed = 0;
        double drBlock = 0, drAdaptive = 0, drGlobalFixed = 0, elBlock = 0, elAdaptive = 0, elFixed = 0;
        run(true, false, peBlock, drBlock, elBlock);
        run(false, true, peAdaptive, drAdaptive, elAdaptive);
        run(false, false, peGlobalFixed, drGlobalFixed, elFixed);
        const double ratio = peBlock > 0 ? double(peAdaptive) / double(peBlock) : 0.0;
        std::printf("  200-body cloud + 1 tight binary, integrated to t=%.1f:\n", T);
        std::printf("    fixed global    drift %.2e   %ld pevals  (t=%.2f)\n", drGlobalFixed, peGlobalFixed, elFixed);
        std::printf("    adaptive global drift %.2e   %ld pevals  (t=%.2f)\n", drAdaptive, peAdaptive, elAdaptive);
        std::printf("    block           drift %.2e   %ld pevals  (t=%.2f)   (%.1fx fewer than adaptive)\n", drBlock,
                    peBlock, elBlock, ratio);
        Check(ok, "block-scheme energy drift", drBlock, 1e-2);
        Check(ok, "block uses >=2x fewer force evals than global-adaptive (matched physical time)",
              (ratio >= 2.0) ? 0.0 : 1.0, 0.0);
    }

    // Barnes-Hut path smoke test: a cold Plummer collapse under the block
    // scheme conserves energy through the first infall (the tree-rebuild-
    // per-substep path is exercised; correctness, not a savings claim,
    // since a shallow collapse keeps most particles on rung 0).
    {
        ic::PlummerParams pp;
        pp.n = 1200;
        pp.seed = 4;
        SoA<3> s = ic::Plummer<3>(pp);
        std::fill(s.vx.begin(), s.vx.end(), 0.0);
        std::fill(s.vy.begin(), s.vy.end(), 0.0);
        std::fill(s.vz.begin(), s.vz.end(), 0.0);
        System<3> sys;
        sys.SetParticles(std::move(s));
        StepParams sp;
        sp.G = 1.0;
        sp.soft = Softening::Plummer(0.02);
        sp.solver = Solver::BarnesHut;
        sp.mac.theta = 0.5;
        sp.dt = 3e-3;
        sp.eta = 0.03;
        sp.block = true;
        sp.blockMaxRung = 5;
        sys.Prime(sp);
        const double e0 = sys.TotalEnergy(sp);
        for (int i = 0; i < 150; ++i) sys.Step(sp);
        const double drift = std::abs((sys.TotalEnergy(sp) - e0) / e0);
        std::printf("  BH cold collapse N=1200, 150 base steps: block-scheme energy drift %.2e\n", drift);
        Check(ok, "block-scheme BH cold-collapse energy drift", drift, 0.05);
    }

    return ok;
}

} // namespace ngrav
