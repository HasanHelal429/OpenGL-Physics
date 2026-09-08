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
    return ok;
}

} // namespace ngrav
