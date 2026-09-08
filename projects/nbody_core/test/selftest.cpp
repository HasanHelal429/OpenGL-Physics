// Standalone validation for nbody_core: the leapfrog integrator, the SoA
// Direct sum, the flat AdaptiveTree, and the Barnes-Hut walk, checked against
// analytic two-body / Lagrange solutions and (BH) against Direct. No GL
// context. This is the seed for 02/03's `--selftest` (P2).
//
// Build: linked as `nbody_core_selftest`. Run: exit 0 = PASS.

#include "ngrav/Integrator.hpp"
#include "ngrav/Solvers.hpp"
#include "ngrav/System.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace ngrav;

namespace {

bool g_ok = true;

void Check(const char* name, double value, double tol) {
    const bool ok = std::isfinite(value) && std::abs(value) <= tol;
    std::printf("  %-38s %12.4e  (tol %.1e)  %s\n", name, value, tol, ok ? "ok" : "WRONG");
    if (!ok) g_ok = false;
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

// ---- two-body Kepler (3D), matches the Python N_Body_Gravity validation ----
void KeplerTest() {
    std::printf("[kepler 3D]\n");
    const double G = 1.0, m = 1.0, s = 1.0;
    const double vCirc = std::sqrt(G * (2.0 * m) / s);
    const double vRel = 0.8 * vCirc; // r0 = apoapsis, e = 0.36
    const double period = 2.0 * M_PI * std::sqrt(s * s * s / (G * 2.0 * m));

    std::vector<Vec<3>> pos = {{-s / 2, 0, 0}, {s / 2, 0, 0}};
    std::vector<Vec<3>> vel = {{0, -vRel / 2, 0}, {0, vRel / 2, 0}};
    std::vector<double> mass = {m, m};

    System<3> sys = MakeSystem<3>(pos, vel, mass);
    StepParams sp;
    sp.G = G;
    sp.soft = Softening::Plummer(1e-6);
    sp.solver = Solver::Direct;
    sp.dt = period / 2000.0;
    sys.Prime(sp);

    const double E0 = sys.TotalEnergy(sp);
    const glm::dvec3 L0 = sys.AngularMomentum();
    double rMin = 1e30, rMax = 0.0, Emin = E0, Emax = E0, Lmin = glm::length(L0), Lmax = Lmin;

    const int steps = 4 * 2000;
    for (int k = 0; k < steps; ++k) {
        sys.Step(sp);
        const SoA<3>& st = sys.State();
        const double r = glm::length(st.Pos(1) - st.Pos(0));
        rMin = std::min(rMin, r);
        rMax = std::max(rMax, r);
        if (k % 50 == 0) {
            const double E = sys.TotalEnergy(sp);
            Emin = std::min(Emin, E);
            Emax = std::max(Emax, E);
            const double Lm = glm::length(sys.AngularMomentum());
            Lmin = std::min(Lmin, Lm);
            Lmax = std::max(Lmax, Lm);
        }
    }

    const double a = s / (1.0 + 0.36); // semi-major from apoapsis r0 = a(1+e)
    const double periAnalytic = a * (1.0 - 0.36);
    Check("periapsis rel err", (rMin - periAnalytic) / periAnalytic, 1e-4);
    Check("apoapsis rel err", (rMax - s) / s, 1e-4);
    Check("energy drift (max-min)/|mean|", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
    Check("|L| drift (max-min)/mean", (Lmax - Lmin) / (0.5 * (Lmax + Lmin)), 1e-6);
}

// ---- Lagrange equilateral triangle (3D) ----
// Side length L = 1; the rigid-rotation rate that balances the net inward
// pull of the two other masses is Omega^2 = 3 G m / L^3 (vertices sit at
// circumradius L / sqrt(3)). Matches the Python N_Body_Gravity validation
// (r = 0.57735, Omega = 1.7320).
void LagrangeTest() {
    std::printf("[lagrange triangle 3D]\n");
    const double G = 1.0, m = 1.0, L = 1.0;
    const double omega = std::sqrt(3.0 * G * m / (L * L * L));
    const double R = L / std::sqrt(3.0);

    std::vector<Vec<3>> pos(3), vel(3);
    std::vector<double> mass = {m, m, m};
    for (int k = 0; k < 3; ++k) {
        const double ang = 2.0 * M_PI * k / 3.0;
        pos[k] = {R * std::cos(ang), R * std::sin(ang), 0};
        vel[k] = {-omega * R * std::sin(ang), omega * R * std::cos(ang), 0};
    }

    System<3> sys = MakeSystem<3>(pos, vel, mass);
    StepParams sp;
    sp.G = G;
    sp.soft = Softening::Plummer(1e-6);
    sp.solver = Solver::Direct;
    sp.dt = (2.0 * M_PI / omega) / 2000.0;
    sys.Prime(sp);

    const double E0 = sys.TotalEnergy(sp);
    double shapeErr = 0.0, Emin = E0, Emax = E0;
    const int steps = 5 * 2000;
    for (int k = 0; k < steps; ++k) {
        sys.Step(sp);
        const SoA<3>& st = sys.State();
        const double s01 = glm::length(st.Pos(1) - st.Pos(0));
        const double s12 = glm::length(st.Pos(2) - st.Pos(1));
        const double s20 = glm::length(st.Pos(0) - st.Pos(2));
        shapeErr = std::max({shapeErr, std::abs(s01 - L) / L, std::abs(s12 - L) / L, std::abs(s20 - L) / L});
        if (k % 50 == 0) {
            const double E = sys.TotalEnergy(sp);
            Emin = std::min(Emin, E);
            Emax = std::max(Emax, E);
        }
    }
    Check("triangle shape deviation", shapeErr, 1e-5);
    Check("energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 2e-8);
}

// ---- Barnes-Hut at theta=0 vs Direct ----
template <int D>
void BHvsDirectTest(const char* label) {
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
    sp.mac.theta = 0.0; // exhaustive -> exact

    SoA<D> aDirect, aBH;
    const PosMassView<D> v = ViewOf(in);
    ComputeAccelDirect<D>(v, sp, aDirect);
    ComputeAccelBarnesHut<D>(v, sp, aBH);

    double maxRel = 0.0;
    for (int i = 0; i < n; ++i) {
        const double rx = aDirect.ax[i], ry = aDirect.ay[i], rz = (D == 3) ? aDirect.az[i] : 0.0;
        const double ex = aBH.ax[i] - rx, ey = aBH.ay[i] - ry, ez = (D == 3) ? (aBH.az[i] - rz) : 0.0;
        const double refMag = std::sqrt(rx * rx + ry * ry + rz * rz);
        const double errMag = std::sqrt(ex * ex + ey * ey + ez * ez);
        maxRel = std::max(maxRel, errMag / std::max(refMag, 1e-30));
    }
    Check("max per-particle |da|/|a| (theta=0)", maxRel, 1e-12);
}

// ---- 2D two-body: precessing orbit, so only conservation is asserted ----
void Kepler2DTest() {
    std::printf("[kepler 2D]\n");
    const double G = 1.0, m = 1.0, s = 1.0;
    const double vCirc = std::sqrt(G * (2.0 * m)); // separation-independent in 2D
    const double vRel = 0.8 * vCirc;

    std::vector<Vec<2>> pos = {{-s / 2, 0}, {s / 2, 0}};
    std::vector<Vec<2>> vel = {{0, -vRel / 2}, {0, vRel / 2}};
    std::vector<double> mass = {m, m};

    System<2> sys = MakeSystem<2>(pos, vel, mass);
    StepParams sp;
    sp.G = G;
    sp.soft = Softening::Plummer(1e-6);
    sp.solver = Solver::Direct;
    sp.dt = (2.0 * M_PI * s / vCirc) / 2000.0;
    sys.Prime(sp);

    const double E0 = sys.TotalEnergy(sp);
    const double L0 = sys.AngularMomentum();
    double Emin = E0, Emax = E0, Lmin = L0, Lmax = L0;
    for (int k = 0; k < 4 * 2000; ++k) {
        sys.Step(sp);
        if (k % 50 == 0) {
            const double E = sys.TotalEnergy(sp);
            Emin = std::min(Emin, E);
            Emax = std::max(Emax, E);
            const double L = sys.AngularMomentum();
            Lmin = std::min(Lmin, L);
            Lmax = std::max(Lmax, L);
        }
    }
    Check("energy drift", (Emax - Emin) / std::abs(0.5 * (Emax + Emin)), 1e-4);
    Check("L drift", (Lmax - Lmin) / std::abs(0.5 * (Lmax + Lmin)), 1e-6);
}

} // namespace

int main() {
    KeplerTest();
    LagrangeTest();
    BHvsDirectTest<3>("barnes-hut vs direct 3D");
    BHvsDirectTest<2>("barnes-hut vs direct 2D");
    Kepler2DTest();
    std::printf("\nnbody_core selftest: %s\n", g_ok ? "PASS" : "FAIL");
    return g_ok ? 0 : 1;
}
