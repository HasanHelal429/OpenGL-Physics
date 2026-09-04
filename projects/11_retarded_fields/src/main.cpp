#include "ChargePath.hpp"
#include "LwFields.hpp"
#include "RetardedFieldsSim.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

using lw::kPi;

struct Args {
    std::string deck, out, renderCheck;
    int steps = 0, substeps = 0;
    bool interactive = false, selftest = false, gpuSelftest = false, cpu = false;
    bool dynSelftest = false, beamSelftest = false, bremsSelftest = false;
    bool inspiralSelftest = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--steps") a.steps = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--interactive") a.interactive = true;
        else if (s == "--selftest") a.selftest = true;
        else if (s == "--gpu-selftest") a.gpuSelftest = true;
        else if (s == "--dynamics-selftest") a.dynSelftest = true;
        else if (s == "--beaming-selftest") a.beamSelftest = true;
        else if (s == "--brems-selftest") a.bremsSelftest = true;
        else if (s == "--inspiral-selftest") a.inspiralSelftest = true;
        else if (s == "--cpu") a.cpu = true;
        else if (s == "--render-check") a.renderCheck = next();
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

lw::PathFn AnalyticPath(const lw::ChargePath& p) {
    return [p](double tau) {
        return lw::PathSample{p.r(tau), p.v(tau), p.a(tau)};
    };
}

bool SelfTest() {
    bool ok = true;
    const double c = 1.0, eps0 = 1.0;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    // (1) static charge: exact Coulomb, B = 0.
    {
        lw::ChargePath p;
        p.kind = lw::ChargePath::Static;
        const auto fn = AnalyticPath(p);
        double maxErr = 0.0, maxB = 0.0;
        for (glm::dvec3 x : {glm::dvec3(3, 0, 0), glm::dvec3(0, 5, 2),
                             glm::dvec3(-4, 1, -3), glm::dvec3(2, -2, 1)}) {
            const lw::LwResult r = lw::LwFields(fn, x, 100.0, 1.0, c, eps0, nan);
            const double R = glm::length(x);
            const glm::dvec3 Ecoul = x / (4.0 * kPi * eps0 * R * R * R);
            maxErr = std::max(maxErr, glm::length(r.E - Ecoul) / glm::length(Ecoul));
            maxB = std::max(maxB, glm::length(r.B));
        }
        std::printf("  static Coulomb: max rel err %.2e   max |B| %.2e\n", maxErr, maxB);
        if (maxErr > 1e-12 || maxB > 1e-14) ok = false;
    }

    // (2) uniformly moving charge: matches the present-position closed form
    //     E = q (1-b^2) Rhat_now / [4 pi eps0 (1 - b^2 sin^2 psi)^{3/2} R_now^2].
    {
        const double beta = 0.3;
        lw::ChargePath p;
        p.kind = lw::ChargePath::Uniform;
        p.v0 = glm::dvec3(beta, 0.0, 0.0);
        const auto fn = AnalyticPath(p);
        const double t = 50.0;
        double maxErr = 0.0;
        for (glm::dvec3 x : {glm::dvec3(beta * t + 4, 0, 0),
                             glm::dvec3(beta * t, 3, 0),
                             glm::dvec3(beta * t + 2, 2, 1),
                             glm::dvec3(beta * t - 3, -1, 2)}) {
            const lw::LwResult r = lw::LwFields(fn, x, t, 1.0, c, eps0, nan);
            const glm::dvec3 Rn = x - glm::dvec3(beta * t, 0, 0);
            const double Rnm = glm::length(Rn);
            const glm::dvec3 nn = Rn / Rnm;
            const double sin2 = 1.0 - nn.x * nn.x;   // angle from v (x axis)
            const double b2 = beta * beta;
            const glm::dvec3 Eexact = (1.0 - b2) * nn /
                (4.0 * kPi * eps0 * std::pow(1.0 - b2 * sin2, 1.5) * Rnm * Rnm);
            maxErr = std::max(maxErr, glm::length(r.E - Eexact) / glm::length(Eexact));
        }
        std::printf("  uniform motion (beta=0.3): max rel err %.2e\n", maxErr);
        if (maxErr > 1e-9) ok = false;
    }

    // (3) Larmor: non-relativistic circular motion, Poynting flux over a
    //     Fibonacci sphere at 4 wavelengths vs P = q^2 a^2 / (6 pi eps0 c^3).
    {
        lw::ChargePath p;
        p.kind = lw::ChargePath::Circular;
        p.omega = 1.0;
        p.radius = 5e-4;                      // v/c = omega*R = 5e-4
        const auto fn = AnalyticPath(p);
        const double accel = p.omega * p.omega * p.radius;
        const double Plarmor = accel * accel / (6.0 * kPi);   // q=c=eps0=1

        const double Rs = 4.0 * (2.0 * kPi / p.omega);         // 4 wavelengths
        const double t = Rs + 10.0;
        const int N = 2000;
        const double golden = kPi * (3.0 - std::sqrt(5.0));
        double flux = 0.0;
        for (int k = 0; k < N; ++k) {
            const double z = 1.0 - 2.0 * (k + 0.5) / N;
            const double rad = std::sqrt(std::max(0.0, 1.0 - z * z));
            const double th = golden * k;
            const glm::dvec3 dir(rad * std::cos(th), rad * std::sin(th), z);
            const glm::dvec3 x = Rs * dir;
            const lw::LwResult r = lw::LwFields(fn, x, t, 1.0, 1.0, 1.0, nan);
            flux += glm::dot(glm::cross(r.E, r.B), dir);
        }
        flux *= 4.0 * kPi * Rs * Rs / N;
        const double rel = std::abs(flux - Plarmor) / Plarmor;
        std::printf("  Larmor: sphere flux %.4e vs P=%.4e   rel err %.2e\n",
                    flux, Plarmor, rel);
        if (rel > 5e-3) ok = false;
    }

    std::printf("selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Phase 4: relativistic beaming. At one instant a charge in circular motion
// radiates into a forward cone about its velocity whose half-angle -> 1/gamma,
// and the total radiated power scales as gamma^4. Both come straight from the
// retarded-quantity angular distribution
//   dP/dOmega ~ |n x ((n - beta) x betadot)|^2 / (1 - n.beta)^5 .
bool BeamingSelfTest() {
    bool ok = true;

    // velocity along +y, centripetal acceleration along -x (unit accel).
    auto integrand = [](glm::dvec3 n, double beta) {
        const glm::dvec3 b(0.0, beta, 0.0);
        const glm::dvec3 bd(-1.0, 0.0, 0.0);
        const glm::dvec3 num = glm::cross(n, glm::cross(n - b, bd));
        const double kappa = 1.0 - glm::dot(n, b);
        return glm::dot(num, num) / std::pow(kappa, 5.0);
    };

    std::printf("  synchrotron beaming (velocity +y):\n");
    std::printf("    gamma   beta      half-angle   1/gamma    ratio     P (a.u.)   P/gamma^4\n");

    std::vector<double> ratios;
    double prevP = 0.0, prevG = 0.0, expMin = 1e9, expMax = -1e9;
    for (double gamma : {2.0, 3.0, 5.0, 8.0, 12.0}) {
        const double beta = std::sqrt(1.0 - 1.0 / (gamma * gamma));

        // in-plane cut: n(theta) = (sin th, cos th, 0), theta from +y.
        const int M = 200000;
        const double thMax = 20.0 / gamma;          // lobe is ~1/gamma wide
        double peak = integrand(glm::dvec3(0, 1, 0), beta);
        double halfAngle = thMax;
        for (int k = 1; k < M; ++k) {
            const double th = thMax * k / M;
            const double g = integrand(glm::dvec3(std::sin(th), std::cos(th), 0.0), beta);
            if (g < 0.5 * peak) { halfAngle = th; break; }
        }
        const double predicted = 1.0 / gamma;
        const double ratio = halfAngle / predicted;
        ratios.push_back(ratio);

        // total power: integrate over the sphere (theta_p from +y axis).
        const int Np = 1200, Na = 240;
        double P = 0.0;
        for (int ip = 0; ip < Np; ++ip) {
            const double tp = kPi * (ip + 0.5) / Np;
            const double st = std::sin(tp), ct = std::cos(tp);
            for (int ia = 0; ia < Na; ++ia) {
                const double ap = 2.0 * kPi * ia / Na;
                // axis = +y; n = ct * y_hat + st*(cos ap x_hat + sin ap z_hat)
                const glm::dvec3 n(st * std::cos(ap), ct, st * std::sin(ap));
                P += integrand(n, beta) * st;
            }
        }
        P *= (kPi / Np) * (2.0 * kPi / Na);
        if (prevG > 0.0) {
            const double e = std::log(P / prevP) / std::log(gamma / prevG);
            expMin = std::min(expMin, e);
            expMax = std::max(expMax, e);
        }
        prevP = P; prevG = gamma;

        std::printf("    %5.1f  %.5f   %9.5f   %8.5f   %.4f   %.4e   %.4f\n",
                    gamma, beta, halfAngle, predicted, ratio, P, P / std::pow(gamma, 4.0));
    }

    double rmin = 1e9, rmax = 0.0;
    for (double r : ratios) { rmin = std::min(rmin, r); rmax = std::max(rmax, r); }
    std::printf("    (half-angle * gamma) in [%.3f, %.3f]  ratio %.3f   "
                "power-law exponent in [%.3f, %.3f]\n",
                rmin, rmax, rmax / rmin, expMin, expMax);
    // The forward-lobe half-max half-width scales as 1/gamma: (half-angle *
    // gamma) is constant to ~10% (approaching ~0.31 as gamma -> inf). Total
    // radiated power scales as gamma^4.
    if (rmax / rmin > 1.15) ok = false;
    if (rmin < 0.25 || rmax > 0.45) ok = false;
    if (expMin < 3.9 || expMax > 4.1) ok = false;

    std::printf("beaming-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Phase 4: Coulomb bremsstrahlung. A light charge is deflected past a heavy
// fixed like charge; in the weak-deflection regime the energy it radiates
// scales as b^-3 with impact parameter. Run the self-consistent solver for
// a sweep of b and fit the log-log slope.
bool BremsSelfTest() {
    std::printf("  Coulomb bremsstrahlung (v=0.5, Q=q=1, m=1):\n");
    std::printf("    b       E_radiated     b^3 * E_rad\n");

    std::vector<double> lb, lE;
    for (double b : {3.0, 4.0, 5.0, 7.0, 10.0}) {
        char dk[900];
        std::snprintf(dk, sizeof(dk), R"(
[domain]
nx = 16
ny = 16
lx = 4.0
ly = 4.0
[mode]
type = "self_consistent"
[history]
buffer_span = 220.0
[time]
dt = 0.04
[[charge]]
q = 1.0
m = 1000000.0
seed_orbit = "static"
x0 = [0.0, 0.0, 0.0]
[[charge]]
q = 1.0
m = 1.0
seed_orbit = "uniform"
x0 = [-25.0, %.4f, 0.0]
v0 = [0.5, 0.0, 0.0]
)", b);
        lw::RetardedFieldsSim sim;
        fw::Deck d = fw::Deck::FromString(dk);
        sim.Configure(d);
        // advance until the light charge has cleared x = +25
        for (int n = 0; n < 6000; ++n) {
            sim.Step(1);
            if (sim.ChargePos(1).x > 25.0) break;
        }
        const double Erad = sim.Energy().radiated;
        std::printf("    %5.1f   %.6e   %.4e\n", b, Erad, b * b * b * Erad);
        lb.push_back(std::log(b));
        lE.push_back(std::log(Erad));
    }

    // least-squares slope of log E vs log b
    const int n = static_cast<int>(lb.size());
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = 0; i < n; ++i) { sx += lb[i]; sy += lE[i]; sxx += lb[i]*lb[i]; sxy += lb[i]*lE[i]; }
    const double slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    std::printf("    log-log slope = %.3f  (expected -3)\n", slope);

    const bool ok = slope < -2.6 && slope > -3.4;
    std::printf("brems-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Phase 4: two-body inspiral rate. The bound pair's mechanical energy
// KE + PE falls at exactly the rate the integrated Larmor diagnostic
// reports (the budget closes) -- that is the real test of the trajectory
// buffer + retarded force. The absolute rate lands at ~0.5 P_dipole,
//   P_dipole = (sum q a)^2 / (6 pi eps0 c^3),  |sum q a| = omega^2 d for
// the equal-mass opposite-charge circular seed: the pairwise retarded
// interaction without the Abraham-Lorentz self-force carries half of the
// symmetric pair's radiation reaction (a known, deliberate limitation).
bool InspiralSelfTest() {
    const char* dk = R"(
[domain]
nx = 16
ny = 16
lx = 4.0
ly = 4.0
[mode]
type = "self_consistent"
seed_orbit = "kepler"
separation = 6.0
[history]
buffer_span = 260.0
[time]
dt = 0.05
[[charge]]
q = 1.0
m = 1.0
[[charge]]
q = -1.0
m = 1.0
)";
    lw::RetardedFieldsSim sim;
    fw::Deck d = fw::Deck::FromString(dk);
    sim.Configure(d);

    const double d0 = glm::length(sim.ChargePos(0) - sim.ChargePos(1));
    const double kc = 1.0 / (4.0 * kPi);
    const double omega = std::sqrt(kc * 2.0 / (d0 * d0 * d0));
    const double T = 2.0 * kPi / omega;
    const double pddot = omega * omega * d0;                 // |sum q a|, q=1
    const double Pdipole = pddot * pddot / (6.0 * kPi);      // eps0 = c = 1

    // sample E_mech and E_rad over 4 periods
    std::vector<double> tt, em, er;
    const long steps = static_cast<long>(4 * T / sim.Dt());
    for (long n = 0; n <= steps; ++n) {
        if (n % 20 == 0) {
            const auto b = sim.Energy();
            tt.push_back(sim.Time());
            em.push_back(b.kinetic + b.interaction);
            er.push_back(b.radiated);
        }
        sim.Step(1);
    }

    // linear fit slope over the middle half of the run
    auto slope = [](const std::vector<double>& x, const std::vector<double>& y) {
        const int n = static_cast<int>(x.size());
        const int a = n / 4, b = 3 * n / 4;
        double sx = 0, sy = 0, sxx = 0, sxy = 0; int m = 0;
        for (int i = a; i < b; ++i) { sx += x[i]; sy += y[i]; sxx += x[i]*x[i]; sxy += x[i]*y[i]; ++m; }
        return (m * sxy - sx * sy) / (m * sxx - sx * sx);
    };
    const double dEmech = slope(tt, em);      // negative
    const double dErad = slope(tt, er);       // positive
    const double budget = std::abs(dEmech + dErad) / std::abs(dEmech);

    std::printf("  two-body inspiral (d0=%.1f, T=%.1f):\n", d0, T);
    std::printf("    dE_mech/dt = %+.3e   dE_rad/dt = %+.3e   P_dipole = %.3e\n",
                dEmech, dErad, Pdipole);
    std::printf("    budget closure |dEmech+dErad|/|dEmech| = %.3f\n", budget);
    std::printf("    (-dE_mech/dt)/P_dipole = %.3f    dE_rad_dt/P_dipole = %.3f\n",
                -dEmech / Pdipole, dErad / Pdipole);

    bool ok = true;
    if (budget > 0.08) ok = false;                     // KE+PE tracks the radiated energy
    if (-dEmech / Pdipole < 0.40 || -dEmech / Pdipole > 0.75) ok = false;
    std::printf("inspiral-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

bool GpuSelfTest() {
    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    bool ok = true;

    const char* decks[] = {
        R"(
[domain]
nx = 200
ny = 200
lx = 24
ly = 24
[time]
dt = 0.1
steps = 1
[[charge]]
q = 1.0
path = "circular"
radius = 2.0
omega = 0.5
)",
        R"(
[domain]
nx = 200
ny = 200
lx = 24
ly = 24
[time]
dt = 0.1
steps = 1
[[charge]]
q = 1.0
path = "linear_oscillator"
amplitude = 2.0
omega = 0.6
axis = [1, 0, 0]
[[charge]]
q = -1.0
path = "circular"
radius = 3.0
omega = 0.4
phase = 1.5
)"};

    for (const char* dk : decks) {
        lw::RetardedFieldsSim cpu, gpu;
        fw::Deck d1 = fw::Deck::FromString(dk);
        fw::Deck d2 = fw::Deck::FromString(dk);
        cpu.Configure(d1);
        gpu.Configure(d2);
        cpu.ForceCpu(true);

        cpu.Step(30);   gpu.Step(30);          // warm-start the tret buffer
        cpu.ComputeFieldCpu();
        gpu.ComputeField();

        const auto& ax = cpu.Ex();  const auto& bx = gpu.Ex();
        const auto& ay = cpu.Ey();  const auto& by = gpu.Ey();
        double amax = 0.0;
        for (std::size_t k = 0; k < ax.size(); ++k)
            amax = std::max(amax, std::hypot(ax[k], ay[k]));

        // The GPU shader runs in fp32; within a cell or two of a charge the
        // 1/R^2 near field amplifies tiny retarded-time differences without
        // bound. Compare on the radiation-relevant region by masking the
        // near-singularity spike (|E| > 20% of the grid maximum).
        const double cut = 0.20 * amax;
        double num2 = 0.0, den2 = 0.0, relmax = 0.0;
        for (std::size_t k = 0; k < ax.size(); ++k) {
            const double a = std::hypot(ax[k], ay[k]);
            if (a > cut || a == 0.0) continue;
            const double d = std::hypot(ax[k] - bx[k], ay[k] - by[k]);
            num2 += d * d;
            den2 += a * a;
            relmax = std::max(relmax, d / a);
        }
        const double relL2 = std::sqrt(num2 / den2);
        std::printf("  %d charge(s): max|E| %.3e  masked rel-L2 %.2e  masked rel-max %.2e\n",
                    (int)fw::Deck::FromString(dk).GetTables("charge").size(),
                    amax, relL2, relmax);
        if (relL2 > 3e-3 || relmax > 3e-2) ok = false;
    }

    std::printf("gpu-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Phase 3 gate: a bound charge pair seeded on the non-radiating circular
// orbit must hold shape for many periods (the trajectory buffer + cubic
// lookup inject no spurious force), and the energy budget
// KE + interaction + radiated must stay constant to a few percent.
bool DynamicsSelfTest() {
    const char* dk = R"(
[domain]
nx = 32
ny = 32
lx = 6.0
ly = 6.0
[mode]
type = "self_consistent"
seed_orbit = "kepler"
separation = 6.0
[time]
dt = 0.1
[[charge]]
q = 1.0
m = 1.0
[[charge]]
q = -1.0
m = 1.0
)";
    lw::RetardedFieldsSim sim;
    fw::Deck d = fw::Deck::FromString(dk);
    sim.Configure(d);

    const double d0 = glm::length(sim.ChargePos(0) - sim.ChargePos(1));
    // Kepler period for this seed: omega^2 = k|q1q2|(m1+m2)/(d^3 m1 m2),
    // k = 1/4pi. d0 = 6, m = 1, |q| = 1  ->  omega ~ 0.02715, T ~ 231.
    const double kc = 1.0 / (4.0 * kPi);
    const double omega = std::sqrt(kc * 2.0 / (d0 * d0 * d0));
    const double T = 2.0 * kPi / omega;
    const int periods = 3;
    const long steps = static_cast<long>(periods * T / sim.Dt());

    double dmin = d0, dmax = d0, errMax = 0.0;
    for (long n = 0; n < steps; ++n) {
        sim.Step(1);
        const double sep = glm::length(sim.ChargePos(0) - sim.ChargePos(1));
        dmin = std::min(dmin, sep);
        dmax = std::max(dmax, sep);
        errMax = std::max(errMax, sim.EnergyError());
    }
    const auto b = sim.Energy();
    const double breathe = (dmax - dmin) / d0;
    const double decay = (d0 - glm::length(sim.ChargePos(0) - sim.ChargePos(1))) / d0;

    std::printf("  kepler pair, %d periods (T=%.1f, %ld steps):\n", periods, T, steps);
    std::printf("    separation  d0=%.4f  range [%.4f, %.4f]  breathe %.2f%%  net decay %.2f%%\n",
                d0, dmin, dmax, 100.0 * breathe, 100.0 * decay);
    std::printf("    energy  KE=%.3e  PE=%.3e  radiated=%.3e  max |dE/E0| = %.2e\n",
                b.kinetic, b.interaction, b.radiated, errMax);

    bool ok = true;
    if (breathe > 0.06) ok = false;       // buffer must not pump the orbit
    if (errMax > 0.02) ok = false;        // budget closes to 2%
    if (!(decay > -0.01)) ok = false;     // radiation removes energy: orbit shrinks
    std::printf("dynamics-selftest: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);
    if (a.selftest) return SelfTest() ? 0 : 1;
    if (a.gpuSelftest) return GpuSelfTest() ? 0 : 1;
    if (a.dynSelftest) return DynamicsSelfTest() ? 0 : 1;
    if (a.beamSelftest) return BeamingSelfTest() ? 0 : 1;
    if (a.bremsSelftest) return BremsSelfTest() ? 0 : 1;
    if (a.inspiralSelftest) return InspiralSelfTest() ? 0 : 1;

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  11_retarded_fields --deck <f.toml> --out <dir> [--cpu]\n"
                     "  11_retarded_fields --interactive --deck <f.toml>\n"
                     "  11_retarded_fields --deck <f.toml> --render-check <png>\n"
                     "  11_retarded_fields --selftest | --gpu-selftest\n"
                     "  11_retarded_fields --dynamics-selftest | --inspiral-selftest\n"
                     "  11_retarded_fields --beaming-selftest | --brems-selftest\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);

    if (a.interactive) {
        lw::RetardedFieldsSim sim;
        fw::SimApp app(sim, deck, deck.GetString("title", "Lienard-Wiechert"));
        app.Run();
        return 0;
    }

    if (!a.renderCheck.empty()) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const int W = 1000, H = 1000;
        GLuint fbo = 0, col = 0, dep = 0;
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(1, &col);
        glBindTexture(GL_TEXTURE_2D, col);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, col, 0);
        glGenRenderbuffers(1, &dep);
        glBindRenderbuffer(GL_RENDERBUFFER, dep);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, W, H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, dep);
        glViewport(0, 0, W, H);
        lw::RetardedFieldsSim sim;
        sim.Configure(deck);
        sim.Step(deck.GetInt("render_check.warm_steps", 60));
        sim.Render(W, H);
        sim.Render(W, H);
        glFinish();
        std::vector<unsigned char> px((size_t)W * H * 4), flip(px.size());
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        for (int y = 0; y < H; ++y)
            std::memcpy(&flip[(size_t)y * W * 4], &px[(size_t)(H - 1 - y) * W * 4], (size_t)W * 4);
        stbi_write_png(a.renderCheck.c_str(), W, H, 4, flip.data(), W * 4);
        std::printf("render-check -> %s\n", a.renderCheck.c_str());
        return 0;
    }

    if (a.out.empty()) {
        std::fprintf(stderr, "error: headless mode needs --out <dir>\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
    lw::RetardedFieldsSim sim;
    sim.Configure(deck);
    if (a.cpu) sim.ForceCpu(true);

    const int totalSteps = a.steps > 0 ? a.steps : deck.GetInt("time.steps", 400);
    const int substeps = a.substeps > 0 ? a.substeps
                                        : deck.GetInt("time.substeps_per_frame", 1);
    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = std::max(1, totalSteps / std::max(1, substeps));
    opts.substeps = substeps;
    return fw::RunHeadless(sim, deck, opts);
}
