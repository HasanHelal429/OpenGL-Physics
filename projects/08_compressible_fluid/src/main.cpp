#include "CompressibleSim.hpp"
#include "Euler1D.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct Args {
    std::string deck;
    std::string out;
    int frames = 0;
    int substeps = 0;
    bool selftest = false;
};

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--deck") a.deck = next();
        else if (s == "--out") a.out = next();
        else if (s == "--frames") a.frames = std::atoi(next());
        else if (s == "--substeps") a.substeps = std::atoi(next());
        else if (s == "--selftest") a.selftest = true;
        else std::fprintf(stderr, "warning: unknown arg '%s'\n", s.c_str());
    }
    return a;
}

// HLLC must reduce to the exact physical flux when the two input states are
// identical (SL<=0<=SR always holds then, and whichever star-state branch
// fires must reproduce F(state) exactly) -- a basic, cheap correctness
// check independent of any Riemann-problem reference solution.
bool SelfTestHllcConsistency() {
    const double gamma = 1.4;
    const cf::Prim states[] = {
        {1.0, 0.0, 1.0}, {0.125, 0.0, 0.1}, {1.0, 0.75, 1.0}, {0.5, -1.2, 0.3},
    };
    double maxErr = 0.0;
    for (const cf::Prim& s : states) {
        const cf::Cons f = cf::HllcFlux(s, s, gamma);
        const cf::Cons fExact = cf::Flux(s, gamma);
        maxErr = std::max({maxErr, std::abs(f.rho - fExact.rho), std::abs(f.mom - fExact.mom),
                            std::abs(f.energy - fExact.energy)});
    }
    std::printf("selftest (hllc consistency): max err = %.3e\n", maxErr);
    const bool ok = maxErr < 1e-12;
    std::printf("selftest (hllc consistency): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// Standard Sod IC (u_l=u_r=0) evolved for a short time, well before either
// wave reaches the domain boundary. At the boundary faces the flux is then
// exactly the (unperturbed) initial uniform-state flux: F_mass=rho*u=0,
// F_energy=u*(E+p)=0, F_mom=rho*u^2+p=p on both sides. So over this window
// mass and energy are EXACTLY conserved (no flux ever leaves), and total
// momentum grows at the exact constant rate p_L-p_R -- both are identities
// of the conservative flux-differencing form itself, not approximate
// physics, so they must hold to near machine precision regardless of
// scheme accuracy elsewhere. (Same identity 07_grhd's plot_shocktube.py
// checks for its relativistic solver.)
bool SelfTestConservation() {
    const double gamma = 1.4;
    const cf::Prim left{1.0, 0.0, 1.0};
    const cf::Prim right{0.125, 0.0, 0.1};
    const int n = 400;

    cf::Euler1D solver;
    solver.Init(n, 0.0, 1.0, gamma);
    solver.SetRiemannIC(0.5, left, right);

    const double dt = 0.4 * solver.Dx() / solver.MaxWaveSpeed();
    const double mass0 = solver.TotalMass();
    const double energy0 = solver.TotalEnergy();
    const double mom0 = solver.TotalMomentum();

    // Short enough that the fastest wave (sound speed ~1.18) stays well
    // inside the domain (dx*n/2 = 0.5 away from either boundary).
    const int steps = 50;
    for (int i = 0; i < steps; ++i) solver.Step(dt);
    const double t = steps * dt;

    const double massErr = std::abs(solver.TotalMass() - mass0) / mass0;
    const double energyErr = std::abs(solver.TotalEnergy() - energy0) / energy0;
    const double predictedMom = mom0 + (left.p - right.p) * t;
    const double momErr = std::abs(solver.TotalMomentum() - predictedMom) / std::abs(predictedMom);

    std::printf("selftest (conservation): mass err=%.3e  energy err=%.3e  momentum err=%.3e (t=%.4f)\n", massErr,
                energyErr, momErr, t);
    const bool ok = massErr < 1e-10 && energyErr < 1e-10 && momErr < 1e-6;
    std::printf("selftest (conservation): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        const bool ok = SelfTestHllcConsistency() && SelfTestConservation();
        return ok ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  08_compressible_fluid --deck <f.toml> --out <dir> [--frames N] [--substeps N]\n"
                     "  08_compressible_fluid --selftest\n");
        return 2;
    }
    if (a.out.empty()) {
        std::fprintf(stderr, "error: needs --out <dir>\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);
    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);

    cf::CompressibleSim sim;
    sim.Configure(deck);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;
    return fw::RunHeadless(sim, deck, opts);
}
