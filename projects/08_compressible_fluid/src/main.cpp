#include "CompressibleSim.hpp"
#include "CompressibleSimScene.hpp"
#include "Euler1D.hpp"
#include "Euler2D.hpp"
#include "kernels_euler1d.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string deck;
    std::string out;
    int frames = 0;
    int substeps = 0;
    bool selftest = false;
    bool scene = false;
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
        else if (s == "--scene") a.scene = true;
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

// Cross-checks kernels_euler1d.hpp's GPU RK2 step sequence against Euler1D's
// (already independently-validated, see SelfTestHllcConsistency and
// SelfTestConservation above, plus tools/plot_shocktube.py's exact-solution
// comparison) CPU implementation on the same Sod IC -- same physics, two
// completely separate code paths (GLSL compute shaders, float32 vs a
// double-precision C++ reference), same role as 04/07's GPU-vs-CPU
// selftest cases. Needs a current GL context (caller's responsibility).
bool SelfTestGpu() {
    const double gamma = 1.4;
    const cf::Prim left{1.0, 0.0, 1.0};
    const cf::Prim right{0.125, 0.0, 0.1};
    const int n = 400;

    cf::Euler1D cpu;
    cpu.Init(n, 0.0, 1.0, gamma);
    cpu.SetRiemannIC(0.5, left, right);
    const double dt = 0.4 * cpu.Dx() / cpu.MaxWaveSpeed();

    std::vector<glm::vec4> initialCons(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const cf::Prim p = cpu.PrimAt(i);
        const cf::Cons c = cf::ToCons(p, gamma);
        initialCons[static_cast<size_t>(i)] =
            glm::vec4(static_cast<float>(c.rho), static_cast<float>(c.mom), static_cast<float>(c.energy), 0.0f);
    }

    GLuint cons[2] = {0, 0}, stage1 = 0, stage2 = 0, prim = 0, slope = 0, flux = 0;
    glCreateBuffers(1, &cons[0]);
    glCreateBuffers(1, &cons[1]);
    glCreateBuffers(1, &stage1);
    glCreateBuffers(1, &stage2);
    glCreateBuffers(1, &prim);
    glCreateBuffers(1, &slope);
    glCreateBuffers(1, &flux);
    glNamedBufferData(cons[0], n * sizeof(glm::vec4), initialCons.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(cons[1], n * sizeof(glm::vec4), initialCons.data(), GL_DYNAMIC_DRAW);
    glNamedBufferData(stage1, n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(stage2, n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(prim, n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(slope, n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(flux, (n + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);

    fw::ComputeShader consToPrim = fw::ComputeShader::FromSource(cf::kernels::ConsToPrim());
    fw::ComputeShader computeSlopes = fw::ComputeShader::FromSource(cf::kernels::ComputeSlopes());
    fw::ComputeShader fluxes = fw::ComputeShader::FromSource(cf::kernels::Fluxes());
    fw::ComputeShader eulerStep = fw::ComputeShader::FromSource(cf::kernels::EulerStep());
    fw::ComputeShader combine = fw::ComputeShader::FromSource(cf::kernels::Combine());

    const GLuint groupsN = static_cast<GLuint>((n + cf::kernels::kWorkgroupSize - 1) / cf::kernels::kWorkgroupSize);
    const GLuint groupsNp1 =
        static_cast<GLuint>((n + 1 + cf::kernels::kWorkgroupSize - 1) / cf::kernels::kWorkgroupSize);
    const float dtOverDx = static_cast<float>(dt / cpu.Dx());

    auto rhsPass = [&](GLuint consBuf) {
        consToPrim.Use();
        consToPrim.SetInt("uN", n);
        consToPrim.SetFloat("uGamma", static_cast<float>(gamma));
        fw::ComputeShader::BindBuffer(0, consBuf);
        fw::ComputeShader::BindBuffer(4, prim);
        consToPrim.Dispatch(groupsN);
        fw::ComputeShader::Barrier();

        computeSlopes.Use();
        computeSlopes.SetInt("uN", n);
        computeSlopes.SetFloat("uGamma", static_cast<float>(gamma));
        fw::ComputeShader::BindBuffer(4, prim);
        fw::ComputeShader::BindBuffer(6, slope);
        computeSlopes.Dispatch(groupsN);
        fw::ComputeShader::Barrier();

        fluxes.Use();
        fluxes.SetInt("uN", n);
        fluxes.SetFloat("uGamma", static_cast<float>(gamma));
        fw::ComputeShader::BindBuffer(4, prim);
        fw::ComputeShader::BindBuffer(6, slope);
        fw::ComputeShader::BindBuffer(5, flux);
        fluxes.Dispatch(groupsNp1);
        fw::ComputeShader::Barrier();
    };
    auto eulerPass = [&](GLuint consIn, GLuint consOut) {
        eulerStep.Use();
        eulerStep.SetInt("uN", n);
        eulerStep.SetFloat("uGamma", static_cast<float>(gamma));
        eulerStep.SetFloat("uDtOverDx", dtOverDx);
        fw::ComputeShader::BindBuffer(0, consIn);
        fw::ComputeShader::BindBuffer(5, flux);
        fw::ComputeShader::BindBuffer(1, consOut);
        eulerStep.Dispatch(groupsN);
        fw::ComputeShader::Barrier();
    };

    int cur = 0;
    const int steps = 50;
    for (int s = 0; s < steps; ++s) {
        const int next = 1 - cur;
        rhsPass(cons[cur]);
        eulerPass(cons[cur], stage1);

        rhsPass(stage1);
        eulerPass(stage1, stage2);

        combine.Use();
        combine.SetInt("uN", n);
        combine.SetFloat("uGamma", static_cast<float>(gamma));
        fw::ComputeShader::BindBuffer(0, cons[cur]);
        fw::ComputeShader::BindBuffer(1, stage2);
        fw::ComputeShader::BindBuffer(2, cons[next]);
        combine.Dispatch(groupsN);
        fw::ComputeShader::Barrier();

        cur = next;
        cpu.Step(dt);
    }

    // Re-run ConsToPrim once more so `prim` reflects the final `cons[cur]`
    // state (the loop's last rhsPass() call read from `stage1`, not `cons`).
    consToPrim.Use();
    consToPrim.SetInt("uN", n);
    consToPrim.SetFloat("uGamma", static_cast<float>(gamma));
    fw::ComputeShader::BindBuffer(0, cons[cur]);
    fw::ComputeShader::BindBuffer(4, prim);
    consToPrim.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    std::vector<glm::vec4> gpuPrim(static_cast<size_t>(n));
    glGetNamedBufferSubData(prim, 0, n * sizeof(glm::vec4), gpuPrim.data());
    GLuint bufs[] = {cons[0], cons[1], stage1, stage2, prim, slope, flux};
    glDeleteBuffers(7, bufs);

    double rhoErr = 0.0, uErr = 0.0, pErr = 0.0;
    double rhoNorm = 1e-12, uNorm = 1e-12, pNorm = 1e-12;
    for (int i = 0; i < n; ++i) {
        const cf::Prim cpuP = cpu.PrimAt(i);
        const glm::vec4& g = gpuPrim[static_cast<size_t>(i)];
        rhoErr = std::max(rhoErr, std::abs(double(g.x) - cpuP.rho));
        uErr = std::max(uErr, std::abs(double(g.y) - cpuP.u));
        pErr = std::max(pErr, std::abs(double(g.z) - cpuP.p));
        rhoNorm = std::max(rhoNorm, std::abs(cpuP.rho));
        uNorm = std::max(uNorm, std::abs(cpuP.u));
        pNorm = std::max(pNorm, std::abs(cpuP.p));
    }
    const double relRho = rhoErr / rhoNorm, relU = uErr / uNorm, relP = pErr / pNorm;
    std::printf("selftest (gpu vs cpu): max relative error  rho=%.3e  u=%.3e  p=%.3e\n", relRho, relU, relP);
    // Looser than the CPU-vs-CPU cases above: this compares float32 GPU
    // arithmetic, accumulated over 2*50 flux evaluations, against a
    // double-precision CPU reference -- the floor is float32 precision, not
    // machine-double precision (same reasoning as 04's GPU-vs-CPU force
    // selftest).
    const bool ok = relRho < 1e-3 && relU < 1e-3 && relP < 1e-3;
    std::printf("selftest (gpu vs cpu): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

// The 2D isentropic vortex (Shu 1997-style): an exact smooth traveling
// solution of the 2D Euler equations superimposed on a uniform background
// flow (Yee, Sandham & Djomehri 1999; Spiegel et al. 2015's review of vortex
// preservation tests). Since it's smooth (no shocks -- MinMod should sit at
// or near unlimited, 2nd-order accuracy almost everywhere) and has a known
// closed-form solution at any later time (rigid advection at the background
// velocity), it is the standard check that dimensional splitting + the
// transverse-momentum HLLC extension haven't broken 2nd-order accuracy or
// introduced a splitting-direction bias (e.g. an X-Y asymmetry) that a
// pure-shock test like Sod can't reveal.
struct VortexParams {
    double x0 = 2.0, y0 = 2.0; // initial center
    double uInf = 1.0, vInf = 1.0; // background flow (diagonal: exercises both sweep directions)
    double beta = 5.0; // vortex strength
    double gamma = 1.4;
};

cf::Prim2D VortexState(double x, double y, double t, const VortexParams& vp) {
    const double kPi = 3.14159265358979323846;
    const double xc = vp.x0 + vp.uInf * t;
    const double yc = vp.y0 + vp.vInf * t;
    const double dx = x - xc, dy = y - yc;
    const double r2 = dx * dx + dy * dy;
    const double expTerm = std::exp(0.5 * (1.0 - r2));
    const double du = -vp.beta / (2.0 * kPi) * expTerm * dy;
    const double dv = vp.beta / (2.0 * kPi) * expTerm * dx;
    const double dT = -(vp.gamma - 1.0) * vp.beta * vp.beta / (8.0 * vp.gamma * kPi * kPi) * std::exp(1.0 - r2);
    const double T = 1.0 + dT; // background T=p/rho=1
    const double rho = std::pow(T, 1.0 / (vp.gamma - 1.0));
    const double p = std::pow(rho, vp.gamma); // isentropic, background entropy p/rho^gamma=1
    return cf::Prim2D{rho, vp.uInf + du, vp.vInf + dv, p};
}

bool SelfTestVortexAdvection() {
    const VortexParams vp;
    const int n = 100;
    const double length = 10.0;
    cf::Euler2D solver;
    solver.Init(n, n, 0.0, length, 0.0, length, vp.gamma);
    solver.SetInitialCondition([&](double x, double y) { return VortexState(x, y, 0.0, vp); });

    const double dt = 0.4 / (solver.MaxWaveSpeedX() / solver.Dx() + solver.MaxWaveSpeedY() / solver.Dy());
    // Vortex core radius ~1; center moves from (2,2) to (4,4) at t=2, still
    // >3 core radii from the nearest boundary (domain is [0,10]^2).
    const double tEnd = 2.0;
    const int steps = static_cast<int>(std::ceil(tEnd / dt));
    for (int s = 0; s < steps; ++s) solver.Step(dt);
    const double tActual = steps * dt;

    double l2Num = 0.0, l2Den = 0.0, linf = 0.0;
    for (int j = 0; j < n; ++j) {
        const double y = (static_cast<double>(j) + 0.5) * solver.Dy();
        for (int i = 0; i < n; ++i) {
            const double x = (static_cast<double>(i) + 0.5) * solver.Dx();
            const double rhoNum = solver.PrimAt(i, j).rho;
            const double rhoEx = VortexState(x, y, tActual, vp).rho;
            const double diff = rhoNum - rhoEx;
            l2Num += diff * diff;
            l2Den += rhoEx * rhoEx;
            linf = std::max(linf, std::abs(diff));
        }
    }
    const double relL2 = std::sqrt(l2Num / l2Den);
    std::printf("selftest (vortex advection): rho relative L2 err=%.3e  Linf err=%.3e (t=%.4f)\n", relL2, linf,
                tActual);
    // MinMod is the most diffusive standard limiter, and n=100 across a
    // length-10 domain gives only ~10 cells across the vortex core (radius
    // 1) -- a coarse, quick-running check, not a formal convergence study,
    // so this tolerance has real headroom above the observed error rather
    // than being tuned tight to it.
    const bool ok = relL2 < 0.05;
    std::printf("selftest (vortex advection): %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

} // namespace

int main(int argc, char** argv) {
    const Args a = ParseArgs(argc, argv);

    if (a.selftest) {
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        const bool ok =
            SelfTestHllcConsistency() && SelfTestConservation() && SelfTestGpu() && SelfTestVortexAdvection();
        return ok ? 0 : 1;
    }

    if (a.deck.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  08_compressible_fluid --deck <f.toml> --out <dir> [--frames N] [--substeps N] [--scene]\n"
                     "  08_compressible_fluid --selftest\n");
        return 2;
    }
    if (a.out.empty()) {
        std::fprintf(stderr, "error: needs --out <dir>\n");
        return 2;
    }

    fw::Deck deck = fw::Deck::FromFile(a.deck);
    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);

    fw::HeadlessOptions opts;
    opts.outDir = a.out;
    opts.frames = a.frames;
    opts.substeps = a.substeps;

    if (a.scene) {
        cf::CompressibleSimScene sim;
        sim.Configure(deck);
        return fw::RunHeadless(sim, deck, opts);
    }
    cf::CompressibleSim sim;
    sim.Configure(deck);
    return fw::RunHeadless(sim, deck, opts);
}
