// 12_tddft -- Stage 2 (GPU) of the rt-TDDFT project. Phase 7: a 3D split-step
// Fourier propagator on the GPU, validated against the Stage-1 Python
// propagator (Physics Simulations/Quantum Mechanics/TDDFT/propagate.py).
//
//   12_tddft --selftest      3D FFT round-trip + free/harmonic vs closed forms
//
// See TDDFT_GPU_Plan.md.
#include "Tddft3D.hpp"
#include "Tddft3DSim.hpp"

#include "framework/Deck.hpp"
#include "framework/GLContext.hpp"
#include "framework/HeadlessRunner.hpp"
#include "framework/SimApp.hpp"

#include <glad/glad.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
using tddft::cf;

int gPass = 0, gFail = 0;
void check(const char* name, bool ok, const std::string& detail) {
    (ok ? gPass : gFail)++;
    std::printf("  [%s] %s  --  %s\n", ok ? "PASS" : "FAIL", name, detail.c_str());
}

// Coordinate along one axis, centered: x[i] = (i - N/2) * dx.
double coord(int i, int n, double dx) { return (i - n / 2) * dx; }

std::vector<cf> Gaussian(int n, double L, std::array<double, 3> c, double sigma,
                         std::array<double, 3> k0) {
    const double dx = L / n;
    std::vector<cf> psi((long)n * n * n);
    const double pref = std::pow(2.0 * kPi * sigma * sigma, -0.25);
    double norm2 = 0.0;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const double rx = coord(x, n, dx) - c[0];
                const double ry = coord(y, n, dx) - c[1];
                const double rz = coord(z, n, dx) - c[2];
                const double g = pref * pref * pref *
                                 std::exp(-(rx * rx + ry * ry + rz * rz) / (4.0 * sigma * sigma));
                const double ph = k0[0] * rx + k0[1] * ry + k0[2] * rz;
                const long i = (long(z) * n + y) * n + x;
                psi[i] = cf((float)(g * std::cos(ph)), (float)(g * std::sin(ph)));
                norm2 += g * g;
            }
    const double s = 1.0 / std::sqrt(norm2 * dx * dx * dx);
    for (auto& v : psi) v *= (float)s;
    return psi;
}

struct Moments {
    double norm, mx, my, mz, sx, sy, sz;
};

Moments moments(const std::vector<cf>& psi, int n, double L) {
    const double dx = L / n;
    const double dv = dx * dx * dx;
    double nrm = 0, mx = 0, my = 0, mz = 0, xx = 0, yy = 0, zz = 0;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const long i = (long(z) * n + y) * n + x;
                const double p = std::norm(psi[i]);  // |psi|^2
                const double cx = coord(x, n, dx), cy = coord(y, n, dx), cz = coord(z, n, dx);
                nrm += p;
                mx += cx * p; my += cy * p; mz += cz * p;
                xx += cx * cx * p; yy += cy * cy * p; zz += cz * cz * p;
            }
    nrm *= dv; mx *= dv; my *= dv; mz *= dv; xx *= dv; yy *= dv; zz *= dv;
    mx /= nrm; my /= nrm; mz /= nrm; xx /= nrm; yy /= nrm; zz /= nrm;
    return {nrm, mx, my, mz,
            std::sqrt(std::max(0.0, xx - mx * mx)),
            std::sqrt(std::max(0.0, yy - my * my)),
            std::sqrt(std::max(0.0, zz - mz * mz))};
}

// --- 1. FFT round-trip + forward-vs-direct-DFT on a small cube -------------
bool FftSelfTest() {
    const int n = 8;
    tddft::Tddft3D prop;
    prop.Configure(n, 8.0, 0.01);

    std::vector<cf> data((long)n * n * n);
    unsigned seed = 12345;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (float)((seed >> 8) & 0xFFFF) / 32768.0f - 1.0f; };
    for (auto& v : data) v = cf(rnd(), rnd());
    const std::vector<cf> original = data;

    GLuint buf = prop.MakeComplexBuffer(data);
    prop.Fft3D(buf, false);
    std::vector<cf> fwd((long)n * n * n);
    glGetNamedBufferSubData(buf, 0, fwd.size() * 2 * sizeof(float), fwd.data());
    prop.Fft3D(buf, true);
    std::vector<cf> rt((long)n * n * n);
    glGetNamedBufferSubData(buf, 0, rt.size() * 2 * sizeof(float), rt.data());
    glDeleteBuffers(1, &buf);

    double rtErr = 0.0;
    for (size_t i = 0; i < rt.size(); ++i) rtErr = std::max(rtErr, (double)std::abs(rt[i] - original[i]));

    // direct 3D DFT of one element (kx,ky,kz)=(1,2,3)
    const int KX = 1, KY = 2, KZ = 3;
    std::complex<double> acc{0, 0};
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const long i = (long(z) * n + y) * n + x;
                const double ang = -2.0 * kPi * (double(KX) * x + double(KY) * y + double(KZ) * z) / n;
                acc += std::complex<double>(original[i].real(), original[i].imag()) *
                       std::complex<double>(std::cos(ang), std::sin(ang));
            }
    const long ik = (long(KZ) * n + KY) * n + KX;
    const double dftErr = std::abs(std::complex<double>(fwd[ik].real(), fwd[ik].imag()) - acc);

    check("3D FFT round-trip", rtErr < 3e-4, "max err " + std::to_string(rtErr));
    check("3D FFT forward vs direct DFT", dftErr < 5e-3, "err " + std::to_string(dftErr));
    return rtErr < 3e-4 && dftErr < 5e-3;
}

// --- 2. free Gaussian spreading -------------------------------------------
void FreeSpreadTest() {
    const int n = 64;
    const double L = 28.0, sigma0 = 2.0, dt = 0.01, T = 4.0;
    const std::array<double, 3> k0{0.4, 0.0, 0.0};
    tddft::Tddft3D prop;
    prop.Configure(n, L, dt);
    prop.SetPsi(Gaussian(n, L, {0, 0, 0}, sigma0, k0));

    const int nsteps = (int)std::lround(T / dt);
    const int chunk = nsteps / 40;
    double wErr = 0.0, mErr = 0.0, normDrift = 0.0;
    for (int done = 0; done < nsteps; done += chunk) {
        prop.Step(chunk);
        const double t = (done + chunk) * dt;
        const Moments m = moments(prop.GetPsi(), n, L);
        const double sAna = sigma0 * std::sqrt(1.0 + std::pow(t / (2.0 * sigma0 * sigma0), 2.0));
        wErr = std::max(wErr, std::abs(m.sx - sAna) / sAna);
        wErr = std::max(wErr, std::abs(m.sy - sigma0 * std::sqrt(1.0 + std::pow(t / (2 * sigma0 * sigma0), 2))) / sAna);
        mErr = std::max(mErr, std::abs(m.mx - k0[0] * t));
        normDrift = std::max(normDrift, std::abs(m.norm - 1.0));
    }
    check("free: sigma(t) matches sigma0 sqrt(1+(t/2sigma0^2)^2)", wErr < 4e-3,
          "max rel err " + std::to_string(wErr));
    check("free: <x>(t) = k0 t", mErr < 2e-2, "max abs err " + std::to_string(mErr));
    check("free: norm conserved (fp32)", normDrift < 5e-4, "max drift " + std::to_string(normDrift));
}

// --- 3. harmonic-oscillator coherent state --------------------------------
void HarmonicTest() {
    const int n = 64;
    const double L = 18.0, omega = 1.0, x0 = 1.5, dt = 0.005;
    const double sigma0 = 1.0 / std::sqrt(2.0 * omega);
    const double period = 2.0 * kPi / omega;
    tddft::Tddft3D prop;
    prop.Configure(n, L, dt);

    std::vector<float> V((long)n * n * n);
    const double dx = L / n;
    for (int z = 0; z < n; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                const double r2 = std::pow(coord(x, n, dx), 2) + std::pow(coord(y, n, dx), 2) +
                                  std::pow(coord(z, n, dx), 2);
                V[(long(z) * n + y) * n + x] = (float)(0.5 * omega * omega * r2);
            }
    prop.SetPotential(V);
    prop.SetPsi(Gaussian(n, L, {x0, 0, 0}, sigma0, {0, 0, 0}));

    const int nsteps = (int)std::lround(1.25 * period / dt);
    const int chunk = nsteps / 60;
    double xErr = 0.0, wDrift = 0.0, normDrift = 0.0;
    for (int done = 0; done < nsteps; done += chunk) {
        prop.Step(chunk);
        const double t = (done + chunk) * dt;
        const Moments m = moments(prop.GetPsi(), n, L);
        xErr = std::max(xErr, std::abs(m.mx - x0 * std::cos(omega * t)));
        wDrift = std::max(wDrift, std::abs(m.sx - sigma0) / sigma0);
        normDrift = std::max(normDrift, std::abs(m.norm - 1.0));
    }
    check("harmonic: <x>(t) = x0 cos(w t)", xErr < 2e-2, "max abs err " + std::to_string(xErr));
    check("harmonic: width stays constant", wDrift < 3e-2, "max rel drift " + std::to_string(wDrift));
    check("harmonic: norm conserved (fp32)", normDrift < 5e-4, "max drift " + std::to_string(normDrift));
}

// --- Phase 8: Helium Kohn-Sham ground state (imaginary time) --------------
void HeliumRelaxTest() {
    // Same grid family as the Stage-1 Python Phase-2/3 He runs: N=32, L=14,
    // soft = 0.5 dx, method = lda.
    const int n = 32;
    const double L = 16.0, dx = L / n, soft = 0.5 * dx;
    tddft::Tddft3D t;
    t.Configure(n, L, 0.05);
    t.SetSoftenedNucleus(2.0, soft);
    const double dtau = 0.2 * dx * dx;
    const auto r = t.RelaxKS(2, dtau, 600, 1e-7, 4, true);
    // Stage-1 Python imaginary_time_ground_state, N=32 L=14 method=lda: eps_1s = -0.76291 Ha
    const double refEps = -0.76291;
    check("He relax: N_electrons = 2", std::abs(r.n_electrons - 2.0) < 1e-3,
          "integral rho = " + std::to_string(r.n_electrons));
    check("He relax: eps_1s matches Stage-1 Python (-0.76291 Ha)", std::abs(r.eps - refEps) < 3e-3,
          "eps = " + std::to_string(r.eps) + " Ha  (" + std::to_string(r.iters) + " iters)");
}

// Absorption analysis of a dipole trace: S(w) = (2w/pi) Im alpha,
// alpha(w) = (1/k) integral_0^inf e^{iwt} e^{-t/tau} [d(t)-d(0)] dt  (direct
// DFT -- the trace is short). Prints the TRK sum rule and the lowest peak and
// checks both against the Stage-1 Python He result.
void AnalyzeDipole(const std::vector<double>& t, const std::vector<double>& d, double k, double Ne) {
    const int nt = (int)t.size();
    const double dt = t[1] - t[0], T = t[nt - 1] - t[0], tau = 0.4 * T;
    std::vector<double> sig(nt);
    for (int i = 0; i < nt; ++i) sig[i] = (d[i] - d[0]) * std::exp(-(t[i] - t[0]) / tau);

    const double dw = 2.0 * kPi / (T * 4.0);            // fine grid
    const int nw = (int)(4.0 / dw);
    std::vector<double> w(nw), S(nw), Im(nw);
    for (int j = 0; j < nw; ++j) {
        const double wj = j * dw;
        double re = 0, im = 0;
        for (int i = 0; i < nt; ++i) { re += sig[i] * std::cos(wj * t[i]); im += sig[i] * std::sin(wj * t[i]); }
        re *= dt; im *= dt;
        // alpha = (1/k) * (re + i im);  S = (2 w / pi) Im alpha
        w[j] = wj;
        Im[j] = im / k;
        S[j] = (2.0 * wj / kPi) * Im[j];
    }
    double sumS = 0.0;
    for (int j = 1; j < nw; ++j) sumS += 0.5 * (S[j] + S[j - 1]) * dw;

    double imLo = 1e9, imHi = -1e9;
    for (int j = 0; j < nw; ++j) if (w[j] > 0.1) { imLo = std::min(imLo, Im[j]); imHi = std::max(imHi, Im[j]); }

    // local maxima in (0.2, 3.0); keep those above 15% of the strongest; the
    // lowest-frequency survivor is the "lowest line" (matches tools/spectrum.py).
    std::vector<int> pk;
    for (int j = 2; j < nw - 2; ++j)
        if (w[j] > 0.2 && w[j] < 3.0 && S[j] > S[j - 1] && S[j] >= S[j + 1]) pk.push_back(j);
    double sMax = 0.0;
    for (int j : pk) sMax = std::max(sMax, S[j]);
    double wLow = 0.0;
    for (int j : pk)
        if (S[j] > 0.15 * sMax) { wLow = w[j]; break; }

    check("He spectrum: TRK sum rule vs Stage-1 Python (97% of N_e)",
          std::abs(sumS - Ne) < 0.2 * Ne,
          "integral S dw = " + std::to_string(sumS) + " = " +
              std::to_string((int)std::lround(100 * sumS / Ne)) + "% of N_e");
    check("He spectrum: Im alpha >= 0 to the noise floor (passivity)",
          imLo / imHi > -0.02, "min/max Im alpha = " + std::to_string(imLo / imHi));
    check("He spectrum: lowest peak vs Stage-1 Python (0.50 Ha = 13.5 eV)",
          std::abs(wLow - 0.50) < 0.08,
          "peak = " + std::to_string(wLow) + " Ha = " + std::to_string(wLow * 27.2114) + " eV");
}

// --- Phase 9: hydrogen HHG in a flat-top laser --------------------------
struct FlatTop {
    double E0, w, tUp, tFlat, tPulse;
    FlatTop(double e0, double omega, int nRamp, int nFlat)
        : E0(e0), w(omega) {
        const double Tc = 2.0 * kPi / omega;
        tUp = nRamp * Tc; tFlat = nFlat * Tc; tPulse = 2 * tUp + tFlat;
    }
    double operator()(double t) const {
        if (t < 0 || t > tPulse) return 0.0;
        double env;
        if (t < tUp) env = std::pow(std::sin(kPi * t / (2 * tUp)), 2);
        else if (t < tUp + tFlat) env = 1.0;
        else env = std::pow(std::sin(kPi * (tPulse - t) / (2 * tUp)), 2);
        return E0 * env * std::sin(w * t);
    }
};

void AnalyzeHHG(const std::vector<double>& t, const std::vector<double>& d, double wL,
                double flat0, double flat1, double ion) {
    // dipole acceleration a(t) = d''(t), Hann-windowed over the flat portion.
    int i0 = 0, i1 = (int)t.size() - 1;
    for (int i = 0; i < (int)t.size(); ++i) { if (t[i] < flat0) i0 = i; if (t[i] <= flat1) i1 = i; }
    const int m = i1 - i0 + 1;
    const double dt = t[1] - t[0];
    std::vector<double> a(m);
    for (int i = 0; i < m; ++i) {
        const int g = i0 + i;
        const double dm = (g > 0) ? d[g - 1] : d[g];
        const double dp = (g < (int)d.size() - 1) ? d[g + 1] : d[g];
        a[i] = (dp - 2.0 * d[g] + dm) / (dt * dt);
    }
    for (int i = 0; i < m; ++i) a[i] *= 0.5 - 0.5 * std::cos(2.0 * kPi * i / (m - 1));

    auto power = [&](double harm) {
        double re = 0, im = 0;
        const double wq = harm * wL;
        for (int i = 0; i < m; ++i) { re += a[i] * std::cos(wq * (t[i0 + i] - t[i0])); im += a[i] * std::sin(wq * (t[i0 + i] - t[i0])); }
        return re * re + im * im;
    };
    const double odd = (power(3) + power(5) + power(7)) / 3.0;
    const double even = (power(2) + power(4) + power(6)) / 3.0;
    const double plateau = 0.5 * (power(3) + power(5));
    int cut = 3;
    for (int h = 3; h < 25; ++h) if (power(h) > plateau * 1e-3) cut = h;

    check("H HHG: odd harmonics dominate over even (inversion symmetry)", odd / even > 8.0,
          "odd/even = " + std::to_string(odd / even));
    check("H HHG: a plateau ends in a sharp cutoff",
          power(cut - 2 > 3 ? cut - 2 : 3) / power(3) > 1e-2 && power(cut + 3) / plateau < 1e-3,
          "plateau to order " + std::to_string(cut));
    check("H HHG: ionization present (mask absorbed norm)", ion > 0.02 && ion < 0.98,
          "ionized fraction = " + std::to_string(ion));
    std::printf("    (Stage-1 Python: odd/even ~5000, plateau + sharp cutoff, cutoff "
                "extends with intensity, ionization 59-92%%)\n");
}

int HydrogenHHG(const std::string& outDir) {
    const int n = 64;
    const double L = 32.0, dx = L / n, soft = 0.5 * dx, dt = 0.04, wL = 0.114;
    tddft::Tddft3D t;
    t.Configure(n, L, dt);
    t.SetBareMode(true);                 // H: no Hartree/XC (SIE-free, like Stage-1 method=None)
    t.SetSoftenedNucleus(1.0, soft);
    t.SetMask(8.0, 2);
    std::printf("relaxing H ground state (bare)...\n");
    const auto r = t.RelaxKS(1, 0.2 * dx * dx, 800, 1e-8, 4, true);
    std::printf("  eps_1s = %.4f Ha = %.1f eV\n", r.eps, r.eps * 27.2114);

    FlatTop E(0.06, wL, 2, 4);
    const int nSteps = (int)std::lround((E.tPulse + 5.0) / dt);
    std::printf("flat-top pulse (E0=0.06, %d ETRS steps)...\n\n", nSteps);
    std::vector<double> ts, ds;
    ts.push_back(0.0); ds.push_back(t.Dipole(2));
    for (int step = 1; step <= nSteps; ++step) {
        const double tn = (step - 1) * dt, tnn = step * dt;
        t.LaserStepKS(E(tn), E(tnn), 2);
        ts.push_back(tnn); ds.push_back(t.Dipole(2));
    }
    const double ion = 1.0 - t.SurvivingNorm() / t.OccWeight();

    AnalyzeHHG(ts, ds, wL, E.tUp, E.tUp + E.tFlat, ion);

    if (!outDir.empty()) {
        std::filesystem::create_directories(outDir);
        std::FILE* f = std::fopen((outDir + "/dipole.csv").c_str(), "w");
        std::fprintf(f, "t,dz\n");
        for (size_t i = 0; i < ts.size(); ++i) std::fprintf(f, "%.6f,%.10e\n", ts[i], ds[i]);
        std::fclose(f);
        f = std::fopen((outDir + "/meta.txt").c_str(), "w");
        std::fprintf(f, "omega_L %.6f\nflat0 %.6f\nflat1 %.6f\nIp %.6f\nE0 0.06\n",
                     wL, E.tUp, E.tUp + E.tFlat, -r.eps);
        std::fclose(f);
        std::printf("\nwrote %s/dipole.csv\n", outDir.c_str());
    }
    std::printf("\n%d/%d checks passed\n", gPass, gPass + gFail);
    return gFail == 0 ? 0 : 1;
}

// --- Phase 8: Helium delta-kick -> dipole trace --------------------------
int HeliumSpectrum(const std::string& outDir) {
    const int n = 32;
    const double L = 16.0, dx = L / n, soft = 0.5 * dx, dt = 0.05;
    tddft::Tddft3D t;
    t.Configure(n, L, dt);
    t.SetSoftenedNucleus(2.0, soft);
    std::printf("relaxing He ground state...\n");
    const auto r = t.RelaxKS(2, 0.2 * dx * dx, 600, 1e-7, 4, true);
    std::printf("  eps_1s = %.4f Ha,  N = %.5f\n", r.eps, r.n_electrons);

    const double kappa = 0.01;
    const int nSteps = 4000;           // T = 200 a.u.
    std::printf("delta-kick (kappa=%.3f) + %d ETRS steps...\n\n", kappa, nSteps);
    const auto tr = t.KickAndRunKS(kappa, 0, nSteps, 1);

    AnalyzeDipole(tr.t, tr.d, kappa, r.n_electrons);

    if (!outDir.empty()) {
        std::filesystem::create_directories(outDir);
        std::FILE* f = std::fopen((outDir + "/dipole.csv").c_str(), "w");
        std::fprintf(f, "t,dx\n");
        for (size_t i = 0; i < tr.t.size(); ++i) std::fprintf(f, "%.6f,%.10e\n", tr.t[i], tr.d[i]);
        std::fclose(f);
        std::fprintf((f = std::fopen((outDir + "/meta.txt").c_str(), "w")),
                     "kappa %.6f\nN_e %.6f\neps_1s %.6f\n", kappa, r.n_electrons, r.eps);
        std::fclose(f);
        std::printf("\nwrote %s/dipole.csv  (%zu samples) -- figure: python tools/spectrum.py %s\n",
                    outDir.c_str(), tr.t.size(), outDir.c_str());
    }
    std::printf("\n%d/%d checks passed\n", gPass, gPass + gFail);
    return gFail == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string mode, out, deck;
    bool interactive = false;
    int frames = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        if (s == "--selftest" || s == "--relax-test" || s == "--he-spectrum" || s == "--h-hhg") mode = s;
        else if (s == "--interactive") interactive = true;
        else if (s == "--deck" && i + 1 < argc) deck = argv[++i];
        else if (s == "--out" && i + 1 < argc) out = argv[++i];
        else if (s == "--frames" && i + 1 < argc) frames = std::atoi(argv[++i]);
    }

    if (!deck.empty()) {
        fw::Deck d = fw::Deck::FromFile(deck);
        if (interactive) {
            tddft::Tddft3DSim sim;
            fw::SimApp app(sim, d, d.GetString("title", "rt-TDDFT (GPU)"));
            app.Run();
            return 0;
        }
        if (out.empty()) { std::fprintf(stderr, "error: --deck headless needs --out DIR\n"); return 2; }
        fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);
        tddft::Tddft3DSim sim;
        sim.Configure(d);
        fw::HeadlessOptions opts;
        opts.outDir = out;
        opts.frames = frames;
        return fw::RunHeadless(sim, d, opts);
    }

    if (mode.empty()) {
        std::fprintf(stderr,
                     "usage:\n"
                     "  12_tddft --selftest                 3D FFT + free/harmonic vs Python\n"
                     "  12_tddft --relax-test               He Kohn-Sham ground state (imag. time)\n"
                     "  12_tddft --he-spectrum [--out DIR]  He delta-kick absorption\n"
                     "  12_tddft --h-hhg [--out DIR]        H atom high-harmonic generation\n"
                     "  12_tddft --deck f.toml --out DIR    headless run -> density frames + diagnostics\n"
                     "  12_tddft --deck f.toml --interactive   live view\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);

    if (mode == "--he-spectrum") {
        std::printf("12_tddft --he-spectrum -- He delta-kick absorption vs Stage-1 Python\n\n");
        return HeliumSpectrum(out);
    }
    if (mode == "--h-hhg") {
        std::printf("12_tddft --h-hhg -- H atom high-harmonic generation vs Stage-1 Python\n\n");
        return HydrogenHHG(out);
    }

    if (mode == "--selftest") {
        std::printf("12_tddft selftest -- 3D GPU split-step vs Stage-1 Python closed forms\n\n");
        std::printf("1. 3D FFT\n");
        FftSelfTest();
        std::printf("\n2. free Gaussian wavepacket\n");
        FreeSpreadTest();
        std::printf("\n3. harmonic-oscillator coherent state\n");
        HarmonicTest();
    } else {  // --relax-test
        std::printf("12_tddft --relax-test -- He Kohn-Sham ground state\n\n");
        HeliumRelaxTest();
    }

    std::printf("\n%d/%d checks passed\n", gPass, gPass + gFail);
    return gFail == 0 ? 0 : 1;
}
