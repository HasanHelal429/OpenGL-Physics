// 12_tddft -- Stage 2 (GPU) of the rt-TDDFT project. Phase 7: a 3D split-step
// Fourier propagator on the GPU, validated against the Stage-1 Python
// propagator (Physics Simulations/Quantum Mechanics/TDDFT/propagate.py).
//
//   12_tddft --selftest      3D FFT round-trip + free/harmonic vs closed forms
//
// See TDDFT_GPU_Plan.md.
#include "Tddft3D.hpp"

#include "framework/GLContext.hpp"

#include <glad/glad.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
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

} // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    bool selftest = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--selftest") == 0) selftest = true;

    if (!selftest) {
        std::fprintf(stderr, "usage: 12_tddft --selftest\n");
        return 2;
    }

    fw::GLContext ctx = fw::GLContext::CreateHidden(4, 6);

    std::printf("12_tddft selftest -- 3D GPU split-step vs Stage-1 Python closed forms\n\n");
    std::printf("1. 3D FFT\n");
    FftSelfTest();
    std::printf("\n2. free Gaussian wavepacket\n");
    FreeSpreadTest();
    std::printf("\n3. harmonic-oscillator coherent state\n");
    HarmonicTest();

    std::printf("\n%d/%d checks passed\n", gPass, gPass + gFail);
    return gFail == 0 ? 0 : 1;
}
