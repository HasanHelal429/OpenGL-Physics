#include "Tddft3D.hpp"

#include "kernels3d.hpp"
#include "kernels_ks.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace tddft {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Tddft3D::~Tddft3D() {
    GLuint bufs[] = {m_psi, m_tmp, m_twiddle, m_vprop, m_kprop,
                     m_vnuc, m_rho, m_veff, m_veffPred, m_psiPred,
                     m_scratchC, m_k2half, m_ktau, m_partials};
    glDeleteBuffers(sizeof(bufs) / sizeof(bufs[0]), bufs);
}

void Tddft3D::Configure(int n, double L, double dt) {
    if ((n & (n - 1)) != 0 || n < 4) throw std::runtime_error("Tddft3D: N must be a power of two >= 4");
    m_n = n;
    m_total = (long)n * n * n;
    m_L = L;
    m_dx = L / n;
    m_dt = dt;

    m_fft = fw::ComputeShader::FromSource(kernels::Fft1D(n));
    m_rotate = fw::ComputeShader::FromSource(kernels::Rotate(n));
    m_cmul = fw::ComputeShader::FromSource(kernels::CMul(n));

    // twiddle table: tw[m] = e^{-i 2 pi m / N}
    std::vector<float> tw(2 * n);
    for (int m = 0; m < n; ++m) {
        tw[2 * m + 0] = std::cos(2.0 * kPi * m / n);
        tw[2 * m + 1] = -std::sin(2.0 * kPi * m / n);
    }
    glCreateBuffers(1, &m_twiddle);
    glNamedBufferData(m_twiddle, tw.size() * sizeof(float), tw.data(), GL_STATIC_DRAW);

    glCreateBuffers(1, &m_psi);
    glNamedBufferData(m_psi, m_total * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glCreateBuffers(1, &m_tmp);
    glNamedBufferData(m_tmp, m_total * 2 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    // kinetic phase multiplier kprop[(z*N+y)*N+x] = exp(-i k^2 dt/2), and the
    // real k^2/2 grid (for imaginary time and <T>).
    const double kscale = 2.0 * kPi / L;
    std::vector<float> kp(m_total * 2), k2h(m_total);
    for (int z = 0; z < n; ++z) {
        const int fz = (z <= n / 2) ? z : z - n;
        for (int y = 0; y < n; ++y) {
            const int fy = (y <= n / 2) ? y : y - n;
            for (int x = 0; x < n; ++x) {
                const int fx = (x <= n / 2) ? x : x - n;
                const double k2 = kscale * kscale * (double(fx) * fx + double(fy) * fy + double(fz) * fz);
                const double ph = -0.5 * k2 * dt;
                const long i = (long(z) * n + y) * n + x;
                kp[2 * i + 0] = (float)std::cos(ph);
                kp[2 * i + 1] = (float)std::sin(ph);
                k2h[i] = (float)(0.5 * k2);
            }
        }
    }
    glCreateBuffers(1, &m_kprop);
    glNamedBufferData(m_kprop, kp.size() * sizeof(float), kp.data(), GL_STATIC_DRAW);
    glCreateBuffers(1, &m_k2half);
    glNamedBufferData(m_k2half, k2h.size() * sizeof(float), k2h.data(), GL_STATIC_DRAW);

    // vprop defaults to identity (V = 0) until SetPotential.
    std::vector<float> ones(m_total * 2, 0.0f);
    for (long i = 0; i < m_total; ++i) ones[2 * i] = 1.0f;
    glCreateBuffers(1, &m_vprop);
    glNamedBufferData(m_vprop, ones.size() * sizeof(float), ones.data(), GL_DYNAMIC_DRAW);

    // Phase-8 buffers (float N^3 unless noted).
    const GLsizeiptr fbytes = m_total * sizeof(float);
    const GLsizeiptr cbytes = m_total * 2 * sizeof(float);
    for (GLuint* b : {&m_vnuc, &m_rho, &m_veff, &m_veffPred, &m_ktau}) {
        glCreateBuffers(1, b);
        glNamedBufferData(*b, fbytes, nullptr, GL_DYNAMIC_DRAW);
    }
    for (GLuint* b : {&m_psiPred, &m_scratchC}) {
        glCreateBuffers(1, b);
        glNamedBufferData(*b, cbytes, nullptr, GL_DYNAMIC_DRAW);
    }
    m_nGroups256 = (int)((m_total + 255) / 256);
    glCreateBuffers(1, &m_partials);
    glNamedBufferData(m_partials, (GLsizeiptr)m_nGroups256 * sizeof(float), nullptr, GL_DYNAMIC_READ);
}

void Tddft3D::EnsureKsShaders() {
    if (m_density.Id() != 0) return;
    const int n = m_n;
    m_density = fw::ComputeShader::FromSource(ks::Density(n));
    m_realToComplex = fw::ComputeShader::FromSource(ks::RealToComplex(n));
    m_poissonK = fw::ComputeShader::FromSource(ks::PoissonK(n));
    m_assembleVeff = fw::ComputeShader::FromSource(ks::AssembleVeff(n));
    m_halfKick = fw::ComputeShader::FromSource(ks::HalfKick(n));
    m_halfKickImag = fw::ComputeShader::FromSource(ks::HalfKickImag(n));
    m_mulRealK = fw::ComputeShader::FromSource(ks::MulRealK(n));
    m_kick = fw::ComputeShader::FromSource(ks::Kick(n));
    m_scale = fw::ComputeShader::FromSource(ks::Scale(n));
    m_reduceMoment = fw::ComputeShader::FromSource(ks::ReduceMoment(n));
    m_reduceWeighted = fw::ComputeShader::FromSource(ks::ReduceWeighted(n));
}

void Tddft3D::SetSoftenedNucleus(double Z, double soft) {
    std::vector<float> v(m_total);
    for (int z = 0; z < m_n; ++z)
        for (int y = 0; y < m_n; ++y)
            for (int x = 0; x < m_n; ++x) {
                const double cx = (x - m_n / 2) * m_dx, cy = (y - m_n / 2) * m_dx, cz = (z - m_n / 2) * m_dx;
                const long i = (long(z) * m_n + y) * m_n + x;
                v[i] = (float)(-Z / std::sqrt(cx * cx + cy * cy + cz * cz + soft * soft));
            }
    glNamedBufferSubData(m_vnuc, 0, v.size() * sizeof(float), v.data());
    m_ksReady = true;
}

void Tddft3D::SetPotential(const std::vector<float>& V) {
    if ((long)V.size() != m_total) throw std::runtime_error("Tddft3D::SetPotential: size mismatch");
    std::vector<float> vp(m_total * 2);
    for (long i = 0; i < m_total; ++i) {
        const double ph = -0.5 * V[i] * m_dt;
        vp[2 * i + 0] = (float)std::cos(ph);
        vp[2 * i + 1] = (float)std::sin(ph);
    }
    glNamedBufferSubData(m_vprop, 0, vp.size() * sizeof(float), vp.data());
}

GLuint Tddft3D::MakeComplexBuffer(const std::vector<cf>& data) const {
    GLuint b = 0;
    glCreateBuffers(1, &b);
    glNamedBufferData(b, data.size() * 2 * sizeof(float), data.data(), GL_DYNAMIC_DRAW);
    return b;
}

void Tddft3D::SetPsi(const std::vector<cf>& psi) {
    if ((long)psi.size() != m_total) throw std::runtime_error("Tddft3D::SetPsi: size mismatch");
    glNamedBufferSubData(m_psi, 0, psi.size() * 2 * sizeof(float), psi.data());
}

std::vector<cf> Tddft3D::GetPsi() const {
    std::vector<cf> out(m_total);
    glGetNamedBufferSubData(m_psi, 0, m_total * 2 * sizeof(float), out.data());
    return out;
}

void Tddft3D::CMul(GLuint dst, GLuint by) {
    m_cmul.Use();
    fw::ComputeShader::BindBuffer(0, dst);
    fw::ComputeShader::BindBuffer(1, by);
    m_cmul.Dispatch((GLuint)((m_total + 63) / 64), 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void Tddft3D::Fft3D(GLuint buf, bool inverse) {
    const GLuint rows = (GLuint)((long)m_n * m_n);
    const GLuint rg = (GLuint)(m_n / 8);
    GLuint a = buf, b = m_tmp;
    for (int pass = 0; pass < 3; ++pass) {
        m_fft.Use();
        m_fft.SetInt("uSign", inverse ? 1 : -1);
        m_fft.SetFloat("uScale", inverse ? 1.0f / m_n : 1.0f);
        fw::ComputeShader::BindBuffer(0, a);
        fw::ComputeShader::BindBuffer(1, m_twiddle);
        m_fft.Dispatch(rows, 1, 1);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        m_rotate.Use();
        fw::ComputeShader::BindBuffer(0, a);
        fw::ComputeShader::BindBuffer(1, b);
        m_rotate.Dispatch(rg, rg, rg);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        std::swap(a, b);
    }
    // three rotations compose to identity -> `a` is back to `buf`'s storage.
    // If not (odd number of swaps would leave it in m_tmp), copy back.
    if (a != buf) {
        glCopyNamedBufferSubData(a, buf, 0, 0, m_total * 2 * sizeof(float));
    }
}

void Tddft3D::StrangStep() {
    CMul(m_psi, m_vprop);          // half potential kick
    Fft3D(m_psi, false);
    CMul(m_psi, m_kprop);          // exact kinetic phase
    Fft3D(m_psi, true);
    CMul(m_psi, m_vprop);          // half potential kick
}

void Tddft3D::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) StrangStep();
    glFinish();
}

// ------------------------------------------------------------------ Phase 8

static void dispatchFlat(fw::ComputeShader& sh, long total) {
    sh.Dispatch((GLuint)((total + 63) / 64), 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

double Tddft3D::SumPartials(int count) {
    std::vector<float> p(count);
    glGetNamedBufferSubData(m_partials, 0, (GLsizeiptr)count * sizeof(float), p.data());
    double s = 0.0;
    for (float v : p) s += v;
    return s;
}

double Tddft3D::NormPsi() {
    m_reduceMoment.Use();
    m_reduceMoment.SetInt("uAxis", -1);
    m_reduceMoment.SetFloat("uDx", (float)m_dx);
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_partials);
    m_reduceMoment.Dispatch((GLuint)m_nGroups256, 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    return SumPartials(m_nGroups256) * m_dx * m_dx * m_dx;
}

double Tddft3D::Moment(int axis) {
    m_reduceMoment.Use();
    m_reduceMoment.SetInt("uAxis", axis);
    m_reduceMoment.SetFloat("uDx", (float)m_dx);
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_partials);
    m_reduceMoment.Dispatch((GLuint)m_nGroups256, 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    return SumPartials(m_nGroups256) * m_dx * m_dx * m_dx;
}

double Tddft3D::VExpectation() {
    m_reduceWeighted.Use();
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_veff);
    fw::ComputeShader::BindBuffer(2, m_partials);
    m_reduceWeighted.Dispatch((GLuint)m_nGroups256, 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    return SumPartials(m_nGroups256) * m_dx * m_dx * m_dx;
}

double Tddft3D::KineticExpectation() {
    glCopyNamedBufferSubData(m_psi, m_scratchC, 0, 0, m_total * 2 * sizeof(float));
    Fft3D(m_scratchC, false);
    m_reduceWeighted.Use();
    fw::ComputeShader::BindBuffer(0, m_scratchC);
    fw::ComputeShader::BindBuffer(1, m_k2half);
    fw::ComputeShader::BindBuffer(2, m_partials);
    m_reduceWeighted.Dispatch((GLuint)m_nGroups256, 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    // Parseval: sum_k |FFT_k|^2 = N^3 sum_n |psi_n|^2  ->  <T> = sum_k (k^2/2)|FFT_k|^2 * dx^3 / N^3
    return SumPartials(m_nGroups256) * (m_dx * m_dx * m_dx) / (double)m_total;
}

void Tddft3D::ScalePsi(double f) {
    m_scale.Use();
    m_scale.SetFloat("uFactor", (float)f);
    fw::ComputeShader::BindBuffer(0, m_psi);
    dispatchFlat(m_scale, m_total);
}

// psiSrc -> m_rho (occ-weighted) -> Poisson -> outVeff = V_nuc + V_H + v_xc
void Tddft3D::BuildVeff(GLuint psiSrc, GLuint outVeff, double occWeight) {
    m_density.Use();
    m_density.SetFloat("uWeight", (float)occWeight);
    m_density.SetInt("uAccumulate", 0);
    fw::ComputeShader::BindBuffer(0, psiSrc);
    fw::ComputeShader::BindBuffer(1, m_rho);
    dispatchFlat(m_density, m_total);

    m_realToComplex.Use();
    fw::ComputeShader::BindBuffer(0, m_rho);
    fw::ComputeShader::BindBuffer(1, m_scratchC);
    dispatchFlat(m_realToComplex, m_total);

    Fft3D(m_scratchC, false);
    m_poissonK.Use();
    m_poissonK.SetFloat("uKScale", (float)(2.0 * kPi / m_L));
    fw::ComputeShader::BindBuffer(0, m_scratchC);
    m_poissonK.Dispatch((GLuint)(m_n / 4), (GLuint)(m_n / 4), (GLuint)(m_n / 4));
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    Fft3D(m_scratchC, true);      // m_scratchC.x = V_H (real)

    m_assembleVeff.Use();
    fw::ComputeShader::BindBuffer(0, m_vnuc);
    fw::ComputeShader::BindBuffer(1, m_scratchC);
    fw::ComputeShader::BindBuffer(2, m_rho);
    fw::ComputeShader::BindBuffer(3, outVeff);
    dispatchFlat(m_assembleVeff, m_total);
}

Tddft3D::RelaxResult Tddft3D::RelaxKS(int nElectrons, double dtau, int maxIter, double tol,
                                     int inner, bool verbose) {
    EnsureKsShaders();
    if (!m_ksReady) throw std::runtime_error("RelaxKS: call SetSoftenedNucleus first");
    const double occ = (nElectrons >= 2) ? 2.0 : 1.0;

    // m_ktau = exp(-k^2 dtau / 2) from the k^2/2 grid.
    std::vector<float> k2h(m_total), ktau(m_total);
    glGetNamedBufferSubData(m_k2half, 0, m_total * sizeof(float), k2h.data());
    for (long i = 0; i < m_total; ++i) ktau[i] = (float)std::exp(-(double)k2h[i] * dtau);
    glNamedBufferSubData(m_ktau, 0, ktau.size() * sizeof(float), ktau.data());

    // seed: a normalized Gaussian blob
    std::vector<cf> seed(m_total);
    double s2 = 0.0;
    for (int z = 0; z < m_n; ++z)
        for (int y = 0; y < m_n; ++y)
            for (int x = 0; x < m_n; ++x) {
                const double cx = (x - m_n / 2) * m_dx, cy = (y - m_n / 2) * m_dx, cz = (z - m_n / 2) * m_dx;
                const double g = std::exp(-(cx * cx + cy * cy + cz * cz));
                seed[(long(z) * m_n + y) * m_n + x] = cf((float)g, 0.0f);
                s2 += g * g;
            }
    const float sc = (float)(1.0 / std::sqrt(s2 * m_dx * m_dx * m_dx));
    for (auto& v : seed) v *= sc;
    SetPsi(seed);

    double epsPrev = 0.0;
    int it = 0;
    for (it = 1; it <= maxIter; ++it) {
        BuildVeff(m_psi, m_veff, occ);
        for (int s = 0; s < inner; ++s) {
            m_halfKickImag.Use();
            m_halfKickImag.SetFloat("uDtau", (float)dtau);
            fw::ComputeShader::BindBuffer(0, m_psi);
            fw::ComputeShader::BindBuffer(1, m_veff);
            dispatchFlat(m_halfKickImag, m_total);

            Fft3D(m_psi, false);
            m_mulRealK.Use();
            fw::ComputeShader::BindBuffer(0, m_psi);
            fw::ComputeShader::BindBuffer(1, m_ktau);
            dispatchFlat(m_mulRealK, m_total);
            Fft3D(m_psi, true);

            m_halfKickImag.Use();
            m_halfKickImag.SetFloat("uDtau", (float)dtau);
            fw::ComputeShader::BindBuffer(0, m_psi);
            fw::ComputeShader::BindBuffer(1, m_veff);
            dispatchFlat(m_halfKickImag, m_total);

            ScalePsi(1.0 / std::sqrt(NormPsi()));
        }
        const double eps = KineticExpectation() + VExpectation();
        if (verbose && it % 20 == 0)
            std::printf("      relax it %3d: eps = %.6f\n", it, eps);
        if (it > 1 && std::abs(eps - epsPrev) < tol) break;
        epsPrev = eps;
    }
    m_occWeight = occ;
    return {epsPrev, occ * NormPsi(), it};
}

void Tddft3D::EtrsStepKS(double occWeight) {
    auto halfKick = [&](GLuint psi, GLuint v) {
        m_halfKick.Use();
        m_halfKick.SetFloat("uDt", (float)m_dt);
        fw::ComputeShader::BindBuffer(0, psi);
        fw::ComputeShader::BindBuffer(1, v);
        dispatchFlat(m_halfKick, m_total);
    };

    // V0 = V_KS[rho(t)]
    BuildVeff(m_psi, m_veff, occWeight);

    // predictor: a full Strang step of a copy under V0 -> psi_pred
    glCopyNamedBufferSubData(m_psi, m_psiPred, 0, 0, m_total * 2 * sizeof(float));
    halfKick(m_psiPred, m_veff);
    Fft3D(m_psiPred, false);
    CMul(m_psiPred, m_kprop);
    Fft3D(m_psiPred, true);
    halfKick(m_psiPred, m_veff);

    // V1 = V_KS[rho(psi_pred)]
    BuildVeff(m_psiPred, m_veffPred, occWeight);

    // real ETRS step: exp(-iV1 dt/2) . T(dt) . exp(-iV0 dt/2)
    halfKick(m_psi, m_veff);
    Fft3D(m_psi, false);
    CMul(m_psi, m_kprop);
    Fft3D(m_psi, true);
    halfKick(m_psi, m_veffPred);
}

Tddft3D::DipoleTrace Tddft3D::KickAndRunKS(double kappa, int axis, int nSteps, int recordEvery) {
    EnsureKsShaders();
    m_kick.Use();
    m_kick.SetFloat("uKappa", (float)kappa);
    m_kick.SetFloat("uDx", (float)m_dx);
    m_kick.SetInt("uAxis", axis);
    fw::ComputeShader::BindBuffer(0, m_psi);
    dispatchFlat(m_kick, m_total);

    DipoleTrace tr;
    auto record = [&](double t) {
        tr.t.push_back(t);
        // first moment integral x_a rho d^3r -- same sign convention as the
        // Stage-1 Python response.moment_observer; spectrum.py applies
        // alpha = +(1/kappa) FT[.] so Im alpha > 0 at resonances.
        tr.d.push_back(m_occWeight * Moment(axis));
    };
    record(0.0);
    for (int step = 1; step <= nSteps; ++step) {
        EtrsStepKS(m_occWeight);
        if (step % recordEvery == 0) record(step * m_dt);
    }
    glFinish();
    return tr;
}

} // namespace tddft
