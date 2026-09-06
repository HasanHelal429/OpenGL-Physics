#include "Tddft3D.hpp"

#include "kernels3d.hpp"

#include <cmath>
#include <stdexcept>

namespace tddft {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Tddft3D::~Tddft3D() {
    GLuint bufs[] = {m_psi, m_tmp, m_twiddle, m_vprop, m_kprop};
    glDeleteBuffers(5, bufs);
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

    // kinetic phase multiplier kprop[(z*N+y)*N+x] = exp(-i k^2 dt/2)
    const double kscale = 2.0 * kPi / L;
    std::vector<float> kp(m_total * 2);
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
            }
        }
    }
    glCreateBuffers(1, &m_kprop);
    glNamedBufferData(m_kprop, kp.size() * sizeof(float), kp.data(), GL_STATIC_DRAW);

    // vprop defaults to identity (V = 0) until SetPotential.
    std::vector<float> ones(m_total * 2, 0.0f);
    for (long i = 0; i < m_total; ++i) ones[2 * i] = 1.0f;
    glCreateBuffers(1, &m_vprop);
    glNamedBufferData(m_vprop, ones.size() * sizeof(float), ones.data(), GL_DYNAMIC_DRAW);
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

} // namespace tddft
