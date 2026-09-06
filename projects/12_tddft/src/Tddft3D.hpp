#pragma once

#include "framework/ComputeShader.hpp"

#include <glad/glad.h>

#include <complex>
#include <vector>

namespace tddft {

using cf = std::complex<float>;

// 3D split-step-Fourier propagator on the GPU. i d/dt psi = [-1/2 grad^2 + V] psi
// (hbar = m = 1), periodic cube of side L, N^3 grid (N a power of two).
//
// Phase 7 scope: a single orbital under a FIXED real potential. The half-kick
// and kinetic multipliers (vprop, kprop) are precomputed on the CPU and applied
// by a pointwise complex-multiply kernel; the 3D FFT is three (1D-FFT + rotate)
// passes. This is the direct 3D analogue of 05_tdse_gpu's 2D method and the
// reference target is the Stage-1 Python propagator (Quantum Mechanics/TDDFT).
class Tddft3D {
public:
    Tddft3D() = default;
    ~Tddft3D();
    Tddft3D(const Tddft3D&) = delete;
    Tddft3D& operator=(const Tddft3D&) = delete;

    // L in Bohr, dt in atomic units. Builds shaders, twiddle table, buffers.
    void Configure(int n, double L, double dt);

    // V is a real N^3 grid, row-major index = (z*N + y)*N + x, in the same
    // centered-coordinate convention as x[i] = (i - N/2) * dx.
    void SetPotential(const std::vector<float>& V);

    void SetPsi(const std::vector<cf>& psi);           // N^3, same layout
    std::vector<cf> GetPsi() const;                    // readback

    void Step(int substeps);                           // `substeps` Strang steps

    // Forward (inverse=false) or inverse 3D FFT of an arbitrary N^3 vec2 buffer,
    // in place. Exposed for the FFT self-test.
    void Fft3D(GLuint buf, bool inverse);

    int N() const { return m_n; }
    double Dx() const { return m_dx; }
    double Dt() const { return m_dt; }
    GLuint MakeComplexBuffer(const std::vector<cf>& data) const;

private:
    void StrangStep();
    void CMul(GLuint dst, GLuint by);

    int m_n = 0;
    long m_total = 0;
    double m_L = 0.0, m_dx = 0.0, m_dt = 0.0;

    GLuint m_psi = 0;
    GLuint m_tmp = 0;       // rotate scratch
    GLuint m_twiddle = 0;   // length N
    GLuint m_vprop = 0;     // exp(-i V dt/2), N^3 vec2
    GLuint m_kprop = 0;     // exp(-i k^2 dt/2), N^3 vec2

    fw::ComputeShader m_fft;
    fw::ComputeShader m_rotate;
    fw::ComputeShader m_cmul;
};

} // namespace tddft
