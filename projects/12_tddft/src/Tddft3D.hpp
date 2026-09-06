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

    // ---- Phase 8: Kohn-Sham (Hartree + ALDA) --------------------------------

    // A softened point charge -Z / sqrt(r^2 + soft^2) at the box centre, uploaded
    // as the nuclear potential. Enables the KS path.
    void SetSoftenedNucleus(double Z, double softening);

    struct RelaxResult { double eps = 0.0; double n_electrons = 0.0; int iters = 0; };
    // Imaginary-time Kohn-Sham ground state for `nElectrons` (>=1, filling one
    // spatial orbital at 2 e- -- He/H are the scope). Returns the KS eigenvalue.
    RelaxResult RelaxKS(int nElectrons, double dtau, int maxIter = 500, double tol = 1e-7,
                        int inner = 4, bool verbose = false);

    // Delta-kick psi -> exp(i kappa x_axis) psi, then ETRS-propagate the KS
    // system for `nSteps` of dt, recording the electronic first moment
    // integral x_axis rho d^3r every `recordEvery` steps.
    struct DipoleTrace { std::vector<double> t, d; };
    DipoleTrace KickAndRunKS(double kappa, int axis, int nSteps, int recordEvery);

    // Bare single-particle mode: V_eff = V_nuc only (no Hartree/XC) -- the
    // self-interaction-free limit used for hydrogen HHG, matching Stage-1's
    // method=None. Call before RelaxKS.
    void SetBareMode(bool bare) { m_bare = bare; }

    // Cosine-taper absorbing boundary over the outer `width` (Bohr), applied
    // to psi once per step (soaks up the ionized flux; norm loss = ionization).
    void SetMask(double width, int order = 2);

    // One ETRS step with a length-gauge laser term E(t) x_axis added to V_eff.
    void LaserStepKS(double fieldNow, double fieldNext, int axis);

    void Kick(double kappa, int axis);                 // psi -> exp(i kappa x) psi
    double Dipole(int axis) { return m_occWeight * Moment(axis); }
    double SurvivingNorm() { return m_occWeight * NormPsi(); }
    void DensitySliceZ0(std::vector<float>& out);      // N*N radial density at z = N/2
    double OccWeight() const { return m_occWeight; }

private:
    void StrangStep();
    void CMul(GLuint dst, GLuint by);

    void BuildVeff(GLuint psiSrc, GLuint outVeff, double occWeight);
    double NormPsi();                           // integral |psi|^2 d^3r  (GPU reduce)
    void ScalePsi(double f);
    double Moment(int axis);                    // integral coord_axis |psi|^2 d^3r
    double VExpectation();                      // integral V_eff |psi|^2 d^3r
    double KineticExpectation();                // <psi| -1/2 grad^2 |psi>
    double SumPartials(int count);

    int m_n = 0;
    long m_total = 0;
    double m_L = 0.0, m_dx = 0.0, m_dt = 0.0;

    GLuint m_psi = 0;
    GLuint m_tmp = 0;       // rotate scratch
    GLuint m_twiddle = 0;   // length N
    GLuint m_vprop = 0;     // exp(-i V dt/2), N^3 vec2   (Phase 7 fixed-V path)
    GLuint m_kprop = 0;     // exp(-i k^2 dt/2), N^3 vec2 (Phase 7)

    fw::ComputeShader m_fft;
    fw::ComputeShader m_rotate;
    fw::ComputeShader m_cmul;

    // Phase 8-9
    bool m_ksReady = false;
    bool m_bare = false;
    GLuint m_mask = 0;     // float N^3 absorbing window (0 = disabled)
    GLuint m_vnuc = 0;      // float N^3
    GLuint m_rho = 0;       // float N^3
    GLuint m_veff = 0;      // float N^3
    GLuint m_veffPred = 0;  // float N^3 (ETRS V1)
    GLuint m_psiPred = 0;   // vec2 N^3
    GLuint m_scratchC = 0;  // vec2 N^3 (Poisson / kinetic-expectation FFT)
    GLuint m_k2half = 0;    // float N^3  = |k|^2 / 2
    GLuint m_ktau = 0;      // float N^3  = exp(-|k|^2 dtau / 2)
    GLuint m_partials = 0;  // float, one per reduction workgroup
    int m_nGroups256 = 0;

    double m_occWeight = 2.0;

    fw::ComputeShader m_density, m_realToComplex, m_poissonK, m_assembleVeff;
    fw::ComputeShader m_halfKick, m_halfKickImag, m_mulRealK, m_kick;
    fw::ComputeShader m_scale, m_reduceMoment, m_reduceWeighted;
    fw::ComputeShader m_addField, m_maskMul;

    void EnsureKsShaders();
    void EtrsStepKS(double occWeight, double fieldNow, double fieldNext, int fieldAxis);
};

} // namespace tddft
