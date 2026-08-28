#pragma once

#include "TdseField.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Shader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <complex>
#include <string>
#include <vector>

namespace tdse {

// 2-D time-dependent Schrodinger equation, i d/dt psi = [-1/2 grad^2 + V] psi
// (hbar = m = 1), integrated by Strang-split split-step Fourier entirely on the
// GPU: half potential kick (a precomputed complex multiplier that also folds in
// the border absorbing potential), a 2-D FFT via row-FFT + transpose, the exact
// kinetic phase in k-space, the inverse transform, and the second half kick.
// The 1-D FFT compute shader (kernels.hpp) is the performance-critical piece.
class TdseSim : public fw::Simulation {
public:
    TdseSim() = default;
    ~TdseSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;
    void OnKey(int key, int action) override;

    struct RelaxOptions {
        int states = 1;
        int steps = 2000;
        double dtau = 0.01;
        int orthoEvery = 1;   // Gram-Schmidt every N steps (SCF warm starts can use >1)
        const std::vector<float>* extraPotential = nullptr;  // added to V0 (e.g. V_Hartree)
        const std::vector<std::vector<std::complex<float>>>* warmStart = nullptr;
        bool quiet = false;
    };
    struct RelaxResult {
        std::vector<std::vector<std::complex<float>>> states;  // K normalized eigenstates
        std::vector<double> energies;                          // ascending
    };
    // Imaginary-time relaxation to the `states` lowest eigenstates of
    // -1/2 grad^2 + V0 (+ extraPotential). Uses the grid/FFT set up by Configure.
    RelaxResult Relax(const RelaxOptions& opt);

    // Self-consistent Poisson-Schrodinger (Phase G4): relax K states in
    // V_ext + V_H, fill `electrons`, solve Poisson, mix, iterate. Writes results
    // to outDir. Returns 0 on success.
    int RunScf(const fw::Deck& deck, const RelaxOptions& relaxOpt, int electrons,
               const std::string& outDir);

    // FFT-Poisson: nabla^2 V_H = -coupling * (rho - <rho>), periodic. rho and
    // the returned V_H are CPU grids [y*n+x]. Uses the GPU FFT.
    std::vector<float> PoissonSolve(const std::vector<float>& rho, double coupling);

    const Grid& GridInfo() const { return m_grid; }
    const std::vector<float>& StaticPotential() const { return m_vCpu; }

private:
    // A time-dependent drive term (see kernels::BuildVprop): type 0 = tilt, 1 = gate.
    struct DriveTerm {
        int type = 0;
        glm::vec4 a{0.0f};
        glm::vec4 b{0.0f};
    };

    void CreateBuffers();
    void UploadInitial();
    void BuildPropagators(const fw::Deck& deck);
    void RebuildVprop(double t);
    void ComputeMeanField();   // rho=|psi|^2 -> FFT-Poisson -> m_vhartree
    void RunStep(double t0);
    void RunStepMagnetic(double t0);
    void Rotate(double alpha);       // exp(i alpha L_z): rotate psi(phi) -> psi(phi + alpha)
    void ShearApply(GLuint buf, double amount, double origin, double step, double kscale);
    void Fft(GLuint buffer, bool inverse);
    void TransposeBuf(GLuint src, GLuint dst);
    void CMulBuf(GLuint dst, GLuint by);
    void ComputeCurrent();
    void Readback() const;

    Grid m_grid;
    double m_dt = 0.002;
    int m_substepsPerFrame = 10;
    double m_transmissionX = 0.0;
    std::string m_title = "2D TDSE (GPU)";
    std::vector<std::string> m_diagNames;
    long m_stepsDone = 0;
    double m_time = 0.0;
    std::vector<DriveTerm> m_drives;
    bool m_hasDrives = false;
    double m_B = 0.0;          // uniform magnetic field (symmetric gauge)
    bool m_hasMagnetic = false;
    bool m_hasMeanfield = false;
    double m_mfCoupling = 0.0;

    // GPU state (all vec2 complex, row-major, except m_potential which is float).
    GLuint m_psi = 0;
    GLuint m_tmp = 0;
    GLuint m_spec = 0;    // scratch for the diagnostic 2D FFT
    GLuint m_vprop = 0;   // exp(-i V dt/2) * exp(-W dt/2)
    GLuint m_kprop = 0;   // exp(-i (kx^2+ky^2)/2 dt), transposed layout
    GLuint m_twiddle = 0; // length N
    GLuint m_potential = 0;   // static external potential V0 (float), also BuildVprop input
    GLuint m_cap = 0;         // absorbing rate W (float), BuildVprop input
    GLuint m_vhartree = 0;    // Hartree potential V_H (float); zero unless [meanfield]
    GLuint m_stat = 0;        // 1 uint: max |psi|^2 for the interactive view
    GLuint m_currentBuf = 0;  // vec2 probability current j
    GLuint m_statJ = 0;       // 1 uint: max |j|^2
    GLuint m_kdisp = 0;       // vec2 psi-hat, centred, for the momentum view
    GLuint m_statK = 0;       // 1 uint: max |psi-hat|^2

    fw::ComputeShader m_fft;
    fw::ComputeShader m_transpose;
    fw::ComputeShader m_cmul;
    fw::ComputeShader m_reduceMax;
    fw::ComputeShader m_currentProg;
    fw::ComputeShader m_fftshift;
    fw::ComputeShader m_buildVprop;
    fw::ComputeShader m_shearPhase;
    fw::ComputeShader m_mfRho;
    fw::ComputeShader m_poissonMul;
    fw::ComputeShader m_extractReal;

    // Interactive view.
    fw::Shader m_view;
    fw::Shader m_arrows;      // probability-current overlay
    GLuint m_vao = 0;
    GLuint m_arrowVao = 0;
    int m_mode = 0;        // 0 phase-colored density, 1 magma density, 2 Re(psi)
    float m_gain = 1.15f;  // linear push before the perceptual curve
    float m_gamma = 0.5f;  // density brightness curve; <1 lifts tails/fringe minima
    float m_zoom = 1.0f;
    glm::vec2 m_panPix{0.0f, 0.0f};
    bool m_showCurrent = false;
    bool m_writeCurrent = false;
    bool m_showMomentum = false;
    double m_kMax = 1.0;   // Nyquist |kx| = pi n / lx
    double m_kView = 1.0;  // half-window shown in momentum mode

    void ComputeSpectrumInto(GLuint dst);  // 2D FFT of psi -> dst (transposed [kxIdx][kyIdx])

    // CPU mirrors.
    std::vector<std::complex<float>> m_initial;
    std::vector<float> m_vCpu;
    mutable std::vector<std::complex<float>> m_scratch;
};

} // namespace tdse
