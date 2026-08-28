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

private:
    void CreateBuffers();
    void UploadInitial();
    void BuildPropagators(const fw::Deck& deck);
    void RunStep();
    void Fft(GLuint buffer, bool inverse);
    void TransposeBuf(GLuint src, GLuint dst);
    void CMulBuf(GLuint dst, GLuint by);
    void Readback() const;

    Grid m_grid;
    double m_dt = 0.002;
    int m_substepsPerFrame = 10;
    double m_transmissionX = 0.0;
    std::string m_title = "2D TDSE (GPU)";
    std::vector<std::string> m_diagNames;
    long m_stepsDone = 0;

    // GPU state (all vec2 complex, row-major, except m_potential which is float).
    GLuint m_psi = 0;
    GLuint m_tmp = 0;
    GLuint m_spec = 0;    // scratch for the diagnostic 2D FFT
    GLuint m_vprop = 0;   // exp(-i V dt/2) * exp(-W dt/2)
    GLuint m_kprop = 0;   // exp(-i (kx^2+ky^2)/2 dt), transposed layout
    GLuint m_twiddle = 0; // length N
    GLuint m_potential = 0;
    GLuint m_stat = 0;    // 1 uint: max |psi|^2 for the interactive view

    fw::ComputeShader m_fft;
    fw::ComputeShader m_transpose;
    fw::ComputeShader m_cmul;
    fw::ComputeShader m_reduceMax;

    // Interactive view.
    fw::Shader m_view;
    GLuint m_vao = 0;
    int m_mode = 0;        // 0 phase-colored density, 1 magma density, 2 Re(psi)
    float m_gain = 1.15f;  // linear push before the perceptual curve
    float m_gamma = 0.5f;  // density brightness curve; <1 lifts tails/fringe minima
    float m_zoom = 1.0f;
    glm::vec2 m_panPix{0.0f, 0.0f};

    // CPU mirrors.
    std::vector<std::complex<float>> m_initial;
    std::vector<float> m_vCpu;
    mutable std::vector<std::complex<float>> m_scratch;
};

} // namespace tdse
