#pragma once

#include "ChargePath.hpp"
#include "LwFields.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Shader.hpp"
#include "framework/Simulation.hpp"
#include "framework/Text.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace fw { class Deck; }

namespace lw {

// Lienard-Wiechert radiated fields of prescribed point charges, evaluated on
// a 2D slice grid by a GPU compute shader (Phase 1). Self-consistent charge
// dynamics come in Phase 3.
class RetardedFieldsSim : public fw::Simulation {
public:
    RetardedFieldsSim() = default;
    ~RetardedFieldsSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;
    void OnKey(int key, int action) override;

    // For --selftest / --gpu-selftest.
    int Nx() const { return m_nx; }
    int Ny() const { return m_ny; }
    double Dt() const { return m_dt; }
    double Time() const { return m_time; }
    void ForceCpu(bool cpu) { m_forceCpu = cpu; }
    PathFn PathFor(int c) const;               // callable (r,v,a)(tau) for charge c
    void ComputeFieldCpu();                     // fill m_Ex..m_Sr on the CPU
    void ComputeField();                        // GPU (or CPU if forced/no GL)
    const std::vector<double>& Ex() const { return m_Ex; }
    const std::vector<double>& Ey() const { return m_Ey; }
    const std::vector<double>& Ez() const { return m_Ez; }
    const std::vector<double>& Bz() const { return m_Bz; }
    double LightSpeed() const { return m_c; }
    double Eps0() const { return m_eps0; }

private:
    void EnsureGpu();
    void EnsureRenderResources();
    void RepackField();

    int m_nx = 256, m_ny = 256;
    double m_lx = 20.0, m_ly = 20.0;
    double m_sliceZ = 0.0;
    double m_c = 1.0, m_eps0 = 1.0;
    double m_dt = 0.05;
    int m_substepsPerFrame = 1;
    long m_totalSteps = 400;
    std::string m_title = "Lienard-Wiechert radiated fields";

    std::vector<ChargePath> m_paths;
    std::vector<ChargePath> m_pathsInit;
    std::vector<double> m_q;

    double m_time = 0.0;
    long m_step = 0;

    // field grids (row-major, index = j*nx + i)
    std::vector<double> m_Ex, m_Ey, m_Ez, m_Emag, m_Bz, m_Sr;

    double x(int i) const { return -0.5 * m_lx + (i + 0.5) * (m_lx / m_nx); }
    double y(int j) const { return -0.5 * m_ly + (j + 0.5) * (m_ly / m_ny); }
    std::size_t idx(int i, int j) const {
        return static_cast<std::size_t>(j) * m_nx + i;
    }

    // GPU
    bool m_forceCpu = false;
    bool m_gpuInit = false;
    fw::ComputeShader m_prog;
    GLuint m_bE = 0, m_bB = 0, m_bTret = 0;
    std::vector<float> m_readE, m_readB;

    // interactive view
    bool m_renderReady = false;
    fw::Shader m_view;
    GLuint m_vao = 0, m_fieldBuf = 0;
    std::vector<float> m_fieldScratch;
    int m_viewMode = 0;          // 0 |E|, 1 Ex, 2 Ez, 3 S_radial
    bool m_logScale = false;
    bool m_radWeight = false;    // multiply display by r to flatten the near field
    float m_zoom = 1.0f, m_gain = 1.0f, m_gamma = 0.8f;
    glm::vec2 m_panPix{0.0f, 0.0f};
    int m_activeCharge = 0;
    std::unique_ptr<fw::TextRenderer> m_text;
    fw::Font m_font;
};

} // namespace lw
