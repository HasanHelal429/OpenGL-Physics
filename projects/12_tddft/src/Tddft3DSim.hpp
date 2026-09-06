#pragma once

#include "Tddft3D.hpp"

#include "framework/Shader.hpp"
#include "framework/Simulation.hpp"

#include <glad/glad.h>

#include <string>
#include <vector>

namespace tddft {

// Deck-driven rt-TDDFT run for the headless writer and the interactive view.
// Wraps Tddft3D: relax the Kohn-Sham ground state, optionally delta-kick, then
// advance under a flat-top laser with an absorbing boundary. The interactive
// Render shows the z = 0 density slice; Snapshot writes that slice plus the
// dipole / ionized-fraction diagnostics.
class Tddft3DSim : public fw::Simulation {
public:
    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnKey(int key, int action) override;

private:
    double Field(double t) const;   // flat-top E(t)

    Tddft3D m_p;
    int m_n = 0;
    double m_L = 0.0, m_dt = 0.0;
    int m_substeps = 4;
    std::string m_title = "rt-TDDFT (GPU)";

    // drive
    double m_E0 = 0.0, m_wL = 0.114, m_tUp = 0.0, m_tFlat = 0.0, m_tPulse = 0.0;
    int m_axis = 2;
    double m_kick = 0.0;
    int m_kickAxis = 0;

    double m_time = 0.0;
    long m_steps = 0;

    // view
    fw::Shader m_view;
    GLuint m_vao = 0;
    float m_gamma = 0.45f;
};

} // namespace tddft
