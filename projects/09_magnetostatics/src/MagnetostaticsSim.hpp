#pragma once

#include "BiotSavart.hpp"
#include "FieldSolver.hpp"
#include "Grid.hpp"
#include "Multigrid.hpp"
#include "SlicePlane.hpp"
#include "Sources.hpp"

#include "framework/Shader.hpp"
#include "framework/Simulation.hpp"
#include "framework/Text.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <memory>
#include <string>
#include <vector>

namespace fw { class Deck; }

namespace mag {

// 2D magnetostatics on the Simulation/Deck/headless model.
//
//   Phase 1  laplacian(A_z) = -mu0 J_z by RB-GS/SOR, B = curl(A_z zhat).
//   Phase 2  geometric multigrid solver (default) + an interactive view:
//            a live heatmap of |B| / B_x / B_y / A_z (cycle with M), an
//            optional B field-line overlay (L), and an "active" wire the
//            arrow keys move -- each move re-rasterises J_z and re-solves
//            (multigrid, warm-started), so the field updates live.
//
// The problem is static: Step() is a no-op and a headless run writes one
// frame. Later phases add 3D Biot-Savart coils and a Boris test-particle
// pusher.
class MagnetostaticsSim : public fw::Simulation {
public:
    MagnetostaticsSim() = default;
    ~MagnetostaticsSim() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;
    void OnKey(int key, int action) override;

    // Read-only access for --selftest / unit checks.
    const Grid& GridInfo() const { return m_grid; }
    const std::vector<double>& Az() const { return m_Az; }
    const std::vector<double>& Bx() const { return m_Bx; }
    const std::vector<double>& By() const { return m_By; }

private:
    void UpdateField(bool warmStart = false);   // Poisson solve OR Biot-Savart
    void SolvePoissonPath(bool warmStart);
    void BiotSavartPath();
    void RebuildAvoidPoints();
    void EnsureRenderResources();
    void RepackField();                       // chosen scalar -> m_fieldScratch
    void RebuildFieldLines();                 // integrate B streamlines -> m_lineVerts

    Grid m_grid;
    double m_mu0 = 1.0;
    std::string m_title = "2D Magnetostatics";
    std::string m_method = "multigrid";       // "multigrid" | "rbgs"
    SolveOptions m_solveOpt;
    MultigridOptions m_mgOpt;

    // --- current sources -------------------------------------------------
    Sources m_sources;                        // 2D out-of-plane wires / strips
    std::vector<CoilSpec> m_coils;            // 3D coils (Biot-Savart path)
    std::vector<WireSegment> m_segments;
    BiotSavartField m_biot;
    SlicePlane m_plane;
    bool m_useBiot = false;

    std::vector<double> m_Jz;
    std::vector<double> m_Az;
    std::vector<double> m_Bx, m_By, m_Bz;     // Bx, By = in-plane; Bz = perpendicular
    std::vector<unsigned char> m_fixedMask;
    std::vector<double> m_fixedValues;
    std::vector<glm::dvec2> m_avoidPts;       // in-plane source locations (field-line stops)

    SolveResult m_lastSolve;
    double m_lastSolveMs = 0.0;

    // --- interactive view (never touched / allocated by the headless path) --
    bool m_renderReady = false;
    fw::Shader m_view;
    fw::Shader m_line;
    GLuint m_vao = 0;
    GLuint m_fieldBuf = 0;
    GLuint m_lineVao = 0, m_lineVbo = 0;
    std::vector<float> m_fieldScratch;
    std::vector<float> m_lineVerts;           // xy pairs, GL_LINES
    int m_lineVertCount = 0;

    int m_mode = 0;                           // 0 |B|, 1 Bx, 2 By, 3 Bz, 4 A_z
    bool m_showLines = true;
    int m_activeWire = 0;
    float m_zoom = 1.0f;
    float m_gain = 1.0f;
    float m_gamma = 1.0f;
    glm::vec2 m_panPix{0.0f, 0.0f};
    bool m_fieldDirty = true;

    std::unique_ptr<fw::TextRenderer> m_text;
    fw::Font m_font;
};

} // namespace mag
