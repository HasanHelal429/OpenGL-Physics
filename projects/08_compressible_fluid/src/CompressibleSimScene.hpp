#pragma once

#include "Euler2D.hpp"

#include "framework/Shader.hpp"
#include "framework/Simulation.hpp"
#include "framework/Text.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace cf {

// A single, generic 2D fw::Simulation driven entirely by deck content --
// grid, physics, boundary conditions (per side, with an optional inflow
// profile), an obstacle list, an initial-condition type (+ optional
// perturbation), named probes, and a tracer toggle. Replaces
// CompressibleSim2D/CompressibleSimChannel/CompressibleSimTaylorGreen/
// CompressibleSimCylinder: every scenario those four covered is now a deck
// under this one schema (see decks/riemann2d_config3.toml,
// decks/poiseuille_channel.toml, decks/taylor_green.toml,
// decks/cylinder_re100.toml), not a new C++ class. A genuinely new SHAPE of
// initial condition, inflow profile, or obstacle still needs new C++ (a new
// named case in CompressibleSimScene.cpp's parsing) -- but every
// combination of the existing pieces (any grid + any per-side BC + any
// obstacle list + any existing IC type + optional tracer) is pure data.
//
// The 1D shock tube (CompressibleSim, Euler1D) is deliberately NOT folded
// into this schema -- boundary "sides" (plural), 2D obstacles, and probes
// are 2D-specific concepts with no natural 1D analog, and forcing them
// into a shared schema would only obscure both.
class CompressibleSimScene : public fw::Simulation {
public:
    ~CompressibleSimScene() override;

    void Configure(const fw::Deck& deck) override;
    void Reset() override;
    void Step(int substeps) override;
    void Snapshot(fw::OutputWriter& writer) override;
    fw::SimInfo Info() const override;

    // Interactive-only (see framework/Simulation.hpp's comment: these
    // default to no-ops for a pure batch sim, which is what this class was
    // until now). Draws a live heatmap of one scalar field at a time
    // (density/speed/vorticity/tracer, cycled with 'M'), with any obstacle
    // cells overlaid as a flat color, plus a small on-screen label naming
    // whichever field is currently displayed (so the mode is visible
    // without having to already know the keybinding). Euler2D's state
    // lives on the CPU (see its m_u comment), not in a GPU buffer the way
    // e.g. 05_tdse_gpu's solver does, so Render() re-packs the chosen field
    // from Euler2D::PrimAt into a small SSBO every frame rather than
    // keeping a persistent GPU-resident copy in sync with the solver. If
    // the deck has an airfoil obstacle, Up/Down also adjust its angle of
    // attack live (see OnKey), with the current angle shown in the label.
    void Render(int fbWidth, int fbHeight) override;
    void OnViewInput(const fw::ViewInput& in) override;
    void OnKey(int key, int action) override;

private:
    // Precomputed once in Configure() from the deck's physical (x,y) probe
    // coordinates -- (i,j) never changes since the grid is fixed after Init().
    struct Probe {
        std::string name;
        int i = 0, j = 0;
    };

    // One obstacle's geometry, kept around (not just baked into the mask
    // lambda and discarded) so an airfoil's angle of attack can be adjusted
    // live in interactive mode -- see OnKey's Up/Down handling and
    // RebuildObstacleMask(). A circle's radius and an airfoil's
    // chord/thickness/angle are mutually exclusive, but a tagged union
    // would be overkill for a struct this small with only ever 1-2
    // instances alive.
    struct ObstacleSpec {
        bool isAirfoil = false;
        double cx = 0.0, cy = 0.0;
        double radius = 0.0;                                    // circle
        double chord = 0.0, thicknessFrac = 0.0, angleRad = 0.0; // airfoil (NACA00xx, symmetric)
    };

    // Lazily creates the shader/VAO/SSBOs Render() needs, sized to the
    // solver's (fixed, post-Configure) grid -- done on first Render() call
    // rather than in Configure() so a headless deck run (which never calls
    // Render()) never allocates any of this.
    void EnsureRenderResources();
    // (Re)builds Euler2D's obstacle mask from the current m_obstacles --
    // called once from Configure() and again every time OnKey adjusts an
    // airfoil's angleRad. Cheap: SetObstacleMask evaluates the returned
    // predicate once per cell (see Euler2D::SetObstacleMask), and this
    // project's decks have at most a handful of obstacles.
    void RebuildObstacleMask();
    // Symmetric NACA00xx thickness distribution, evaluated in the airfoil's
    // own body frame and rotated by `o.angleRad` -- see the .cpp definition
    // for the formula and derivation. A static member (not a free function)
    // only because ObstacleSpec is private; it doesn't touch `this`.
    static bool IsInsideAirfoil(double x, double y, const ObstacleSpec& o);

    Euler2D m_solver;
    // Captures the initial-condition type + any perturbation, fully baked
    // (all deck-read parameters closed over by value) at Configure() time,
    // so Reset() can restore the initial state without re-reading the deck
    // -- the same convention every Simulation in this project follows (see
    // e.g. MDSim.hpp's comment on Reset() not re-parsing the deck).
    std::function<Prim2D(double x, double y)> m_icFn;
    std::vector<Probe> m_probes;
    // The deck's obstacle list, kept live (not just captured into the mask
    // lambda) so interactive mode can rewrite an airfoil's angleRad and
    // rebuild the mask without re-parsing the deck. Reset() deliberately
    // does NOT restore this to the deck's original angle -- see OnKey's
    // comment for why angle of attack behaves like a live control knob,
    // not part of the "initial condition" Reset() restores.
    std::vector<ObstacleSpec> m_obstacles;
    bool m_hasAirfoil = false;
    bool m_writeTracer = false;
    double m_cfl = 0.4;
    int m_substepsPerFrame = 20;
    double m_dt = 0.0;
    std::string m_title = "Compressible Navier-Stokes -- generic 2D scene";

    // Interactive rendering state -- untouched, and never allocated, by the
    // headless path (Render() is only ever called from fw::SimApp).
    fw::Shader m_view;
    GLuint m_vao = 0;
    GLuint m_fieldBuf = 0, m_maskBuf = 0;
    bool m_renderReady = false;
    int m_renderMode = 0; // 0=density, 1=speed, 2=vorticity (dv/dx-du/dy), 3=tracer
    float m_viewGain = 1.0f, m_viewGamma = 1.0f, m_zoom = 1.0f;
    glm::vec2 m_panPix{0.0f, 0.0f};
    std::vector<float> m_fieldScratch, m_maskScratch;
    // TextRenderer's real constructor needs a current GL context (it builds
    // a shader/VAO immediately), so unlike m_font (default-constructible,
    // see fw::Font) it can't be a plain member -- constructed lazily in
    // EnsureRenderResources(), same reason/pattern as MDSim.hpp's m_text.
    fw::Font m_font;
    std::unique_ptr<fw::TextRenderer> m_text;
};

} // namespace cf
