#include "CompressibleSimScene.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace cf {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Fullscreen-triangle trick (LearnOpenGL/Mardern GL folklore, also used by
// 05_tdse_gpu's TdseSim::kViewVert): 3 vertices, no VBO, covering the whole
// viewport -- gl_VertexID picks a corner far enough outside [-1,1] that the
// rasterizer still clips it to a full-screen quad.
const char* kViewVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";

// Screen pixel -> world coords -> grid cell, exactly 05_tdse_gpu's
// TdseSim::kViewFrag's approach generalized to a non-square nx*ny grid: wx/wy
// are offsets from the DOMAIN CENTER (not its lower-left corner), so
// `wx + 0.5*uLx` recovers a position relative to the domain's own local
// origin regardless of the solver's absolute xMin/yMin -- the same reason
// CompressibleSimScene::Render() only ever needs to pass uLx/uLy, not xMin/yMin.
const char* kViewFrag = R"(#version 460 core
out vec4 FragColor;

layout(std430, binding = 0) readonly buffer FieldBuf { float field[]; };
layout(std430, binding = 1) readonly buffer MaskBuf { float mask[]; };

uniform vec2 uRes;
uniform int uNx;
uniform int uNy;
uniform float uLx;
uniform float uLy;
uniform float uPixPerUnit;
uniform vec2 uPanPix;
uniform float uFieldMin;
uniform float uFieldMax;
uniform float uGain;
uniform float uGammaView;

// Polynomial fit of matplotlib 'magma' -- identical LUT to
// 05_tdse_gpu/src/TdseSim.cpp's kViewFrag, reused verbatim.
vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(-0.002136, -0.000750, -0.005386);
    const vec3 c1 = vec3(0.251723, 0.677631, 2.494027);
    const vec3 c2 = vec3(8.353717, -3.577720, 0.311613);
    const vec3 c3 = vec3(-27.668733, 14.264731, -13.649213);
    const vec3 c4 = vec3(52.176140, -27.943606, 12.944169);
    const vec3 c5 = vec3(-50.768525, 29.046583, 4.234153);
    const vec3 c6 = vec3(18.655705, -11.489774, -5.601962);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}

void main() {
    float wx = (gl_FragCoord.x - 0.5 * uRes.x - uPanPix.x) / uPixPerUnit;
    float wy = (gl_FragCoord.y - 0.5 * uRes.y - uPanPix.y) / uPixPerUnit;
    float dx = uLx / float(uNx);
    float dy = uLy / float(uNy);
    int i = int(floor((wx + 0.5 * uLx) / dx));
    int j = int(floor((wy + 0.5 * uLy) / dy));
    if (i < 0 || j < 0 || i >= uNx || j >= uNy) {
        FragColor = vec4(0.03, 0.03, 0.045, 1.0); // outside the domain
        return;
    }
    int idx = j * uNx + i;
    if (mask[idx] > 0.5) {
        FragColor = vec4(0.16, 0.16, 0.18, 1.0); // obstacle: flat, no field color
        return;
    }
    float t = clamp((field[idx] - uFieldMin) / max(uFieldMax - uFieldMin, 1e-12), 0.0, 1.0);
    t = pow(clamp(uGain * t, 0.0, 1.0), uGammaView);
    FragColor = vec4(magma(t), 1.0);
}
)";

WallBC ParseWallBC(const std::string& s) {
    if (s == "outflow") return WallBC::Outflow;
    if (s == "periodic") return WallBC::Periodic;
    if (s == "no_slip") return WallBC::NoSlipReflective;
    if (s == "free_slip") return WallBC::FreeSlipReflective;
    if (s == "inflow") return WallBC::Inflow;
    std::fprintf(stderr, "warning: unknown boundary type '%s', defaulting to outflow\n", s.c_str());
    return WallBC::Outflow;
}

// One side's parsed [boundary.<side>] table: its WallBC, and (only
// meaningful for Inflow) the fixed base state plus an optional named
// profile pattern varying that state along the side's own coordinate.
struct SideConfig {
    WallBC bc = WallBC::Outflow;
    Prim2D state;
    bool hasProfile = false;
    std::string profileType;
    double profileStripeWidth = 1.0;
};

SideConfig ParseSide(const fw::Deck& deck, const std::string& sideKey) {
    SideConfig sc;
    sc.bc = ParseWallBC(deck.GetString(sideKey + ".type", "outflow"));
    if (sc.bc != WallBC::Inflow) return sc;
    sc.state = Prim2D{
        deck.GetDouble(sideKey + ".rho", 1.0),
        deck.GetDouble(sideKey + ".u", 0.0),
        deck.GetDouble(sideKey + ".v", 0.0),
        deck.GetDouble(sideKey + ".p", 1.0),
        deck.GetDouble(sideKey + ".tracer", 0.0),
    };
    if (deck.Has(sideKey + ".profile.type")) {
        sc.hasProfile = true;
        sc.profileType = deck.GetString(sideKey + ".profile.type", "uniform");
        sc.profileStripeWidth = deck.GetDouble(sideKey + ".profile.stripe_width", 1.0);
    }
    return sc;
}

Prim2D ReadPrim(const fw::Deck& deck, const std::string& key, const Prim2D& fallback) {
    return Prim2D{
        deck.GetDouble(key + ".rho", fallback.rho),
        deck.GetDouble(key + ".u", fallback.u),
        deck.GetDouble(key + ".v", fallback.v),
        deck.GetDouble(key + ".p", fallback.p),
        deck.GetDouble(key + ".tracer", fallback.tracer),
    };
}

Prim2D ReadQuadrant(const fw::Deck& deck, const std::string& key) {
    const std::vector<double> v = deck.GetDoubleArray(key);
    return Prim2D{v[0], v[1], v[2], v[3]};
}

// Builds the full initial-condition function from [initial_condition]
// (+ an optional [initial_condition.perturbation]) -- see
// CompressibleSimScene.hpp's class comment for why this is a closed-over
// std::function rather than something re-read from the deck at Reset() time.
std::function<Prim2D(double, double)> BuildInitialCondition(const fw::Deck& deck) {
    const std::string icType = deck.GetString("initial_condition.type", "uniform");
    std::function<Prim2D(double, double)> baseIc;

    if (icType == "uniform") {
        const Prim2D state = ReadPrim(deck, "initial_condition", Prim2D{1.0, 0.0, 0.0, 1.0, 0.0});
        baseIc = [state](double, double) { return state; };
    } else if (icType == "riemann_quadrants") {
        // Four constant states in the four quadrants of the domain, split
        // at (x0,y0) -- the classic genuinely-2D Riemann problem (e.g.
        // Kurganov & Tadmor 2002's "Configuration 3", decks/
        // riemann2d_config3.toml).
        const double x0 = deck.GetDouble("initial_condition.x0", 0.5);
        const double y0 = deck.GetDouble("initial_condition.y0", 0.5);
        const Prim2D ne = ReadQuadrant(deck, "initial_condition.ne");
        const Prim2D nw = ReadQuadrant(deck, "initial_condition.nw");
        const Prim2D sw = ReadQuadrant(deck, "initial_condition.sw");
        const Prim2D se = ReadQuadrant(deck, "initial_condition.se");
        baseIc = [=](double x, double y) {
            if (y >= y0) return x >= x0 ? ne : nw;
            return x >= x0 ? se : sw;
        };
    } else if (icType == "taylor_green_vortex") {
        // The classic Taylor-Green (1937) vortex pair -- see
        // decks/taylor_green.toml and docs/SIMULATION.md's Phase 4 section
        // for the analytic viscous-decay solution this validates against.
        // Assumes a square domain (length derived from the x-extent alone);
        // every deck that actually uses this IC type has one.
        const double rho0 = deck.GetDouble("initial_condition.rho0", 1.0);
        const double p0 = deck.GetDouble("initial_condition.p0", 1.0);
        const double u0 = deck.GetDouble("initial_condition.u0", 0.1);
        const double length = deck.GetDouble("grid.x_max", 1.0) - deck.GetDouble("grid.x_min", 0.0);
        const int waveNumberMultiplier = deck.GetInt("initial_condition.wavenumber_multiplier", 1);
        const double k = 2.0 * kPi * waveNumberMultiplier / length;
        baseIc = [=](double x, double y) {
            const double u = u0 * std::cos(k * x) * std::sin(k * y);
            const double v = -u0 * std::sin(k * x) * std::cos(k * y);
            const double p = p0 - 0.25 * rho0 * u0 * u0 * (std::cos(2.0 * k * x) + std::cos(2.0 * k * y));
            return Prim2D{rho0, u, v, p};
        };
    } else {
        std::fprintf(stderr, "warning: unknown initial_condition.type '%s', defaulting to uniform rest\n",
                     icType.c_str());
        baseIc = [](double, double) { return Prim2D{1.0, 0.0, 0.0, 1.0}; };
    }

    if (!deck.Has("initial_condition.perturbation.type")) return baseIc;
    const std::string pertType = deck.GetString("initial_condition.perturbation.type", "");
    if (pertType != "antisymmetric_gaussian") {
        std::fprintf(stderr, "warning: unknown initial_condition.perturbation.type '%s', ignoring\n",
                     pertType.c_str());
        return baseIc;
    }
    // A one-time antisymmetric velocity bump, localized near `center`
    // (Gaussian, width `sigma`), opposite sign above/below `center`'s y --
    // the belt-and-suspenders symmetry-breaker cylinder_re100.toml needs on
    // top of its off-center obstacle placement (see docs/SIMULATION.md
    // section 6.2 for why a perfectly symmetric setup never sheds at all
    // without one). `amplitude` is a fraction of the base state's own u.
    const std::vector<double> center = deck.GetDoubleArray("initial_condition.perturbation.center");
    const double cx = center.size() > 0 ? center[0] : 0.0;
    const double cy = center.size() > 1 ? center[1] : 0.0;
    const double sigma = deck.GetDouble("initial_condition.perturbation.sigma", 1.0);
    const double amplitude = deck.GetDouble("initial_condition.perturbation.amplitude", 0.0);
    const std::string component = deck.GetString("initial_condition.perturbation.component", "v");
    return [=](double x, double y) {
        Prim2D p = baseIc(x, y);
        const double dxp = x - cx, dyp = y - cy;
        const double r2 = dxp * dxp + dyp * dyp;
        const double sign = dyp >= 0.0 ? 1.0 : -1.0;
        const double pert = amplitude * p.u * sign * std::exp(-r2 / (2.0 * sigma * sigma));
        if (component == "u") p.u += pert; else p.v += pert;
        return p;
    };
}

} // namespace

CompressibleSimScene::~CompressibleSimScene() {
    if (m_fieldBuf) glDeleteBuffers(1, &m_fieldBuf);
    if (m_maskBuf) glDeleteBuffers(1, &m_maskBuf);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

// Symmetric NACA00xx thickness distribution (Abbott & von Doenhoff's
// standard open-trailing-edge coefficients), evaluated in the airfoil's own
// body frame (leading edge at the origin, chord along +x) and rotated by
// `angleRad` -- the angle of attack, positive nose-up -- to place it in
// world coordinates. This slots into Euler2D::SetObstacleMask exactly the
// way a circle's `dx*dx+dy*dy<=r*r` test does (see Euler2D.hpp's comment):
// just another pointwise inside/outside predicate, so the immersed-boundary
// cell-masking machinery needs no changes at all to support a new shape.
bool CompressibleSimScene::IsInsideAirfoil(double x, double y, const ObstacleSpec& o) {
    const double dx = x - o.cx, dy = y - o.cy;
    const double ca = std::cos(o.angleRad), sa = std::sin(o.angleRad);
    // Rotate world-frame (dx,dy) by -angleRad into the body frame.
    const double xBody = dx * ca + dy * sa;
    const double yBody = -dx * sa + dy * ca;
    if (xBody < 0.0 || xBody > o.chord) return false;
    const double xoc = xBody / o.chord;
    const double halfThickness =
        5.0 * o.thicknessFrac * o.chord *
        (0.2969 * std::sqrt(xoc) - 0.1260 * xoc - 0.3516 * xoc * xoc + 0.2843 * xoc * xoc * xoc -
         0.1015 * xoc * xoc * xoc * xoc);
    return std::abs(yBody) <= halfThickness;
}

void CompressibleSimScene::RebuildObstacleMask() {
    if (m_obstacles.empty()) return;
    const std::vector<ObstacleSpec> obstacles = m_obstacles; // captured by value: cheap, and stays
                                                               // valid after this function returns
    m_solver.SetObstacleMask([obstacles](double x, double y) {
        for (const ObstacleSpec& o : obstacles) {
            if (o.isAirfoil) {
                if (IsInsideAirfoil(x, y, o)) return true;
            } else {
                const double dx = x - o.cx, dy = y - o.cy;
                if (dx * dx + dy * dy <= o.radius * o.radius) return true;
            }
        }
        return false;
    });
}

void CompressibleSimScene::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);

    const int nx = deck.GetInt("grid.nx", 100);
    const int ny = deck.GetInt("grid.ny", 100);
    const double xMin = deck.GetDouble("grid.x_min", 0.0);
    const double xMax = deck.GetDouble("grid.x_max", 1.0);
    const double yMin = deck.GetDouble("grid.y_min", 0.0);
    const double yMax = deck.GetDouble("grid.y_max", 1.0);
    const double gamma = deck.GetDouble("physics.gamma", 1.4);
    const double mu = deck.GetDouble("physics.mu", 0.0);
    const double conductivity = deck.GetDouble("physics.conductivity", 0.0);
    const double tracerDiffusivity = deck.GetDouble("physics.tracer_diffusivity", 0.0);
    const double bodyForceX = deck.GetDouble("physics.body_force_x", 0.0);

    m_solver.Init(nx, ny, xMin, xMax, yMin, yMax, gamma);
    m_solver.SetViscosity(mu, conductivity);
    m_solver.SetTracerDiffusivity(tracerDiffusivity);
    m_solver.SetBodyForceX(bodyForceX);

    const SideConfig left = ParseSide(deck, "boundary.left");
    const SideConfig right = ParseSide(deck, "boundary.right");
    const SideConfig bottom = ParseSide(deck, "boundary.bottom");
    const SideConfig top = ParseSide(deck, "boundary.top");
    m_solver.SetBoundaryConditions(left.bc, right.bc, bottom.bc, top.bc);

    // Euler2D's Inflow BC uses one shared state/profile for every side
    // that's Inflow (see Euler2D::SetInflowState/SetInflowProfile) -- fine
    // as long as no deck needs two DIFFERENT inflow states on two
    // different sides (none does), so only the first Inflow side found is
    // used here.
    for (const SideConfig* sc : {&left, &right, &bottom, &top}) {
        if (sc->bc != WallBC::Inflow) continue;
        m_solver.SetInflowState(sc->state);
        if (sc->hasProfile && sc->profileType == "tracer_stripes") {
            const Prim2D base = sc->state;
            const double stripeWidth = sc->profileStripeWidth;
            m_solver.SetInflowProfile([=](double coord) {
                Prim2D p = base;
                const int band = static_cast<int>(std::floor(coord / stripeWidth));
                p.tracer = (band % 2 == 0) ? 1.0 : 0.0;
                return p;
            });
        } else if (sc->hasProfile) {
            std::fprintf(stderr, "warning: unknown inflow profile type '%s', using the fixed state instead\n",
                         sc->profileType.c_str());
        }
        break;
    }

    m_obstacles.clear();
    m_hasAirfoil = false;
    for (const fw::Deck& obs : deck.GetTables("obstacles")) {
        const std::string shape = obs.GetString("shape", "circle");
        const std::vector<double> center = obs.GetDoubleArray("center");
        ObstacleSpec spec;
        spec.cx = center.size() > 0 ? center[0] : 0.0;
        spec.cy = center.size() > 1 ? center[1] : 0.0;
        if (shape == "circle") {
            spec.radius = obs.GetDouble("radius", 0.0);
        } else if (shape == "airfoil") {
            // Symmetric NACA00xx profile (see RebuildObstacleMask's
            // thickness formula) -- `center` is the leading edge, so the
            // body occupies x in [cx, cx+chord] before rotation.
            spec.isAirfoil = true;
            spec.chord = obs.GetDouble("chord", 1.0);
            spec.thicknessFrac = obs.GetDouble("thickness", 0.12);
            spec.angleRad = obs.GetDouble("angle_deg", 0.0) * kPi / 180.0;
            m_hasAirfoil = true;
        } else {
            std::fprintf(stderr, "warning: unsupported obstacle shape '%s', ignoring\n", shape.c_str());
            continue;
        }
        m_obstacles.push_back(spec);
    }
    RebuildObstacleMask();

    m_icFn = BuildInitialCondition(deck);
    Reset();

    m_probes.clear();
    for (const fw::Deck& p : deck.GetTables("probes")) {
        const std::string name = p.GetString("name", "probe");
        const double x = p.GetDouble("x", 0.0);
        const double y = p.GetDouble("y", 0.0);
        const int i = std::clamp(static_cast<int>((x - xMin) / m_solver.Dx()), 0, nx - 1);
        const int j = std::clamp(static_cast<int>((y - yMin) / m_solver.Dy()), 0, ny - 1);
        m_probes.push_back({name, i, j});
    }

    m_writeTracer = deck.GetBool("output.write_tracer", false);

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 20);
    // Fixed dt from the (post-Reset, i.e. including any perturbation) max
    // wave speed -- see decks/cylinder_re100.toml's comment via
    // CompressibleSimCylinder's original version of this line for why a
    // fixed dt is safe for every scenario this schema currently expresses.
    m_dt = m_cfl / (m_solver.MaxWaveSpeedX() / m_solver.Dx() + m_solver.MaxWaveSpeedY() / m_solver.Dy());
}

void CompressibleSimScene::Reset() {
    m_solver.SetInitialCondition(m_icFn);
}

void CompressibleSimScene::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_solver.Step(m_dt);
}

void CompressibleSimScene::Snapshot(fw::OutputWriter& writer) {
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    std::vector<float> rho(static_cast<size_t>(nx * ny)), u(static_cast<size_t>(nx * ny)),
        v(static_cast<size_t>(nx * ny)), p(static_cast<size_t>(nx * ny));
    std::vector<float> tracer;
    if (m_writeTracer) tracer.resize(static_cast<size_t>(nx * ny));

    double kineticEnergy = 0.0, uMax = 0.0;
    const double cellArea = m_solver.Dx() * m_solver.Dy();
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const Prim2D pr = m_solver.PrimAt(i, j);
            const size_t idx = static_cast<size_t>(j * nx + i);
            rho[idx] = static_cast<float>(pr.rho);
            u[idx] = static_cast<float>(pr.u);
            v[idx] = static_cast<float>(pr.v);
            p[idx] = static_cast<float>(pr.p);
            if (m_writeTracer) tracer[idx] = static_cast<float>(pr.tracer);
            kineticEnergy += 0.5 * pr.rho * (pr.u * pr.u + pr.v * pr.v) * cellArea;
            uMax = std::max(uMax, pr.u);
        }
    }
    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("u", u.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, ny, nx);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, ny, nx);
    if (m_writeTracer) writer.WriteField("tracer", tracer.data(), fw::NpyDtype::F4, ny, nx);

    writer.WriteScalar("mass_total", m_solver.TotalMass());
    writer.WriteScalar("energy_total", m_solver.TotalEnergy());
    writer.WriteScalar("kinetic_energy", kineticEnergy);
    writer.WriteScalar("u_max", uMax);
    for (const Probe& probe : m_probes) {
        const Prim2D val = m_solver.PrimAt(probe.i, probe.j);
        writer.WriteScalar(probe.name + "_rho", val.rho);
        writer.WriteScalar(probe.name + "_u", val.u);
        writer.WriteScalar(probe.name + "_v", val.v);
        writer.WriteScalar(probe.name + "_p", val.p);
    }
}

fw::SimInfo CompressibleSimScene::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_solver.Nx();
    info.gridNy = m_solver.Ny();
    info.lx = m_solver.Nx() * m_solver.Dx();
    info.ly = m_solver.Ny() * m_solver.Dy();
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "u", "v", "p"};
    if (m_writeTracer) info.frameFields.push_back("tracer");
    info.diagnostics = {"mass_total", "energy_total", "kinetic_energy", "u_max"};
    for (const Probe& probe : m_probes) {
        info.diagnostics.push_back(probe.name + "_rho");
        info.diagnostics.push_back(probe.name + "_u");
        info.diagnostics.push_back(probe.name + "_v");
        info.diagnostics.push_back(probe.name + "_p");
    }
    return info;
}

void CompressibleSimScene::EnsureRenderResources() {
    if (m_renderReady) return;
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    glGenVertexArrays(1, &m_vao);
    glCreateBuffers(1, &m_fieldBuf);
    glCreateBuffers(1, &m_maskBuf);
    const size_t numCells = static_cast<size_t>(m_solver.Nx()) * static_cast<size_t>(m_solver.Ny());
    glNamedBufferData(m_fieldBuf, static_cast<GLsizeiptr>(numCells * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_maskBuf, static_cast<GLsizeiptr>(numCells * sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
    m_fieldScratch.assign(numCells, 0.0f);
    m_maskScratch.assign(numCells, 0.0f);
    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);
    m_renderReady = true;
}

void CompressibleSimScene::Render(int fbWidth, int fbHeight) {
    EnsureRenderResources();
    const int nx = m_solver.Nx(), ny = m_solver.Ny();
    const double dx = m_solver.Dx(), dy = m_solver.Dy();

    // Re-pack the currently selected scalar field from the CPU-resident
    // solver state every frame (see the class comment for why there's no
    // persistent GPU copy to just re-bind), tracking min/max as we go so
    // the colormap always auto-scales to what's actually on screen right
    // now -- there's no fixed physical range for e.g. vorticity to
    // normalize against ahead of time.
    float fieldMin = std::numeric_limits<float>::max();
    float fieldMax = std::numeric_limits<float>::lowest();
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const size_t idx = static_cast<size_t>(j * nx + i);
            const Prim2D pr = m_solver.PrimAt(i, j);
            float value;
            switch (m_renderMode) {
            case 1: // speed
                value = static_cast<float>(std::sqrt(pr.u * pr.u + pr.v * pr.v));
                break;
            case 2: { // vorticity: dv/dx - du/dy, central difference (clamped at the domain edge,
                      // like Euler2D.cpp's own zero-gradient boundary treatment -- an approximation
                      // there, fine for a live view that isn't a quantitative diagnostic)
                const int im1 = std::max(i - 1, 0), ip1 = std::min(i + 1, nx - 1);
                const int jm1 = std::max(j - 1, 0), jp1 = std::min(j + 1, ny - 1);
                const double dvdx = (m_solver.PrimAt(ip1, j).v - m_solver.PrimAt(im1, j).v) / (2.0 * dx);
                const double dudy = (m_solver.PrimAt(i, jp1).u - m_solver.PrimAt(i, jm1).u) / (2.0 * dy);
                value = static_cast<float>(dvdx - dudy);
                break;
            }
            case 3: // tracer
                value = static_cast<float>(pr.tracer);
                break;
            default: // density
                value = static_cast<float>(pr.rho);
                break;
            }
            m_fieldScratch[idx] = value;
            m_maskScratch[idx] = m_solver.IsSolidAt(i, j) ? 1.0f : 0.0f;
            fieldMin = std::min(fieldMin, value);
            fieldMax = std::max(fieldMax, value);
        }
    }
    glNamedBufferSubData(m_fieldBuf, 0, static_cast<GLsizeiptr>(m_fieldScratch.size() * sizeof(float)),
                         m_fieldScratch.data());
    glNamedBufferSubData(m_maskBuf, 0, static_cast<GLsizeiptr>(m_maskScratch.size() * sizeof(float)),
                         m_maskScratch.data());

    const double lx = nx * dx, ly = ny * dy;
    // Fit the WHOLE domain inside the window regardless of its aspect ratio
    // (the cylinder deck's 25x8 domain is far from square, unlike
    // 05_tdse_gpu's always-square grid, which is why this differs from that
    // sim's pixPerUnit formula) -- min() of the two per-axis fits, so
    // neither dimension overflows the viewport.
    const float pixPerUnit =
        static_cast<float>(std::min(fbWidth / lx, fbHeight / ly)) * m_zoom;

    glDisable(GL_DEPTH_TEST);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2(static_cast<float>(fbWidth), static_cast<float>(fbHeight)));
    m_view.SetInt("uNx", nx);
    m_view.SetInt("uNy", ny);
    m_view.SetFloat("uLx", static_cast<float>(lx));
    m_view.SetFloat("uLy", static_cast<float>(ly));
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetFloat("uFieldMin", fieldMin);
    m_view.SetFloat("uFieldMax", fieldMax);
    m_view.SetFloat("uGain", m_viewGain);
    m_view.SetFloat("uGammaView", m_viewGamma);

    fw::ComputeShader::BindBuffer(0, m_fieldBuf);
    fw::ComputeShader::BindBuffer(1, m_maskBuf);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    // Name the field on screen -- otherwise there's no way to tell what's
    // currently displayed (or that 'M' switches it) without already
    // knowing the keybinding from the README.
    static const char* kModeNames[4] = {"density", "speed", "vorticity", "tracer"};
    char line[96];
    std::snprintf(line, sizeof(line), "field: %s  ('M' to cycle)", kModeNames[m_renderMode]);
    m_text->SetViewport(fbWidth, fbHeight);
    m_text->Draw(m_font, line, glm::vec2(12.0f, 24.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.9f));
    if (m_hasAirfoil) {
        // Same rationale as the field label: the angle is a live control
        // (Up/Down, see OnKey), so its current value needs to be visible
        // without the user having tracked every keypress themselves.
        double angleDeg = 0.0;
        for (const ObstacleSpec& o : m_obstacles) {
            if (o.isAirfoil) { angleDeg = o.angleRad * 180.0 / kPi; break; }
        }
        std::snprintf(line, sizeof(line), "angle of attack: %.1f deg  (Up/Down)", angleDeg);
        m_text->Draw(m_font, line, glm::vec2(12.0f, 44.0f), glm::vec4(1.0f, 1.0f, 1.0f, 0.9f));
    }

    glEnable(GL_DEPTH_TEST);
}

void CompressibleSimScene::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) {
        m_panPix.x += static_cast<float>(in.dx);
        m_panPix.y -= static_cast<float>(in.dy); // screen y-down -> world y-up
    }
    if (in.scrollDelta != 0.0) {
        m_zoom *= std::exp(0.12f * static_cast<float>(in.scrollDelta));
        m_zoom = std::clamp(m_zoom, 0.2f, 80.0f);
    }
}

void CompressibleSimScene::OnKey(int key, int action) {
    (void)action;
    switch (key) {
    case 'M': m_renderMode = (m_renderMode + 1) % 4; break;
    case '[': m_viewGain = std::max(0.15f, m_viewGain * 0.85f); break;
    case ']': m_viewGain = std::min(12.0f, m_viewGain * 1.18f); break;
    case '-': m_viewGamma = std::max(0.2f, m_viewGamma - 0.05f); break;
    case '=': m_viewGamma = std::min(2.5f, m_viewGamma + 0.05f); break;
    case '0':
        m_zoom = 1.0f;
        m_panPix = {0.0f, 0.0f};
        m_viewGain = 1.0f;
        m_viewGamma = 1.0f;
        break;
    // Angle of attack is treated as a live control knob, not part of the
    // "initial condition" -- unlike m_icFn, Reset() deliberately leaves
    // m_obstacles (and therefore the current angle) untouched, so sweeping
    // the angle and then resetting the flow (e.g. after a spurious
    // transient) doesn't also snap the airfoil back to the deck's angle.
    // No-op if the deck has no airfoil obstacle (nothing to rotate).
    case GLFW_KEY_UP:
    case GLFW_KEY_DOWN: {
        if (!m_hasAirfoil) break;
        const double stepRad = (key == GLFW_KEY_UP ? 1.0 : -1.0) * kPi / 180.0;
        for (ObstacleSpec& o : m_obstacles) {
            if (!o.isAirfoil) continue;
            const double newDeg = std::clamp((o.angleRad + stepRad) * 180.0 / kPi, -45.0, 45.0);
            o.angleRad = newDeg * kPi / 180.0;
        }
        RebuildObstacleMask();
        break;
    }
    default:
        break;
    }
}

} // namespace cf
