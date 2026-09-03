#include "MagnetostaticsSim.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace mag {

namespace {

// Fullscreen triangle (no VBO); gl_VertexID picks a corner outside [-1,1].
const char* kViewVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";

// Screen pixel -> world -> grid cell (same mapping as 08_compressible_fluid's
// view), then colour: magma for the unsigned |B|, a blue-white-red diverging
// map centred on zero for the signed B_x / B_y / A_z.
const char* kViewFrag = R"(#version 460 core
out vec4 FragColor;
layout(std430, binding = 0) readonly buffer FieldBuf { float field[]; };

uniform vec2  uRes;
uniform int   uNx;
uniform int   uNy;
uniform float uLx;
uniform float uLy;
uniform float uPixPerUnit;
uniform vec2  uPanPix;
uniform float uFieldMin;
uniform float uFieldMax;   // unsigned: max; signed: max|.|
uniform int   uSigned;
uniform float uGain;
uniform float uGamma;

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(-0.002136,-0.000750,-0.005386);
    const vec3 c1 = vec3(0.251723,0.677631,2.494027);
    const vec3 c2 = vec3(8.353717,-3.577720,0.311613);
    const vec3 c3 = vec3(-27.668733,14.264731,-13.649213);
    const vec3 c4 = vec3(52.176140,-27.943606,12.944169);
    const vec3 c5 = vec3(-50.768525,29.046583,4.234153);
    const vec3 c6 = vec3(18.655705,-11.489774,-5.601962);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}
vec3 diverging(float s) {   // s in [-1, 1]
    vec3 lo  = vec3(0.19, 0.31, 0.75);
    vec3 mid = vec3(0.96, 0.96, 0.96);
    vec3 hi  = vec3(0.79, 0.16, 0.16);
    return s < 0.0 ? mix(mid, lo, clamp(-s, 0.0, 1.0))
                   : mix(mid, hi, clamp( s, 0.0, 1.0));
}

void main() {
    float wx = (gl_FragCoord.x - 0.5*uRes.x - uPanPix.x) / uPixPerUnit;
    float wy = (gl_FragCoord.y - 0.5*uRes.y - uPanPix.y) / uPixPerUnit;
    float dx = uLx / float(uNx);
    float dy = uLy / float(uNy);
    int i = int(floor((wx + 0.5*uLx) / dx));
    int j = int(floor((wy + 0.5*uLy) / dy));
    if (i < 0 || j < 0 || i >= uNx || j >= uNy) {
        FragColor = vec4(0.03, 0.03, 0.045, 1.0);
        return;
    }
    float v = field[j*uNx + i];
    if (uSigned == 1) {
        float s = v / max(uFieldMax, 1e-20);
        s = sign(s) * pow(clamp(abs(s)*uGain, 0.0, 1.0), uGamma);
        FragColor = vec4(diverging(clamp(s, -1.0, 1.0)), 1.0);
    } else {
        float t = (v - uFieldMin) / max(uFieldMax - uFieldMin, 1e-20);
        t = pow(clamp(uGain*t, 0.0, 1.0), uGamma);
        FragColor = vec4(magma(t), 1.0);
    }
}
)";

const char* kLineVert = R"(#version 460 core
layout(location = 0) in vec2 aPos;   // world coords
uniform vec2  uRes;
uniform float uPixPerUnit;
uniform vec2  uPanPix;
void main() {
    vec2 px = aPos * uPixPerUnit + 0.5*uRes + uPanPix;
    vec2 ndc = (px / uRes) * 2.0 - 1.0;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)";
const char* kLineFrag = R"(#version 460 core
out vec4 FragColor;
uniform vec4 uColor;
void main() { FragColor = uColor; }
)";

constexpr double kPi = 3.14159265358979323846;

} // namespace

MagnetostaticsSim::~MagnetostaticsSim() {
    if (m_fieldBuf) glDeleteBuffers(1, &m_fieldBuf);
    if (m_lineVbo) glDeleteBuffers(1, &m_lineVbo);
    if (m_trailVbo) glDeleteBuffers(1, &m_trailVbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_lineVao) glDeleteVertexArrays(1, &m_lineVao);
    if (m_trailVao) glDeleteVertexArrays(1, &m_trailVao);
}

void MagnetostaticsSim::Configure(const fw::Deck& deck) {
    m_grid.nx = deck.GetInt("grid.nx", 256);
    m_grid.ny = deck.GetInt("grid.ny", m_grid.nx);
    m_grid.lx = deck.GetDouble("grid.lx", 4.0);
    m_grid.ly = deck.GetDouble("grid.ly", m_grid.lx);
    m_mu0 = deck.GetDouble("physics.mu0", 1.0);
    m_title = deck.GetString("title", "2D Magnetostatics");

    m_solveOpt.tol = deck.GetDouble("solver.tol", 1e-9);
    m_solveOpt.maxIterations = deck.GetInt("solver.max_iterations", 200000);
    m_solveOpt.omega = deck.GetDouble("solver.omega", -1.0);
    m_mgOpt.tol = deck.GetDouble("solver.tol", 1e-9);
    m_mgOpt.maxCycles = deck.GetInt("solver.max_cycles", 100);

    m_method = deck.GetString("solver.method", "multigrid");
    const std::string bc = deck.GetString("solver.boundary", "dirichlet");
    m_solveOpt.edgeBC = (bc == "neumann") ? BC::Neumann : BC::Dirichlet;
    if (m_solveOpt.edgeBC == BC::Neumann) m_method = "rbgs";

    m_sources = ParseSources(m_grid, deck);
    m_coils = ParseCoils(deck);
    m_useBiot = !m_coils.empty();
    m_plane.kind = SlicePlane::ParseKind(deck.GetString("grid.plane", "xy"));
    m_plane.offset = deck.GetDouble("grid.plane_offset", 0.0);

    m_Jz = RasterizeCurrent(m_grid, m_sources);
    if (m_solveOpt.edgeBC == BC::Neumann) {
        m_fixedMask.assign(m_grid.count(), 0);
        m_fixedValues.assign(m_grid.count(), 0.0);
    } else {
        MakeEdgeDirichlet(m_grid, m_fixedMask, m_fixedValues);
    }
    if (m_useBiot) m_biot.Init(m_grid);
    UpdateField();

    // --- test-particle pusher -----------------------------------------
    m_c = deck.GetDouble("physics.c", 1.0);
    auto vec3 = [](const std::vector<double>& a, glm::dvec3 def) {
        return a.size() == 3 ? glm::dvec3(a[0], a[1], a[2]) : def;
    };
    m_bgB = vec3(deck.GetDoubleArray("background.B"), glm::dvec3(0.0));
    m_bgE = vec3(deck.GetDoubleArray("background.E"), glm::dvec3(0.0));
    m_dt = deck.GetDouble("time.dt", 0.0);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_charges.clear();
    for (const auto& ct : deck.GetTables("charge")) {
        TestCharge p;
        p.qm = ct.GetDouble("q_over_m", 1.0);
        p.x = vec3(ct.GetDoubleArray("x0"), glm::dvec3(0.0));
        const glm::dvec3 v0 = vec3(ct.GetDoubleArray("v0"), glm::dvec3(0.0));
        const double v2 = glm::dot(v0, v0);
        const double gamma = 1.0 / std::sqrt(std::max(1e-30, 1.0 - v2 / (m_c * m_c)));
        p.u = gamma * v0;
        const auto col = ct.GetDoubleArray("color");
        if (col.size() == 3)
            p.color = glm::vec3((float)col[0], (float)col[1], (float)col[2]);
        m_charges.push_back(p);
    }
    m_chargesInit = m_charges;
    m_step = 0;
    m_time = 0.0;
}

void MagnetostaticsSim::SampleB2D(double x, double y, double& bx, double& by) const {
    const double dx = m_grid.dx(), dy = m_grid.dy();
    double fi = (x + 0.5 * m_grid.lx) / dx - 0.5;
    double fj = (y + 0.5 * m_grid.ly) / dy - 0.5;
    int i = std::clamp(static_cast<int>(std::floor(fi)), 0, m_grid.nx - 2);
    int j = std::clamp(static_cast<int>(std::floor(fj)), 0, m_grid.ny - 2);
    double tx = std::clamp(fi - i, 0.0, 1.0), ty = std::clamp(fj - j, 0.0, 1.0);
    auto lerp2 = [&](const std::vector<double>& f) {
        return (1 - tx) * (1 - ty) * f[m_grid.idx(i, j)] +
               tx * (1 - ty) * f[m_grid.idx(i + 1, j)] +
               (1 - tx) * ty * f[m_grid.idx(i, j + 1)] +
               tx * ty * f[m_grid.idx(i + 1, j + 1)];
    };
    bx = lerp2(m_Bx);
    by = lerp2(m_By);
}

void MagnetostaticsSim::SampleField(const glm::dvec3& r, glm::dvec3& E,
                                    glm::dvec3& B) const {
    E = m_bgE;
    B = m_bgB;
    if (m_useBiot) {
        B += BiotSavartAt(m_segments, r, m_mu0);
    } else if (!m_sources.wires.empty()) {
        // 2D in-plane field (z-invariant for the out-of-plane line currents).
        double bx, by;
        SampleB2D(r.x, r.y, bx, by);
        B += glm::dvec3(bx, by, 0.0);
    }
}

void MagnetostaticsSim::UpdateField(bool warmStart) {
    const auto t0 = std::chrono::steady_clock::now();
    if (m_useBiot) BiotSavartPath();
    else           SolvePoissonPath(warmStart);
    m_lastSolveMs = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();

    // Fold a uniform background field into the display grid so a
    // background-only deck (cyclotron / E x B) still shows a field, not black.
    if (glm::dot(m_bgB, m_bgB) > 0.0) {
        const glm::dvec2 ip = m_plane.InPlane(m_bgB);
        const double perp = m_plane.kind == SlicePlane::XZ ? m_bgB.y
                          : m_plane.kind == SlicePlane::YZ ? m_bgB.x
                          : m_bgB.z;
        for (int k = 0; k < m_grid.count(); ++k) {
            m_Bx[k] += ip.x;
            m_By[k] += ip.y;
            m_Bz[k] += perp;
        }
    }

    RebuildAvoidPoints();
    m_fieldDirty = true;

    if (!warmStart) {
        if (m_useBiot)
            std::printf("[magnetostatics] biot-savart: %zu segments, slice %s "
                        "(%.1f ms)\n",
                        m_segments.size(),
                        m_plane.kind == SlicePlane::XY ? "xy" :
                        m_plane.kind == SlicePlane::XZ ? "xz" : "yz",
                        m_lastSolveMs);
        else
            std::printf("[magnetostatics] %s: %d %s, residual %.3e%s  (%.1f ms)\n",
                        m_method.c_str(), m_lastSolve.iterations,
                        m_method == "multigrid" ? "W-cycles" : "iterations",
                        m_lastSolve.residual,
                        m_lastSolve.converged ? "" : "  (NOT converged)",
                        m_lastSolveMs);
    }
}

void MagnetostaticsSim::SolvePoissonPath(bool warmStart) {
    std::vector<double> rhs(m_grid.count(), 0.0);
    for (int k = 0; k < m_grid.count(); ++k) rhs[k] = -m_mu0 * m_Jz[k];

    if (!warmStart || static_cast<int>(m_Az.size()) != m_grid.count())
        m_Az.assign(m_grid.count(), 0.0);

    if (m_method == "multigrid") {
        const MultigridResult r =
            SolvePoissonMultigrid(m_grid, rhs, m_fixedValues, m_Az, m_mgOpt);
        m_lastSolve = {r.cycles, r.residual, r.converged};
    } else {
        m_lastSolve = SolvePoisson(m_grid, rhs, m_fixedMask, m_fixedValues,
                                   m_Az, m_solveOpt);
    }
    CurlZ(m_grid, m_Az, m_Bx, m_By);
    m_Bz.assign(m_grid.count(), 0.0);
}

void MagnetostaticsSim::BiotSavartPath() {
    m_segments = BuildSegments(m_coils);
    m_biot.Evaluate(m_grid, m_plane, m_mu0, m_segments, m_Bx, m_By, m_Bz);
    m_Az.assign(m_grid.count(), 0.0);   // no vector potential on this path
}

void MagnetostaticsSim::RebuildAvoidPoints() {
    m_avoidPts.clear();
    for (const WireSpec& w : m_sources.wires)
        m_avoidPts.emplace_back(w.x, w.y);
    // Where a coil segment pierces (or nearly grazes) the slice plane, project
    // that point into the plane -- field lines spiral into it like a wire core.
    const double tol = 0.03 * std::min(m_grid.lx, m_grid.ly);
    auto perp = [&](const glm::dvec3& p) {
        switch (m_plane.kind) {
        case SlicePlane::XZ: return p.y;
        case SlicePlane::YZ: return p.x;
        default:             return p.z;
        }
    };
    for (const WireSegment& s : m_segments)
        if (std::abs(perp(s.mid) - m_plane.offset) < tol)
            m_avoidPts.push_back(m_plane.InPlane(s.mid));
}

void MagnetostaticsSim::Reset() {
    UpdateField();
    m_charges = m_chargesInit;
    m_step = 0;
    m_time = 0.0;
    m_fieldDirty = true;
}

void MagnetostaticsSim::Step(int substeps) {
    if (m_charges.empty() || m_dt == 0.0) return;
    for (int s = 0; s < substeps; ++s) {
        for (TestCharge& p : m_charges) {
            glm::dvec3 E, B;
            SampleField(p.x, E, B);
            BorisPush(p, E, B, m_dt, m_c);
            p.trail.push_back(glm::vec3(p.x));
            if (p.trail.size() > p.trailCap) p.trail.pop_front();
        }
        m_time += m_dt;
        ++m_step;
    }
    m_fieldDirty = true;   // refresh the trail VBO
}

void MagnetostaticsSim::Snapshot(fw::OutputWriter& writer) {
    // Fields are static -- write them once, on the first frame. Bx, By are the
    // two in-plane components of the slice; Bz is the out-of-plane one
    // (identically zero on the Poisson / xy path).
    if (m_step == 0) {
        writer.WriteField("A_z", m_Az.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
        writer.WriteField("Bx", m_Bx.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
        writer.WriteField("By", m_By.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
        writer.WriteField("Bz", m_Bz.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
        writer.WriteField("Jz", m_Jz.data(), fw::NpyDtype::F8, m_grid.ny, m_grid.nx);
    }

    double bmax = 0.0;
    for (int k = 0; k < m_grid.count(); ++k)
        bmax = std::max(bmax, m_Bx[k] * m_Bx[k] + m_By[k] * m_By[k] +
                              m_Bz[k] * m_Bz[k]);
    writer.WriteScalar("B_max", std::sqrt(bmax));
    writer.WriteScalar("solve_iterations", m_lastSolve.iterations);
    writer.WriteScalar("solve_residual", m_lastSolve.residual);

    char name[24];
    for (std::size_t k = 0; k < m_charges.size(); ++k) {
        const TestCharge& p = m_charges[k];
        glm::dvec3 E, B;
        SampleField(p.x, E, B);
        const glm::dvec3 v = p.velocity(m_c);
        const double bmag = glm::length(B);
        const double vpar = bmag > 1e-30 ? glm::dot(v, B) / bmag : 0.0;
        const double vperp2 = std::max(0.0, glm::dot(v, v) - vpar * vpar);
        auto put = [&](const char* f, double val) {
            std::snprintf(name, sizeof(name), "q%zu_%s", k, f);
            writer.WriteScalar(name, val);
        };
        put("x", p.x.x); put("y", p.x.y); put("z", p.x.z);
        put("speed", glm::length(v));
        put("gamma", p.gamma(m_c));
        put("ke", p.keSpecific(m_c));
        put("vpar", vpar);
        put("mu", bmag > 1e-30 ? vperp2 / bmag : 0.0);
    }
}

fw::SimInfo MagnetostaticsSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_grid.nx;
    info.gridNy = m_grid.ny;
    info.lx = m_grid.lx;
    info.ly = m_grid.ly;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"A_z", "Bx", "By", "Bz", "Jz"};
    info.diagnostics = {"B_max", "solve_iterations", "solve_residual"};
    char name[24];
    for (std::size_t k = 0; k < m_charges.size(); ++k)
        for (const char* f : {"x", "y", "z", "speed", "gamma", "ke", "vpar", "mu"}) {
            std::snprintf(name, sizeof(name), "q%zu_%s", k, f);
            info.diagnostics.emplace_back(name);
        }
    return info;
}

// ---------------------------------------------------------------- interactive

void MagnetostaticsSim::EnsureRenderResources() {
    if (m_renderReady) return;
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    m_line = fw::Shader::FromSource(kLineVert, kLineFrag);
    glGenVertexArrays(1, &m_vao);
    glCreateBuffers(1, &m_fieldBuf);
    glNamedBufferData(m_fieldBuf,
                      static_cast<GLsizeiptr>(m_grid.count() * sizeof(float)),
                      nullptr, GL_DYNAMIC_DRAW);
    m_fieldScratch.assign(m_grid.count(), 0.0f);

    glGenVertexArrays(1, &m_lineVao);
    glGenBuffers(1, &m_lineVbo);
    glBindVertexArray(m_lineVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);

    glGenVertexArrays(1, &m_trailVao);
    glGenBuffers(1, &m_trailVbo);
    glBindVertexArray(m_trailVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_trailVbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), nullptr);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);
    m_renderReady = true;
}

void MagnetostaticsSim::RepackField() {
    const int n = m_grid.count();
    for (int k = 0; k < n; ++k) {
        switch (m_mode) {
        case 1:  m_fieldScratch[k] = static_cast<float>(m_Bx[k]); break;
        case 2:  m_fieldScratch[k] = static_cast<float>(m_By[k]); break;
        case 3:  m_fieldScratch[k] = static_cast<float>(m_Bz[k]); break;
        case 4:  m_fieldScratch[k] = static_cast<float>(m_Az[k]); break;
        default: m_fieldScratch[k] = static_cast<float>(
            std::sqrt(m_Bx[k] * m_Bx[k] + m_By[k] * m_By[k] + m_Bz[k] * m_Bz[k]));
        }
    }
    glNamedBufferSubData(m_fieldBuf, 0,
                         static_cast<GLsizeiptr>(n * sizeof(float)),
                         m_fieldScratch.data());
}

void MagnetostaticsSim::RebuildFieldLines() {
    m_lineVerts.clear();
    const double dx = m_grid.dx(), dy = m_grid.dy();
    auto sampleB = [&](double x, double y, double& bx, double& by) {
        double fi = (x + 0.5 * m_grid.lx) / dx - 0.5;
        double fj = (y + 0.5 * m_grid.ly) / dy - 0.5;
        int i = std::clamp(static_cast<int>(std::floor(fi)), 0, m_grid.nx - 2);
        int j = std::clamp(static_cast<int>(std::floor(fj)), 0, m_grid.ny - 2);
        double tx = std::clamp(fi - i, 0.0, 1.0), ty = std::clamp(fj - j, 0.0, 1.0);
        auto lerp2 = [&](const std::vector<double>& f) {
            return (1 - tx) * (1 - ty) * f[m_grid.idx(i, j)] +
                   tx * (1 - ty) * f[m_grid.idx(i + 1, j)] +
                   (1 - tx) * ty * f[m_grid.idx(i, j + 1)] +
                   tx * ty * f[m_grid.idx(i + 1, j + 1)];
        };
        bx = lerp2(m_Bx);
        by = lerp2(m_By);
    };

    // A streamline that wanders within this distance of a current source is
    // spiralling into the (integrable) singular core -- stop it there rather
    // than let it wind up into a dense blob.
    const double avoidR = std::max(0.04 * std::min(m_grid.lx, m_grid.ly),
                                   3.0 * std::min(dx, dy));
    auto nearWire = [&](double x, double y) {
        for (const glm::dvec2& a : m_avoidPts)
            if (std::hypot(x - a.x, y - a.y) < avoidR) return true;
        return false;
    };

    const int seedsX = 12, seedsY = 12;
    const double step = 1.1 * std::min(dx, dy);
    const int maxSteps = 4000;
    const double xlo = -0.48 * m_grid.lx, xhi = 0.48 * m_grid.lx;
    const double ylo = -0.48 * m_grid.ly, yhi = 0.48 * m_grid.ly;
    const double closeEps = 1.5 * step;   // stop once a closed loop returns home

    for (int sy = 0; sy < seedsY; ++sy) {
        for (int sx = 0; sx < seedsX; ++sx) {
            const double x0 = xlo + (xhi - xlo) * (sx + 0.5) / seedsX;
            const double y0 = ylo + (yhi - ylo) * (sy + 0.5) / seedsY;
            if (nearWire(x0, y0)) continue;
            for (int dir = -1; dir <= 1; dir += 2) {
                double x = x0, y = y0;
                for (int s = 0; s < maxSteps; ++s) {
                    double bx, by;
                    sampleB(x, y, bx, by);
                    double b = std::sqrt(bx * bx + by * by);
                    if (b < 1e-12 || nearWire(x, y)) break;
                    if (s > 8 && std::hypot(x - x0, y - y0) < closeEps) break;
                    // RK4 on the normalised field.
                    auto f = [&](double px, double py, double& ox, double& oy) {
                        double lx, ly;
                        sampleB(px, py, lx, ly);
                        double m = std::sqrt(lx * lx + ly * ly) + 1e-30;
                        ox = dir * lx / m;
                        oy = dir * ly / m;
                    };
                    double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y;
                    f(x, y, k1x, k1y);
                    f(x + 0.5 * step * k1x, y + 0.5 * step * k1y, k2x, k2y);
                    f(x + 0.5 * step * k2x, y + 0.5 * step * k2y, k3x, k3y);
                    f(x + step * k3x, y + step * k3y, k4x, k4y);
                    double nx = x + step / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x);
                    double ny = y + step / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y);
                    if (nx < xlo || nx > xhi || ny < ylo || ny > yhi) break;
                    m_lineVerts.push_back(static_cast<float>(x));
                    m_lineVerts.push_back(static_cast<float>(y));
                    m_lineVerts.push_back(static_cast<float>(nx));
                    m_lineVerts.push_back(static_cast<float>(ny));
                    x = nx;
                    y = ny;
                }
            }
        }
    }
    m_lineVertCount = static_cast<int>(m_lineVerts.size() / 2);
    glBindVertexArray(m_lineVao);
    glBindBuffer(GL_ARRAY_BUFFER, m_lineVbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_lineVerts.size() * sizeof(float)),
                 m_lineVerts.data(), GL_DYNAMIC_DRAW);
    glBindVertexArray(0);
}

void MagnetostaticsSim::Render(int fbWidth, int fbHeight) {
    EnsureRenderResources();

    if (m_fieldDirty) {
        RepackField();
        if (m_showLines) RebuildFieldLines();
        m_fieldDirty = false;
    }

    // auto-scale the colormap to what's on screen
    float fmin = std::numeric_limits<float>::max();
    float fmax = std::numeric_limits<float>::lowest();
    float amax = 0.0f;
    for (float v : m_fieldScratch) {
        fmin = std::min(fmin, v);
        fmax = std::max(fmax, v);
        amax = std::max(amax, std::abs(v));
    }
    const bool signedMode = m_mode != 0;

    const double lx = m_grid.lx, ly = m_grid.ly;
    const float pixPerUnit =
        static_cast<float>(std::min(fbWidth / lx, fbHeight / ly)) * m_zoom;

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.03f, 0.03f, 0.045f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2((float)fbWidth, (float)fbHeight));
    m_view.SetInt("uNx", m_grid.nx);
    m_view.SetInt("uNy", m_grid.ny);
    m_view.SetFloat("uLx", (float)lx);
    m_view.SetFloat("uLy", (float)ly);
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetFloat("uFieldMin", fmin);
    m_view.SetFloat("uFieldMax", signedMode ? amax : fmax);
    m_view.SetInt("uSigned", signedMode ? 1 : 0);
    m_view.SetFloat("uGain", m_gain);
    m_view.SetFloat("uGamma", m_gamma);
    fw::ComputeShader::BindBuffer(0, m_fieldBuf);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    if (m_showLines && m_lineVertCount > 0) {
        m_line.Use();
        m_line.SetVec2("uRes", glm::vec2((float)fbWidth, (float)fbHeight));
        m_line.SetFloat("uPixPerUnit", pixPerUnit);
        m_line.SetVec2("uPanPix", m_panPix);
        m_line.SetVec4("uColor", glm::vec4(0.88f, 0.90f, 0.96f, 0.55f));
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBindVertexArray(m_lineVao);
        glDrawArrays(GL_LINES, 0, m_lineVertCount);
        glBindVertexArray(0);
        glDisable(GL_BLEND);
    }

    // Test-particle trails (3D positions projected into the slice plane).
    if (!m_charges.empty()) {
        m_line.Use();
        m_line.SetVec2("uRes", glm::vec2((float)fbWidth, (float)fbHeight));
        m_line.SetFloat("uPixPerUnit", pixPerUnit);
        m_line.SetVec2("uPanPix", m_panPix);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        for (const TestCharge& p : m_charges) {
            if (p.trail.size() < 2) continue;
            std::vector<float> v;
            v.reserve(p.trail.size() * 2);
            for (const glm::vec3& t : p.trail) {
                const glm::dvec2 ip = m_plane.InPlane(glm::dvec3(t));
                v.push_back((float)ip.x);
                v.push_back((float)ip.y);
            }
            glBindVertexArray(m_trailVao);
            glBindBuffer(GL_ARRAY_BUFFER, m_trailVbo);
            glBufferData(GL_ARRAY_BUFFER,
                         static_cast<GLsizeiptr>(v.size() * sizeof(float)),
                         v.data(), GL_DYNAMIC_DRAW);
            m_line.SetVec4("uColor", glm::vec4(p.color, 0.9f));
            glDrawArrays(GL_LINE_STRIP, 0, static_cast<int>(v.size() / 2));
        }
        glBindVertexArray(0);
        glDisable(GL_BLEND);
    }

    static const char* kModeNames[5] = {"|B|", "Bx", "By", "Bz", "A_z"};
    const std::size_t nSrc =
        m_useBiot ? m_coils.size() : m_sources.wires.size();
    char line[192];
    if (m_charges.empty())
        std::snprintf(line, sizeof(line),
                      "field: %s (M)   lines: %s (L)   %s %d/%zu: arrows move   (%.1f ms/update)",
                      kModeNames[m_mode], m_showLines ? "on" : "off",
                      m_useBiot ? "coil" : "wire",
                      nSrc ? m_activeWire + 1 : 0, nSrc, m_lastSolveMs);
    else
        std::snprintf(line, sizeof(line),
                      "field: %s (M)   lines: %s (L)   %zu charge(s)   t = %.2f",
                      kModeNames[m_mode], m_showLines ? "on" : "off",
                      m_charges.size(), m_time);
    m_text->SetViewport(fbWidth, fbHeight);
    m_text->Draw(m_font, line, glm::vec2(12.0f, 24.0f),
                 glm::vec4(1.0f, 1.0f, 1.0f, 0.9f));

    glEnable(GL_DEPTH_TEST);
}

void MagnetostaticsSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) {
        m_panPix.x += static_cast<float>(in.dx);
        m_panPix.y -= static_cast<float>(in.dy);
    }
    if (in.scrollDelta != 0.0) {
        m_zoom *= std::exp(0.12f * static_cast<float>(in.scrollDelta));
        m_zoom = std::clamp(m_zoom, 0.2f, 60.0f);
    }
}

void MagnetostaticsSim::OnKey(int key, int action) {
    (void)action;
    const double moveStep = 0.03 * std::min(m_grid.lx, m_grid.ly);
    const int nSrc = m_useBiot ? static_cast<int>(m_coils.size())
                               : static_cast<int>(m_sources.wires.size());
    switch (key) {
    case 'M': m_mode = (m_mode + 1) % 5; m_fieldDirty = true; break;
    case 'L': m_showLines = !m_showLines; m_fieldDirty = true; break;
    case '[': m_gain = std::max(0.15f, m_gain * 0.85f); break;
    case ']': m_gain = std::min(12.0f, m_gain * 1.18f); break;
    case '-': m_gamma = std::max(0.2f, m_gamma - 0.05f); break;
    case '=': m_gamma = std::min(2.5f, m_gamma + 0.05f); break;
    case GLFW_KEY_TAB:
        if (nSrc) m_activeWire = (m_activeWire + 1) % nSrc;
        break;
    case '0':
        m_zoom = 1.0f; m_panPix = {0.0f, 0.0f};
        m_gain = 1.0f; m_gamma = 1.0f;
        break;
    case GLFW_KEY_LEFT:  case GLFW_KEY_RIGHT:
    case GLFW_KEY_UP:    case GLFW_KEY_DOWN: {
        if (!nSrc) break;
        const double sx = (key == GLFW_KEY_LEFT ? -moveStep :
                           key == GLFW_KEY_RIGHT ? moveStep : 0.0);
        const double sy = (key == GLFW_KEY_DOWN ? -moveStep :
                           key == GLFW_KEY_UP ? moveStep : 0.0);
        if (m_useBiot) {
            // move the coil centre in the slice plane
            CoilSpec& c = m_coils[m_activeWire];
            switch (m_plane.kind) {
            case SlicePlane::XZ: c.center.x += sx; c.center.z += sy; break;
            case SlicePlane::YZ: c.center.y += sx; c.center.z += sy; break;
            default:             c.center.x += sx; c.center.y += sy; break;
            }
        } else {
            WireSpec& w = m_sources.wires[m_activeWire];
            w.x = std::clamp(w.x + sx, -0.45 * m_grid.lx, 0.45 * m_grid.lx);
            w.y = std::clamp(w.y + sy, -0.45 * m_grid.ly, 0.45 * m_grid.ly);
            m_Jz = RasterizeCurrent(m_grid, m_sources);
        }
        UpdateField(/*warmStart=*/true);
        break;
    }
    default: break;
    }
}

} // namespace mag
