#include "RetardedFieldsSim.hpp"

#include "kernels_lw.hpp"

#include "framework/ComputeShader.hpp"
#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

namespace lw {

namespace {
glm::dvec3 Vec3(const std::vector<double>& a, glm::dvec3 def) {
    return a.size() == 3 ? glm::dvec3(a[0], a[1], a[2]) : def;
}

// Relativistic Larmor power: P = (q^2 / 6 pi eps0 c^3) gamma^6
//   ( |a|^2 - |v x a|^2 / c^2 ).
double LarmorPower(double q, const glm::dvec3& v, const glm::dvec3& a,
                   double c, double eps0) {
    const double b2 = glm::dot(v, v) / (c * c);
    const double g2 = 1.0 / std::max(1e-12, 1.0 - b2);
    const double a2 = glm::dot(a, a);
    const double va2 = glm::dot(glm::cross(v, a), glm::cross(v, a)) / (c * c);
    return q * q / (6.0 * kPi * eps0 * c * c * c) * g2 * g2 * g2 * (a2 - va2);
}
} // namespace

RetardedFieldsSim::~RetardedFieldsSim() {
    for (GLuint b : {m_bE, m_bB, m_bTret, m_fieldBuf})
        if (b) glDeleteBuffers(1, &b);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void RetardedFieldsSim::Configure(const fw::Deck& deck) {
    m_nx = deck.GetInt("domain.nx", 256);
    m_ny = deck.GetInt("domain.ny", m_nx);
    m_lx = deck.GetDouble("domain.lx", 20.0);
    m_ly = deck.GetDouble("domain.ly", m_lx);
    m_sliceZ = deck.GetDouble("domain.slice_z", 0.0);
    m_c = deck.GetDouble("domain.c", 1.0);
    m_eps0 = deck.GetDouble("domain.eps0", 1.0);
    m_dt = deck.GetDouble("time.dt", 0.05);
    m_totalSteps = deck.GetInt("time.steps", 400);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);
    m_title = deck.GetString("title", "Lienard-Wiechert radiated fields");

    m_radWeight = deck.GetBool("render.r_weight", false);
    m_logScale = deck.GetBool("render.log_scale", false);
    { const std::string vm = deck.GetString("render.field", "E_mag");
      m_viewMode = vm == "Ex" ? 1 : vm == "Ez" ? 2
                 : (vm == "S_radial" || vm == "S_mag") ? 3 : 0; }

    const std::string mode = deck.GetString("mode.type", deck.GetString("mode", "prescribed"));
    m_selfConsistent = (mode == "self_consistent");
    m_extB = Vec3(deck.GetDoubleArray("external.B0"), glm::dvec3(0.0));
    m_extE = Vec3(deck.GetDoubleArray("external.E0"), glm::dvec3(0.0));

    m_paths.clear();
    m_q.clear();
    m_mass.clear();
    for (const auto& ct : deck.GetTables("charge")) {
        ChargePath p;
        p.kind = ChargePath::ParseKind(ct.GetString("path", "circular"));
        p.center = Vec3(ct.GetDoubleArray("center"), glm::dvec3(0.0));
        p.v0 = Vec3(ct.GetDoubleArray("v0"), glm::dvec3(0.0));
        p.axis = glm::normalize(Vec3(ct.GetDoubleArray("axis"),
                                     glm::dvec3(1.0, 0.0, 0.0)));
        p.radius = ct.GetDouble("radius", 1.0);
        p.amplitude = ct.GetDouble("amplitude", 1.0);
        p.omega = ct.GetDouble("omega", 1.0);
        p.phase = ct.GetDouble("phase", 0.0);
        m_paths.push_back(p);
        m_q.push_back(ct.GetDouble("q", 1.0));
        m_mass.push_back(ct.GetDouble("m", ct.GetDouble("mass", 1.0)));
    }
    if (m_paths.empty()) {           // a default so a bare deck still runs
        m_paths.push_back(ChargePath{});
        m_q.push_back(1.0);
        m_mass.push_back(1.0);
    }
    if (m_paths.size() > 8) { m_paths.resize(8); m_q.resize(8); m_mass.resize(8); }
    m_pathsInit = m_paths;

    const int n = m_nx * m_ny;
    for (auto* v : {&m_Ex, &m_Ey, &m_Ez, &m_Emag, &m_Bz, &m_Sr})
        v->assign(n, 0.0);

    m_time = 0.0;
    m_step = 0;
    m_radiated = 0.0;
    m_e0Set = false;

    if (m_selfConsistent) {
        m_forceCpu = true;            // GPU grid path is prescribed-only for now
        SeedOrbits(deck);
    }

    std::printf("[lw] %dx%d  domain %.1fx%.1f  %zu charge(s)  dt=%.3g  steps=%ld  mode=%s\n",
                m_nx, m_ny, m_lx, m_ly, m_paths.size(), m_dt, m_totalSteps,
                m_selfConsistent ? "self_consistent" : "prescribed");
}

void RetardedFieldsSim::SeedOrbits(const fw::Deck& deck) {
    const auto tables = deck.GetTables("charge");
    const int nc = static_cast<int>(m_q.size());
    const double k = 1.0 / (4.0 * kPi * m_eps0);

    m_movers.assign(nc, Mover{});
    m_buf.assign(nc, TrajectoryBuffer{});
    std::vector<ChargePath>& seed = m_seed;
    seed.assign(nc, ChargePath{});             // steady past for the prefill

    // Global seed selection: "kepler" builds a 2-body bound circular orbit
    // from [mode].separation; otherwise each charge follows its own uniform
    // drift / fixed point from x0, v0.
    std::string sorb = deck.GetString("mode.seed_orbit", "uniform");
    if (!tables.empty()) sorb = tables[0].GetString("seed_orbit", sorb);

    double maxR = 1.0, omega = 0.0;
    if (sorb == "kepler" && nc == 2) {
        const double d = deck.GetDouble("mode.separation",
                                        deck.GetDouble("history.separation", 4.0));
        const double m1 = m_mass[0], m2 = m_mass[1];
        const double R1 = d * m2 / (m1 + m2), R2 = d * m1 / (m1 + m2);
        const double Fmag = k * std::abs(m_q[0] * m_q[1]) / (d * d);
        omega = std::sqrt(Fmag / (m1 * R1));
        maxR = std::max(R1, R2);
        seed[0] = ChargePath{ChargePath::Circular, glm::dvec3(0.0), glm::dvec3(0.0),
                             R1, R1, omega, 0.0, glm::dvec3(1, 0, 0)};
        seed[1] = ChargePath{ChargePath::Circular, glm::dvec3(0.0), glm::dvec3(0.0),
                             R2, R2, omega, kPi, glm::dvec3(1, 0, 0)};
        std::printf("[lw] kepler seed: d=%.3f  R=(%.3f,%.3f)  omega=%.4f  "
                    "v=(%.3f,%.3f)  T=%.3f\n",
                    d, R1, R2, omega, omega * R1, omega * R2, 2.0 * kPi / omega);
    } else {
        for (int c = 0; c < nc; ++c) {
            const glm::dvec3 x0 = c < (int)tables.size()
                ? Vec3(tables[c].GetDoubleArray("x0"), m_paths[c].center)
                : m_paths[c].center;
            const glm::dvec3 v0 = c < (int)tables.size()
                ? Vec3(tables[c].GetDoubleArray("v0"), glm::dvec3(0.0))
                : glm::dvec3(0.0);
            const std::string sc = c < (int)tables.size()
                ? tables[c].GetString("seed_orbit", sorb) : sorb;
            ChargePath p;
            p.kind = (sc == "static") ? ChargePath::Static : ChargePath::Uniform;
            p.center = x0;
            p.v0 = (sc == "static") ? glm::dvec3(0.0) : v0;
            seed[c] = p;
            maxR = std::max(maxR, glm::length(x0));
        }
    }

    // Buffer span: cover the longest retarded lookback an in-domain observer
    // can need -- domain diagonal plus a few orbit radii of source travel.
    const double diag = std::sqrt(m_lx * m_lx + m_ly * m_ly);
    m_bufferSpan = deck.GetDouble("history.buffer_span",
                                  1.5 * (diag + 4.0 * maxR) / m_c);
    // Hard floor: the charge-charge retarded query reaches back ~2 maxR / c.
    // Below that the ring buffer clamps and injects a divergent spurious
    // force (see Studies/retarded_fields/history_convergence).
    const double floorSpan = 4.0 * std::max(maxR, 1.0) / m_c;
    if (m_bufferSpan < floorSpan) {
        std::printf("[lw] history.buffer_span %.2f below the %.2f floor "
                    "(2 * pair reach); raising it\n", m_bufferSpan, floorSpan);
        m_bufferSpan = floorSpan;
    }
    std::size_t cap = static_cast<std::size_t>(m_bufferSpan / m_dt) + 4;
    cap = std::min<std::size_t>(cap, 400000);
    m_bufCap = cap;
    m_settleSteps = static_cast<long>(cap);
    LaySeed();
}

void RetardedFieldsSim::LaySeed() {
    const int nc = static_cast<int>(m_q.size());
    m_movers.assign(nc, Mover{});
    m_buf.assign(nc, TrajectoryBuffer{});
    for (int c = 0; c < nc; ++c) {
        const ChargePath sp = m_seed[c];
        m_buf[c].Init(m_dt, m_bufCap, -m_bufferSpan);
        m_buf[c].Prefill(0.0, [sp](double t) {
            return PathSample{sp.r(t), sp.v(t), sp.a(t)};
        });
        m_movers[c].x = sp.r(0.0);
        m_movers[c].qm = m_q[c] / m_mass[c];
        const glm::dvec3 v = sp.v(0.0);
        const double g = 1.0 / std::sqrt(std::max(1e-12,
                              1.0 - glm::dot(v, v) / (m_c * m_c)));
        m_movers[c].u = g * v;
    }
    m_time = 0.0;
    m_step = 0;
    m_radiated = 0.0;
    const EnergyBudget b0 = Energy();          // radiated is 0 here
    m_e0 = b0.kinetic + b0.interaction;
    m_e0Set = true;
}

void RetardedFieldsSim::FieldAt(const glm::dvec3& xp, double t, int skip,
                                glm::dvec3& E, glm::dvec3& B) const {
    E = m_extE;
    B = m_extB;
    for (int j = 0; j < (int)m_q.size(); ++j) {
        if (j == skip) continue;
        const LwResult lr = LwFields(m_buf[j].PathFn_(), xp, t, m_q[j], m_c, m_eps0,
                                     std::numeric_limits<double>::quiet_NaN());
        E += lr.E;
        B += lr.B;
    }
}

void RetardedFieldsSim::StepSelfConsistent(int substeps) {
    const int nc = static_cast<int>(m_q.size());
    for (int s = 0; s < substeps; ++s) {
        // Fields on each charge from every other charge's retarded history.
        std::vector<glm::dvec3> Ei(nc), Bi(nc), acc(nc);
        for (int i = 0; i < nc; ++i)
            FieldAt(m_movers[i].x, m_time, i, Ei[i], Bi[i]);

        // Coordinate acceleration (for the buffer) at the pre-push state.
        for (int i = 0; i < nc; ++i)
            acc[i] = CoordAccel(m_movers[i], Ei[i], Bi[i], m_c);

        // Radiated energy over this step (relativistic Larmor, midpoint-ish).
        for (int i = 0; i < nc; ++i)
            m_radiated += LarmorPower(m_q[i], m_movers[i].v(m_c), acc[i],
                                      m_c, m_eps0) * m_dt;

        // Push and record.
        for (int i = 0; i < nc; ++i)
            BorisPush(m_movers[i], Ei[i], Bi[i], m_dt, m_c);
        m_time += m_dt;
        m_step += 1;
        for (int i = 0; i < nc; ++i) {
            const glm::dvec3 v = m_movers[i].v(m_c);
            // recompute a at the new position for the sample we store
            glm::dvec3 En, Bn;
            FieldAt(m_movers[i].x, m_time, i, En, Bn);
            m_buf[i].Push(m_movers[i].x, v, CoordAccel(m_movers[i], En, Bn, m_c));
        }
    }

}

RetardedFieldsSim::EnergyBudget RetardedFieldsSim::Energy() const {
    EnergyBudget b;
    const double kc = 1.0 / (4.0 * kPi * m_eps0);
    const int nc = static_cast<int>(m_q.size());
    for (int i = 0; i < nc; ++i)
        b.kinetic += m_mass[i] * m_movers[i].keSpecific(m_c);
    for (int i = 0; i < nc; ++i)
        for (int j = i + 1; j < nc; ++j) {
            const double r = glm::length(m_movers[i].x - m_movers[j].x);
            if (r > 1e-9) b.interaction += kc * m_q[i] * m_q[j] / r;
        }
    b.radiated = m_radiated;
    return b;
}

double RetardedFieldsSim::EnergyError() const {
    if (!m_e0Set) return 0.0;
    const double e = Energy().total();
    const double denom = std::max(1e-12, std::abs(m_e0));
    return std::abs(e - m_e0) / denom;
}

glm::dvec3 RetardedFieldsSim::ChargePos(int c) const {
    if (m_selfConsistent && c < (int)m_movers.size()) return m_movers[c].x;
    return m_paths[c].r(m_time);
}

void RetardedFieldsSim::Reset() {
    m_paths = m_pathsInit;
    m_time = 0.0;
    m_step = 0;
    m_radiated = 0.0;
    m_e0Set = false;
    if (m_selfConsistent && !m_seed.empty()) LaySeed();
    if (m_gpuInit) {
        std::vector<float> sentinel(m_nx * m_ny * m_paths.size(), 1e30f);
        glNamedBufferSubData(m_bTret, 0,
                             static_cast<GLsizeiptr>(sentinel.size() * sizeof(float)),
                             sentinel.data());
    }
}

void RetardedFieldsSim::Step(int substeps) {
    if (m_selfConsistent) { StepSelfConsistent(substeps); return; }
    m_time += m_dt * substeps;
    m_step += substeps;
}

PathFn RetardedFieldsSim::PathFor(int c) const {
    if (m_selfConsistent && c < (int)m_buf.size())
        return m_buf[c].PathFn_();
    const ChargePath p = m_paths[c];
    return [p](double tau) {
        return PathSample{p.r(tau), p.v(tau), p.a(tau)};
    };
}

void RetardedFieldsSim::ComputeFieldCpu() {
    const int nc = static_cast<int>(m_paths.size());
    std::vector<PathFn> fns;
    for (int c = 0; c < nc; ++c) fns.push_back(PathFor(c));

    for (int j = 0; j < m_ny; ++j) {
        for (int i = 0; i < m_nx; ++i) {
            const glm::dvec3 xp(x(i), y(j), m_sliceZ);
            glm::dvec3 E(0.0), B(0.0);
            for (int c = 0; c < nc; ++c) {
                const LwResult lr = LwFields(fns[c], xp, m_time, m_q[c], m_c,
                                             m_eps0,
                                             std::numeric_limits<double>::quiet_NaN());
                E += lr.E;
                B += lr.B;
            }
            const std::size_t p = idx(i, j);
            m_Ex[p] = E.x; m_Ey[p] = E.y; m_Ez[p] = E.z;
            m_Emag[p] = glm::length(E);
            m_Bz[p] = B.z;
            const glm::dvec3 S = glm::cross(E, B);   // mu0 = 1
            const double rr = glm::length(xp);
            m_Sr[p] = rr > 1e-12 ? glm::dot(S, xp / rr) : 0.0;
        }
    }
}

void RetardedFieldsSim::EnsureGpu() {
    if (m_gpuInit) return;
    m_prog = fw::ComputeShader::FromSource(kernels::LwField());
    const int n = m_nx * m_ny;
    glCreateBuffers(1, &m_bE);
    glCreateBuffers(1, &m_bB);
    glCreateBuffers(1, &m_bTret);
    glNamedBufferData(m_bE, n * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_bB, n * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    std::vector<float> sentinel(static_cast<std::size_t>(n) * m_paths.size(), 1e30f);
    glNamedBufferData(m_bTret,
                      static_cast<GLsizeiptr>(sentinel.size() * sizeof(float)),
                      sentinel.data(), GL_DYNAMIC_DRAW);
    m_readE.assign(n * 4, 0.0f);
    m_readB.assign(n * 4, 0.0f);
    m_gpuInit = true;
}

void RetardedFieldsSim::ComputeField() {
    if (m_forceCpu) { ComputeFieldCpu(); return; }
    EnsureGpu();
    const int nc = static_cast<int>(m_paths.size());

    m_prog.Use();
    m_prog.SetInt("uNx", m_nx);   m_prog.SetInt("uNy", m_ny);
    m_prog.SetFloat("uX0", static_cast<float>(-0.5 * m_lx));
    m_prog.SetFloat("uDx", static_cast<float>(m_lx / m_nx));
    m_prog.SetFloat("uY0", static_cast<float>(-0.5 * m_ly));
    m_prog.SetFloat("uDy", static_cast<float>(m_ly / m_ny));
    m_prog.SetFloat("uZ", static_cast<float>(m_sliceZ));
    m_prog.SetFloat("uT", static_cast<float>(m_time));
    m_prog.SetFloat("uC", static_cast<float>(m_c));
    m_prog.SetFloat("uEps0", static_cast<float>(m_eps0));
    m_prog.SetInt("uNCharge", nc);

    int kind[8];
    glm::vec4 center[8], v0[8], geom[8], axis[8];
    for (int c = 0; c < nc; ++c) {
        const ChargePath& p = m_paths[c];
        kind[c] = static_cast<int>(p.kind);
        center[c] = glm::vec4(p.center, static_cast<float>(m_q[c]));
        v0[c] = glm::vec4(p.v0, static_cast<float>(p.omega));
        geom[c] = glm::vec4((float)p.radius, (float)p.amplitude, (float)p.phase, 0.0f);
        axis[c] = glm::vec4(p.axis, 0.0f);
    }
    m_prog.SetIntArray("uKind", kind, nc);
    m_prog.SetVec4Array("uCenter", center, nc);
    m_prog.SetVec4Array("uV0", v0, nc);
    m_prog.SetVec4Array("uGeom", geom, nc);
    m_prog.SetVec4Array("uAxis", axis, nc);

    fw::ComputeShader::BindBuffer(0, m_bE);
    fw::ComputeShader::BindBuffer(1, m_bB);
    fw::ComputeShader::BindBuffer(2, m_bTret);
    m_prog.Dispatch((m_nx + 15) / 16, (m_ny + 15) / 16);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT |
                               GL_BUFFER_UPDATE_BARRIER_BIT);

    const int n = m_nx * m_ny;
    glGetNamedBufferSubData(m_bE, 0, n * 4 * sizeof(float), m_readE.data());
    glGetNamedBufferSubData(m_bB, 0, n * 4 * sizeof(float), m_readB.data());
    for (int j = 0; j < m_ny; ++j)
        for (int i = 0; i < m_nx; ++i) {
            const std::size_t p = idx(i, j);
            const glm::dvec3 E(m_readE[p * 4], m_readE[p * 4 + 1], m_readE[p * 4 + 2]);
            const glm::dvec3 B(m_readB[p * 4], m_readB[p * 4 + 1], m_readB[p * 4 + 2]);
            m_Ex[p] = E.x; m_Ey[p] = E.y; m_Ez[p] = E.z;
            m_Emag[p] = glm::length(E);
            m_Bz[p] = B.z;
            const glm::dvec3 xp(x(i), y(j), m_sliceZ);
            const double rr = glm::length(xp);
            m_Sr[p] = rr > 1e-12 ? glm::dot(glm::cross(E, B), xp / rr) : 0.0;
        }
}

void RetardedFieldsSim::Snapshot(fw::OutputWriter& writer) {
    ComputeField();
    writer.WriteField("Ex", m_Ex.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteField("Ey", m_Ey.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteField("Ez", m_Ez.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteField("E_mag", m_Emag.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteField("Bz", m_Bz.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteField("S_radial", m_Sr.data(), fw::NpyDtype::F8, m_ny, m_nx);

    double emax = 0.0, srsum = 0.0;
    for (int k = 0; k < m_nx * m_ny; ++k) {
        emax = std::max(emax, m_Emag[k]);
        srsum += m_Sr[k];
    }
    writer.WriteScalar("E_max", emax);
    writer.WriteScalar("S_radial_sum", srsum);

    if (m_selfConsistent) {
        const EnergyBudget b = Energy();
        writer.WriteScalar("KE", b.kinetic);
        writer.WriteScalar("PE_interaction", b.interaction);
        writer.WriteScalar("E_radiated", b.radiated);
        writer.WriteScalar("E_total", b.total());
        writer.WriteScalar("energy_error", EnergyError());
        const int nc = static_cast<int>(m_q.size());
        if (nc >= 2) {
            writer.WriteScalar("separation",
                               glm::length(m_movers[0].x - m_movers[1].x));
        }
        for (int c = 0; c < nc && c < 4; ++c) {
            const glm::dvec3 p = m_movers[c].x;
            const glm::dvec3 v = m_movers[c].v(m_c);
            writer.WriteScalar("x" + std::to_string(c), p.x);
            writer.WriteScalar("y" + std::to_string(c), p.y);
            writer.WriteScalar("gamma" + std::to_string(c), m_movers[c].gamma(m_c));
            writer.WriteScalar("speed" + std::to_string(c), glm::length(v));
        }
    }
}

fw::SimInfo RetardedFieldsSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_nx;
    info.gridNy = m_ny;
    info.lx = m_lx;
    info.ly = m_ly;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"Ex", "Ey", "Ez", "E_mag", "Bz", "S_radial"};
    info.diagnostics = {"E_max", "S_radial_sum"};
    if (m_selfConsistent) {
        info.diagnostics = {"E_max", "S_radial_sum", "KE", "PE_interaction",
                            "E_radiated", "E_total", "energy_error", "separation"};
    }
    return info;
}

// ------------------------------------------------------------- interactive
// (Phase 2 fleshes this out; a minimal heatmap for --render-check now.)

namespace {
const char* kViewVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";
const char* kViewFrag = R"(#version 460 core
out vec4 FragColor;
layout(std430, binding = 0) readonly buffer FieldBuf { float field[]; };
uniform vec2 uRes; uniform int uNx; uniform int uNy;
uniform float uLx; uniform float uLy; uniform float uPixPerUnit; uniform vec2 uPanPix;
uniform float uScale; uniform int uSigned; uniform float uGain; uniform float uGamma;
uniform int uLog;
vec3 magma(float t){t=clamp(t,0.0,1.0);
  const vec3 c0=vec3(-0.002136,-0.000750,-0.005386),c1=vec3(0.251723,0.677631,2.494027);
  const vec3 c2=vec3(8.353717,-3.577720,0.311613),c3=vec3(-27.668733,14.264731,-13.649213);
  const vec3 c4=vec3(52.176140,-27.943606,12.944169),c5=vec3(-50.768525,29.046583,4.234153);
  const vec3 c6=vec3(18.655705,-11.489774,-5.601962);
  return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))),0.0,1.0);}
vec3 diverging(float s){vec3 lo=vec3(0.19,0.31,0.75),mid=vec3(0.98,0.98,0.98),hi=vec3(0.79,0.16,0.16);
  return s<0.0?mix(mid,lo,clamp(-s,0.0,1.0)):mix(mid,hi,clamp(s,0.0,1.0));}
void main(){
  float wx=(gl_FragCoord.x-0.5*uRes.x-uPanPix.x)/uPixPerUnit;
  float wy=(gl_FragCoord.y-0.5*uRes.y-uPanPix.y)/uPixPerUnit;
  int i=int(floor((wx+0.5*uLx)/(uLx/float(uNx))));
  int j=int(floor((wy+0.5*uLy)/(uLy/float(uNy))));
  if(i<0||j<0||i>=uNx||j>=uNy){FragColor=vec4(0.03,0.03,0.045,1.0);return;}
  float v=field[j*uNx+i];
  if(uSigned==1){
    float a=abs(v)/uScale;
    if(uLog==1) a=clamp((log(a*1e6+1.0))/log(1e6+1.0),0.0,1.0);
    float s=sign(v)*pow(clamp(a*uGain,0.0,1.0),uGamma);
    FragColor=vec4(diverging(clamp(s,-1.0,1.0)),1.0);
  } else {
    float a=v/uScale;
    if(uLog==1) a=clamp((log(a*1e6+1.0))/log(1e6+1.0),0.0,1.0);
    FragColor=vec4(magma(pow(clamp(a*uGain,0.0,1.0),uGamma)),1.0);
  }
}
)";
} // namespace

void RetardedFieldsSim::EnsureRenderResources() {
    if (m_renderReady) return;
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    glGenVertexArrays(1, &m_vao);
    glCreateBuffers(1, &m_fieldBuf);
    glNamedBufferData(m_fieldBuf, m_nx * m_ny * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    m_fieldScratch.assign(m_nx * m_ny, 0.0f);
    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);
    m_renderReady = true;
}

void RetardedFieldsSim::RepackField() {
    const int n = m_nx * m_ny;
    // r measured from the centroid of the active charge's orbit -- close
    // enough to the true source for the 1/r flattening to work everywhere
    // outside the near zone.
    const int nc = NumCharges();
    std::vector<glm::dvec2> src(nc);
    glm::dvec2 centroid(0.0);
    for (int c = 0; c < nc; ++c) {
        const glm::dvec3 p = ChargePos(c);
        src[c] = glm::dvec2(p.x, p.y);
        centroid += src[c];
    }
    centroid /= static_cast<double>(nc);
    const double rNear = std::max(1.0, 3.0 * m_lx / m_nx);   // hide each near zone
    for (int j = 0; j < m_ny; ++j) {
        for (int i = 0; i < m_nx; ++i) {
            const std::size_t k = idx(i, j);
            float v;
            switch (m_viewMode) {
            case 1:  v = (float)m_Ex[k]; break;
            case 2:  v = (float)m_Ez[k]; break;
            case 3:  v = (float)m_Sr[k]; break;
            default: v = (float)m_Emag[k]; break;
            }
            if (m_radWeight) {
                double rMin = 1e30;
                for (const auto& s : src)
                    rMin = std::min(rMin, std::hypot(x(i) - s.x, y(j) - s.y));
                const double rc = std::hypot(x(i) - centroid.x, y(j) - centroid.y);
                v = rMin < rNear ? 0.0f : v * (float)rc;
            }
            m_fieldScratch[k] = v;
        }
    }
    glNamedBufferSubData(m_fieldBuf, 0, n * sizeof(float), m_fieldScratch.data());
}

void RetardedFieldsSim::Render(int fbWidth, int fbHeight) {
    EnsureRenderResources();
    ComputeField();
    RepackField();

    // Normalisation: with r-weighting on, a few cells clinging to a masked
    // near zone shouldn't set the whole scale, so use the 99.5th percentile
    // of |field|. Without it (raw 1/r^2, log scale) the near-field maximum is
    // what the log compression is meant to tame -- keep the true max.
    const bool signedMode = m_viewMode != 0;
    float scale = 0.0f;
    if (m_radWeight) {
        std::vector<float> mag(m_fieldScratch.size());
        for (std::size_t k = 0; k < mag.size(); ++k)
            mag[k] = std::abs(m_fieldScratch[k]);
        const std::size_t q = static_cast<std::size_t>(0.995 * (mag.size() - 1));
        std::nth_element(mag.begin(), mag.begin() + q, mag.end());
        scale = mag[q];
    } else {
        for (float v : m_fieldScratch) scale = std::max(scale, std::abs(v));
    }
    if (scale < 1e-30f) scale = 1.0f;

    const float pixPerUnit =
        static_cast<float>(std::min(fbWidth / m_lx, fbHeight / m_ly)) * m_zoom;
    glDisable(GL_DEPTH_TEST);
    glClearColor(0.03f, 0.03f, 0.045f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2((float)fbWidth, (float)fbHeight));
    m_view.SetInt("uNx", m_nx); m_view.SetInt("uNy", m_ny);
    m_view.SetFloat("uLx", (float)m_lx); m_view.SetFloat("uLy", (float)m_ly);
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetFloat("uScale", scale);
    m_view.SetInt("uSigned", signedMode ? 1 : 0);
    m_view.SetInt("uLog", m_logScale ? 1 : 0);
    m_view.SetFloat("uGain", m_gain);
    m_view.SetFloat("uGamma", m_gamma);
    fw::ComputeShader::BindBuffer(0, m_fieldBuf);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    static const char* kNames[4] = {"|E|", "Ex", "Ez", "S_radial"};
    const int nc = NumCharges();
    double gmax = 1.0;
    for (int c = 0; c < nc; ++c) {
        const glm::dvec3 v = m_selfConsistent ? m_movers[c].v(m_c)
                                              : m_paths[c].v(m_time);
        gmax = std::max(gmax, 1.0 / std::sqrt(std::max(1e-9,
                        1.0 - glm::dot(v, v) / (m_c * m_c))));
    }
    char line[256];
    std::snprintf(line, sizeof(line),
                  "field: %s (M)   log %s (L)   r-weight %s (R)   t=%.2f",
                  kNames[m_viewMode], m_logScale ? "on" : "off",
                  m_radWeight ? "on" : "off", m_time);
    char line2[256];
    if (m_selfConsistent) {
        const EnergyBudget b = Energy();
        const double sep = nc >= 2
            ? glm::length(m_movers[0].x - m_movers[1].x) : 0.0;
        std::snprintf(line2, sizeof(line2),
                      "self-consistent  %d charges   separation=%.3f   "
                      "gamma_max=%.3f   dE/E0=%.1e   E_rad=%.2e",
                      nc, sep, gmax, EnergyError(), b.radiated);
    } else {
        const ChargePath& ap = m_paths[m_activeCharge];
        std::snprintf(line2, sizeof(line2),
                      "charge %d/%d (Tab)   omega=%.3f (,/.)   radius=%.2f (;/')"
                      "   gamma_max=%.2f",
                      m_activeCharge + 1, nc, ap.omega, ap.radius, gmax);
    }
    m_text->SetViewport(fbWidth, fbHeight);

    // charge markers
    for (int c = 0; c < nc; ++c) {
        const glm::dvec3 pw = ChargePos(c);
        const float fx = (float)pw.x * pixPerUnit + 0.5f * fbWidth + m_panPix.x;
        const float fy = (float)pw.y * pixPerUnit + 0.5f * fbHeight + m_panPix.y;
        const float tx = fx, ty = fbHeight - fy;
        if (tx < -20 || tx > fbWidth + 20 || ty < -20 || ty > fbHeight + 20)
            continue;
        const double qc = m_q.empty() ? 1.0 : m_q[c];
        const char* mk = qc >= 0.0 ? "(+)" : "(-)";
        const glm::vec4 col = qc >= 0.0 ? glm::vec4(1.0f, 0.9f, 0.5f, 1.0f)
                                        : glm::vec4(0.55f, 0.8f, 1.0f, 1.0f);
        m_text->Draw(m_font, mk, glm::vec2(tx - 11, ty + 7), glm::vec4(0, 0, 0, 0.7f));
        m_text->Draw(m_font, mk, glm::vec2(tx - 12, ty + 6), col);
    }

    for (int p = 0; p < 2; ++p) {
        const char* s = p == 0 ? line : line2;
        const float yy = 24.0f + p * 20.0f;
        m_text->Draw(m_font, s, glm::vec2(13, yy + 1), glm::vec4(0, 0, 0, 0.55f));
        m_text->Draw(m_font, s, glm::vec2(12, yy), glm::vec4(1, 1, 0.85f, 0.95f));
    }
    glEnable(GL_DEPTH_TEST);
}

void RetardedFieldsSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) { m_panPix.x += (float)in.dx; m_panPix.y -= (float)in.dy; }
    if (in.scrollDelta != 0.0) {
        m_zoom *= std::exp(0.12f * (float)in.scrollDelta);
        m_zoom = std::clamp(m_zoom, 0.25f, 40.0f);
    }
}

void RetardedFieldsSim::OnKey(int key, int action) {
    (void)action;
    const double step = 0.03 * std::min(m_lx, m_ly);
    switch (key) {
    case 'M': m_viewMode = (m_viewMode + 1) % 4; return;
    case 'L': m_logScale = !m_logScale; return;
    case 'R': m_radWeight = !m_radWeight; return;
    case '[': m_gain = std::max(0.1f, m_gain * 0.8f); return;
    case ']': m_gain = std::min(30.0f, m_gain * 1.25f); return;
    case '-': m_gamma = std::max(0.2f, m_gamma - 0.05f); return;
    case '=': m_gamma = std::min(2.0f, m_gamma + 0.05f); return;
    case '\t':
        m_activeCharge = (m_activeCharge + 1) % NumCharges();
        return;
    case '0':
        m_zoom = 1.0f; m_panPix = {0, 0}; m_gain = 1.0f; m_gamma = 0.8f;
        return;
    default: break;
    }

    if (m_selfConsistent) return;   // charges are dynamical -- no live editing

    ChargePath& p = m_paths[m_activeCharge];
    switch (key) {
    case ',': p.omega *= 0.92; break;
    case '.': p.omega *= 1.08; break;
    case ';': p.radius = std::max(0.05, p.radius * 0.92); break;
    case '\'': p.radius *= 1.08; break;
    case GLFW_KEY_LEFT:  p.center.x -= step; break;
    case GLFW_KEY_RIGHT: p.center.x += step; break;
    case GLFW_KEY_DOWN:  p.center.y -= step; break;
    case GLFW_KEY_UP:    p.center.y += step; break;
    default: break;
    }
}

} // namespace lw
