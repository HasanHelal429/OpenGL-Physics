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
}

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

    m_paths.clear();
    m_q.clear();
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
    }
    if (m_paths.empty()) {           // a default so a bare deck still runs
        m_paths.push_back(ChargePath{});
        m_q.push_back(1.0);
    }
    if (m_paths.size() > 8) { m_paths.resize(8); m_q.resize(8); }
    m_pathsInit = m_paths;

    const int n = m_nx * m_ny;
    for (auto* v : {&m_Ex, &m_Ey, &m_Ez, &m_Emag, &m_Bz, &m_Sr})
        v->assign(n, 0.0);

    m_time = 0.0;
    m_step = 0;
    std::printf("[lw] %dx%d  domain %.1fx%.1f  %zu charge(s)  dt=%.3g  steps=%ld\n",
                m_nx, m_ny, m_lx, m_ly, m_paths.size(), m_dt, m_totalSteps);
}

void RetardedFieldsSim::Reset() {
    m_paths = m_pathsInit;
    m_time = 0.0;
    m_step = 0;
    if (m_gpuInit) {
        std::vector<float> sentinel(m_nx * m_ny * m_paths.size(), 1e30f);
        glNamedBufferSubData(m_bTret, 0,
                             static_cast<GLsizeiptr>(sentinel.size() * sizeof(float)),
                             sentinel.data());
    }
}

void RetardedFieldsSim::Step(int substeps) {
    m_time += m_dt * substeps;
    m_step += substeps;
}

PathFn RetardedFieldsSim::PathFor(int c) const {
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
    for (int k = 0; k < n; ++k) {
        switch (m_viewMode) {
        case 1:  m_fieldScratch[k] = (float)m_Ex[k]; break;
        case 2:  m_fieldScratch[k] = (float)m_Ez[k]; break;
        case 3:  m_fieldScratch[k] = (float)m_Sr[k]; break;
        default: m_fieldScratch[k] = (float)m_Emag[k]; break;
        }
    }
    glNamedBufferSubData(m_fieldBuf, 0, n * sizeof(float), m_fieldScratch.data());
}

void RetardedFieldsSim::Render(int fbWidth, int fbHeight) {
    EnsureRenderResources();
    ComputeField();
    RepackField();

    const bool signedMode = m_viewMode != 0;
    float scale = 0.0f;
    if (signedMode)
        for (float v : m_fieldScratch) scale = std::max(scale, std::abs(v));
    else
        for (float v : m_fieldScratch) scale = std::max(scale, v);
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
    double gmax = 1.0;
    for (int c = 0; c < (int)m_paths.size(); ++c) {
        const glm::dvec3 v = m_paths[c].v(m_time);
        gmax = std::max(gmax, 1.0 / std::sqrt(std::max(1e-9,
                        1.0 - glm::dot(v, v) / (m_c * m_c))));
    }
    char line[192];
    std::snprintf(line, sizeof(line),
                  "field: %s (M)   log %s (L)   t=%.2f   %zu charge(s)  gamma_max=%.2f",
                  kNames[m_viewMode], m_logScale ? "on" : "off", m_time,
                  m_paths.size(), gmax);
    m_text->SetViewport(fbWidth, fbHeight);
    m_text->Draw(m_font, line, glm::vec2(13, 25), glm::vec4(0, 0, 0, 0.55f));
    m_text->Draw(m_font, line, glm::vec2(12, 24), glm::vec4(1, 1, 0.85f, 0.95f));
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
    ChargePath& p = m_paths[m_activeCharge];
    const double step = 0.03 * std::min(m_lx, m_ly);
    switch (key) {
    case 'M': m_viewMode = (m_viewMode + 1) % 4; break;
    case 'L': m_logScale = !m_logScale; break;
    case '[': m_gain = std::max(0.1f, m_gain * 0.8f); break;
    case ']': m_gain = std::min(30.0f, m_gain * 1.25f); break;
    case '-': m_gamma = std::max(0.2f, m_gamma - 0.05f); break;
    case '=': m_gamma = std::min(2.0f, m_gamma + 0.05f); break;
    case ',': p.omega *= 0.92; break;
    case '.': p.omega *= 1.08; break;
    case ';': p.radius = std::max(0.05, p.radius * 0.92); break;
    case '\'': p.radius *= 1.08; break;
    case '\t':
        m_activeCharge = (m_activeCharge + 1) % static_cast<int>(m_paths.size());
        break;
    case '0':
        m_zoom = 1.0f; m_panPix = {0, 0}; m_gain = 1.0f; m_gamma = 0.5f;
        break;
    case GLFW_KEY_LEFT:  p.center.x -= step; break;
    case GLFW_KEY_RIGHT: p.center.x += step; break;
    case GLFW_KEY_DOWN:  p.center.y -= step; break;
    case GLFW_KEY_UP:    p.center.y += step; break;
    default: break;
    }
}

} // namespace lw
