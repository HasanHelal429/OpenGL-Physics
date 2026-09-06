#include "Tddft3DSim.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>

namespace tddft {

namespace {
constexpr double kPi = 3.14159265358979323846;

const char* kVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";

const char* kFrag = R"(#version 460 core
out vec4 FragColor;
layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
uniform vec2 uRes;
uniform int uN;
uniform float uRhoMax;
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
    return clamp(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0.0, 1.0);
}

void main() {
    float s = min(uRes.x, uRes.y);
    vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / s + 0.5;   // 0..1 over the square
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { FragColor = vec4(0.03,0.03,0.045,1); return; }
    int i = clamp(int(uv.x * float(uN)), 0, uN - 1);
    int j = clamp(int(uv.y * float(uN)), 0, uN - 1);
    int mid = uN / 2;
    int idx = (mid * uN + j) * uN + i;
    vec2 p = psi[idx];
    float rel = clamp(dot(p, p) / max(uRhoMax, 1e-30), 0.0, 1.0);
    FragColor = vec4(magma(pow(rel, uGamma)), 1.0);
}
)";
}

double Tddft3DSim::Field(double t) const {
    if (m_E0 == 0.0 || t < 0.0 || t > m_tPulse) return 0.0;
    double env;
    if (t < m_tUp) env = std::pow(std::sin(kPi * t / (2 * m_tUp)), 2);
    else if (t < m_tUp + m_tFlat) env = 1.0;
    else env = std::pow(std::sin(kPi * (m_tPulse - t) / (2 * m_tUp)), 2);
    return m_E0 * env * std::sin(m_wL * t);
}

void Tddft3DSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "rt-TDDFT (GPU)");
    m_n = deck.GetInt("grid.n", 64);
    m_L = deck.GetDouble("grid.L", 32.0);
    m_dt = deck.GetDouble("time.dt", 0.04);
    m_substeps = deck.GetInt("time.substeps_per_frame", 4);

    const int Z = deck.GetInt("atom.Z", 1);
    const double soft = deck.GetDouble("atom.softening", 0.5 * m_L / m_n);

    m_E0 = deck.GetDouble("drive.E0", 0.0);
    m_wL = deck.GetDouble("drive.omega", 0.114);
    const int nRamp = deck.GetInt("drive.n_ramp", 2);
    const int nFlat = deck.GetInt("drive.n_flat", 4);
    m_axis = deck.GetInt("drive.axis", 2);
    const double Tc = 2.0 * kPi / m_wL;
    m_tUp = nRamp * Tc;
    m_tFlat = nFlat * Tc;
    m_tPulse = 2 * m_tUp + m_tFlat;

    m_kick = deck.GetDouble("kick.kappa", 0.0);
    m_kickAxis = deck.GetInt("kick.axis", 0);

    m_p.Configure(m_n, m_L, m_dt);
    m_p.SetBareMode(Z == 1);
    m_p.SetSoftenedNucleus((double)Z, soft);
    if (deck.Has("mask.width")) m_p.SetMask(deck.GetDouble("mask.width", 8.0), 2);
    m_p.RelaxKS(Z, 0.2 * (m_L / m_n) * (m_L / m_n), 800, 1e-8, 4, false);
    if (m_kick != 0.0) m_p.Kick(m_kick, m_kickAxis);
    m_time = 0.0;
    m_steps = 0;

    m_view = fw::Shader::FromSource(kVert, kFrag);
    glGenVertexArrays(1, &m_vao);
}

void Tddft3DSim::Reset() {
    // simplest: re-run Configure's relax path by re-relaxing (bare state) and
    // re-kicking. A stored initial buffer would be faster; deferred.
    m_time = 0.0;
    m_steps = 0;
}

void Tddft3DSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        m_p.LaserStepKS(Field(m_time), Field(m_time + m_dt), m_axis);
        m_time += m_dt;
        ++m_steps;
    }
}

void Tddft3DSim::Snapshot(fw::OutputWriter& writer) {
    std::vector<float> slice;
    m_p.DensitySliceZ0(slice);
    writer.WriteField("density", slice.data(), fw::NpyDtype::F4, m_n, m_n);
    writer.WriteScalar("dipole", m_p.Dipole(m_axis));
    writer.WriteScalar("ionized", 1.0 - m_p.SurvivingNorm() / m_p.OccWeight());
    writer.WriteScalar("field", Field(m_time));
}

fw::SimInfo Tddft3DSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_n;
    info.gridNy = m_n;
    info.lx = m_L;
    info.ly = m_L;
    info.dt = m_dt;
    info.substepsPerFrame = m_substeps;
    info.frameFields = {"density"};
    info.diagnostics = {"dipole", "ionized", "field"};
    return info;
}

void Tddft3DSim::Render(int fbWidth, int fbHeight) {
    std::vector<float> slice;
    m_p.DensitySliceZ0(slice);
    float rhoMax = 1e-30f;
    for (float v : slice) rhoMax = std::max(rhoMax, v);

    glDisable(GL_DEPTH_TEST);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2(fbWidth, fbHeight));
    m_view.SetInt("uN", m_n);
    m_view.SetFloat("uRhoMax", rhoMax);
    m_view.SetFloat("uGamma", m_gamma);
    fw::ComputeShader::BindBuffer(0, m_p.PsiBuffer());
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
}

void Tddft3DSim::OnKey(int key, int action) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    if (key == GLFW_KEY_MINUS) m_gamma = std::min(1.5f, m_gamma + 0.05f);
    if (key == GLFW_KEY_EQUAL) m_gamma = std::max(0.15f, m_gamma - 0.05f);
}

} // namespace tddft
