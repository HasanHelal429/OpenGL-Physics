#include "GrhdSim.hpp"
#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>

namespace grhd {

GrhdSim::~GrhdSim() {
    GLuint bufs[] = {m_cons[0], m_cons[1], m_stage1, m_stage2, m_prim, m_flux};
    glDeleteBuffers(6, bufs);
}

void GrhdSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    m_n = deck.GetInt("grid.n", 400);
    m_length = deck.GetDouble("grid.length", 1.0);
    m_dx = m_length / static_cast<double>(m_n);

    m_gamma = deck.GetDouble("physics.gamma", 5.0 / 3.0);

    m_rhoL = deck.GetDouble("riemann.rho_l", 1.0);
    m_vL = deck.GetDouble("riemann.v_l", 0.0);
    m_pL = deck.GetDouble("riemann.p_l", 1.0);
    m_rhoR = deck.GetDouble("riemann.rho_r", 0.125);
    m_vR = deck.GetDouble("riemann.v_r", 0.0);
    m_pR = deck.GetDouble("riemann.p_r", 0.1);
    m_x0 = deck.GetDouble("riemann.x0", 0.5);

    m_cfl = deck.GetDouble("time.cfl", 0.4);
    m_dt = m_cfl * m_dx; // no signal in SR moves faster than c=1 -- always a safe CFL bound
    m_consToPrimIters = deck.GetInt("solver.cons_to_prim_iters", 25);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    CreateBuffers();
    UploadInitial();

    m_consToPrim = fw::ComputeShader::FromSource(kernels::ConsToPrim());
    m_fluxes = fw::ComputeShader::FromSource(kernels::Fluxes());
    m_eulerStep = fw::ComputeShader::FromSource(kernels::EulerStep());
    m_combine = fw::ComputeShader::FromSource(kernels::Combine());
}

namespace {
glm::vec4 PrimToCons(double rho, double v, double p, double gamma) {
    const double W = 1.0 / std::sqrt(1.0 - v * v);
    const double h = 1.0 + gamma * p / ((gamma - 1.0) * rho);
    const double D = rho * W;
    const double S = rho * h * W * W * v;
    const double tau = rho * h * W * W - p - D;
    return glm::vec4(static_cast<float>(D), static_cast<float>(S), static_cast<float>(tau), 0.0f);
}
} // namespace

void GrhdSim::CreateBuffers() {
    glCreateBuffers(1, &m_cons[0]);
    glCreateBuffers(1, &m_cons[1]);
    glCreateBuffers(1, &m_stage1);
    glCreateBuffers(1, &m_stage2);
    glCreateBuffers(1, &m_prim);
    glCreateBuffers(1, &m_flux);

    glNamedBufferData(m_cons[0], m_n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_cons[1], m_n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage1, m_n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_stage2, m_n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_prim, m_n * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
    glNamedBufferData(m_flux, (m_n + 1) * sizeof(glm::vec4), nullptr, GL_DYNAMIC_DRAW);
}

void GrhdSim::UploadInitial() {
    m_initialCons.resize(m_n);
    std::vector<glm::vec4> initialPrim(m_n);
    for (int i = 0; i < m_n; ++i) {
        const double x = (static_cast<double>(i) + 0.5) * m_dx;
        const bool left = x < m_x0 * m_length;
        const double rho = left ? m_rhoL : m_rhoR;
        const double v = left ? m_vL : m_vR;
        const double p = left ? m_pL : m_pR;
        m_initialCons[i] = PrimToCons(rho, v, p, m_gamma);
        initialPrim[i] = glm::vec4(static_cast<float>(rho), static_cast<float>(v), static_cast<float>(p), 0.0f);
    }
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), initialPrim.data());
    m_cur = 0;
}

void GrhdSim::Reset() {
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    m_cur = 0;
}

void GrhdSim::RunOneRk2Step() {
    const GLuint groupsN = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const GLuint groupsNp1 = static_cast<GLuint>((m_n + 1 + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const float dtOverDx = static_cast<float>(m_dt / m_dx);
    const int next = 1 - m_cur;

    auto consToPrimPass = [&](GLuint consBuf) {
        m_consToPrim.Use();
        m_consToPrim.SetInt("uN", m_n);
        m_consToPrim.SetFloat("uGamma", static_cast<float>(m_gamma));
        m_consToPrim.SetInt("uIters", m_consToPrimIters);
        fw::ComputeShader::BindBuffer(0, consBuf);
        fw::ComputeShader::BindBuffer(4, m_prim);
        m_consToPrim.Dispatch(groupsN);
        fw::ComputeShader::Barrier();
    };
    auto fluxesPass = [&](GLuint consBuf) {
        m_fluxes.Use();
        m_fluxes.SetInt("uN", m_n);
        m_fluxes.SetFloat("uGamma", static_cast<float>(m_gamma));
        fw::ComputeShader::BindBuffer(0, consBuf);
        fw::ComputeShader::BindBuffer(4, m_prim);
        fw::ComputeShader::BindBuffer(5, m_flux);
        m_fluxes.Dispatch(groupsNp1);
        fw::ComputeShader::Barrier();
    };
    auto eulerPass = [&](GLuint consIn, GLuint consOut) {
        m_eulerStep.Use();
        m_eulerStep.SetInt("uN", m_n);
        m_eulerStep.SetFloat("uGamma", static_cast<float>(m_gamma));
        m_eulerStep.SetFloat("uDtOverDx", dtOverDx);
        fw::ComputeShader::BindBuffer(0, consIn);
        fw::ComputeShader::BindBuffer(5, m_flux);
        fw::ComputeShader::BindBuffer(1, consOut);
        m_eulerStep.Dispatch(groupsN);
        fw::ComputeShader::Barrier();
    };

    // Stage A: Euler predictor from U0 (m_cons[m_cur]) to U1 (m_stage1).
    consToPrimPass(m_cons[m_cur]);
    fluxesPass(m_cons[m_cur]);
    eulerPass(m_cons[m_cur], m_stage1);

    // Stage B: another Euler step from U1 (m_stage1) to U2 (m_stage2).
    consToPrimPass(m_stage1);
    fluxesPass(m_stage1);
    eulerPass(m_stage1, m_stage2);

    // Combine: U_next = 0.5*(U0 + U2) -- Heun's method (RK2, SSP).
    m_combine.Use();
    m_combine.SetInt("uN", m_n);
    m_combine.SetFloat("uGamma", static_cast<float>(m_gamma));
    fw::ComputeShader::BindBuffer(0, m_cons[m_cur]);
    fw::ComputeShader::BindBuffer(1, m_stage2);
    fw::ComputeShader::BindBuffer(2, m_cons[next]);
    m_combine.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    m_cur = next;
}

void GrhdSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        RunOneRk2Step();
    }
}

void GrhdSim::ReadBack() {
    m_consCpu.resize(m_n);
    m_primCpu.resize(m_n);
    glGetNamedBufferSubData(m_cons[m_cur], 0, m_n * sizeof(glm::vec4), m_consCpu.data());
    glGetNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), m_primCpu.data());
}

void GrhdSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();

    std::vector<float> rho(m_n), v(m_n), p(m_n);
    double massTotal = 0.0, momentumTotal = 0.0, energyTotal = 0.0;
    for (int i = 0; i < m_n; ++i) {
        rho[i] = m_primCpu[i].x;
        v[i] = m_primCpu[i].y;
        p[i] = m_primCpu[i].z;
        massTotal += m_consCpu[i].x * m_dx;
        momentumTotal += m_consCpu[i].y * m_dx;
        energyTotal += m_consCpu[i].z * m_dx;
    }

    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteScalar("mass_total", massTotal);
    writer.WriteScalar("momentum_total", momentumTotal);
    writer.WriteScalar("energy_total", energyTotal);
}

fw::SimInfo GrhdSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_n;
    info.gridNy = 1;
    info.lx = m_length;
    info.ly = 0.0;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "v", "p"};
    info.diagnostics = {"mass_total", "momentum_total", "energy_total"};
    return info;
}

} // namespace grhd
