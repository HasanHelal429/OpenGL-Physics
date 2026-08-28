#include "SchwarzschildSim.hpp"
#include "kernels_schwarzschild.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace grhd {

SchwarzschildSim::~SchwarzschildSim() {
    GLuint bufs[] = {m_cons[0], m_cons[1], m_stage1, m_stage2, m_prim, m_flux};
    glDeleteBuffers(6, bufs);
}

namespace {
glm::vec4 PrimToConsSch(double rho, double v, double P, double r, double alpha, double gamma) {
    const double W = 1.0 / std::sqrt(1.0 - v * v);
    const double h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    const double r2 = r * r;
    const double D = r2 * rho * W / alpha;
    const double S = r2 * rho * h * W * W * v;
    const double tau = r2 * (rho * h * W * W - P) - D;
    return glm::vec4(static_cast<float>(D), static_cast<float>(S), static_cast<float>(tau), 0.0f);
}
} // namespace

void SchwarzschildSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    m_n = deck.GetInt("grid.n", 400);
    m_rMin = deck.GetDouble("grid.r_min", 3.0);
    m_rMax = deck.GetDouble("grid.r_max", 40.0);
    m_dr = (m_rMax - m_rMin) / static_cast<double>(m_n);

    m_M = deck.GetDouble("physics.M", 1.0);
    m_gamma = deck.GetDouble("physics.gamma", 5.0 / 3.0);

    m_cfl = deck.GetDouble("time.cfl", 0.2);
    m_dt = m_cfl * m_dr;
    m_consToPrimIters = deck.GetInt("solver.cons_to_prim_iters", 30);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_icFile = deck.GetString("bondi.ic_file", "");
    if (m_icFile.empty()) {
        throw std::runtime_error("SchwarzschildSim::Configure: deck needs bondi.ic_file "
                                  "(see tools/make_bondi_ic.py)");
    }

    CreateBuffers();
    UploadInitial();

    m_consToPrim = fw::ComputeShader::FromSource(kernels_sch::ConsToPrim());
    m_fluxes = fw::ComputeShader::FromSource(kernels_sch::Fluxes());
    m_eulerStep = fw::ComputeShader::FromSource(kernels_sch::EulerStep());
    m_combine = fw::ComputeShader::FromSource(kernels_sch::Combine());
}

void SchwarzschildSim::CreateBuffers() {
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

void SchwarzschildSim::UploadInitial() {
    std::ifstream f(m_icFile, std::ios::binary);
    if (!f) throw std::runtime_error("SchwarzschildSim: cannot open bondi.ic_file '" + m_icFile + "'");
    // grid.n cell records + one extra outer-ghost record (see
    // tools/make_bondi_ic.py) -- the fixed exterior (Dirichlet) state at
    // r_max, needed because the outer edge is subsonic (one characteristic
    // there comes from outside the domain, so plain zero-gradient outflow
    // is ill-posed and lets the solution drift, confirmed the hard way).
    std::vector<float> raw(static_cast<size_t>(m_n + 1) * 3);
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!f) throw std::runtime_error("SchwarzschildSim: '" + m_icFile + "' is smaller than (grid.n+1)*3 floats");
    m_ghostHi = glm::vec3(raw[3 * m_n + 0], raw[3 * m_n + 1], raw[3 * m_n + 2]);

    m_initialCons.resize(m_n);
    std::vector<glm::vec4> initialPrim(m_n);
    for (int i = 0; i < m_n; ++i) {
        const double rho = raw[3 * i + 0];
        const double v = raw[3 * i + 1];
        const double P = raw[3 * i + 2];
        const double r = m_rMin + (static_cast<double>(i) + 0.5) * m_dr;
        const double alpha = std::sqrt(1.0 - 2.0 * m_M / r);
        m_initialCons[i] = PrimToConsSch(rho, v, P, r, alpha, m_gamma);
        initialPrim[i] = glm::vec4(static_cast<float>(rho), static_cast<float>(v), static_cast<float>(P), 0.0f);
    }
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), initialPrim.data());
    m_cur = 0;
}

void SchwarzschildSim::Reset() {
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    m_cur = 0;
}

void SchwarzschildSim::RunOneRk2Step() {
    const GLuint groupsN = static_cast<GLuint>((m_n + kernels_sch::kWorkgroupSize - 1) / kernels_sch::kWorkgroupSize);
    const GLuint groupsNp1 = static_cast<GLuint>((m_n + 1 + kernels_sch::kWorkgroupSize - 1) / kernels_sch::kWorkgroupSize);
    const int next = 1 - m_cur;

    auto setCommon = [&](fw::ComputeShader& sh) {
        sh.Use();
        sh.SetInt("uN", m_n);
        sh.SetFloat("uGamma", static_cast<float>(m_gamma));
        sh.SetFloat("uM", static_cast<float>(m_M));
        sh.SetFloat("uRmin", static_cast<float>(m_rMin));
        sh.SetFloat("uDr", static_cast<float>(m_dr));
    };

    auto consToPrimPass = [&](GLuint consBuf) {
        setCommon(m_consToPrim);
        m_consToPrim.SetInt("uIters", m_consToPrimIters);
        fw::ComputeShader::BindBuffer(0, consBuf);
        fw::ComputeShader::BindBuffer(4, m_prim);
        m_consToPrim.Dispatch(groupsN);
        fw::ComputeShader::Barrier();
    };
    auto fluxesPass = [&]() {
        setCommon(m_fluxes);
        m_fluxes.SetVec4("uGhostHi", glm::vec4(m_ghostHi, 0.0f));
        fw::ComputeShader::BindBuffer(4, m_prim);
        fw::ComputeShader::BindBuffer(5, m_flux);
        m_fluxes.Dispatch(groupsNp1);
        fw::ComputeShader::Barrier();
    };
    auto eulerPass = [&](GLuint consIn, GLuint consOut) {
        setCommon(m_eulerStep);
        m_eulerStep.SetFloat("uDt", static_cast<float>(m_dt));
        fw::ComputeShader::BindBuffer(0, consIn);
        fw::ComputeShader::BindBuffer(4, m_prim);
        fw::ComputeShader::BindBuffer(5, m_flux);
        fw::ComputeShader::BindBuffer(1, consOut);
        m_eulerStep.Dispatch(groupsN);
        fw::ComputeShader::Barrier();
    };

    consToPrimPass(m_cons[m_cur]);
    fluxesPass();
    eulerPass(m_cons[m_cur], m_stage1);

    consToPrimPass(m_stage1);
    fluxesPass();
    eulerPass(m_stage1, m_stage2);

    setCommon(m_combine);
    fw::ComputeShader::BindBuffer(0, m_cons[m_cur]);
    fw::ComputeShader::BindBuffer(1, m_stage2);
    fw::ComputeShader::BindBuffer(2, m_cons[next]);
    m_combine.Dispatch(groupsN);
    fw::ComputeShader::Barrier();

    m_cur = next;
}

void SchwarzschildSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        RunOneRk2Step();
    }
}

void SchwarzschildSim::ReadBack() {
    m_consCpu.resize(m_n);
    m_primCpu.resize(m_n);
    glGetNamedBufferSubData(m_cons[m_cur], 0, m_n * sizeof(glm::vec4), m_consCpu.data());
    glGetNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), m_primCpu.data());
}

void SchwarzschildSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();

    std::vector<float> rho(m_n), v(m_n), p(m_n), mdot(m_n);
    for (int i = 0; i < m_n; ++i) {
        rho[i] = m_primCpu[i].x;
        v[i] = m_primCpu[i].y;
        p[i] = m_primCpu[i].z;
        // f*v*D at the cell center: exactly conserved (=const in r) in the
        // true steady state -- a strong, cheap per-frame diagnostic (see
        // README's Validation section).
        const double r = m_rMin + (static_cast<double>(i) + 0.5) * m_dr;
        const double f = 1.0 - 2.0 * m_M / r;
        mdot[i] = static_cast<float>(f * v[i] * m_consCpu[i].x); // f*v*D
    }

    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("v", v.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("mdot", mdot.data(), fw::NpyDtype::F4, 1, m_n);

    double mdotMin = mdot[0], mdotMax = mdot[0];
    for (int i = 1; i < m_n; ++i) {
        mdotMin = std::min<double>(mdotMin, mdot[i]);
        mdotMax = std::max<double>(mdotMax, mdot[i]);
    }
    writer.WriteScalar("mdot_min", mdotMin);
    writer.WriteScalar("mdot_max", mdotMax);
    writer.WriteScalar("mdot_spread", (mdotMax - mdotMin) / std::abs(0.5 * (mdotMax + mdotMin)));
}

fw::SimInfo SchwarzschildSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_n;
    info.gridNy = 1;
    info.lx = m_rMax - m_rMin;
    info.ly = 0.0;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "v", "p", "mdot"};
    info.diagnostics = {"mdot_min", "mdot_max", "mdot_spread"};
    return info;
}

} // namespace grhd
