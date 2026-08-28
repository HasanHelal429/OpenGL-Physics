#include "KerrEquatorialSim.hpp"
#include "kernels_kerr.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <fstream>
#include <stdexcept>

namespace grhd {

KerrEquatorialSim::~KerrEquatorialSim() {
    GLuint bufs[] = {m_cons[0], m_cons[1], m_stage1, m_stage2, m_prim, m_flux};
    glDeleteBuffers(6, bufs);
}

namespace {
// CPU mirror of kernels_kerr.hpp's kinematics()+PrimToCons, double
// precision, used only to build the initial conserved-variable buffer
// from the IC file's primitives (see tools/kerr_equatorial_ref.py for the
// derivation and its cross-checks -- this is a direct transcription of
// that already-validated reference, not a fresh derivation).
struct Metric { double gtt, gtphi, grr, gphiphi; };
Metric MetricAt(double r, double M, double a) {
    const double Delta = r * r - 2.0 * M * r + a * a;
    Metric m;
    m.gtt = -(1.0 - 2.0 * M / r);
    m.gtphi = -2.0 * M * a / r;
    m.grr = r * r / Delta;
    m.gphiphi = r * r + a * a + 2.0 * M * a * a / r;
    return m;
}

glm::vec4 PrimToConsKerr(double rho, double vr, double vphi, double P, double r, double M, double a, double gamma) {
    const Metric m = MetricAt(r, M, a);
    const double Delta = r * r - 2.0 * M * r + a * a;
    const double A = (r * r + a * a) * (r * r + a * a) - a * a * Delta;
    const double alpha = std::sqrt(r * r * Delta / A);

    const double W = 1.0 / std::sqrt(1.0 - vr * vr - vphi * vphi);
    const double h = 1.0 + gamma * P / ((gamma - 1.0) * rho);
    const double ut = W / alpha;
    const double urCov = W * std::sqrt(m.grr) * vr;
    const double uPhiCov = W * std::sqrt(m.gphiphi) * vphi;
    const double E = W * (alpha - m.gtphi * vphi / std::sqrt(m.gphiphi));

    const double r2 = r * r;
    const double D = r2 * rho * ut;
    const double Sr = r2 * rho * h * ut * urCov;
    const double L = r2 * rho * h * ut * uPhiCov;
    const double tau = r2 * rho * h * ut * E - r2 * P - D;
    return glm::vec4(static_cast<float>(D), static_cast<float>(Sr), static_cast<float>(L), static_cast<float>(tau));
}
} // namespace

void KerrEquatorialSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", m_title);
    m_n = deck.GetInt("grid.n", 400);
    m_rMin = deck.GetDouble("grid.r_min", 6.0);
    m_rMax = deck.GetDouble("grid.r_max", 20.0);
    m_dr = (m_rMax - m_rMin) / static_cast<double>(m_n);

    m_M = deck.GetDouble("physics.M", 1.0);
    m_a = deck.GetDouble("physics.a", 0.0);
    m_gamma = deck.GetDouble("physics.gamma", 4.0 / 3.0);

    m_cfl = deck.GetDouble("time.cfl", 0.2);
    m_dt = m_cfl * m_dr;
    m_consToPrimIters = deck.GetInt("solver.cons_to_prim_iters", 40);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_icFile = deck.GetString("kerr.ic_file", "");
    if (m_icFile.empty()) {
        throw std::runtime_error("KerrEquatorialSim::Configure: deck needs kerr.ic_file "
                                  "(see tools/make_kerr_orbit_ic.py)");
    }

    CreateBuffers();
    UploadInitial();

    m_consToPrim = fw::ComputeShader::FromSource(kernels_kerr::ConsToPrim());
    m_fluxes = fw::ComputeShader::FromSource(kernels_kerr::Fluxes());
    m_eulerStep = fw::ComputeShader::FromSource(kernels_kerr::EulerStep());
    m_combine = fw::ComputeShader::FromSource(kernels_kerr::Combine());
}

void KerrEquatorialSim::CreateBuffers() {
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

void KerrEquatorialSim::UploadInitial() {
    std::ifstream f(m_icFile, std::ios::binary);
    if (!f) throw std::runtime_error("KerrEquatorialSim: cannot open kerr.ic_file '" + m_icFile + "'");
    std::vector<float> raw(static_cast<size_t>(m_n) * 4); // rho, v_r, v_phi, P per cell
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size() * sizeof(float)));
    if (!f) throw std::runtime_error("KerrEquatorialSim: '" + m_icFile + "' is smaller than grid.n*4 floats");

    m_initialCons.resize(m_n);
    std::vector<glm::vec4> initialPrim(m_n);
    for (int i = 0; i < m_n; ++i) {
        const double rho = raw[4 * i + 0];
        const double vr = raw[4 * i + 1];
        const double vphi = raw[4 * i + 2];
        const double P = raw[4 * i + 3];
        const double r = m_rMin + (static_cast<double>(i) + 0.5) * m_dr;
        m_initialCons[i] = PrimToConsKerr(rho, vr, vphi, P, r, m_M, m_a, m_gamma);
        initialPrim[i] = glm::vec4(static_cast<float>(rho), static_cast<float>(vr),
                                    static_cast<float>(vphi), static_cast<float>(P));
    }
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), initialPrim.data());
    m_cur = 0;
}

void KerrEquatorialSim::Reset() {
    glNamedBufferSubData(m_cons[0], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    glNamedBufferSubData(m_cons[1], 0, m_n * sizeof(glm::vec4), m_initialCons.data());
    m_cur = 0;
}

void KerrEquatorialSim::RunOneRk2Step() {
    const GLuint groupsN = static_cast<GLuint>((m_n + kernels_kerr::kWorkgroupSize - 1) / kernels_kerr::kWorkgroupSize);
    const GLuint groupsNp1 = static_cast<GLuint>((m_n + 1 + kernels_kerr::kWorkgroupSize - 1) / kernels_kerr::kWorkgroupSize);
    const int next = 1 - m_cur;

    auto setCommon = [&](fw::ComputeShader& sh) {
        sh.Use();
        sh.SetInt("uN", m_n);
        sh.SetFloat("uGamma", static_cast<float>(m_gamma));
        sh.SetFloat("uM", static_cast<float>(m_M));
        sh.SetFloat("uA", static_cast<float>(m_a));
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

void KerrEquatorialSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        RunOneRk2Step();
    }
}

void KerrEquatorialSim::ReadBack() {
    m_consCpu.resize(m_n);
    m_primCpu.resize(m_n);
    glGetNamedBufferSubData(m_cons[m_cur], 0, m_n * sizeof(glm::vec4), m_consCpu.data());
    glGetNamedBufferSubData(m_prim, 0, m_n * sizeof(glm::vec4), m_primCpu.data());
}

void KerrEquatorialSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();

    std::vector<float> rho(m_n), vr(m_n), vphi(m_n), p(m_n);
    for (int i = 0; i < m_n; ++i) {
        rho[i] = m_primCpu[i].x;
        vr[i] = m_primCpu[i].y;
        vphi[i] = m_primCpu[i].z;
        p[i] = m_primCpu[i].w;
    }

    writer.WriteField("rho", rho.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("v_r", vr.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("v_phi", vphi.data(), fw::NpyDtype::F4, 1, m_n);
    writer.WriteField("p", p.data(), fw::NpyDtype::F4, 1, m_n);

    double maxAbsVr = 0.0;
    for (int i = 0; i < m_n; ++i) maxAbsVr = std::max<double>(maxAbsVr, std::abs(vr[i]));
    writer.WriteScalar("max_abs_vr", maxAbsVr);
}

fw::SimInfo KerrEquatorialSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_n;
    info.gridNy = 1;
    info.lx = m_rMax - m_rMin;
    info.ly = 0.0;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"rho", "v_r", "v_phi", "p"};
    info.diagnostics = {"max_abs_vr"};
    return info;
}

} // namespace grhd
