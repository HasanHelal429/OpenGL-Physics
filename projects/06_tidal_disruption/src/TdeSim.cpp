#include "TdeSim.hpp"
#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace tde {

namespace {
GLuint MakeVec4Buffer(GLsizei count) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferStorage(buf, count * static_cast<GLsizeiptr>(sizeof(glm::vec4)), nullptr,
                         GL_DYNAMIC_STORAGE_BIT);
    return buf;
}
GLuint MakeFloatBuffer(GLsizei count) {
    GLuint buf = 0;
    glCreateBuffers(1, &buf);
    glNamedBufferStorage(buf, count * static_cast<GLsizeiptr>(sizeof(float)), nullptr,
                         GL_DYNAMIC_STORAGE_BIT);
    return buf;
}
} // namespace

TdeSim::~TdeSim() {
    GLuint bufs[5] = {m_posMass, m_vel, m_acc, m_rhoPress, m_h};
    glDeleteBuffers(5, bufs);
}

void TdeSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "Tidal Disruption -- SPH star");

    const std::string icFile = deck.GetString("star.ic_file", "");
    if (icFile.empty()) throw std::runtime_error("deck missing [star] ic_file");

    std::ifstream f(icFile, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open ic_file: " + icFile);
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    constexpr int kFloatsPerParticle = 7; // x,y,z,mass,vx,vy,vz -- tools/make_star_ic.py
    m_n = static_cast<int>(bytes / (kFloatsPerParticle * static_cast<std::streamsize>(sizeof(float))));
    if (m_n <= 0) throw std::runtime_error("ic_file has no particles: " + icFile);

    std::vector<float> raw(static_cast<size_t>(m_n) * kFloatsPerParticle);
    f.read(reinterpret_cast<char*>(raw.data()), bytes);

    m_initial.resize(m_n);
    for (int i = 0; i < m_n; ++i) {
        const float* p = &raw[static_cast<size_t>(i) * kFloatsPerParticle];
        m_initial[i].pos = glm::vec3(p[0], p[1], p[2]);
        m_initial[i].mass = p[3];
        m_initial[i].vel = glm::vec3(p[4], p[5], p[6]);
    }

    m_G = deck.GetDouble("gravity.G", 1.0);
    m_softening = deck.GetDouble("gravity.softening", 0.05);

    m_hInit = deck.GetDouble("sph.h_init", 0.1);
    m_eta = deck.GetDouble("sph.eta", 1.2);
    m_hIters = deck.GetInt("sph.h_iters", 3);
    m_hMin = deck.GetDouble("sph.h_min_factor", 0.2) * m_hInit;
    m_hMax = deck.GetDouble("sph.h_max_factor", 8.0) * m_hInit;
    m_K = deck.GetDouble("sph.K", 1.0);
    m_gamma = deck.GetDouble("sph.gamma", 5.0 / 3.0);
    m_viscAlpha = deck.GetDouble("sph.visc_alpha", 1.0);
    m_viscBeta = deck.GetDouble("sph.visc_beta", 2.0);
    m_damping = deck.GetDouble("time.damping", 0.0);

    m_dt = deck.GetDouble("time.dt", 1e-3);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 1);

    m_diagNames = {"kinetic", "thermal", "potential", "energy", "virial_2T_over_W",
                   "com_x", "com_y", "com_z", "com_speed"};

    m_density = fw::ComputeShader::FromSource(kernels::Density());
    m_forces = fw::ComputeShader::FromSource(kernels::Forces());
    m_kick = fw::ComputeShader::FromSource(kernels::Kick());
    m_drift = fw::ComputeShader::FromSource(kernels::Drift());

    CreateBuffers();
    UploadInitial();
    ComputeDensityAndForces(); // acc(t=0), needed before Step's first half-kick

    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = 4.0f;
    m_camera.pitch = 20.0f;
}

void TdeSim::CreateBuffers() {
    m_posMass = MakeVec4Buffer(m_n);
    m_vel = MakeVec4Buffer(m_n);
    m_acc = MakeVec4Buffer(m_n);
    m_rhoPress = MakeVec4Buffer(m_n);
    m_h = MakeFloatBuffer(m_n);
}

void TdeSim::UploadInitial() {
    std::vector<glm::vec4> posMass(m_n), vel(m_n);
    for (int i = 0; i < m_n; ++i) {
        posMass[i] = glm::vec4(m_initial[i].pos, m_initial[i].mass);
        vel[i] = glm::vec4(m_initial[i].vel, 0.0f);
    }
    glNamedBufferSubData(m_posMass, 0, m_n * sizeof(glm::vec4), posMass.data());
    glNamedBufferSubData(m_vel, 0, m_n * sizeof(glm::vec4), vel.data());
    const std::vector<glm::vec4> zero(m_n, glm::vec4(0.0f));
    glNamedBufferSubData(m_acc, 0, m_n * sizeof(glm::vec4), zero.data());
    glNamedBufferSubData(m_rhoPress, 0, m_n * sizeof(glm::vec4), zero.data());
    const std::vector<float> hInit(m_n, static_cast<float>(m_hInit));
    glNamedBufferSubData(m_h, 0, m_n * sizeof(float), hInit.data());
}

void TdeSim::Reset() {
    UploadInitial();
    ComputeDensityAndForces();
}

void TdeSim::ComputeDensityAndForces() {
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);

    m_density.Use();
    m_density.SetInt("uN", m_n);
    m_density.SetFloat("uK", static_cast<float>(m_K));
    m_density.SetFloat("uGamma", static_cast<float>(m_gamma));
    m_density.SetFloat("uEta", static_cast<float>(m_eta));
    m_density.SetInt("uHIters", m_hIters);
    m_density.SetFloat("uHMin", static_cast<float>(m_hMin));
    m_density.SetFloat("uHMax", static_cast<float>(m_hMax));
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(3, m_rhoPress);
    fw::ComputeShader::BindBuffer(4, m_h);
    m_density.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    m_forces.Use();
    m_forces.SetInt("uN", m_n);
    m_forces.SetFloat("uG", static_cast<float>(m_G));
    m_forces.SetFloat("uSoftening2", static_cast<float>(m_softening * m_softening));
    m_forces.SetFloat("uViscAlpha", static_cast<float>(m_viscAlpha));
    m_forces.SetFloat("uViscBeta", static_cast<float>(m_viscBeta));
    fw::ComputeShader::BindBuffer(0, m_posMass);
    fw::ComputeShader::BindBuffer(1, m_vel);
    fw::ComputeShader::BindBuffer(2, m_acc);
    fw::ComputeShader::BindBuffer(3, m_rhoPress);
    fw::ComputeShader::BindBuffer(4, m_h);
    m_forces.Dispatch(groups);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdeSim::Step(int substeps) {
    const GLuint groups = static_cast<GLuint>((m_n + kernels::kWorkgroupSize - 1) / kernels::kWorkgroupSize);
    const float halfDt = static_cast<float>(0.5 * m_dt);
    const float dampingFactor = static_cast<float>(std::exp(-m_damping * halfDt));

    auto kick = [&]() {
        m_kick.Use();
        m_kick.SetInt("uN", m_n);
        m_kick.SetFloat("uHalfDt", halfDt);
        m_kick.SetFloat("uDampingFactor", dampingFactor);
        fw::ComputeShader::BindBuffer(1, m_vel);
        fw::ComputeShader::BindBuffer(2, m_acc);
        m_kick.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };
    auto drift = [&]() {
        m_drift.Use();
        m_drift.SetInt("uN", m_n);
        m_drift.SetFloat("uDt", static_cast<float>(m_dt));
        fw::ComputeShader::BindBuffer(0, m_posMass);
        fw::ComputeShader::BindBuffer(1, m_vel);
        m_drift.Dispatch(groups);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
    };

    for (int s = 0; s < substeps; ++s) {
        kick();                    // v(t+dt/2) = v(t) + (dt/2) a(t)
        drift();                   // x(t+dt) = x(t) + dt v(t+dt/2)
        ComputeDensityAndForces(); // a(t+dt)
        kick();                    // v(t+dt) = v(t+dt/2) + (dt/2) a(t+dt)
    }
}

void TdeSim::ReadBack() {
    m_posMassCpu.resize(m_n);
    m_velCpu.resize(m_n);
    m_rhoPressCpu.resize(m_n);
    m_hCpu.resize(m_n);
    glGetNamedBufferSubData(m_posMass, 0, m_n * sizeof(glm::vec4), m_posMassCpu.data());
    glGetNamedBufferSubData(m_vel, 0, m_n * sizeof(glm::vec4), m_velCpu.data());
    glGetNamedBufferSubData(m_rhoPress, 0, m_n * sizeof(glm::vec4), m_rhoPressCpu.data());
    glGetNamedBufferSubData(m_h, 0, m_n * sizeof(float), m_hCpu.data());
}

void TdeSim::Snapshot(fw::OutputWriter& writer) {
    ReadBack();

    double kinetic = 0.0, thermal = 0.0, potential = 0.0, totalMass = 0.0;
    glm::dvec3 com(0.0), comP(0.0); // comP = mass-weighted momentum, for COM velocity
    for (int i = 0; i < m_n; ++i) {
        const double m = m_posMassCpu[i].w;
        const glm::dvec3 v(m_velCpu[i]);
        const double rho = m_rhoPressCpu[i].x;
        const double P = m_rhoPressCpu[i].y;
        totalMass += m;
        com += m * glm::dvec3(m_posMassCpu[i]);
        comP += m * v;
        kinetic += 0.5 * m * glm::dot(v, v);
        if (rho > 0.0) thermal += m * P / ((m_gamma - 1.0) * rho);
    }
    for (int i = 0; i < m_n; ++i) {
        const glm::dvec3 ri(m_posMassCpu[i]);
        for (int j = i + 1; j < m_n; ++j) {
            const glm::dvec3 rij = ri - glm::dvec3(m_posMassCpu[j]);
            const double r = std::sqrt(glm::dot(rij, rij) + m_softening * m_softening);
            potential -= m_G * m_posMassCpu[i].w * m_posMassCpu[j].w / r;
        }
    }
    const glm::dvec3 comVel = comP / totalMass;
    com /= totalMass;
    const double energy = kinetic + thermal + potential;
    const double virial = (potential != 0.0) ? 2.0 * kinetic / std::abs(potential) : 0.0;

    writer.WriteField("pos_mass", m_posMassCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("vel", m_velCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("rho_press", m_rhoPressCpu.data(), fw::NpyDtype::F4, m_n, 4);
    writer.WriteField("h", m_hCpu.data(), fw::NpyDtype::F4, m_n, 1);
    writer.WriteScalar("kinetic", kinetic);
    writer.WriteScalar("thermal", thermal);
    writer.WriteScalar("potential", potential);
    writer.WriteScalar("energy", energy);
    writer.WriteScalar("virial_2T_over_W", virial);
    writer.WriteScalar("com_x", com.x);
    writer.WriteScalar("com_y", com.y);
    writer.WriteScalar("com_z", com.z);
    writer.WriteScalar("com_speed", glm::length(comVel));
}

fw::SimInfo TdeSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"pos_mass", "vel", "rho_press", "h"};
    info.diagnostics = m_diagNames;
    return info;
}

void TdeSim::Render(int fbWidth, int fbHeight) {
    ReadBack();

    double rhoMax = 1e-30;
    for (int i = 0; i < m_n; ++i) rhoMax = std::max(rhoMax, static_cast<double>(m_rhoPressCpu[i].x));

    std::vector<fw::ParticleInstance> particles(m_n);
    for (int i = 0; i < m_n; ++i) {
        const float t = static_cast<float>(std::pow(m_rhoPressCpu[i].x / rhoMax, 0.35)); // gamma-lift for dim outskirts
        particles[i].position = glm::vec3(m_posMassCpu[i]);
        particles[i].color = glm::vec4(0.3f + 0.7f * t, 0.5f * (1.0f - t) + 0.2f, 1.0f - 0.6f * t, 1.0f);
        particles[i].size = 0.02f;
    }
    m_particles.SetParticles(particles);

    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const float aspect = static_cast<float>(fbWidth) / static_cast<float>(std::max(fbHeight, 1));
    m_particles.Draw(m_camera.ViewMatrix(), m_camera.ProjectionMatrix(aspect), static_cast<float>(fbHeight));
}

void TdeSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) m_camera.Orbit(static_cast<float>(in.dx), static_cast<float>(in.dy));
    if (in.scrollDelta != 0.0) m_camera.Zoom(static_cast<float>(in.scrollDelta));
}

} // namespace tde
