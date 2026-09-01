#include "MDSim.hpp"
#include "Scenarios.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace md {

void MDSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "Molecular Dynamics -- Lennard-Jones fluid");

    const int targetN = deck.GetInt("system.target_n", 500);
    const double density = deck.GetDouble("system.density", 0.70);
    const double temperature = deck.GetDouble("system.temperature", 1.00);
    const unsigned seed = static_cast<unsigned>(deck.GetInt("system.seed", 42));

    const ScenarioResult ic = BuildFccLattice(targetN, density, temperature, seed);
    m_initialPos = ic.pos;
    m_initialVel = ic.vel;
    m_initialL = ic.boxLength;

    m_params.cutoff = deck.GetDouble("potential.cutoff", 2.5);
    m_params.skin = deck.GetDouble("potential.skin", 0.3);
    m_params.neighborRebuildEvery = deck.GetInt("time.neighbor_rebuild_every", 5);

    const std::string thermoType = deck.GetString("thermostat.type", "berendsen");
    if (thermoType == "none") m_params.thermostat = Thermostat::None;
    else if (thermoType == "velocity_rescale") m_params.thermostat = Thermostat::VelocityRescale;
    else if (thermoType == "nose_hoover") m_params.thermostat = Thermostat::NoseHoover;
    else m_params.thermostat = Thermostat::Berendsen;
    m_params.targetT = deck.GetDouble("thermostat.target_t", temperature);
    // One deck key ("thermostat.tau") drives whichever of the two relaxation-
    // time-shaped thermostats is actually selected -- they're never both
    // active at once, so there's no ambiguity in sharing it.
    m_params.berendsenTau = deck.GetDouble("thermostat.tau", 1.0);
    m_params.noseHooverTau = deck.GetDouble("thermostat.tau", 1.0);
    m_params.rescaleEvery = deck.GetInt("thermostat.rescale_every", 20);

    if (deck.GetBool("barostat.enabled", false)) {
        m_params.barostat = Barostat::Berendsen;
        m_params.targetP = deck.GetDouble("barostat.target_p", 0.0);
        m_params.berendsenTauP = deck.GetDouble("barostat.tau_p", 1.0);
        m_params.compressibility = deck.GetDouble("barostat.compressibility", 1.0);
    } else {
        m_params.barostat = Barostat::None;
    }

    m_dt = deck.GetDouble("time.dt", 0.002);
    m_params.dt = m_dt;
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 20);

    m_diagNames = {"kinetic",  "potential", "potential_tail", "total_energy", "temperature",
                   "pressure", "pressure_tail", "box_length", "density", "nh_xi", "nh_invariant"};

    m_system.SetParticles(m_initialPos, m_initialVel, m_initialL);
    m_system.PrimeForces(m_params);

    // Deferred construction -- see MDSim.hpp's member comment.
    m_particles = std::make_unique<fw::ParticleCloud>();
    m_box = std::make_unique<WireBox>();
    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);

    const float half = static_cast<float>(m_initialL) * 0.5f;
    m_camera.target = glm::vec3(half);
    m_camera.distance = static_cast<float>(m_initialL) * 1.5f;
    m_camera.pitch = 20.0f;
}

void MDSim::Reset() {
    m_system.SetParticles(m_initialPos, m_initialVel, m_initialL);
    m_system.PrimeForces(m_params);
}

void MDSim::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) m_system.Step(m_params);
}

void MDSim::Snapshot(fw::OutputWriter& writer) {
    const int n = static_cast<int>(m_system.Count());
    const std::vector<glm::dvec3>& pos = m_system.Positions();
    const std::vector<glm::dvec3>& vel = m_system.Velocities();
    std::vector<glm::vec3> posF(static_cast<size_t>(n)), velF(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        posF[static_cast<size_t>(i)] = glm::vec3(pos[static_cast<size_t>(i)]);
        velF[static_cast<size_t>(i)] = glm::vec3(vel[static_cast<size_t>(i)]);
    }
    writer.WriteField("pos", posF.data(), fw::NpyDtype::F4, n, 3);
    writer.WriteField("vel", velF.data(), fw::NpyDtype::F4, n, 3);

    const double ke = m_system.KineticEnergy();
    const double pe = m_system.PotentialEnergy();
    const double p = m_system.Pressure();
    const double L = m_system.BoxLength();
    writer.WriteScalar("kinetic", ke);
    writer.WriteScalar("potential", pe);
    writer.WriteScalar("potential_tail", pe + m_system.TailEnergyCorrection(m_params.cutoff));
    writer.WriteScalar("total_energy", ke + pe);
    writer.WriteScalar("temperature", m_system.Temperature());
    writer.WriteScalar("pressure", p);
    writer.WriteScalar("pressure_tail", p + m_system.TailPressureCorrection(m_params.cutoff));
    writer.WriteScalar("box_length", L);
    writer.WriteScalar("density", n / (L * L * L));
    writer.WriteScalar("nh_xi", m_system.ThermostatXi());
    writer.WriteScalar("nh_invariant", m_system.NoseHooverInvariant(m_params));
}

fw::SimInfo MDSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"pos", "vel"};
    info.diagnostics = m_diagNames;
    return info;
}

void MDSim::Render(int fbWidth, int fbHeight) {
    const int n = static_cast<int>(m_system.Count());
    const std::vector<glm::dvec3>& pos = m_system.Positions();
    const std::vector<glm::dvec3>& vel = m_system.Velocities();

    double vMax = 1e-9;
    for (const glm::dvec3& v : vel) vMax = std::max(vMax, glm::dot(v, v));
    vMax = std::sqrt(vMax);

    std::vector<fw::ParticleInstance> particles;
    particles.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        const float t = static_cast<float>(glm::length(vel[static_cast<size_t>(i)]) / vMax);
        fw::ParticleInstance p;
        p.position = glm::vec3(pos[static_cast<size_t>(i)]);
        p.color = glm::vec4(0.3f + 0.7f * t, 0.45f, 1.0f - 0.6f * t, 1.0f); // blue (slow) -> orange (fast)
        p.size = 0.06f;
        particles.push_back(p);
    }
    m_particles->SetParticles(particles);

    glViewport(0, 0, fbWidth, fbHeight);
    glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    const float aspect = static_cast<float>(fbWidth) / static_cast<float>(std::max(fbHeight, 1));
    const glm::mat4 view = m_camera.ViewMatrix();
    const glm::mat4 proj = m_camera.ProjectionMatrix(aspect);

    m_box->Draw(view, proj, m_system.BoxLength(), glm::vec4(0.5f, 0.65f, 0.9f, 0.5f));
    m_particles->Draw(view, proj, static_cast<float>(fbHeight));

    m_text->SetViewport(fbWidth, fbHeight);
    const glm::vec4 white(1.0f, 1.0f, 1.0f, 0.9f);
    const glm::vec4 dim(0.75f, 0.85f, 1.0f, 0.85f);
    char line[256];
    float y = 22.0f;
    std::snprintf(line, sizeof(line), "T=%.3f  P=%.3f  E=%.4f", m_system.Temperature(), m_system.Pressure(),
                  m_system.TotalEnergy());
    m_text->Draw(m_font, line, glm::vec2(12.0f, y), white);
    y += 20.0f;
    const double L = m_system.BoxLength();
    std::snprintf(line, sizeof(line), "N=%d  rho=%.3f  L=%.3f", n, n / (L * L * L), L);
    m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    if (m_params.thermostat == Thermostat::NoseHoover) {
        y += 20.0f;
        std::snprintf(line, sizeof(line), "Nose-Hoover xi=%.4f  invariant=%.4f", m_system.ThermostatXi(),
                      m_system.NoseHooverInvariant(m_params));
        m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    }
}

void MDSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) m_camera.Orbit(static_cast<float>(in.dx), static_cast<float>(in.dy));
    if (in.scrollDelta != 0.0) m_camera.Zoom(static_cast<float>(in.scrollDelta));
}

} // namespace md
