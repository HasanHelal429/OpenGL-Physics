#include "MDSim.hpp"
#include "Scenarios.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <glad/glad.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

namespace md {

void MDSim::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "Molecular Dynamics -- Lennard-Jones fluid");

    const int targetN = deck.GetInt("system.target_n", 500);
    const double density = deck.GetDouble("system.density", 0.70);
    const double temperature = deck.GetDouble("system.temperature", 1.00);
    const unsigned seed = static_cast<unsigned>(deck.GetInt("system.seed", 42));

    const double fracB = deck.GetDouble("system.species_b_fraction", 0.0);
    m_isMixture = fracB > 0.0;
    ScenarioResult ic;
    if (m_isMixture) {
        ic = BuildFccLatticeBinary(targetN, density, temperature, 1.0 - fracB, seed);
        // Kob-Andersen (Kob & Andersen, Phys. Rev. E 51, 4626, 1995)
        // defaults -- lengths/energies quoted in sigma_AA/epsilon_AA, the
        // canonical 80:20 glass-forming mixture at these values.
        const double sigmaBB = deck.GetDouble("mixture.sigma_bb", 0.88);
        const double epsBB = deck.GetDouble("mixture.eps_bb", 0.50);
        const double sigmaAB = deck.GetDouble("mixture.sigma_ab", 0.80);
        const double epsAB = deck.GetDouble("mixture.eps_ab", 1.50);
        m_system.SetSpeciesLJParams(sigmaBB, epsBB, sigmaAB, epsAB);
    } else {
        ic = BuildFccLattice(targetN, density, temperature, seed);
    }
    m_initialPos = ic.pos;
    m_initialVel = ic.vel;
    m_initialSpecies = ic.species;
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

    m_ramp.clear();
    if (deck.GetBool("ramp.enabled", false)) {
        for (const fw::Deck& seg : deck.GetTables("ramp.segments")) {
            RampSegment r;
            r.t0 = seg.GetDouble("t0", 0.0);
            r.t1 = seg.GetDouble("t1", 0.0);
            r.tStart = seg.GetDouble("t_start", m_params.targetT);
            r.tEnd = seg.GetDouble("t_end", m_params.targetT);
            m_ramp.push_back(r);
        }
    }
    m_simTime = 0.0;

    // FCC nearest-neighbor distance at this density, independent of
    // cellsPerAxis: density = N/V = 4/a_lattice^3 for any FCC conventional
    // cell count, so a_lattice = cbrt(4/density) always; nn distance in FCC
    // is a_lattice/sqrt(2) (face-diagonal half-length).
    m_nnDistance = std::cbrt(4.0 / density) / std::sqrt(2.0);

    m_diagNames = {"kinetic",  "potential", "potential_tail", "total_energy", "temperature",
                   "pressure", "pressure_tail", "box_length", "density", "nh_xi", "nh_invariant",
                   "target_t", "lindemann", "rdf_peak"};
    if (m_isMixture) {
        m_diagNames.insert(m_diagNames.end(), {"rdf_peak_aa", "rdf_peak_bb", "rdf_peak_ab", "fraction_b"});
    }

    m_system.SetParticles(m_initialPos, m_initialVel, m_initialL, m_initialSpecies);
    m_system.PrimeForces(m_params);
    m_unwrappedPos = m_initialPos;

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
    m_system.SetParticles(m_initialPos, m_initialVel, m_initialL, m_initialSpecies);
    m_system.PrimeForces(m_params);
    m_unwrappedPos = m_initialPos;
    m_simTime = 0.0;
}

void MDSim::UpdateRampTargetT() {
    if (m_ramp.empty()) return;
    if (m_simTime <= m_ramp.front().t0) {
        m_params.targetT = m_ramp.front().tStart;
        return;
    }
    for (const RampSegment& r : m_ramp) {
        if (m_simTime >= r.t0 && m_simTime < r.t1) {
            const double frac = (m_simTime - r.t0) / (r.t1 - r.t0);
            m_params.targetT = r.tStart + (r.tEnd - r.tStart) * frac;
            return;
        }
    }
    m_params.targetT = m_ramp.back().tEnd; // past the last segment: hold
}

void MDSim::UpdateUnwrappedPositions(const std::vector<glm::dvec3>& prevPos) {
    const std::vector<glm::dvec3>& newPos = m_system.Positions();
    const double L = m_system.BoxLength();
    const size_t n = newPos.size();
    for (size_t i = 0; i < n; ++i) {
        glm::dvec3 raw = newPos[i] - prevPos[i];
        raw.x -= L * std::round(raw.x / L);
        raw.y -= L * std::round(raw.y / L);
        raw.z -= L * std::round(raw.z / L);
        m_unwrappedPos[i] += raw;
    }
}

double MDSim::LindemannParameter() const {
    const size_t n = m_unwrappedPos.size();
    if (n == 0) return 0.0;
    double sumSq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const glm::dvec3 d = m_unwrappedPos[i] - m_initialPos[i];
        sumSq += glm::dot(d, d);
    }
    return std::sqrt(sumSq / static_cast<double>(n)) / m_nnDistance;
}

void MDSim::Step(int substeps) {
    for (int i = 0; i < substeps; ++i) {
        UpdateRampTargetT();
        const std::vector<glm::dvec3> prevPos = m_system.Positions();
        m_system.Step(m_params);
        UpdateUnwrappedPositions(prevPos);
        m_simTime += m_dt;
    }
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
    // Species is static (never changes after Configure()) -- only needs
    // writing once, on the first frame (see fw::SimInfo's comment on
    // frameFields).
    if (m_isMixture && writer.FramesWritten() == 0) {
        std::vector<int32_t> speciesI(m_system.Species().begin(), m_system.Species().end());
        writer.WriteField("species", speciesI.data(), fw::NpyDtype::I4, n, 1);
    }

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
    writer.WriteScalar("target_t", m_params.targetT);
    writer.WriteScalar("lindemann", LindemannParameter());

    // Structural order parameter that, unlike lindemann above, depends only
    // on the INSTANTANEOUS configuration (not displacement from the
    // original lattice site) -- so it can register refreezing into a new
    // arrangement on a cooling branch, not just the one-way melting jump
    // lindemann catches. A tall, sharp first peak means solid- or dense-
    // liquid-like local order; a short, smoothed-out one means a disordered
    // fluid. Computed once per FRAME (not substep) -- see the README's
    // melting/freezing hysteresis validation for why lindemann alone isn't
    // enough here.
    const std::vector<double> g = m_system.ComputeRDF(40, 2.0);
    const double rdfPeak = g.empty() ? 0.0 : *std::max_element(g.begin(), g.end());
    writer.WriteScalar("rdf_peak", rdfPeak);

    if (m_isMixture) {
        auto peakOf = [](const std::vector<double>& gg) {
            return gg.empty() ? 0.0 : *std::max_element(gg.begin(), gg.end());
        };
        writer.WriteScalar("rdf_peak_aa", peakOf(m_system.ComputePartialRDF(40, 2.0, 0, 0)));
        writer.WriteScalar("rdf_peak_bb", peakOf(m_system.ComputePartialRDF(40, 2.0, 1, 1)));
        writer.WriteScalar("rdf_peak_ab", peakOf(m_system.ComputePartialRDF(40, 2.0, 0, 1)));
        const int nB = static_cast<int>(std::count(m_system.Species().begin(), m_system.Species().end(), 1));
        writer.WriteScalar("fraction_b", static_cast<double>(nB) / std::max(n, 1));
    }
}

fw::SimInfo MDSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = m_isMixture ? std::vector<std::string>{"pos", "vel", "species"}
                                    : std::vector<std::string>{"pos", "vel"};
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

    const std::vector<int>& species = m_system.Species();
    std::vector<fw::ParticleInstance> particles;
    particles.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        fw::ParticleInstance p;
        p.position = glm::vec3(pos[static_cast<size_t>(i)]);
        if (m_isMixture) {
            // Fixed per-species colors (not speed) -- what matters for a
            // mixture is seeing the two components, not the instantaneous
            // kinetic state.
            p.color = species[static_cast<size_t>(i)] == 0 ? glm::vec4(0.35f, 0.55f, 1.0f, 1.0f)   // A: blue
                                                             : glm::vec4(1.0f, 0.45f, 0.25f, 1.0f);  // B: orange
        } else {
            const float t = static_cast<float>(glm::length(vel[static_cast<size_t>(i)]) / vMax);
            p.color = glm::vec4(0.3f + 0.7f * t, 0.45f, 1.0f - 0.6f * t, 1.0f); // blue (slow) -> orange (fast)
        }
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
    if (!m_ramp.empty()) {
        y += 20.0f;
        std::snprintf(line, sizeof(line), "ramp: t=%.2f  target_t=%.3f  Lindemann=%.4f", m_simTime,
                      m_params.targetT, LindemannParameter());
        m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    }
    if (m_isMixture) {
        y += 20.0f;
        const int nB = static_cast<int>(std::count(species.begin(), species.end(), 1));
        std::snprintf(line, sizeof(line), "mixture: A=blue (%d)  B=orange (%d)", n - nB, nB);
        m_text->Draw(m_font, line, glm::vec2(12.0f, y), dim);
    }
}

void MDSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) m_camera.Orbit(static_cast<float>(in.dx), static_cast<float>(in.dy));
    if (in.scrollDelta != 0.0) m_camera.Zoom(static_cast<float>(in.scrollDelta));
}

} // namespace md
