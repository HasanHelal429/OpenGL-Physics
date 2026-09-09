#include "ngrav/DeckSim.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ngrav {

namespace {

Solver SolverFromString(const std::string& s) {
    if (s == "direct") return Solver::Direct;
    if (s == "barnes_hut" || s == "bh") return Solver::BarnesHut;
    if (s == "fmm") return Solver::Fmm;
    if (s == "spherical_fmm") return Solver::SphericalFmm;
    if (s == "complex_fmm") return Solver::ComplexFmm;
    return Solver::BarnesHut;
}

} // namespace

template <int D>
void DeckSim<D>::Configure(const fw::Deck& deck) {
    m_title = deck.GetString("title", "N-Body Gravity");

    m_sp = StepParams{};
    m_sp.adaptive = deck.GetBool("time.adaptive", false);
    m_sp.eta = deck.GetDouble("time.eta", 0.03);
    // time.scheme = "global" (default) | "block". Block => power-of-2 rung
    // timesteps (Direct/Barnes-Hut only; falls back to `adaptive` otherwise).
    m_sp.block = (deck.GetString("time.scheme", "global") == "block");
    m_sp.blockMaxRung = deck.GetInt("time.block_max_rung", 8);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 4);

    m_sp.solver = SolverFromString(deck.GetString("solver.kind", "barnes_hut"));
    m_sp.mac.theta = deck.GetDouble("solver.theta", 0.5);
    m_sp.mac.kind = (deck.GetString("solver.mac", "geometric") == "relative") ? MacKind::Relative : MacKind::Geometric;
    m_sp.mac.alpha = deck.GetDouble("solver.alpha", 0.001);
    m_sp.fmmOrder = deck.GetInt("solver.fmm_order", 3);
    m_sp.sphericalOrder = deck.GetInt("solver.spherical_order", 5);

    m_scenarioType = ScenarioFromString(deck.GetString("scenario.type", "plummer"), ScenarioType::PlummerSphere);
    m_scenarioParams.n = deck.GetInt("scenario.n", 2000);
    m_scenarioParams.seed = static_cast<std::uint32_t>(deck.GetInt("scenario.seed", 1));
    m_scenarioParams.diskRotationFraction = deck.GetDouble("scenario.disk_rotation_fraction", 0.7);
    m_scenarioParams.eccentricity = deck.GetDouble("scenario.eccentricity", 0.36);
    m_scenarioParams.w0 = deck.GetDouble("scenario.king_w0", 6.0);
    m_scenarioParams.icFile = deck.GetString("scenario.ic_file", "");

    // A scenario carries tuned G / softening / dt; the deck overrides each
    // only when it explicitly sets the key.
    m_deckHasG = deck.Has("sim.G");
    m_deckHasEps = deck.Has("softening.eps");
    m_deckHasDt = deck.Has("time.dt");
    m_deckG = deck.GetDouble("sim.G", 1.0);
    m_deckEps = deck.GetDouble("softening.eps", 0.05);
    m_deckDt = deck.GetDouble("time.dt", 1e-3);
    m_deckSoftKind =
        (deck.GetString("softening.kind", "plummer") == "spline") ? SofteningKind::Spline : SofteningKind::Plummer;

    LoadScenario();

    if (m_registerAux) m_registerAux(m_sys);
    m_sys.Prime(m_sp);

    m_haveGL = true;
    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = static_cast<float>(m_cameraScale);
    m_camera.pitch = (D == 2) ? 89.0f : 25.0f;
    m_camera.maxDistance = 5000.0f;
}

template <int D>
void DeckSim<D>::LoadScenario() {
    Scenario<D> sc = BuildScenario<D>(m_scenarioType, m_scenarioParams);
    m_sp.G = m_deckHasG ? m_deckG : sc.G;
    const double softEps = m_deckHasEps ? m_deckEps : sc.soft.eps;
    m_sp.soft = (m_deckSoftKind == SofteningKind::Spline) ? Softening::Spline(softEps) : Softening::Plummer(softEps);
    m_sp.dt = m_deckHasDt ? m_deckDt : sc.suggestedDt;
    m_lastDt = m_sp.dt;
    m_cameraScale = sc.cameraScale;

    m_sys.SetParticles(std::move(sc.state));
    m_time = 0.0;
    m_step = 0;
    m_energy0 = m_sys.TotalEnergy(m_sp);
    if constexpr (D == 3) {
        m_lmag0 = glm::length(m_sys.AngularMomentum());
    } else {
        m_lmag0 = std::abs(m_sys.AngularMomentum());
    }
}

template <int D>
void DeckSim<D>::Reset() {
    LoadScenario();
    if (m_registerAux) m_registerAux(m_sys);
    m_sys.Prime(m_sp);
}

template <int D>
void DeckSim<D>::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        m_lastDt = m_sys.Step(m_sp);
        m_time += m_lastDt;
        ++m_step;
    }
}

template <int D>
void DeckSim<D>::Snapshot(fw::OutputWriter& writer) {
    const SoA<D>& st = m_sys.State();
    const int n = static_cast<int>(st.Count());

    std::vector<float> pos(static_cast<std::size_t>(n) * D), vel(static_cast<std::size_t>(n) * D);
    for (int i = 0; i < n; ++i) {
        pos[static_cast<std::size_t>(i * D + 0)] = static_cast<float>(st.x[static_cast<std::size_t>(i)]);
        pos[static_cast<std::size_t>(i * D + 1)] = static_cast<float>(st.y[static_cast<std::size_t>(i)]);
        vel[static_cast<std::size_t>(i * D + 0)] = static_cast<float>(st.vx[static_cast<std::size_t>(i)]);
        vel[static_cast<std::size_t>(i * D + 1)] = static_cast<float>(st.vy[static_cast<std::size_t>(i)]);
        if constexpr (D == 3) {
            pos[static_cast<std::size_t>(i * D + 2)] = static_cast<float>(st.z[static_cast<std::size_t>(i)]);
            vel[static_cast<std::size_t>(i * D + 2)] = static_cast<float>(st.vz[static_cast<std::size_t>(i)]);
        }
    }
    writer.WriteField("pos", pos.data(), fw::NpyDtype::F4, n, D);
    writer.WriteField("vel", vel.data(), fw::NpyDtype::F4, n, D);

    const double E = m_sys.TotalEnergy(m_sp);
    const Vec<D> com = m_sys.CenterOfMass();
    double lmag;
    if constexpr (D == 3)
        lmag = glm::length(m_sys.AngularMomentum());
    else
        lmag = std::abs(m_sys.AngularMomentum());

    // Total linear momentum P = sum m_i v_i. Conserved by any solver whose
    // *integrator* is symplectic (this always holds here) -- but a force
    // solver that isn't itself momentum-conserving (a one-directional FMM,
    // or any particle-cluster method with no explicit symmetry) can still
    // leak P through accumulated per-step force imbalance. This is the
    // deck-level, directly-observable version of the mutual FMM's headline
    // claim (see decks/cold_collapse_fmm.toml).
    double px = 0.0, py = 0.0, pz = 0.0;
    for (std::size_t i = 0; i < st.Count(); ++i) {
        px += st.m[i] * st.vx[i];
        py += st.m[i] * st.vy[i];
        if constexpr (D == 3) pz += st.m[i] * st.vz[i];
    }
    const double pmag = std::sqrt(px * px + py * py + pz * pz);

    writer.WriteScalar("energy", E);
    writer.WriteScalar("energy_drift_pct", (m_energy0 != 0.0) ? 100.0 * (E - m_energy0) / std::abs(m_energy0) : 0.0);
    writer.WriteScalar("L_mag", lmag);
    writer.WriteScalar("L_drift_pct", (m_lmag0 > 1e-12) ? 100.0 * (lmag - m_lmag0) / m_lmag0 : 100.0 * lmag);
    writer.WriteScalar("p_mag", pmag);
    writer.WriteScalar("com_x", com.x);
    writer.WriteScalar("com_y", com.y);
    writer.WriteScalar("com_z", (D == 3) ? Zc(com) : 0.0);
    writer.WriteScalar("dt_last", m_lastDt);
    // P15: deepest rung in use (0 for the global scheme). rung_max=k means
    // the finest particle is stepping at dt/2^k this frame.
    int rungMax = 0;
    for (int r : m_sys.Rungs()) rungMax = std::max(rungMax, r);
    writer.WriteScalar("rung_max", static_cast<double>(rungMax));
}

template <int D>
fw::SimInfo DeckSim<D>::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.dt = m_sp.dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"pos", "vel"};
    info.diagnostics = {"energy", "energy_drift_pct", "L_mag",   "L_drift_pct", "p_mag",
                        "com_x",  "com_y",            "com_z",   "dt_last",     "rung_max"};
    return info;
}

template <int D>
void DeckSim<D>::Render(int fbWidth, int fbHeight) {
    if (!m_haveGL) return;
    const SoA<D>& st = m_sys.State();
    const int n = static_cast<int>(st.Count());
    m_scratch.resize(static_cast<std::size_t>(n));

    double maxSpeed = 1e-9;
    for (int i = 0; i < n; ++i) {
        const double vx = st.vx[static_cast<std::size_t>(i)], vy = st.vy[static_cast<std::size_t>(i)];
        const double vz = (D == 3) ? st.vz[static_cast<std::size_t>(i)] : 0.0;
        maxSpeed = std::max(maxSpeed, std::sqrt(vx * vx + vy * vy + vz * vz));
    }
    const float baseSize =
        static_cast<float>(std::clamp(0.6 / std::sqrt(std::max<double>(1.0, n)), 0.006, 0.08));
    for (int i = 0; i < n; ++i) {
        const std::size_t ii = static_cast<std::size_t>(i);
        const double vx = st.vx[ii], vy = st.vy[ii];
        const double vz = (D == 3) ? st.vz[ii] : 0.0;
        const float t = static_cast<float>(std::clamp(std::sqrt(vx * vx + vy * vy + vz * vz) / maxSpeed, 0.0, 1.0));
        const glm::vec3 col = (t < 0.5f) ? glm::mix(glm::vec3(0.25f, 0.45f, 1.0f), glm::vec3(1.0f), t * 2.0f)
                                         : glm::mix(glm::vec3(1.0f), glm::vec3(1.0f, 0.55f, 0.15f), (t - 0.5f) * 2.0f);
        m_scratch[ii].position =
            glm::vec3(static_cast<float>(st.x[ii]), static_cast<float>(st.y[ii]),
                      (D == 3) ? static_cast<float>(st.z[ii]) : 0.0f);
        m_scratch[ii].color = glm::vec4(col, 0.9f);
        m_scratch[ii].size = baseSize;
    }
    m_cloud.SetParticles(m_scratch);

    const float aspect = static_cast<float>(fbWidth) / static_cast<float>(std::max(fbHeight, 1));
    m_cloud.Draw(m_camera.ViewMatrix(), m_camera.ProjectionMatrix(aspect), static_cast<float>(fbHeight));
}

template class DeckSim<2>;
template class DeckSim<3>;

} // namespace ngrav
