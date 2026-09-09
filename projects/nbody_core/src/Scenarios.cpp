#include "ngrav/Scenarios.hpp"

#include "ngrav/ic/DiskBulgeHalo.hpp"
#include "ngrav/ic/Hernquist.hpp"
#include "ngrav/ic/King.hpp"
#include "ngrav/ic/Plummer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace ngrav {

namespace {

template <int D>
void Fill(SoA<D>& s, int n) {
    s.Resize(static_cast<std::size_t>(n));
}

// --- two-body Kepler --------------------------------------------------------
// `pp.eccentricity` sets the 3D orbit's shape via vis-viva at apoapsis r0=1:
// vRel = vCirc * sqrt(1-e), a = r0/(1+e). e=0.36 (the default) reproduces the
// original fixed "0.8*vCirc" orbit exactly (sqrt(1-0.36) = 0.8) -- so this is
// a strict generalization, not a behavior change, at the default. 2D has no
// closed-orbit eccentricity in the usual sense (see ComplexFmm.hpp's header
// comment on the 1/r force law); its velocity stays the fixed 0.8*vCirc.
template <int D>
Scenario<D> Kepler(const ScenarioParams& pp) {
    Scenario<D> r;
    const double G = 1.0, m = 1.0, sep = 1.0;
    const double vCirc = (D == 3) ? std::sqrt(G * 2.0 * m / sep) : std::sqrt(G * 2.0 * m);
    double vRel;
    if constexpr (D == 3) {
        const double e = std::clamp(pp.eccentricity, 0.0, 0.99);
        vRel = vCirc * std::sqrt(1.0 - e);
    } else {
        vRel = 0.8 * vCirc;
    }
    Fill<D>(r.state, 2);
    Vec<D> p0(0.0), p1(0.0), v0(0.0), v1(0.0);
    p0.x = -sep / 2;
    p1.x = sep / 2;
    v0.y = -vRel / 2;
    v1.y = vRel / 2;
    r.state.SetPos(0, p0);
    r.state.SetPos(1, p1);
    r.state.SetVel(0, v0);
    r.state.SetVel(1, v1);
    r.state.m[0] = m;
    r.state.m[1] = m;
    r.G = G;
    r.soft = Softening::Plummer(0.02);
    if constexpr (D == 3) {
        const double e = std::clamp(pp.eccentricity, 0.0, 0.99);
        const double a = sep / (1.0 + e);
        const double period = 2.0 * M_PI * std::sqrt(a * a * a / (G * 2.0 * m));
        // Apoapsis-scale step (2000/period, same convention at every e) --
        // deliberately NOT scaled up for a tight periapsis: a high-e orbit
        // is exactly the case adaptive global timestepping exists for (see
        // eccentric_orbit_{fixed,adaptive}.toml), so the "suggested" fixed
        // dt should stay the cheap, apoapsis-appropriate one and let a fixed
        // run visibly struggle at periapsis rather than papering over it.
        r.suggestedDt = period / 2000.0;
    } else {
        r.suggestedDt = (2.0 * M_PI * sep / vCirc) / 2000.0;
    }
    r.suggestedTheta = 0.5;
    r.cameraScale = 3.0;
    return r;
}

// --- Lagrange equilateral triangle (side length L = 1) ---------------------
template <int D>
Scenario<D> Lagrange() {
    Scenario<D> r;
    const double G = 1.0, m = 1.0, L = 1.0;
    const double omega = (D == 3) ? std::sqrt(3.0 * G * m / (L * L * L)) : (std::sqrt(G * m) / L);
    const double R = (D == 3) ? (L / std::sqrt(3.0)) : L; // 2D: place at radius L (its own convention)
    Fill<D>(r.state, 3);
    for (int k = 0; k < 3; ++k) {
        const double a = 2.0 * M_PI * k / 3.0;
        Vec<D> p(0.0), v(0.0);
        p.x = R * std::cos(a);
        p.y = R * std::sin(a);
        v.x = -omega * R * std::sin(a);
        v.y = omega * R * std::cos(a);
        r.state.SetPos(static_cast<std::size_t>(k), p);
        r.state.SetVel(static_cast<std::size_t>(k), v);
        r.state.m[static_cast<std::size_t>(k)] = m;
    }
    r.G = G;
    r.soft = Softening::Plummer(0.02);
    r.suggestedDt = (2.0 * M_PI / omega) / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraScale = 3.0;
    return r;
}

// --- roughly-virialised uniform sphere / disk ----------------------------
template <int D>
Scenario<D> Cluster(const ScenarioParams& pp) {
    Scenario<D> r;
    std::mt19937 rng(pp.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const int n = std::max(pp.n, 1);
    const double G = 1.0, R = 1.0, M = 1.0;
    Fill<D>(r.state, n);
    for (int i = 0; i < n; ++i) {
        double radius;
        Vec<D> dir(0.0);
        if constexpr (D == 3) {
            radius = R * std::cbrt(u(rng));
            const double cosT = 2.0 * u(rng) - 1.0, sinT = std::sqrt(std::max(0.0, 1.0 - cosT * cosT));
            const double phi = 2.0 * M_PI * u(rng);
            dir = Vec<3>(sinT * std::cos(phi), sinT * std::sin(phi), cosT);
        } else {
            radius = R * std::sqrt(u(rng));
            const double phi = 2.0 * M_PI * u(rng);
            dir = Vec<2>(std::cos(phi), std::sin(phi));
        }
        r.state.SetPos(static_cast<std::size_t>(i), dir * radius);
        const double vScale = (D == 3) ? std::sqrt(G * M / R) : std::sqrt(G * M);
        Vec<D> v(0.0);
        v.x = 0.3 * vScale * gauss(rng);
        v.y = 0.3 * vScale * gauss(rng);
        if constexpr (D == 3) v.z = 0.3 * vScale * gauss(rng);
        r.state.SetVel(static_cast<std::size_t>(i), v);
        r.state.m[static_cast<std::size_t>(i)] = M / n;
    }
    r.G = G;
    const double meanSpacing = (D == 3) ? (R / std::cbrt((double)n)) : (R / std::sqrt((double)n));
    r.soft = Softening::Plummer(2.0 * meanSpacing);
    const double tDyn = (D == 3) ? std::sqrt(R * R * R / (G * M)) : (R / std::sqrt(G * M));
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = 0.5;
    r.cameraScale = (D == 3) ? 3.5 : 1.75;
    return r;
}

// --- rotating disk (partial support) -------------------------------------
template <int D>
Scenario<D> RotatingDisk(const ScenarioParams& pp) {
    Scenario<D> r;
    std::mt19937 rng(pp.seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::normal_distribution<double> gz(0.0, 1.0);
    const int n = std::max(pp.n, 1);
    const double G = 1.0, R = 1.0, M = 1.0, f = pp.diskRotationFraction;
    Fill<D>(r.state, n);
    const double thickness = 0.03 * R;
    for (int i = 0; i < n; ++i) {
        const double radius = R * std::sqrt(u(rng));
        const double angle = 2.0 * M_PI * u(rng);
        Vec<D> p(0.0);
        p.x = radius * std::cos(angle);
        p.y = radius * std::sin(angle);
        if constexpr (D == 3) p.z = thickness * gz(rng);
        r.state.SetPos(static_cast<std::size_t>(i), p);
        const double mEnc = M * (radius * radius) / (R * R);
        const double vCirc = (D == 3) ? (radius > 1e-9 ? std::sqrt(G * mEnc / radius) : 0.0) : std::sqrt(G * mEnc);
        const double v = f * vCirc;
        Vec<D> vel(0.0);
        vel.x = -v * std::sin(angle);
        vel.y = v * std::cos(angle);
        r.state.SetVel(static_cast<std::size_t>(i), vel);
        r.state.m[static_cast<std::size_t>(i)] = M / n;
    }
    r.G = G;
    const double meanSpacing = R / std::sqrt((double)n);
    r.soft = Softening::Plummer(3.0 * meanSpacing);
    const double tDyn = (D == 3) ? std::sqrt(R * R * R / (G * M)) : (R / std::sqrt(G * M));
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = (D == 3) ? 0.5 : 0.2;
    r.cameraScale = (D == 3) ? 3.5 : 1.75;
    return r;
}

// --- Plummer sphere (equilibrium) --------------------------------------
template <int D>
Scenario<D> PlummerEquilibrium(const ScenarioParams& pp) {
    Scenario<D> r;
    ic::PlummerParams ip;
    ip.n = std::max(pp.n, 1);
    ip.seed = pp.seed;
    r.state = ic::Plummer<D>(ip);
    r.G = 1.0;
    const double meanSpacing = (D == 3) ? (1.0 / std::cbrt((double)ip.n)) : (1.0 / std::sqrt((double)ip.n));
    r.soft = Softening::Plummer(std::max(0.01, 1.5 * meanSpacing));
    r.suggestedDt = (2.0 * M_PI) / 800.0; // t_cross ~ 2*pi
    r.suggestedTheta = 0.5;
    r.cameraScale = (D == 3) ? 6.0 : 3.0;
    return r;
}

// --- cold collapse: uniform sphere/disk at rest -----------------------
template <int D>
Scenario<D> ColdUniform(const ScenarioParams& pp) {
    Scenario<D> r = Cluster<D>(pp); // uniform positions, then zero the velocities
    std::fill(r.state.vx.begin(), r.state.vx.end(), 0.0);
    std::fill(r.state.vy.begin(), r.state.vy.end(), 0.0);
    if constexpr (D == 3) std::fill(r.state.vz.begin(), r.state.vz.end(), 0.0);
    const int n = std::max(pp.n, 1);
    const double meanSpacing = (D == 3) ? (1.0 / std::cbrt((double)n)) : (1.0 / std::sqrt((double)n));
    r.soft = Softening::Plummer(std::max(0.01, 1.0 * meanSpacing)); // tighter -> real collapse
    const double tFF = std::sqrt(1.0);                              // free-fall ~ 1/sqrt(G rho); G=M=R=1
    r.suggestedDt = tFF / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraScale = (D == 3) ? 3.5 : 1.75;
    return r;
}

// --- Hernquist sphere (P13) -------------------------------------------
template <int D>
Scenario<D> HernquistScenario(const ScenarioParams& pp) {
    if constexpr (D == 2) {
        return PlummerEquilibrium<2>(pp); // no 2D Hernquist -- fall back, documented in Scenarios.hpp
    } else {
        Scenario<3> r;
        ic::HernquistParams hp;
        hp.n = std::max(pp.n, 1);
        hp.seed = pp.seed;
        r.state = ic::Hernquist<3>(hp);
        r.G = 1.0;
        const double meanSpacing = 1.0 / std::cbrt((double)hp.n);
        r.soft = Softening::Plummer(std::max(0.01, 1.5 * meanSpacing));
        r.suggestedDt = (2.0 * M_PI) / 800.0;
        r.suggestedTheta = 0.5;
        r.cameraScale = 8.0;
        return r;
    }
}

// --- King sphere (P13) ------------------------------------------------
template <int D>
Scenario<D> KingScenario(const ScenarioParams& pp) {
    if constexpr (D == 2) {
        return PlummerEquilibrium<2>(pp);
    } else {
        Scenario<3> r;
        ic::KingParams kp;
        kp.n = std::max(pp.n, 1);
        kp.seed = pp.seed;
        kp.w0 = pp.w0;
        r.state = ic::King<3>(kp);
        r.G = 1.0;
        const double meanSpacing = 1.0 / std::cbrt((double)kp.n);
        r.soft = Softening::Plummer(std::max(0.01, 1.2 * meanSpacing));
        r.suggestedDt = (2.0 * M_PI) / 800.0;
        r.suggestedTheta = 0.5;
        r.cameraScale = 6.0;
        return r;
    }
}

// --- disk galaxy: exp disk + Hernquist bulge + NFW halo (P13) --------
template <int D>
Scenario<D> DiskGalaxyScenario(const ScenarioParams& pp) {
    if constexpr (D == 2) {
        return RotatingDisk<2>(pp); // no 3-component 2D galaxy -- fall back to the flat rotating disk
    } else {
        Scenario<3> r;
        ic::DiskBulgeHaloParams dp;
        // Split the requested N across components in a fixed 40/10/50 ratio.
        const int n = std::max(pp.n, 100);
        dp.nDisk = std::max(1, static_cast<int>(0.40 * n));
        dp.nBulge = std::max(1, static_cast<int>(0.10 * n));
        dp.nHalo = std::max(1, n - dp.nDisk - dp.nBulge);
        dp.seed = pp.seed;
        dp.diskRotationFraction = (pp.diskRotationFraction > 0.0) ? pp.diskRotationFraction : 1.0;
        r.state = ic::DiskBulgeHalo<3>(dp);
        r.G = 1.0;
        r.soft = Softening::Plummer(0.15);
        // Disk dynamical time at ~1 scale length.
        const double vc = std::sqrt(1.0 * (dp.bulgeMass + dp.diskMass * 0.25) / dp.diskScaleLength);
        r.suggestedDt = (2.0 * M_PI * dp.diskScaleLength / std::max(vc, 1e-3)) / 400.0;
        r.suggestedTheta = 0.5;
        r.cameraScale = 25.0;
        return r;
    }
}

// --- raw float32 particle dump (P13) ---------------------------------
// 06_tidal_disruption's format: N records of 7 float32, x,y,z,mass,vx,vy,vz
// (see that project's tools/make_star_ic.py / src/TdeSim.cpp). For D==2 the
// z and vz columns are read and discarded. Written by this project's own
// tools/make_ic.py.
template <int D>
Scenario<D> IcFileScenario(const ScenarioParams& pp) {
    Scenario<D> r;
    std::ifstream f(pp.icFile, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("ic_file: cannot open '" + pp.icFile + "'");
    const std::streamsize bytes = f.tellg();
    f.seekg(0);
    constexpr int kFloatsPerParticle = 7; // always 7 on disk, even for 2D (z/vz discarded)
    const int n = static_cast<int>(bytes / (kFloatsPerParticle * static_cast<std::streamsize>(sizeof(float))));
    if (n <= 0) throw std::runtime_error("ic_file: '" + pp.icFile + "' has no particles");
    std::vector<float> raw(static_cast<std::size_t>(n) * kFloatsPerParticle);
    f.read(reinterpret_cast<char*>(raw.data()), bytes);

    Fill<D>(r.state, n);
    for (int i = 0; i < n; ++i) {
        const float* p = &raw[static_cast<std::size_t>(i) * kFloatsPerParticle];
        Vec<D> pos(0.0), vel(0.0);
        pos.x = p[0];
        pos.y = p[1];
        vel.x = p[4];
        vel.y = p[5];
        if constexpr (D == 3) {
            pos.z = p[2];
            vel.z = p[6];
        }
        r.state.SetPos(static_cast<std::size_t>(i), pos);
        r.state.SetVel(static_cast<std::size_t>(i), vel);
        r.state.m[static_cast<std::size_t>(i)] = p[3];
    }
    r.G = 1.0;
    // No embedded tuning in the file -- pick middle-of-the-road defaults the
    // deck is expected to override (tools/make_ic.py prints suggestions).
    r.soft = Softening::Plummer(0.05);
    r.suggestedDt = 1e-3;
    r.suggestedTheta = 0.5;
    r.cameraScale = (D == 3) ? 6.0 : 3.0;
    return r;
}

} // namespace

const char* ScenarioName(ScenarioType t) {
    switch (t) {
        case ScenarioType::TwoBodyKepler: return "Two-body Kepler orbit";
        case ScenarioType::LagrangeTriangle: return "Lagrange equilateral triangle";
        case ScenarioType::Cluster: return "Cluster (N stress test)";
        case ScenarioType::RotatingDisk: return "Rotating disk / collapse";
        case ScenarioType::PlummerSphere: return "Plummer sphere (equilibrium)";
        case ScenarioType::ColdCollapse: return "Cold collapse";
        case ScenarioType::HernquistSphere: return "Hernquist sphere (equilibrium)";
        case ScenarioType::KingSphere: return "King sphere (equilibrium)";
        case ScenarioType::DiskGalaxy: return "Disk galaxy (disk + bulge + NFW halo)";
        case ScenarioType::IcFile: return "IC file (raw float32 dump)";
    }
    return "?";
}

bool ScenarioNeedsN(ScenarioType t) {
    return t == ScenarioType::Cluster || t == ScenarioType::RotatingDisk || t == ScenarioType::PlummerSphere ||
           t == ScenarioType::ColdCollapse || t == ScenarioType::HernquistSphere || t == ScenarioType::KingSphere ||
           t == ScenarioType::DiskGalaxy;
}

ScenarioType ScenarioFromString(const std::string& s, ScenarioType fallback) {
    if (s == "two_body_kepler" || s == "kepler") return ScenarioType::TwoBodyKepler;
    if (s == "lagrange_triangle" || s == "lagrange") return ScenarioType::LagrangeTriangle;
    if (s == "cluster") return ScenarioType::Cluster;
    if (s == "rotating_disk") return ScenarioType::RotatingDisk;
    if (s == "plummer" || s == "plummer_sphere") return ScenarioType::PlummerSphere;
    if (s == "cold_collapse") return ScenarioType::ColdCollapse;
    if (s == "hernquist" || s == "hernquist_sphere") return ScenarioType::HernquistSphere;
    if (s == "king" || s == "king_sphere") return ScenarioType::KingSphere;
    if (s == "disk_galaxy" || s == "galaxy") return ScenarioType::DiskGalaxy;
    if (s == "ic_file") return ScenarioType::IcFile;
    return fallback;
}

template <int D>
Scenario<D> BuildScenario(ScenarioType type, const ScenarioParams& params) {
    switch (type) {
        case ScenarioType::TwoBodyKepler: return Kepler<D>(params);
        case ScenarioType::LagrangeTriangle: return Lagrange<D>();
        case ScenarioType::Cluster: return Cluster<D>(params);
        case ScenarioType::RotatingDisk: return RotatingDisk<D>(params);
        case ScenarioType::PlummerSphere: return PlummerEquilibrium<D>(params);
        case ScenarioType::ColdCollapse: return ColdUniform<D>(params);
        case ScenarioType::HernquistSphere: return HernquistScenario<D>(params);
        case ScenarioType::KingSphere: return KingScenario<D>(params);
        case ScenarioType::DiskGalaxy: return DiskGalaxyScenario<D>(params);
        case ScenarioType::IcFile: return IcFileScenario<D>(params);
    }
    return Kepler<D>(params);
}

template Scenario<2> BuildScenario<2>(ScenarioType, const ScenarioParams&);
template Scenario<3> BuildScenario<3>(ScenarioType, const ScenarioParams&);

} // namespace ngrav
