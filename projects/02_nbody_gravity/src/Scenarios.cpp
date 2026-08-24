#include "Scenarios.hpp"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace nbody {

namespace {

// Two equal masses on a bound Kepler orbit (v_rel = 0.8 * v_circ, so
// eccentric but bound), placed symmetric about their (stationary) center
// of mass. suggestedDt uses the standard two-body period formula with the
// initial separation as a stand-in for the semi-major axis -- not exact
// for the eccentric case, just a sane timestep scale.
ScenarioResult MakeTwoBodyKepler() {
    ScenarioResult r;
    const double m = 1.0;
    const double G = 1.0;
    const double s = 1.0;
    const double vCirc = std::sqrt(G * (2.0 * m) / s);
    const double vRel = 0.8 * vCirc;

    r.pos = {glm::dvec3(-s / 2.0, 0.0, 0.0), glm::dvec3(s / 2.0, 0.0, 0.0)};
    r.vel = {glm::dvec3(0.0, -vRel / 2.0, 0.0), glm::dvec3(0.0, vRel / 2.0, 0.0)};
    r.mass = {m, m};

    r.G = G;
    r.softening = 0.02;
    const double period = 2.0 * glm::pi<double>() * std::sqrt(s * s * s / (G * 2.0 * m));
    r.suggestedDt = period / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraDistance = 3.0f;
    return r;
}

// Three equal masses at the vertices of an equilateral triangle, rotating
// rigidly about their shared centroid -- the Lagrange equilateral-triangle
// exact solution (Omega^2 = G * 3m / s^3), with no close encounters, so it
// stays numerically well-behaved at any reasonable dt.
ScenarioResult MakeLagrangeTriangle() {
    ScenarioResult r;
    const double m = 1.0;
    const double G = 1.0;
    const double s = 1.0; // circumradius from centroid
    const double omega = std::sqrt(G * 3.0 * m / (s * s * s));

    r.pos.resize(3);
    r.vel.resize(3);
    r.mass = {m, m, m};
    for (int k = 0; k < 3; ++k) {
        const double angle = glm::pi<double>() * 2.0 * static_cast<double>(k) / 3.0;
        const glm::dvec3 dir(std::cos(angle), std::sin(angle), 0.0);
        const glm::dvec3 tangent(-std::sin(angle), std::cos(angle), 0.0);
        r.pos[static_cast<size_t>(k)] = s * dir;
        r.vel[static_cast<size_t>(k)] = omega * s * tangent;
    }

    r.G = G;
    r.softening = 0.02;
    r.suggestedDt = (2.0 * glm::pi<double>() / omega) / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraDistance = 3.0f;
    return r;
}

// A roughly-virialized spherical cluster: uniform-density sphere (radius ~
// R*u^(1/3)) with isotropic Gaussian velocities scaled to the system's
// crossing-time velocity scale sqrt(G*M/R). This is a crude order-of-
// magnitude draw, not a real Plummer/King-model equilibrium -- just enough
// to keep the cluster from instantly collapsing or flying apart, so it
// reads as a stable N-scaling / solver-comparison stress test rather than
// a one-shot collapse.
ScenarioResult MakeCluster(const ScenarioParams& params) {
    ScenarioResult r;
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);

    const int n = std::max(params.n, 1);
    const double G = 1.0;
    const double R = 1.0;
    const double totalMass = 1.0;
    const double m = totalMass / static_cast<double>(n);

    r.pos.resize(static_cast<size_t>(n));
    r.vel.resize(static_cast<size_t>(n));
    r.mass.assign(static_cast<size_t>(n), m);

    for (int i = 0; i < n; ++i) {
        const double radius = R * std::cbrt(uni(rng));
        const double theta = std::acos(2.0 * uni(rng) - 1.0);
        const double phi = 2.0 * glm::pi<double>() * uni(rng);
        r.pos[static_cast<size_t>(i)] =
            radius * glm::dvec3(std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta));
    }

    const double vScale = std::sqrt(G * totalMass / R);
    for (int i = 0; i < n; ++i) {
        r.vel[static_cast<size_t>(i)] = 0.3 * vScale * glm::dvec3(gauss(rng), gauss(rng), gauss(rng));
    }

    r.G = G;
    const double meanSpacing = R / std::cbrt(static_cast<double>(n));
    r.softening = 2.0 * meanSpacing;
    const double tDyn = std::sqrt(R * R * R / (G * totalMass));
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = 0.5;
    r.cameraDistance = 3.5f;
    return r;
}

// A thin disk (uniform surface density -> radius ~ R*sqrt(u)) with
// circular velocity set from each radius's enclosed mass, scaled by
// diskRotationFraction f: f=1 is (approximately) rotationally supported,
// f=0 starts completely at rest and free-falls straight in. Intermediate f
// gives a slow-rotation collapse that retains real angular momentum
// through the crash. suggestedDt is based on the free-fall time (not an
// orbital period, which is undefined at f=0), so it stays sane across the
// whole f range.
ScenarioResult MakeRotatingDisk(const ScenarioParams& params) {
    ScenarioResult r;
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::normal_distribution<double> gaussZ(0.0, 1.0);

    const int n = std::max(params.n, 1);
    const double G = 1.0;
    const double R = 1.0;
    const double totalMass = 1.0;
    const double m = totalMass / static_cast<double>(n);
    const double f = params.diskRotationFraction;

    r.pos.resize(static_cast<size_t>(n));
    r.vel.resize(static_cast<size_t>(n));
    r.mass.assign(static_cast<size_t>(n), m);

    const double thickness = 0.03 * R;
    for (int i = 0; i < n; ++i) {
        const double radius = R * std::sqrt(uni(rng));
        const double angle = 2.0 * glm::pi<double>() * uni(rng);
        const double z = thickness * gaussZ(rng);
        r.pos[static_cast<size_t>(i)] = glm::dvec3(radius * std::cos(angle), radius * std::sin(angle), z);

        // Uniform surface density -> enclosed mass ~ (radius/R)^2 * totalMass.
        const double mEnclosed = totalMass * (radius * radius) / (R * R);
        const double vCirc = radius > 1e-9 ? std::sqrt(G * mEnclosed / radius) : 0.0;
        const double v = f * vCirc;
        const glm::dvec3 tangent(-std::sin(angle), std::cos(angle), 0.0);
        r.vel[static_cast<size_t>(i)] = v * tangent;
    }

    r.G = G;
    const double meanSpacing = R / std::sqrt(static_cast<double>(n)); // 2D surface-density spacing
    r.softening = 3.0 * meanSpacing;
    const double tDyn = std::sqrt(R * R * R / (G * totalMass)); // free-fall time, independent of f
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = 0.5;
    r.cameraDistance = 3.5f;
    return r;
}

} // namespace

const char* ScenarioName(ScenarioType type) {
    switch (type) {
        case ScenarioType::TwoBodyKepler: return "Two-body Kepler orbit";
        case ScenarioType::LagrangeTriangle: return "Lagrange equilateral triangle";
        case ScenarioType::Cluster: return "Star cluster (N stress test)";
        case ScenarioType::RotatingDisk: return "Rotating disk / collapse";
    }
    return "?";
}

ScenarioResult BuildScenario(ScenarioType type, const ScenarioParams& params) {
    switch (type) {
        case ScenarioType::TwoBodyKepler: return MakeTwoBodyKepler();
        case ScenarioType::LagrangeTriangle: return MakeLagrangeTriangle();
        case ScenarioType::Cluster: return MakeCluster(params);
        case ScenarioType::RotatingDisk: return MakeRotatingDisk(params);
    }
    return MakeTwoBodyKepler();
}

} // namespace nbody
