#include "Scenarios.hpp"

#include <glm/gtc/constants.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace nbody2d {

namespace {

// Two equal masses on a bound orbit, placed symmetric about their
// (stationary) center of mass. Unlike 3D, 2D gravity's force falls off as
// 1/r (not 1/r^2 -- see ComplexFmm.hpp's header comment), so the relative
// acceleration is G*(m1+m2)/s (not /s^2) and the circular speed works out
// to v_circ = sqrt(G*(m1+m2)) -- *independent of separation*, a genuine
// feature of 2D gravity, not a simplification. Also unlike 3D, a 1/r force
// law isn't one of the two power laws Bertrand's theorem allows closed
// orbits for, so this orbit precesses rather than retracing a fixed
// ellipse -- suggestedDt is therefore a dimensional-analysis timescale
// (s/v_circ), not an exact period.
ScenarioResult MakeTwoBodyKepler() {
    ScenarioResult r;
    const double m = 1.0;
    const double G = 1.0;
    const double s = 1.0;
    const double vCirc = std::sqrt(G * (2.0 * m));
    const double vRel = 0.8 * vCirc;

    r.pos = {glm::dvec2(-s / 2.0, 0.0), glm::dvec2(s / 2.0, 0.0)};
    r.vel = {glm::dvec2(0.0, -vRel / 2.0), glm::dvec2(0.0, vRel / 2.0)};
    r.mass = {m, m};

    r.G = G;
    r.softening = 0.02;
    r.suggestedDt = (2.0 * glm::pi<double>() * s / vCirc) / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraExtent = 1.5f;
    return r;
}

// Three equal masses at the vertices of an equilateral triangle, rotating
// rigidly about their shared centroid. The equilateral configuration's net
// inward force per mass works for *any* central pair force law (a
// symmetry argument, not specific to 1/r^2) -- for 2D's 1/r force this
// works out to Omega = sqrt(G*m)/s (vs. 3D's sqrt(G*3m/s^3)).
ScenarioResult MakeLagrangeTriangle() {
    ScenarioResult r;
    const double m = 1.0;
    const double G = 1.0;
    const double s = 1.0; // circumradius from centroid
    const double omega = std::sqrt(G * m) / s;

    r.pos.resize(3);
    r.vel.resize(3);
    r.mass = {m, m, m};
    for (int k = 0; k < 3; ++k) {
        const double angle = glm::pi<double>() * 2.0 * static_cast<double>(k) / 3.0;
        const glm::dvec2 dir(std::cos(angle), std::sin(angle));
        const glm::dvec2 tangent(-std::sin(angle), std::cos(angle));
        r.pos[static_cast<size_t>(k)] = s * dir;
        r.vel[static_cast<size_t>(k)] = omega * s * tangent;
    }

    r.G = G;
    r.softening = 0.02;
    r.suggestedDt = (2.0 * glm::pi<double>() / omega) / 2000.0;
    r.suggestedTheta = 0.5;
    r.cameraExtent = 1.5f;
    return r;
}

// A roughly-virialized disk cluster: uniform-density filled circle (radius
// ~ R*sqrt(u), the 2D analogue of 3D's uniform sphere) with isotropic
// Gaussian velocities scaled to the system's crossing-time velocity scale
// -- same crude-equilibrium spirit as the 3D project's Cluster scenario, a
// stable N-scaling/solver-comparison stress test rather than a real
// equilibrium model.
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
        const double radius = R * std::sqrt(uni(rng));
        const double angle = 2.0 * glm::pi<double>() * uni(rng);
        r.pos[static_cast<size_t>(i)] = radius * glm::dvec2(std::cos(angle), std::sin(angle));
    }

    // v_circ for 2D gravity is independent of radius (see MakeTwoBodyKepler's
    // own comment) -- G*totalMass is the natural velocity-squared scale here.
    const double vScale = std::sqrt(G * totalMass);
    for (int i = 0; i < n; ++i) {
        r.vel[static_cast<size_t>(i)] = 0.3 * vScale * glm::dvec2(gauss(rng), gauss(rng));
    }

    r.G = G;
    const double meanSpacing = R / std::sqrt(static_cast<double>(n));
    r.softening = 2.0 * meanSpacing;
    const double tDyn = R / vScale; // crossing time
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = 0.5;
    r.cameraExtent = 1.75f;
    return r;
}

// A filled circle with circular velocity set from each radius's enclosed
// mass, scaled by diskRotationFraction f -- same role as the 3D project's
// RotatingDisk, minus the out-of-plane thickness (already 2D). Enclosed
// mass for a uniform-surface-density disk is ~(radius/R)^2 * totalMass,
// same as 3D's RotatingDisk; what differs is 2D's own force law, which
// makes v_circ = sqrt(G*mEnclosed) with *no* division by radius (see
// MakeTwoBodyKepler's comment) -- so unlike 3D's v_circ ~ sqrt(1/r) falloff
// convention, here v_circ actually *grows* with radius out to R (mEnclosed
// grows as r^2, so v_circ ~ r) before the enclosed-mass model runs out at
// r=R. This is a real feature of 2D gravity (not a bug): with a uniform
// disk's mass profile, 2D's 1/r force gives solid-body-like rotation
// within the disk.
ScenarioResult MakeRotatingDisk(const ScenarioParams& params) {
    ScenarioResult r;
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    const int n = std::max(params.n, 1);
    const double G = 1.0;
    const double R = 1.0;
    const double totalMass = 1.0;
    const double m = totalMass / static_cast<double>(n);
    const double f = params.diskRotationFraction;

    r.pos.resize(static_cast<size_t>(n));
    r.vel.resize(static_cast<size_t>(n));
    r.mass.assign(static_cast<size_t>(n), m);

    for (int i = 0; i < n; ++i) {
        const double radius = R * std::sqrt(uni(rng));
        const double angle = 2.0 * glm::pi<double>() * uni(rng);
        r.pos[static_cast<size_t>(i)] = radius * glm::dvec2(std::cos(angle), std::sin(angle));

        const double mEnclosed = totalMass * (radius * radius) / (R * R);
        const double vCirc = std::sqrt(G * mEnclosed);
        const double v = f * vCirc;
        const glm::dvec2 tangent(-std::sin(angle), std::cos(angle));
        r.vel[static_cast<size_t>(i)] = v * tangent;
    }

    r.G = G;
    const double meanSpacing = R / std::sqrt(static_cast<double>(n));
    r.softening = 3.0 * meanSpacing;
    const double vScale = std::sqrt(G * totalMass);
    const double tDyn = R / vScale;
    r.suggestedDt = tDyn / 1000.0;
    r.suggestedTheta = 0.5;
    r.cameraExtent = 1.75f;
    return r;
}

} // namespace

const char* ScenarioName(ScenarioType type) {
    switch (type) {
        case ScenarioType::TwoBodyKepler: return "Two-body orbit";
        case ScenarioType::LagrangeTriangle: return "Lagrange equilateral triangle";
        case ScenarioType::Cluster: return "Disk cluster (N stress test)";
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

} // namespace nbody2d
