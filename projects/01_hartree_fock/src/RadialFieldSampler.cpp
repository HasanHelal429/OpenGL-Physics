#include "RadialFieldSampler.hpp"

#include "Quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>
#include <random>

namespace hf {

namespace {
constexpr double kPi = std::numbers::pi;
}

std::vector<glm::vec3> SampleRadialParticles(const std::vector<double>& r, const std::vector<double>& rho, int count,
                                              unsigned seed) {
    std::vector<double> radialProb(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        radialProb[i] = 4.0 * kPi * r[i] * r[i] * rho[i];
    }
    std::vector<double> cdf = CumulativeTrapezoid(radialProb, r);
    const double total = cdf.back();
    for (double& v : cdf) v /= total;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    std::vector<glm::vec3> points;
    points.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i) {
        const double u = unit(rng);
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        size_t idx = static_cast<size_t>(std::distance(cdf.begin(), it));
        idx = std::clamp(idx, size_t{1}, cdf.size() - 1);

        const double c0 = cdf[idx - 1], c1 = cdf[idx];
        const double frac = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.0;
        const double sampledR = r[idx - 1] + frac * (r[idx] - r[idx - 1]);

        const double z = 2.0 * unit(rng) - 1.0;
        const double phi = 2.0 * kPi * unit(rng);
        const double sinTheta = std::sqrt(std::max(0.0, 1.0 - z * z));
        const glm::dvec3 dir(sinTheta * std::cos(phi), sinTheta * std::sin(phi), z);

        points.push_back(glm::vec3(dir * sampledR));
    }
    return points;
}

std::vector<double> FindIsosurfaceRadii(const std::vector<double>& r, const std::vector<double>& rho, double threshold) {
    std::vector<double> radii;
    for (size_t i = 1; i < r.size(); ++i) {
        const double a = rho[i - 1] - threshold;
        const double b = rho[i] - threshold;
        if ((a <= 0.0 && b > 0.0) || (a > 0.0 && b <= 0.0)) {
            const double frac = a / (a - b);
            radii.push_back(r[i - 1] + frac * (r[i] - r[i - 1]));
        }
    }
    return radii;
}

double SuggestIsosurfaceThreshold(const std::vector<double>& r, const std::vector<double>& rho, double rWindowMin,
                                   double rWindowMax) {
    double peak = 0.0;
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i] >= rWindowMin && r[i] <= rWindowMax) {
            peak = std::max(peak, rho[i]);
        }
    }
    return peak * 0.1;
}

} // namespace hf
