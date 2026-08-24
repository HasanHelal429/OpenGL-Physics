#pragma once

#include <vector>

namespace hf {

// integral(y dr) via the trapezoid rule.
inline double Trapezoid(const std::vector<double>& y, const std::vector<double>& r) {
    double sum = 0.0;
    for (size_t i = 1; i < y.size(); ++i) {
        sum += 0.5 * (y[i] + y[i - 1]) * (r[i] - r[i - 1]);
    }
    return sum;
}

// result[0] = 0, result[i] = result[i-1] + 0.5*(y[i]+y[i-1])*(r[i]-r[i-1]).
inline std::vector<double> CumulativeTrapezoid(const std::vector<double>& y, const std::vector<double>& r) {
    std::vector<double> result(y.size(), 0.0);
    for (size_t i = 1; i < y.size(); ++i) {
        result[i] = result[i - 1] + 0.5 * (y[i] + y[i - 1]) * (r[i] - r[i - 1]);
    }
    return result;
}

} // namespace hf
