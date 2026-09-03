#include "Materials.hpp"

#include "framework/Deck.hpp"

#include <cmath>
#include <functional>

namespace fdtd {

void Materials::Build(int nx_, int ny_, double dx, double dy,
                      const fw::Deck& deck) {
    nx = nx_;
    ny = ny_;
    const std::size_t n = static_cast<std::size_t>(nx) * ny;
    epsR.assign(n, 1.0);
    muR.assign(n, 1.0);
    sigma.assign(n, 0.0);
    pec.assign(n, 0);

    auto forEachCell = [&](auto&& inside, auto&& apply) {
        for (int j = 0; j < ny; ++j) {
            const double y = j * dy;
            for (int i = 0; i < nx; ++i) {
                const double x = i * dx;
                if (inside(x, y)) apply(static_cast<std::size_t>(j) * nx + i);
            }
        }
    };

    auto shapePredicate = [&](const fw::Deck& t) {
        const std::string shape = t.GetString("shape", "box");
        if (shape == "slab") {
            const bool axisX = t.GetString("axis", "y") == "x";
            const double pos = t.GetDouble("pos", 0.5 * (axisX ? nx * dx : ny * dy));
            const double half = 0.5 * t.GetDouble("thickness", 10.0);
            return std::function<bool(double, double)>(
                [=](double x, double y) {
                    return std::abs((axisX ? x : y) - pos) <= half;
                });
        }
        if (shape == "halfspace") {
            const bool axisX = t.GetString("axis", "y") == "x";
            const double pos = t.GetDouble("pos", 0.5 * (axisX ? nx * dx : ny * dy));
            const bool hi = t.GetString("side", "lo") == "hi";
            return std::function<bool(double, double)>(
                [=](double x, double y) {
                    const double c = axisX ? x : y;
                    return hi ? c >= pos : c <= pos;
                });
        }
        if (shape == "cylinder") {
            const double cx = t.GetDouble("x", 0.5 * nx * dx);
            const double cy = t.GetDouble("y", 0.5 * ny * dy);
            const double r = t.GetDouble("radius", 10.0);
            return std::function<bool(double, double)>(
                [=](double x, double y) {
                    return (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r;
                });
        }
        // box
        const double cx = t.GetDouble("x", 0.5 * nx * dx);
        const double cy = t.GetDouble("y", 0.5 * ny * dy);
        const double hx = t.GetDouble("half_x", 10.0);
        const double hy = t.GetDouble("half_y", 10.0);
        return std::function<bool(double, double)>(
            [=](double x, double y) {
                return std::abs(x - cx) <= hx && std::abs(y - cy) <= hy;
            });
    };

    for (const auto& t : deck.GetTables("material")) {
        const double er = t.GetDouble("eps_r", 1.0);
        const double mr = t.GetDouble("mu_r", 1.0);
        const double sg = t.GetDouble("sigma", 0.0);
        const auto in = shapePredicate(t);
        forEachCell(in, [&](std::size_t p) {
            epsR[p] = er;
            muR[p] = mr;
            sigma[p] = sg;
        });
    }
    for (const auto& t : deck.GetTables("pec")) {
        const auto in = shapePredicate(t);
        forEachCell(in, [&](std::size_t p) { pec[p] = 1; });
    }
}

bool Materials::AnyPec() const {
    for (unsigned char c : pec)
        if (c) return true;
    return false;
}

bool Materials::AnyDielectric() const {
    for (std::size_t p = 0; p < epsR.size(); ++p)
        if (epsR[p] != 1.0 || muR[p] != 1.0 || sigma[p] != 0.0) return true;
    return false;
}

} // namespace fdtd
