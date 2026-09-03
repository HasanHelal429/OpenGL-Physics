#include "Sources.hpp"

#include "framework/Deck.hpp"

#include <algorithm>
#include <cmath>

namespace mag {

std::vector<double> BuildCurrent(const Grid& g, const fw::Deck& deck) {
    std::vector<double> Jz(g.count(), 0.0);
    const double cellArea = g.dx() * g.dy();

    for (const auto& w : deck.GetTables("wire")) {
        const double x0 = w.GetDouble("x", 0.0);
        const double y0 = w.GetDouble("y", 0.0);
        const double I = w.GetDouble("current", 1.0);
        const double radius =
            w.GetDouble("radius", 1.5 * std::max(g.dx(), g.dy()));
        const double r2 = radius * radius;

        // Count the covered cells first so the total integrates to exactly I.
        int covered = 0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double dx = g.x(i) - x0, dy = g.y(j) - y0;
                if (dx * dx + dy * dy <= r2) ++covered;
            }
        if (covered == 0) {
            // Radius smaller than a cell: dump it all in the nearest cell.
            const int i = std::clamp(
                static_cast<int>((x0 + 0.5 * g.lx) / g.dx()), 0, g.nx - 1);
            const int j = std::clamp(
                static_cast<int>((y0 + 0.5 * g.ly) / g.dy()), 0, g.ny - 1);
            Jz[g.idx(i, j)] += I / cellArea;
            continue;
        }
        const double jval = I / (covered * cellArea);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double dx = g.x(i) - x0, dy = g.y(j) - y0;
                if (dx * dx + dy * dy <= r2) Jz[g.idx(i, j)] += jval;
            }
    }

    for (const auto& s : deck.GetTables("strip")) {
        const double x0 = s.GetDouble("x", 0.0);
        const double y0 = s.GetDouble("y", 0.0);
        const double halfW = 0.5 * s.GetDouble("width", g.lx);
        const double halfH = 0.5 * s.GetDouble("height", g.dy());
        const double I = s.GetDouble("current", 1.0);

        int covered = 0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                if (std::abs(g.x(i) - x0) <= halfW &&
                    std::abs(g.y(j) - y0) <= halfH)
                    ++covered;
            }
        if (covered == 0) continue;
        const double jval = I / (covered * cellArea);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                if (std::abs(g.x(i) - x0) <= halfW &&
                    std::abs(g.y(j) - y0) <= halfH)
                    Jz[g.idx(i, j)] += jval;
            }
    }

    return Jz;
}

} // namespace mag
