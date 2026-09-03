#include "Sources.hpp"

#include "framework/Deck.hpp"

#include <algorithm>
#include <cmath>

namespace mag {

Sources ParseSources(const Grid& g, const fw::Deck& deck) {
    Sources s;
    const double defRadius = 1.5 * std::max(g.dx(), g.dy());
    for (const auto& w : deck.GetTables("wire")) {
        WireSpec ws;
        ws.x = w.GetDouble("x", 0.0);
        ws.y = w.GetDouble("y", 0.0);
        ws.current = w.GetDouble("current", 1.0);
        ws.radius = w.GetDouble("radius", defRadius);
        if (ws.radius <= 0.0) ws.radius = defRadius;
        s.wires.push_back(ws);
    }
    for (const auto& t : deck.GetTables("strip")) {
        StripSpec st;
        st.x = t.GetDouble("x", 0.0);
        st.y = t.GetDouble("y", 0.0);
        st.width = t.GetDouble("width", g.lx);
        st.height = t.GetDouble("height", g.dy());
        st.current = t.GetDouble("current", 1.0);
        s.strips.push_back(st);
    }
    return s;
}

std::vector<double> RasterizeCurrent(const Grid& g, const Sources& src) {
    std::vector<double> Jz(g.count(), 0.0);
    const double cellArea = g.dx() * g.dy();

    for (const auto& w : src.wires) {
        const double r2 = w.radius * w.radius;
        int covered = 0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double dx = g.x(i) - w.x, dy = g.y(j) - w.y;
                if (dx * dx + dy * dy <= r2) ++covered;
            }
        if (covered == 0) {
            const int i = std::clamp(
                static_cast<int>((w.x + 0.5 * g.lx) / g.dx()), 0, g.nx - 1);
            const int j = std::clamp(
                static_cast<int>((w.y + 0.5 * g.ly) / g.dy()), 0, g.ny - 1);
            Jz[g.idx(i, j)] += w.current / cellArea;
            continue;
        }
        const double jval = w.current / (covered * cellArea);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                const double dx = g.x(i) - w.x, dy = g.y(j) - w.y;
                if (dx * dx + dy * dy <= r2) Jz[g.idx(i, j)] += jval;
            }
    }

    for (const auto& s : src.strips) {
        const double halfW = 0.5 * s.width, halfH = 0.5 * s.height;
        int covered = 0;
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (std::abs(g.x(i) - s.x) <= halfW &&
                    std::abs(g.y(j) - s.y) <= halfH)
                    ++covered;
        if (covered == 0) continue;
        const double jval = s.current / (covered * cellArea);
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i)
                if (std::abs(g.x(i) - s.x) <= halfW &&
                    std::abs(g.y(j) - s.y) <= halfH)
                    Jz[g.idx(i, j)] += jval;
    }

    return Jz;
}

} // namespace mag
