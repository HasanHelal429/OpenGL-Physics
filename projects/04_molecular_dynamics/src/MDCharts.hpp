#pragma once

#include "framework/Chart2D.hpp"

#include <vector>

namespace md {

// Two live charts, laid out side by side in a caller-provided strip:
// temperature (orange) and pressure (blue) vs. simulation time on the
// left, the radial distribution function g(r) -- the standard MD
// structural fingerprint that distinguishes gas (g~1 everywhere), liquid
// (a decaying series of shells), and solid (sharp lattice peaks) -- on the
// right.
class MDCharts {
public:
    MDCharts(fw::Font& font, fw::TextRenderer& textRenderer);

    void SetRegions(int tpX, int tpY, int tpW, int tpH, int rdfX, int rdfY, int rdfW, int rdfH);
    void Reset(); // call when loading a new scenario

    void AddSample(double simTime, double temperature, double pressure);
    void SetRdf(const std::vector<double>& g, double rMax);

    void Render();

private:
    fw::Chart2D m_tpChart;
    fw::Chart2D m_rdfChart;

    std::vector<glm::dvec2> m_tPoints;
    std::vector<glm::dvec2> m_pPoints;
};

} // namespace md
