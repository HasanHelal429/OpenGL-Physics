#pragma once

#include "framework/Chart2D.hpp"

#include <vector>

namespace nbody {

// Energy and angular-momentum drift (%, relative to the values captured at
// scenario load) vs. simulation time -- the live, always-on version of the
// conservation checks the Python project's notebooks ran once per run.
class DriftChart {
public:
    DriftChart(fw::Font& font, fw::TextRenderer& textRenderer);

    void SetRegion(int x, int y, int width, int height);
    void Reset(); // call when loading a new scenario
    void AddPoint(double simTime, double energyDriftPct, double angularMomentumDriftPct);
    void Render();

private:
    fw::Chart2D m_chart;
    std::vector<glm::dvec2> m_energyPoints;
    std::vector<glm::dvec2> m_lPoints;
};

} // namespace nbody
