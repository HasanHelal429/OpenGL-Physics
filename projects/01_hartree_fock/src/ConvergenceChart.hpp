#pragma once

#include "framework/Chart2D.hpp"

#include <vector>

namespace hf {

// E_total vs. iteration, replacing the old ImPlot convergence panel.
class ConvergenceChart {
public:
    ConvergenceChart(fw::Font& font, fw::TextRenderer& textRenderer);

    void SetRegion(int x, int y, int width, int height);
    void Reset(); // call when starting a new SCF run

    // Call when a new snapshot arrives, not every frame.
    void AddPoint(int iteration, double eTotal);

    void Render();

private:
    fw::Chart2D m_chart;
    std::vector<glm::dvec2> m_points;
};

} // namespace hf
