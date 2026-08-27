#include "DriftChart.hpp"

#include <algorithm>

namespace nbody2d {

DriftChart::DriftChart(fw::Font& font, fw::TextRenderer& textRenderer) : m_chart(font, textRenderer) {
    m_chart.SetTitle("Energy (orange) / L (blue) drift, %");
}

void DriftChart::SetRegion(int x, int y, int width, int height) { m_chart.SetRegion(x, y, width, height); }

void DriftChart::Reset() {
    m_energyPoints.clear();
    m_lPoints.clear();
}

void DriftChart::AddPoint(double simTime, double energyDriftPct, double angularMomentumDriftPct) {
    m_energyPoints.emplace_back(simTime, energyDriftPct);
    m_lPoints.emplace_back(simTime, angularMomentumDriftPct);

    double xMin = m_energyPoints.front().x, xMax = simTime;
    double yMin = m_energyPoints.front().y, yMax = m_energyPoints.front().y;
    for (const glm::dvec2& p : m_energyPoints) {
        xMin = std::min(xMin, p.x);
        xMax = std::max(xMax, p.x);
        yMin = std::min(yMin, p.y);
        yMax = std::max(yMax, p.y);
    }
    for (const glm::dvec2& p : m_lPoints) {
        yMin = std::min(yMin, p.y);
        yMax = std::max(yMax, p.y);
    }
    if (xMax <= xMin) xMax = xMin + 1.0;
    if (yMax <= yMin) {
        yMin -= 1.0;
        yMax += 1.0;
    }
    const double yPad = std::max((yMax - yMin) * 0.1, 1e-6);

    fw::ChartAxis xAxis;
    xAxis.scale = fw::AxisScale::Linear;
    xAxis.min = xMin;
    xAxis.max = xMax;
    xAxis.label = "sim time";

    fw::ChartAxis yAxis;
    yAxis.scale = fw::AxisScale::Linear;
    yAxis.min = yMin - yPad;
    yAxis.max = yMax + yPad;
    yAxis.label = "drift %";

    m_chart.SetAxes(xAxis, yAxis);

    fw::ChartLine energyLine;
    energyLine.points = m_energyPoints;
    energyLine.color = glm::vec4(0.95f, 0.75f, 0.25f, 1.0f);
    energyLine.thicknessPx = 2.5f;

    fw::ChartLine lLine;
    lLine.points = m_lPoints;
    lLine.color = glm::vec4(0.4f, 0.8f, 1.0f, 1.0f);
    lLine.thicknessPx = 2.5f;

    m_chart.SetContent({energyLine, lLine}, {});
}

void DriftChart::Render() { m_chart.Render(); }

} // namespace nbody2d
