#include "ConvergenceChart.hpp"

#include <algorithm>

namespace hf {

ConvergenceChart::ConvergenceChart(fw::Font& font, fw::TextRenderer& textRenderer) : m_chart(font, textRenderer) {
    m_chart.SetTitle("Total energy convergence");
}

void ConvergenceChart::SetRegion(int x, int y, int width, int height) { m_chart.SetRegion(x, y, width, height); }

void ConvergenceChart::Reset() { m_points.clear(); }

void ConvergenceChart::AddPoint(int iteration, double eTotal) {
    m_points.emplace_back(static_cast<double>(iteration), eTotal);

    double xMin = m_points.front().x, xMax = m_points.front().x;
    double yMin = m_points.front().y, yMax = m_points.front().y;
    for (const glm::dvec2& p : m_points) {
        xMin = std::min(xMin, p.x);
        xMax = std::max(xMax, p.x);
        yMin = std::min(yMin, p.y);
        yMax = std::max(yMax, p.y);
    }
    if (xMax <= xMin) xMax = xMin + 1.0;
    const double yPad = std::max((yMax - yMin) * 0.1, 1e-6);

    fw::ChartAxis xAxis;
    xAxis.scale = fw::AxisScale::Linear;
    xAxis.min = xMin;
    xAxis.max = xMax;
    xAxis.label = "SCF iteration";

    fw::ChartAxis yAxis;
    yAxis.scale = fw::AxisScale::Linear;
    yAxis.min = yMin - yPad;
    yAxis.max = yMax + yPad;
    yAxis.label = "E_total (Ha)";

    m_chart.SetAxes(xAxis, yAxis);

    fw::ChartLine line;
    line.points = m_points;
    line.color = glm::vec4(0.95f, 0.75f, 0.25f, 1.0f);
    line.thicknessPx = 2.5f;
    m_chart.SetContent({line}, {});
}

void ConvergenceChart::Render() { m_chart.Render(); }

} // namespace hf
