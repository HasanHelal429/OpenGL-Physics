#include "MDCharts.hpp"

#include <algorithm>

namespace md {

MDCharts::MDCharts(fw::Font& font, fw::TextRenderer& textRenderer) : m_tpChart(font, textRenderer), m_rdfChart(font, textRenderer) {
    m_tpChart.SetTitle("Temperature T* (orange) / Pressure P* (blue) vs time");
    m_rdfChart.SetTitle("Radial distribution g(r)");
}

void MDCharts::SetRegions(int tpX, int tpY, int tpW, int tpH, int rdfX, int rdfY, int rdfW, int rdfH) {
    m_tpChart.SetRegion(tpX, tpY, tpW, tpH);
    m_rdfChart.SetRegion(rdfX, rdfY, rdfW, rdfH);
}

void MDCharts::Reset() {
    m_tPoints.clear();
    m_pPoints.clear();
}

void MDCharts::AddSample(double simTime, double temperature, double pressure) {
    m_tPoints.emplace_back(simTime, temperature);
    m_pPoints.emplace_back(simTime, pressure);

    double xMin = m_tPoints.front().x, xMax = simTime;
    double yMin = m_tPoints.front().y, yMax = m_tPoints.front().y;
    for (const glm::dvec2& p : m_tPoints) {
        yMin = std::min(yMin, p.y);
        yMax = std::max(yMax, p.y);
    }
    for (const glm::dvec2& p : m_pPoints) {
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
    xAxis.min = xMin;
    xAxis.max = xMax;
    xAxis.label = "sim time";

    fw::ChartAxis yAxis;
    yAxis.min = yMin - yPad;
    yAxis.max = yMax + yPad;
    yAxis.label = "T* / P*";

    m_tpChart.SetAxes(xAxis, yAxis);

    fw::ChartLine tLine;
    tLine.points = m_tPoints;
    tLine.color = glm::vec4(0.95f, 0.75f, 0.25f, 1.0f);
    tLine.thicknessPx = 2.5f;

    fw::ChartLine pLine;
    pLine.points = m_pPoints;
    pLine.color = glm::vec4(0.4f, 0.8f, 1.0f, 1.0f);
    pLine.thicknessPx = 2.5f;

    m_tpChart.SetContent({tLine, pLine}, {});
}

void MDCharts::SetRdf(const std::vector<double>& g, double rMax) {
    if (g.empty() || rMax <= 0.0) {
        m_rdfChart.SetContent({}, {});
        return;
    }

    const double dr = rMax / static_cast<double>(g.size());
    double yMax = 1.0;
    std::vector<glm::dvec2> points;
    points.reserve(g.size());
    for (size_t i = 0; i < g.size(); ++i) {
        const double r = (static_cast<double>(i) + 0.5) * dr;
        points.emplace_back(r, g[i]);
        yMax = std::max(yMax, g[i]);
    }

    fw::ChartAxis xAxis;
    xAxis.min = 0.0;
    xAxis.max = rMax;
    xAxis.label = "r / sigma";

    fw::ChartAxis yAxis;
    yAxis.min = 0.0;
    yAxis.max = yMax * 1.1;
    yAxis.label = "g(r)";

    m_rdfChart.SetAxes(xAxis, yAxis);

    fw::ChartLine gLine;
    gLine.points = std::move(points);
    gLine.color = glm::vec4(0.55f, 0.95f, 0.55f, 1.0f);
    gLine.thicknessPx = 2.5f;

    m_rdfChart.SetContent({gLine}, {});
}

void MDCharts::Render() {
    m_tpChart.Render();
    m_rdfChart.Render();
}

} // namespace md
