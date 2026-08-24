#include "SpectrumView.hpp"

#include <algorithm>

namespace hf {

namespace {

glm::vec4 ColorForL(int l) {
    switch (l) {
        case 0: return {0.30f, 0.55f, 0.95f, 1.0f}; // s: blue
        case 1: return {0.95f, 0.55f, 0.20f, 1.0f}; // p: orange
        case 2: return {0.35f, 0.75f, 0.35f, 1.0f}; // d: green
        case 3: return {0.90f, 0.30f, 0.35f, 1.0f}; // f: red
        default: return {0.7f, 0.7f, 0.7f, 1.0f};
    }
}

} // namespace

SpectrumView::SpectrumView(fw::Font& font, fw::TextRenderer& textRenderer) : m_chart(font, textRenderer) {
    m_chart.SetTitle("Orbital energy levels");
}

void SpectrumView::SetRegion(int x, int y, int width, int height) { m_chart.SetRegion(x, y, width, height); }

void SpectrumView::Reset() { m_hasRange = false; }

void SpectrumView::UpdateEnergies(const std::map<ShellKey, double>& orbitalEnergies) {
    if (orbitalEnergies.empty()) return;

    double lo = m_hasRange ? m_rangeMin : orbitalEnergies.begin()->second;
    double hi = m_hasRange ? m_rangeMax : orbitalEnergies.begin()->second;
    for (const auto& [nl, eps] : orbitalEnergies) {
        lo = std::min(lo, eps);
        hi = std::max(hi, eps);
    }
    hi = std::min(hi, -1e-3); // keep strictly negative: avoids a degenerate symlog range touching 0
    if (lo >= hi) lo = hi - 1.0;
    m_rangeMin = lo;
    m_rangeMax = hi;
    m_hasRange = true;

    fw::ChartAxis xAxis;
    xAxis.scale = fw::AxisScale::SymLog;
    xAxis.min = m_rangeMin;
    xAxis.max = m_rangeMax;
    xAxis.symLogLinThresh = 0.5;
    xAxis.label = "Orbital energy (Ha)";

    fw::ChartAxis yAxis;
    yAxis.scale = fw::AxisScale::Linear;
    yAxis.min = 0.0;
    yAxis.max = 1.0;

    m_chart.SetAxes(xAxis, yAxis);

    std::vector<fw::ChartTick> ticks;
    ticks.reserve(orbitalEnergies.size());
    for (const auto& [nl, eps] : orbitalEnergies) {
        fw::ChartTick tick;
        tick.value = eps;
        tick.yFrom = 0.08;
        tick.yTo = 0.92;
        tick.color = ColorForL(nl.second);
        tick.thicknessPx = 3.0f;
        ticks.push_back(tick);
    }
    m_chart.SetContent({}, ticks);
}

void SpectrumView::Render() { m_chart.Render(); }

} // namespace hf
