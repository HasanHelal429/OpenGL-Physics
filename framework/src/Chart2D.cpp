#include "framework/Chart2D.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>

namespace fw {

namespace {

const char* kVertexShader = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 aColor;
layout (location = 2) in float aDist;
layout (location = 3) in float aHalfWidth;

uniform mat4 uProjection;

out vec4 vColor;
out float vDist;
out float vHalfWidth;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
    vDist = aDist;
    vHalfWidth = aHalfWidth;
}
)";

const char* kFragmentShader = R"(
#version 460 core
in vec4 vColor;
in float vDist;
in float vHalfWidth;
out vec4 FragColor;

void main() {
    float alpha = 1.0 - smoothstep(vHalfWidth - 1.0, vHalfWidth + 1.0, abs(vDist));
    if (alpha <= 0.001) discard;
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
)";

double ForwardTransform(const ChartAxis& axis, double value) {
    switch (axis.scale) {
        case AxisScale::Linear:
            return (value - axis.min) / (axis.max - axis.min);
        case AxisScale::Log10: {
            const double lo = std::log10(axis.min), hi = std::log10(axis.max);
            return (std::log10(value) - lo) / (hi - lo);
        }
        case AxisScale::SymLog: {
            auto f = [&](double v) { return (v < 0 ? -1.0 : 1.0) * std::log10(1.0 + std::abs(v) / axis.symLogLinThresh); };
            const double lo = f(axis.min), hi = f(axis.max);
            return (f(value) - lo) / (hi - lo);
        }
    }
    return 0.0;
}

double InverseTransform(const ChartAxis& axis, double t) {
    switch (axis.scale) {
        case AxisScale::Linear:
            return axis.min + t * (axis.max - axis.min);
        case AxisScale::Log10: {
            const double lo = std::log10(axis.min), hi = std::log10(axis.max);
            return std::pow(10.0, lo + t * (hi - lo));
        }
        case AxisScale::SymLog: {
            auto f = [&](double v) { return (v < 0 ? -1.0 : 1.0) * std::log10(1.0 + std::abs(v) / axis.symLogLinThresh); };
            const double lo = f(axis.min), hi = f(axis.max);
            const double fval = lo + t * (hi - lo);
            const double sign = fval < 0 ? -1.0 : 1.0;
            return sign * axis.symLogLinThresh * (std::pow(10.0, std::abs(fval)) - 1.0);
        }
    }
    return 0.0;
}

std::string FormatTick(double value) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", value);
    return buf;
}

constexpr int kMarginLeft = 64;
constexpr int kMarginBottom = 36;
constexpr int kMarginRight = 12;
constexpr int kTicksPerAxis = 5;

} // namespace

Chart2D::Chart2D(Font& font, TextRenderer& textRenderer) : m_font(font), m_textRenderer(textRenderer) {
    m_shader = Shader::FromSource(kVertexShader, kFragmentShader);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offsetof(LineVertex, position)));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offsetof(LineVertex, color)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offsetof(LineVertex, dist)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, sizeof(LineVertex), reinterpret_cast<void*>(offsetof(LineVertex, halfWidth)));
    glEnableVertexAttribArray(3);
    glBindVertexArray(0);
}

Chart2D::~Chart2D() {
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void Chart2D::SetRegion(int x, int y, int width, int height) {
    m_regionX = x;
    m_regionY = y;
    m_regionWidth = std::max(width, 1);
    m_regionHeight = std::max(height, 1);
}

void Chart2D::SetAxes(ChartAxis xAxis, ChartAxis yAxis) {
    m_xAxis = xAxis;
    m_yAxis = yAxis;
}

double Chart2D::TransformX(double value) const { return ForwardTransform(m_xAxis, value); }
double Chart2D::TransformY(double value) const { return ForwardTransform(m_yAxis, value); }

glm::vec2 Chart2D::ToPixels(double xNorm, double yNorm) const {
    const float plotX0 = static_cast<float>(kMarginLeft);
    const float plotX1 = static_cast<float>(m_regionWidth - kMarginRight);
    const float plotY0 = m_title.empty() ? 10.0f : 26.0f; // top
    const float plotY1 = static_cast<float>(m_regionHeight - kMarginBottom); // bottom

    const float px = plotX0 + static_cast<float>(xNorm) * (plotX1 - plotX0);
    const float py = plotY1 - static_cast<float>(yNorm) * (plotY1 - plotY0);
    return {px, py};
}

void Chart2D::AppendThickLine(std::vector<LineVertex>& out, glm::vec2 a, glm::vec2 b, glm::vec4 color, float halfWidth) const {
    glm::vec2 dir = b - a;
    const float len = glm::length(dir);
    if (len < 1e-6f) return;
    dir /= len;
    const glm::vec2 perp(-dir.y, dir.x);
    const float extrude = halfWidth + 1.5f;

    const glm::vec2 a0 = a + perp * extrude, a1 = a - perp * extrude;
    const glm::vec2 b0 = b + perp * extrude, b1 = b - perp * extrude;

    const LineVertex va0{a0, color, extrude, halfWidth};
    const LineVertex va1{a1, color, -extrude, halfWidth};
    const LineVertex vb0{b0, color, extrude, halfWidth};
    const LineVertex vb1{b1, color, -extrude, halfWidth};

    out.push_back(va0);
    out.push_back(va1);
    out.push_back(vb0);
    out.push_back(va1);
    out.push_back(vb1);
    out.push_back(vb0);
}

void Chart2D::BuildChrome(std::vector<LineVertex>& out, std::vector<std::pair<std::string, glm::vec2>>& labels) const {
    const glm::vec4 axisColor(0.55f, 0.58f, 0.63f, 1.0f);
    const glm::vec4 gridColor(0.30f, 0.32f, 0.36f, 0.6f);

    const glm::vec2 origin = ToPixels(0.0, 0.0);
    const glm::vec2 topLeft = ToPixels(0.0, 1.0);
    const glm::vec2 bottomRight = ToPixels(1.0, 0.0);

    // Axis border (left + bottom).
    AppendThickLine(out, topLeft, origin, axisColor, 1.0f);
    AppendThickLine(out, origin, bottomRight, axisColor, 1.0f);

    for (int i = 0; i < kTicksPerAxis; ++i) {
        const double t = static_cast<double>(i) / (kTicksPerAxis - 1);

        const glm::vec2 xTickTop = ToPixels(t, 0.0);
        const glm::vec2 xTickBottom = xTickTop + glm::vec2(0.0f, 5.0f);
        AppendThickLine(out, xTickTop, xTickBottom, axisColor, 1.0f);
        if (i > 0) AppendThickLine(out, ToPixels(t, 0.0), ToPixels(t, 1.0), gridColor, 0.5f);

        const std::string xLabel = FormatTick(InverseTransform(m_xAxis, t));
        const float xLabelWidth = TextRenderer::MeasureWidth(m_font, xLabel);
        labels.emplace_back(xLabel, glm::vec2(xTickBottom.x - xLabelWidth * 0.5f, xTickBottom.y + 4.0f));

        const glm::vec2 yTickRight = ToPixels(0.0, t);
        const glm::vec2 yTickLeft = yTickRight - glm::vec2(5.0f, 0.0f);
        AppendThickLine(out, yTickLeft, yTickRight, axisColor, 1.0f);
        if (i > 0) AppendThickLine(out, ToPixels(0.0, t), ToPixels(1.0, t), gridColor, 0.5f);

        const std::string yLabel = FormatTick(InverseTransform(m_yAxis, t));
        const float yLabelWidth = TextRenderer::MeasureWidth(m_font, yLabel);
        labels.emplace_back(yLabel, glm::vec2(yTickLeft.x - yLabelWidth - 4.0f, yTickRight.y - m_font.PixelHeight() * 0.35f));
    }

    if (!m_title.empty()) {
        labels.emplace_back(m_title, glm::vec2(topLeft.x, 4.0f));
    }
    if (!m_xAxis.label.empty()) {
        const float w = TextRenderer::MeasureWidth(m_font, m_xAxis.label);
        labels.emplace_back(m_xAxis.label, glm::vec2((topLeft.x + bottomRight.x) * 0.5f - w * 0.5f,
                                                       static_cast<float>(m_regionHeight) - 4.0f));
    }
}

void Chart2D::SetContent(std::vector<ChartLine> lines, std::vector<ChartTick> ticks) {
    std::vector<LineVertex> vertices;
    m_labels.clear();

    BuildChrome(vertices, m_labels);

    for (const ChartLine& line : lines) {
        for (size_t i = 1; i < line.points.size(); ++i) {
            const glm::vec2 a = ToPixels(TransformX(line.points[i - 1].x), TransformY(line.points[i - 1].y));
            const glm::vec2 b = ToPixels(TransformX(line.points[i].x), TransformY(line.points[i].y));
            AppendThickLine(vertices, a, b, line.color, line.thicknessPx * 0.5f);
        }
    }

    for (const ChartTick& tick : ticks) {
        const glm::vec2 a = ToPixels(TransformX(tick.value), TransformY(tick.yFrom));
        const glm::vec2 b = ToPixels(TransformX(tick.value), TransformY(tick.yTo));
        AppendThickLine(vertices, a, b, tick.color, tick.thicknessPx * 0.5f);
    }

    m_vertexCount = static_cast<GLsizei>(vertices.size());
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(LineVertex)), vertices.data(),
                 GL_DYNAMIC_DRAW);
}

void Chart2D::Render() {
    glViewport(m_regionX, m_regionY, m_regionWidth, m_regionHeight);
    const glm::mat4 projection =
        glm::ortho(0.0f, static_cast<float>(m_regionWidth), static_cast<float>(m_regionHeight), 0.0f);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    if (m_vertexCount > 0) {
        m_shader.Use();
        m_shader.SetMat4("uProjection", projection);
        glBindVertexArray(m_vao);
        glDrawArrays(GL_TRIANGLES, 0, m_vertexCount);
        glBindVertexArray(0);
    }

    m_textRenderer.SetViewport(m_regionWidth, m_regionHeight);
    const glm::vec4 textColor(0.85f, 0.87f, 0.90f, 1.0f);
    for (const auto& [text, pos] : m_labels) {
        m_textRenderer.Draw(m_font, text, pos, textColor);
    }

    glEnable(GL_DEPTH_TEST);
}

} // namespace fw
