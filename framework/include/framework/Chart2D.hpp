#pragma once

#include "framework/Shader.hpp"
#include "framework/Text.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>

#include <string>
#include <vector>

namespace fw {

enum class AxisScale { Linear, Log10, SymLog };

struct ChartAxis {
    AxisScale scale = AxisScale::Linear;
    double min = 0.0;
    double max = 1.0;
    std::string label;
    // SymLog only: values with |value| < linThresh are treated linearly;
    // beyond it, log-scaled. Matches the qualitative shape of matplotlib's
    // symlog (not bit-for-bit identical), sign-preserving so it also works
    // for the all-negative energy case (still ordered correctly).
    double symLogLinThresh = 1.0;
};

struct ChartLine {
    std::vector<glm::dvec2> points; // data space
    glm::vec4 color{1.0f};
    float thicknessPx = 2.0f;
};

// A single vertical tick (the atomic-spectrum "stick" look): one segment at
// x=value spanning [yFrom, yTo] in data space.
struct ChartTick {
    double value = 0.0;
    double yFrom = 0.0;
    double yTo = 1.0;
    glm::vec4 color{1.0f};
    float thicknessPx = 2.0f;
};

// A 2D scientific chart rendered as real OpenGL geometry: axes, gridlines,
// line/tick series, and text labels. All line-ish content (axes, gridlines,
// series, ticks) is batched into a single VBO and drawn with ONE
// glDrawArrays call per frame via a data-driven shader (each vertex carries
// its own signed distance-from-centerline and half-width, so segments of
// different color/thickness can share one draw call) -- lines are triangle
// strips expanded on the CPU, not GL_LINE_STRIP, so width is consistent
// across drivers and edges are antialiased via a smoothstep falloff in the
// fragment shader rather than relying on MSAA.
//
// Uses a shared Font/TextRenderer (pass the same instances to every chart
// in an app to avoid baking multiple font atlases). Operates in a
// top-left-origin, y-down pixel region of the window (SetRegion), matching
// TextRenderer's convention.
class Chart2D {
public:
    Chart2D(Font& font, TextRenderer& textRenderer);
    ~Chart2D();

    Chart2D(const Chart2D&) = delete;
    Chart2D& operator=(const Chart2D&) = delete;

    // x,y,width,height are standard glViewport parameters (window-space,
    // origin bottom-left). Content *within* the region (text, ticks) is
    // placed using a top-left-origin, y-down local coordinate system.
    void SetRegion(int x, int y, int width, int height);
    void SetAxes(ChartAxis xAxis, ChartAxis yAxis);
    void SetTitle(std::string title) { m_title = std::move(title); }

    // Rebuilds all chart geometry (axes/gridlines/series/ticks) from the
    // given content and uploads it to the GPU. Call when data changes
    // (e.g. once per SCF snapshot) -- not required every frame.
    void SetContent(std::vector<ChartLine> lines, std::vector<ChartTick> ticks);

    // Issues the batched draw call plus text labels using whatever content
    // was last set. Call every frame.
    void Render();

private:
    struct LineVertex {
        glm::vec2 position; // pixels, region-local
        glm::vec4 color;
        float dist = 0.0f;
        float halfWidth = 0.0f;
    };

    double TransformX(double value) const;
    double TransformY(double value) const;
    glm::vec2 ToPixels(double xNorm, double yNorm) const;
    void AppendThickLine(std::vector<LineVertex>& out, glm::vec2 a, glm::vec2 b, glm::vec4 color, float halfWidth) const;
    void BuildChrome(std::vector<LineVertex>& out, std::vector<std::pair<std::string, glm::vec2>>& labels) const;

    Font& m_font;
    TextRenderer& m_textRenderer;

    int m_regionX = 0, m_regionY = 0, m_regionWidth = 1, m_regionHeight = 1;
    ChartAxis m_xAxis, m_yAxis;
    std::string m_title;

    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    Shader m_shader;
    GLsizei m_vertexCount = 0;

    std::vector<std::pair<std::string, glm::vec2>> m_labels; // text + region-local pixel position
};

} // namespace fw
