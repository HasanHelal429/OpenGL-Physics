#pragma once

#include "framework/Shader.hpp"

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <stb_truetype.h>

#include <filesystem>
#include <string_view>
#include <vector>

namespace fw {

// A single baked font atlas (fixed pixel size) loaded from a .ttf file,
// covering ASCII 32..126. Scale text at draw time rather than re-baking for
// different sizes — good enough for chart labels/HUD text.
class Font {
public:
    static Font FromFile(const std::filesystem::path& ttfPath, float pixelHeight);

    Font() = default;
    ~Font();

    Font(const Font&) = delete;
    Font& operator=(const Font&) = delete;
    Font(Font&& other) noexcept;
    Font& operator=(Font&& other) noexcept;

    GLuint AtlasTexture() const { return m_atlasTexture; }
    float PixelHeight() const { return m_pixelHeight; }

private:
    friend class TextRenderer;

    static constexpr int kAtlasSize = 512;
    static constexpr int kFirstChar = 32;
    static constexpr int kNumChars = 96;

    GLuint m_atlasTexture = 0;
    float m_pixelHeight = 0.0f;
    std::vector<stbtt_bakedchar> m_charData; // size kNumChars
};

// Draws text baked from a Font as textured quads. Uses a top-left-origin,
// y-down pixel coordinate system local to whatever viewport SetViewport was
// last called with -- matches stb_truetype's native baseline convention
// directly (no coordinate flip needed) and matches how most UI/chart text
// placement is naturally reasoned about ("this label starts here, flows
// right and down").
class TextRenderer {
public:
    TextRenderer();
    ~TextRenderer();

    TextRenderer(const TextRenderer&) = delete;
    TextRenderer& operator=(const TextRenderer&) = delete;

    void SetViewport(int width, int height);

    // position is the text's baseline-left anchor, in this renderer's
    // top-left-origin/y-down pixel space.
    void Draw(const Font& font, std::string_view text, glm::vec2 position, glm::vec4 color, float scale = 1.0f);

    // Width in pixels of `text` at `font`'s baked size * scale, without drawing it.
    static float MeasureWidth(const Font& font, std::string_view text, float scale = 1.0f);

private:
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    Shader m_shader;
    glm::mat4 m_projection{1.0f};
};

} // namespace fw
