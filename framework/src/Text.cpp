#include "framework/Text.hpp"

#define STB_TRUETYPE_IMPLEMENTATION
#include <stb_truetype.h>

#include <glm/gtc/matrix_transform.hpp>

#include <fstream>
#include <stdexcept>

namespace fw {

namespace {

const char* kVertexShader = R"(
#version 460 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aUV;

uniform mat4 uProjection;

out vec2 vUV;

void main() {
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    vUV = aUV;
}
)";

const char* kFragmentShader = R"(
#version 460 core
in vec2 vUV;
out vec4 FragColor;

uniform sampler2D uAtlas;
uniform vec4 uColor;

void main() {
    float alpha = texture(uAtlas, vUV).r;
    FragColor = vec4(uColor.rgb, uColor.a * alpha);
}
)";

std::vector<unsigned char> ReadFileBytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Font::FromFile: failed to open " + path.string());
    }
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    if (!file.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Font::FromFile: failed to read " + path.string());
    }
    return bytes;
}

} // namespace

Font Font::FromFile(const std::filesystem::path& ttfPath, float pixelHeight) {
    const std::vector<unsigned char> ttfBytes = ReadFileBytes(ttfPath);

    Font font;
    font.m_pixelHeight = pixelHeight;
    font.m_charData.resize(kNumChars);

    std::vector<unsigned char> bitmap(static_cast<size_t>(kAtlasSize) * kAtlasSize);
    const int result = stbtt_BakeFontBitmap(ttfBytes.data(), 0, pixelHeight, bitmap.data(), kAtlasSize, kAtlasSize,
                                             kFirstChar, kNumChars, font.m_charData.data());
    if (result < 0) {
        throw std::runtime_error("Font::FromFile: glyphs did not fit the atlas for " + ttfPath.string());
    }

    glGenTextures(1, &font.m_atlasTexture);
    glBindTexture(GL_TEXTURE_2D, font.m_atlasTexture);
    GLint prevAlignment = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &prevAlignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, kAtlasSize, kAtlasSize, 0, GL_RED, GL_UNSIGNED_BYTE, bitmap.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, prevAlignment);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, 0);

    return font;
}

Font::~Font() {
    if (m_atlasTexture) {
        glDeleteTextures(1, &m_atlasTexture);
    }
}

Font::Font(Font&& other) noexcept
    : m_atlasTexture(other.m_atlasTexture), m_pixelHeight(other.m_pixelHeight), m_charData(std::move(other.m_charData)) {
    other.m_atlasTexture = 0;
}

Font& Font::operator=(Font&& other) noexcept {
    if (this != &other) {
        if (m_atlasTexture) glDeleteTextures(1, &m_atlasTexture);
        m_atlasTexture = other.m_atlasTexture;
        m_pixelHeight = other.m_pixelHeight;
        m_charData = std::move(other.m_charData);
        other.m_atlasTexture = 0;
    }
    return *this;
}

TextRenderer::TextRenderer() {
    m_shader = Shader::FromSource(kVertexShader, kFragmentShader);

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

TextRenderer::~TextRenderer() {
    if (m_vbo) glDeleteBuffers(1, &m_vbo);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
}

void TextRenderer::SetViewport(int width, int height) {
    // Top-left origin, y-down: matches stb_truetype's native baseline
    // convention directly, no flip needed when placing glyph quads.
    m_projection = glm::ortho(0.0f, static_cast<float>(width), static_cast<float>(height), 0.0f);
}

void TextRenderer::Draw(const Font& font, std::string_view text, glm::vec2 position, glm::vec4 color, float scale) {
    if (text.empty()) return;

    std::vector<float> vertices;
    vertices.reserve(text.size() * 6 * 4);

    float cursorX = 0.0f;
    float cursorY = 0.0f;

    for (char c : text) {
        if (c < Font::kFirstChar || c >= Font::kFirstChar + Font::kNumChars) {
            continue;
        }

        stbtt_aligned_quad q;
        stbtt_GetBakedQuad(font.m_charData.data(), Font::kAtlasSize, Font::kAtlasSize, c - Font::kFirstChar, &cursorX,
                            &cursorY, &q, 1);

        const float x0 = position.x + q.x0 * scale;
        const float x1 = position.x + q.x1 * scale;
        const float y0 = position.y + q.y0 * scale;
        const float y1 = position.y + q.y1 * scale;

        // Two triangles: (x0,y0)-(x1,y0)-(x1,y1) and (x0,y0)-(x1,y1)-(x0,y1).
        const float quad[6][4] = {
            {x0, y0, q.s0, q.t0}, {x1, y0, q.s1, q.t0}, {x1, y1, q.s1, q.t1},
            {x0, y0, q.s0, q.t0}, {x1, y1, q.s1, q.t1}, {x0, y1, q.s0, q.t1},
        };
        for (const auto& v : quad) {
            vertices.insert(vertices.end(), v, v + 4);
        }
    }

    if (vertices.empty()) return;

    m_shader.Use();
    m_shader.SetMat4("uProjection", m_projection);
    m_shader.SetVec4("uColor", color);
    m_shader.SetInt("uAtlas", 0);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, font.AtlasTexture());

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(),
                 GL_DYNAMIC_DRAW);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size() / 4));
    glDisable(GL_BLEND);

    glBindVertexArray(0);
}

float TextRenderer::MeasureWidth(const Font& font, std::string_view text, float scale) {
    float cursorX = 0.0f;
    float cursorY = 0.0f;
    for (char c : text) {
        if (c < Font::kFirstChar || c >= Font::kFirstChar + Font::kNumChars) {
            continue;
        }
        stbtt_aligned_quad q;
        stbtt_GetBakedQuad(font.m_charData.data(), Font::kAtlasSize, Font::kAtlasSize, c - Font::kFirstChar, &cursorX,
                            &cursorY, &q, 1);
    }
    return cursorX * scale;
}

} // namespace fw
