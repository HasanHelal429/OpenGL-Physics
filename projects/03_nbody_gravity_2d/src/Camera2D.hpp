#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace nbody2d {

// Minimal pan/zoom orthographic camera for a flat 2D scene -- deliberately
// not a repurposed 3D fw::Camera (perspective, orbit-based): 2D gravity
// lives in a plane, so an orthographic top-down view with no perspective
// distortion is the natural fit, and building a small project-local
// camera keeps this addition self-contained (see this project's plan doc)
// rather than changing framework/. Reuses fw::ParticleCloud for rendering
// (position.z = 0 for every particle) -- its point-size shader assumes a
// perspective projection's [1][1] term and divides by view-space depth,
// but with every particle at the same depth (kEyeDistance below) that
// just becomes a constant scale factor, so it still renders correctly-
// scaled, constant-size points under this orthographic setup. That factor
// is 1/kEyeDistance on top of the size the shader would otherwise produce
// for an orthographic projection -- kEyeDistance=5 (the original value)
// made every point 5x smaller than intended, small enough that most
// particle counts rendered as sub-pixel points that never rasterized a
// fragment at all. 1.0 cancels that extra shrink exactly (still safely
// inside the projection's [0.01, 100] near/far range for target z=0).
struct Camera2D {
    glm::vec2 center{0.0f};    // world-space center of view
    float halfHeight = 2.0f;   // world-space half-height of view -- the zoom level
    float minHalfHeight = 0.05f;
    float maxHalfHeight = 500.0f;

    static constexpr float kEyeDistance = 1.0f;

    glm::mat4 ViewMatrix() const {
        const glm::vec3 eye(center.x, center.y, kEyeDistance);
        const glm::vec3 target(center.x, center.y, 0.0f);
        return glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    }

    glm::mat4 ProjectionMatrix(float aspectRatio) const {
        const float halfWidth = halfHeight * aspectRatio;
        return glm::ortho(-halfWidth, halfWidth, -halfHeight, halfHeight, 0.01f, 100.0f);
    }

    // dxPixels/dyPixels: mouse delta since last sample (screen space,
    // y-down). viewportHeightPx: current 3D-view viewport height, needed
    // to convert pixel deltas into world-space units at the current zoom.
    void Pan(float dxPixels, float dyPixels, float viewportHeightPx) {
        const float worldPerPixel = (2.0f * halfHeight) / std::max(viewportHeightPx, 1.0f);
        center.x -= dxPixels * worldPerPixel;
        center.y += dyPixels * worldPerPixel; // screen y-down vs. world y-up
    }

    void Zoom(float scrollAmount) {
        halfHeight *= std::pow(0.9f, scrollAmount);
        halfHeight = glm::clamp(halfHeight, minHalfHeight, maxHalfHeight);
    }
};

} // namespace nbody2d
