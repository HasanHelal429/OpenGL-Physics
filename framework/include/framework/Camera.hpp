#pragma once

#include <glm/glm.hpp>

namespace fw {

// Orbit (arcball-style) camera: rotates around a target point at a fixed
// distance. Good default for looking at particle systems / N-body scenes.
class Camera {
public:
    glm::vec3 target = glm::vec3(0.0f);
    float distance = 10.0f;
    float yaw = -90.0f;   // degrees, around Y
    float pitch = 20.0f;  // degrees, clamped to (-89, 89)
    float fovDegrees = 45.0f;
    float nearPlane = 0.05f;
    float farPlane = 500.0f;

    float minDistance = 0.5f;
    float maxDistance = 500.0f;

    glm::vec3 Position() const;
    glm::mat4 ViewMatrix() const;
    glm::mat4 ProjectionMatrix(float aspectRatio) const;

    // Call from mouse-drag handling: dx/dy in pixels since last sample.
    void Orbit(float dx, float dy);
    // Call from scroll handling.
    void Zoom(float scrollAmount);
};

} // namespace fw
