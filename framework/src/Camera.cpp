#include "framework/Camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace fw {

glm::vec3 Camera::Position() const {
    const float yawRad = glm::radians(yaw);
    const float pitchRad = glm::radians(pitch);
    glm::vec3 offset;
    offset.x = distance * std::cos(pitchRad) * std::cos(yawRad);
    offset.y = distance * std::sin(pitchRad);
    offset.z = distance * std::cos(pitchRad) * std::sin(yawRad);
    return target + offset;
}

glm::mat4 Camera::ViewMatrix() const {
    return glm::lookAt(Position(), target, glm::vec3(0.0f, 1.0f, 0.0f));
}

glm::mat4 Camera::ProjectionMatrix(float aspectRatio) const {
    return glm::perspective(glm::radians(fovDegrees), aspectRatio, nearPlane, farPlane);
}

void Camera::Orbit(float dx, float dy) {
    const float sensitivity = 0.25f;
    yaw += dx * sensitivity;
    pitch = std::clamp(pitch - dy * sensitivity, -89.0f, 89.0f);
}

void Camera::Zoom(float scrollAmount) {
    distance = std::clamp(distance - scrollAmount * (distance * 0.1f), minDistance, maxDistance);
}

} // namespace fw
