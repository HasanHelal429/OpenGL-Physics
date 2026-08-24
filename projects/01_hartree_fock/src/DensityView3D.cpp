#include "DensityView3D.hpp"

#include "RadialFieldSampler.hpp"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>

namespace hf {

namespace {

const char* kShellVertexShader = R"(
#version 460 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vNormalWorld;
out vec3 vWorldPos;

void main() {
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormalWorld = mat3(uModel) * aNormal; // uniform scale here, no inverse-transpose needed
    gl_Position = uProjection * uView * worldPos;
}
)";

// Rim-lit translucent shell: brighter/more opaque at grazing angles
// (Fresnel-style), so a sphere reads as a hollow shell rather than a flat
// disc -- the look a "surface of constant probability" should have.
const char* kShellFragmentShader = R"(
#version 460 core
in vec3 vNormalWorld;
in vec3 vWorldPos;
out vec4 FragColor;

uniform vec3 uCameraPos;
uniform vec3 uColor;
uniform float uAlpha;

void main() {
    vec3 N = normalize(vNormalWorld);
    vec3 V = normalize(uCameraPos - vWorldPos);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 2.0);
    float alpha = uAlpha * mix(0.15, 1.0, fresnel);

    vec3 lightDir = normalize(vec3(0.4, 0.8, 0.5));
    float diff = max(dot(N, lightDir), 0.0);
    vec3 color = uColor * (0.35 + 0.65 * diff);

    FragColor = vec4(color, alpha);
}
)";

} // namespace

DensityView3D::DensityView3D() {
    m_camera.target = glm::vec3(0.0f);
    m_camera.distance = 8.0f;
    m_camera.pitch = 15.0f;
    m_camera.minDistance = 1.0f;
    m_camera.maxDistance = 40.0f;

    m_sphereMesh = fw::Mesh::Upload(fw::GenerateIcosphere(2));
    m_shellShader = fw::Shader::FromSource(kShellVertexShader, kShellFragmentShader);
}

double DensityView3D::DisplayRadius(double physicalR) {
    // Log-compressive remap: physical r spans ~1e-5 (core, heavy Z) to
    // ~1e2 Bohr (outer grid edge); this keeps both core and valence shells
    // simultaneously visible in one 3D view instead of the core collapsing
    // to an invisible point at linear scale.
    constexpr double kR0 = 0.3;
    constexpr double kScale = 2.5;
    return kScale * std::log10(1.0 + physicalR / kR0);
}

void DensityView3D::UpdateField(const std::vector<double>& r, const std::vector<double>& rho, double isoThreshold,
                                 int particleCount) {
    const std::vector<glm::vec3> physicalPoints = SampleRadialParticles(r, rho, particleCount);
    std::vector<fw::ParticleInstance> instances;
    instances.reserve(physicalPoints.size());
    for (const glm::vec3& p : physicalPoints) {
        const float physR = glm::length(p);
        if (physR < 1e-9f) continue;
        const glm::vec3 dir = p / physR;
        const float dispR = static_cast<float>(DisplayRadius(physR));

        fw::ParticleInstance inst;
        inst.position = dir * dispR;
        inst.color = glm::vec4(0.45f, 0.65f, 1.0f, 0.35f);
        inst.size = 0.035f;
        instances.push_back(inst);
    }
    m_particles.SetParticles(instances);

    const std::vector<double> radiiPhysical = FindIsosurfaceRadii(r, rho, isoThreshold);
    m_isoRadiiDisplay.clear();
    m_isoRadiiDisplay.reserve(radiiPhysical.size());
    for (double rp : radiiPhysical) {
        m_isoRadiiDisplay.push_back(DisplayRadius(rp));
    }
}

void DensityView3D::RenderIsosurface(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPos) {
    if (m_isoRadiiDisplay.empty()) return;

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    m_shellShader.Use();
    m_shellShader.SetMat4("uView", view);
    m_shellShader.SetMat4("uProjection", projection);
    m_shellShader.SetVec3("uCameraPos", cameraPos);

    const size_t n = m_isoRadiiDisplay.size();
    for (size_t i = n; i-- > 0;) { // largest radius first: back-to-front given camera outside all shells
        const float radius = static_cast<float>(m_isoRadiiDisplay[i]);
        const glm::mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(radius));
        const float t = n > 1 ? static_cast<float>(i) / static_cast<float>(n - 1) : 0.0f;
        const glm::vec3 color = glm::mix(glm::vec3(1.0f, 0.55f, 0.3f), glm::vec3(0.35f, 0.55f, 1.0f), t);

        m_shellShader.SetMat4("uModel", model);
        m_shellShader.SetVec3("uColor", color);
        m_shellShader.SetFloat("uAlpha", 0.35f);
        m_sphereMesh.Draw();
    }

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void DensityView3D::Render(int viewportX, int viewportY, int viewportWidth, int viewportHeight) {
    glViewport(viewportX, viewportY, viewportWidth, viewportHeight);
    glEnable(GL_DEPTH_TEST);

    const float aspect = static_cast<float>(viewportWidth) / static_cast<float>(std::max(viewportHeight, 1));
    const glm::mat4 view = m_camera.ViewMatrix();
    const glm::mat4 projection = m_camera.ProjectionMatrix(aspect);

    if (m_mode == DensityViewMode::ParticleCloud) {
        m_particles.Draw(view, projection, static_cast<float>(viewportHeight));
    } else {
        RenderIsosurface(view, projection, m_camera.Position());
    }
}

} // namespace hf
