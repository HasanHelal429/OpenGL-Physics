#pragma once

#include "framework/Camera.hpp"
#include "framework/Mesh.hpp"
#include "framework/ParticleCloud.hpp"
#include "framework/Shader.hpp"

#include <vector>

namespace hf {

enum class DensityViewMode { ParticleCloud, Isosurface };

// The 3D "wave function" view: this solver's density rho(r) is isotropic,
// so both modes are exact closed-form geometry (see RadialFieldSampler) --
// no marching cubes or raymarching. Physical radii (Bohr) are remapped
// through a log-compressive DisplayRadius so both the compact core (r~1/Z)
// and the much larger valence shells (r~few Bohr) are simultaneously
// legible in one 3D view.
class DensityView3D {
public:
    DensityView3D();

    fw::Camera& GetCamera() { return m_camera; }
    void SetMode(DensityViewMode mode) { m_mode = mode; }
    DensityViewMode Mode() const { return m_mode; }

    // Rebuilds sampled points / isosurface shell radii from a new
    // snapshot. Call when a new snapshot arrives, not every frame.
    void UpdateField(const std::vector<double>& r, const std::vector<double>& rho, double isoThreshold, int particleCount = 60000);

    void Render(int viewportX, int viewportY, int viewportWidth, int viewportHeight);

private:
    static double DisplayRadius(double physicalR);

    void RenderIsosurface(const glm::mat4& view, const glm::mat4& projection, const glm::vec3& cameraPos);

    fw::Camera m_camera;
    fw::ParticleCloud m_particles;
    fw::Mesh m_sphereMesh;
    fw::Shader m_shellShader;

    std::vector<double> m_isoRadiiDisplay; // ascending
    DensityViewMode m_mode = DensityViewMode::ParticleCloud;
};

} // namespace hf
