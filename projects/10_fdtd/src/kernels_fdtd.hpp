#pragma once

#include <string>

// GLSL compute-shader source for the TM^z FDTD update. Fields are float
// SSBOs, row-major, index = j*nx + i. Bindings:
//   0 Ez   1 Hx   2 Hy   3 ca   4 cb   5 EzPrev   6 Src (vec2: idx, value)
//
// These are line-for-line the CPU Fdtd2D::UpdateH / UpdateE / ApplyMur /
// InjectSources; the --selftest cross-check steps both and compares Ez.

namespace fdtd::kernels {

inline std::string UpdateH() {
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) readonly buffer Ez { float ez[]; };
layout(std430, binding = 1) buffer Hx { float hx[]; };
layout(std430, binding = 2) buffer Hy { float hy[]; };
uniform int uNx; uniform int uNy;
uniform float uCx;   // dt*(1/mu)/dx
uniform float uCy;   // dt*(1/mu)/dy
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= uNx || j >= uNy) return;
    int p = j * uNx + i;
    if (j < uNy - 1) hx[p] -= uCy * (ez[p + uNx] - ez[p]);
    if (i < uNx - 1) hy[p] += uCx * (ez[p + 1] - ez[p]);
}
)";
}

inline std::string UpdateE() {
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 1) readonly buffer Hx { float hx[]; };
layout(std430, binding = 2) readonly buffer Hy { float hy[]; };
layout(std430, binding = 3) readonly buffer Ca { float ca[]; };
layout(std430, binding = 4) readonly buffer Cb { float cb[]; };
uniform int uNx; uniform int uNy;
uniform float uInvDx; uniform float uInvDy;
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i < 1 || j < 1 || i >= uNx - 1 || j >= uNy - 1) return;
    int p = j * uNx + i;
    float curl = (hy[p] - hy[p - 1]) * uInvDx - (hx[p] - hx[p - uNx]) * uInvDy;
    ez[p] = ca[p] * ez[p] + cb[p] * curl;
}
)";
}

inline std::string CopyPrev() {
    return R"(#version 460 core
layout(local_size_x = 256) in;
layout(std430, binding = 0) readonly buffer Ez { float ez[]; };
layout(std430, binding = 5) writeonly buffer EzPrev { float ezPrev[]; };
uniform int uN;
void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k < uint(uN)) ezPrev[k] = ez[k];
}
)";
}

inline std::string Inject() {
    return R"(#version 460 core
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 4) readonly buffer Cb { float cb[]; };
layout(std430, binding = 6) readonly buffer Src { vec2 src[]; };  // (idx, value)
uniform int uCount;
void main() {
    uint k = gl_GlobalInvocationID.x;
    if (k >= uint(uCount)) return;
    int p = int(src[k].x);
    ez[p] += cb[p] * src[k].y;
}
)";
}

inline std::string Mur() {
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(std430, binding = 0) buffer Ez { float ez[]; };
layout(std430, binding = 5) readonly buffer EzPrev { float ezPrev[]; };
uniform int uNx; uniform int uNy;
uniform float uCoefX; uniform float uCoefY;
void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= uNx || j >= uNy) return;
    int p = j * uNx + i;
    // x edges first, then y edges (y wins at the corners) -- matches the CPU order.
    if (i == 0)
        ez[p] = ezPrev[j * uNx + 1] + uCoefX * (ez[j * uNx + 1] - ezPrev[p]);
    else if (i == uNx - 1)
        ez[p] = ezPrev[j * uNx + uNx - 2] + uCoefX * (ez[j * uNx + uNx - 2] - ezPrev[p]);
    if (j == 0)
        ez[p] = ezPrev[uNx + i] + uCoefY * (ez[uNx + i] - ezPrev[p]);
    else if (j == uNy - 1)
        ez[p] = ezPrev[(uNy - 2) * uNx + i] + uCoefY * (ez[(uNy - 2) * uNx + i] - ezPrev[p]);
}
)";
}

} // namespace fdtd::kernels
