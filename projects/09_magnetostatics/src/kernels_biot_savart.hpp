#pragma once

#include <string>

namespace mag::kernels {

// Biot-Savart over a slice plane: one thread per grid cell (i, j), inner loop
// over every segment. Segments carry (I dl) already, so the sum is just
//   B += (mu0/4pi) * cross(seg.dl, R) / |R|^3,   R = fieldPoint - seg.mid
//
// Slice mapping (uPlane): 0 = xy at z=uOffset, 1 = xz at y=uOffset,
// 2 = yz at x=uOffset -- matches SlicePlane::Point.
inline std::string BiotSavart() {
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

struct Seg { vec4 mid; vec4 dl; };   // std430: .xyz used, .w padding
layout(std430, binding = 0) readonly buffer Segs { Seg segs[]; };
layout(std430, binding = 1) writeonly buffer Out { vec4 B[]; };  // (Bx,By,Bz,0)

uniform int   uNx;
uniform int   uNy;
uniform float uX0;    // world x(0) = uX0 + (i+0.5)*uDx  -- grid axis 0
uniform float uDx;
uniform float uY0;    // grid axis 1
uniform float uDy;
uniform int   uNseg;
uniform int   uPlane;
uniform float uOffset;
uniform float uK;     // mu0 / (4 pi)

void main() {
    int i = int(gl_GlobalInvocationID.x);
    int j = int(gl_GlobalInvocationID.y);
    if (i >= uNx || j >= uNy) return;

    float a = uX0 + (float(i) + 0.5) * uDx;
    float b = uY0 + (float(j) + 0.5) * uDy;
    vec3 p;
    if      (uPlane == 1) p = vec3(a, uOffset, b);
    else if (uPlane == 2) p = vec3(uOffset, a, b);
    else                  p = vec3(a, b, uOffset);

    vec3 acc = vec3(0.0);
    for (int s = 0; s < uNseg; ++s) {
        vec3 R = p - segs[s].mid.xyz;
        float r2 = dot(R, R);
        float inv = inversesqrt(max(r2, 1e-20));
        float inv3 = inv * inv * inv;
        acc += cross(segs[s].dl.xyz, R) * inv3;
    }
    B[j * uNx + i] = vec4(uK * acc, 0.0);
}
)";
}

} // namespace mag::kernels
