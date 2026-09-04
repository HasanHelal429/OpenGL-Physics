#pragma once

#include <string>

// GLSL compute shader for the Lienard-Wiechert field on a 2D slice grid.
// One thread per cell: for each charge, solve the retarded time (Newton +
// bisection, warm-started from last frame's per-cell value), evaluate the
// velocity + acceleration field split, and superpose. A line-for-line port
// of LwFields.hpp / retarded_fields.py's _lw_fields_jit_core.
//
// Bindings:  0 outE (vec4 Ex,Ey,Ez,|E|)   1 outB (vec4)   2 tretBuf (float,
// nx*ny*nCharge, warm-start; sentinel 1e30 = "no guess")
//
// Charge params (max 8), one array slot each:
//   uKind[c]   0 static  1 uniform  3 linear_oscillator  4 figure8  else circular
//   uCenter[c] xyz = centre,          w = q
//   uV0[c]     xyz = uniform velocity, w = omega
//   uGeom[c]   x = radius, y = amplitude, z = phase
//   uAxis[c]   xyz = oscillator axis

namespace lw::kernels {

inline std::string LwField() {
    return R"(#version 460 core
layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) writeonly buffer OutE { vec4 outE[]; };
layout(std430, binding = 1) writeonly buffer OutB { vec4 outB[]; };
layout(std430, binding = 2) buffer Tret { float tret[]; };

uniform int   uNx;
uniform int   uNy;
uniform float uX0;      // world x(0)
uniform float uDx;
uniform float uY0;
uniform float uDy;
uniform float uZ;       // slice offset
uniform float uT;
uniform float uC;
uniform float uEps0;
uniform int   uNCharge;
uniform int   uKind[8];
uniform vec4  uCenter[8];
uniform vec4  uV0[8];
uniform vec4  uGeom[8];
uniform vec4  uAxis[8];

void pathRVA(int c, float tau, out vec3 rr, out vec3 vv, out vec3 aa) {
    int k = uKind[c];
    vec3 ctr = uCenter[c].xyz;
    float R = uGeom[c].x, A = uGeom[c].y, ph = uGeom[c].z;
    float w = uV0[c].w;
    float p = w * tau + ph;
    if (k == 0) { rr = ctr; vv = vec3(0.0); aa = vec3(0.0); }
    else if (k == 1) { rr = ctr + uV0[c].xyz * tau; vv = uV0[c].xyz; aa = vec3(0.0); }
    else if (k == 3) {
        vec3 ax = uAxis[c].xyz;
        rr = ctr + ax * (A * sin(p));
        vv = ax * (A * w * cos(p));
        aa = ax * (-A * w * w * sin(p));
    } else if (k == 4) {
        rr = ctr + vec3(R * sin(p), A * sin(2.0 * p), 0.0);
        vv = vec3(R * w * cos(p), 2.0 * A * w * cos(2.0 * p), 0.0);
        aa = vec3(-R * w * w * sin(p), -4.0 * A * w * w * sin(2.0 * p), 0.0);
    } else {
        rr = ctr + vec3(R * cos(p), R * sin(p), 0.0);
        vv = vec3(-R * w * sin(p), R * w * cos(p), 0.0);
        aa = vec3(-R * w * w * cos(p), -R * w * w * sin(p), 0.0);
    }
}

float retardedTime(int c, vec3 x, float guess) {
    float window = 1.0;
    float tLo = uT - window;
    vec3 r, v, a;
    for (int e = 0; e < 64; ++e) {
        pathRVA(c, tLo, r, v, a);
        if (uC * (uT - tLo) - length(x - r) > 0.0) break;
        window *= 2.0;
        tLo = uT - window;
    }
    float tHi = uT;
    float tRet = (guess < 1e29 && tLo < guess && guess < tHi)
                     ? guess : 0.5 * (tLo + tHi);
    for (int it = 0; it < 80; ++it) {
        pathRVA(c, tRet, r, v, a);
        vec3 Rv = x - r;
        float R = length(Rv);
        float gv = uC * (uT - tRet) - R;
        if (gv > 0.0) tLo = tRet;
        else if (gv < 0.0) tHi = tRet;
        else return tRet;
        vec3 n = Rv / R;
        float gp = -uC + dot(n, v);
        float step = gp != 0.0 ? gv / gp : 0.0;
        float tN = tRet - step;
        if (tLo < tN && tN < tHi) {
            if (abs(step) < 1e-6 * max(1.0, abs(tRet))) return tN;
            tRet = tN;
        } else {
            tRet = 0.5 * (tLo + tHi);
        }
    }
    return tRet;
}

void main() {
    int i = int(gl_GlobalInvocationID.x), j = int(gl_GlobalInvocationID.y);
    if (i >= uNx || j >= uNy) return;
    int cell = j * uNx + i;
    vec3 x = vec3(uX0 + (float(i) + 0.5) * uDx, uY0 + (float(j) + 0.5) * uDy, uZ);

    vec3 Eacc = vec3(0.0), Bacc = vec3(0.0);
    for (int c = 0; c < uNCharge; ++c) {
        int slot = cell * uNCharge + c;
        float tRet = retardedTime(c, x, tret[slot]);
        tret[slot] = tRet;

        vec3 r, v, a;
        pathRVA(c, tRet, r, v, a);
        vec3 Rv = x - r;
        float R = length(Rv);
        vec3 n = Rv / R;
        vec3 beta = v / uC;
        vec3 betaDot = a / uC;
        float kappa = 1.0 - dot(n, beta);
        float denom = kappa * kappa * kappa;
        float b2 = dot(beta, beta);
        vec3 nearT = (n - beta) * (1.0 - b2) / (R * R * denom);
        vec3 farT = cross(n, cross(n - beta, betaDot)) / (uC * R * denom);
        vec3 E = (uCenter[c].w / (4.0 * 3.14159265358979 * uEps0)) * (nearT + farT);
        Eacc += E;
        Bacc += cross(n, E) / uC;
    }
    outE[cell] = vec4(Eacc, length(Eacc));
    outB[cell] = vec4(Bacc, 0.0);
}
)";
}

} // namespace lw::kernels
