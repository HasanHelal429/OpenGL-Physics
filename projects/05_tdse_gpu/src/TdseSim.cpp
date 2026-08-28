#include "TdseSim.hpp"

#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace tdse {

namespace {
constexpr double kPi = 3.14159265358979323846;

bool IsPow2(int n) { return n >= 2 && (n & (n - 1)) == 0; }

const char* kViewVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";

const char* kViewFrag = R"(#version 460 core
out vec4 FragColor;

layout(std430, binding = 0) readonly buffer Psi  { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer Pot  { float pot[]; };
layout(std430, binding = 2) readonly buffer Stat { uint sMaxBits; }; // max |psi|^2 this frame

uniform vec2 uRes;
uniform int uN;
uniform float uLx;
uniform float uLy;
uniform float uPixPerUnit;
uniform vec2 uPanPix;
uniform int uMode;
uniform float uGain;   // linear push before the perceptual curve
uniform float uGamma;  // <1 lifts low density so tails/fringe minima stay visible
uniform float uVref;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Polynomial fit of matplotlib 'magma' (Bhaskaran / Stefan-Gustavson style).
vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(-0.002136, -0.000750, -0.005386);
    const vec3 c1 = vec3(0.251723, 0.677631, 2.494027);
    const vec3 c2 = vec3(8.353717, -3.577720, 0.311613);
    const vec3 c3 = vec3(-27.668733, 14.264731, -13.649213);
    const vec3 c4 = vec3(52.176140, -27.943606, 12.944169);
    const vec3 c5 = vec3(-50.768525, 29.046583, 4.234153);
    const vec3 c6 = vec3(18.655705, -11.489774, -5.601962);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}

void main() {
    float wx = (gl_FragCoord.x - 0.5 * uRes.x - uPanPix.x) / uPixPerUnit;
    float wy = (gl_FragCoord.y - 0.5 * uRes.y - uPanPix.y) / uPixPerUnit;
    float dx = uLx / float(uN);
    float dy = uLy / float(uN);
    int i = int(floor((wx + 0.5 * uLx) / dx));
    int j = int(floor((wy + 0.5 * uLy) / dy));
    if (i < 0 || j < 0 || i >= uN || j >= uN) {
        FragColor = vec4(0.03, 0.03, 0.045, 1.0);
        return;
    }
    int idx = j * uN + i;
    vec2 p = psi[idx];
    float mag2 = dot(p, p);
    float v = pot[idx];

    // Brightness tracks probability density, auto-scaled to the current peak so
    // the whole distribution stays legible as the packet spreads or is absorbed,
    // then a gamma < 1 pulls the low end up so tails and fringe minima show.
    float rhoMax = max(uintBitsToFloat(sMaxBits), 1e-30);
    float rel = clamp(uGain * mag2 / rhoMax, 0.0, 1.0);
    float dens = pow(rel, uGamma);

    vec3 col;
    if (uMode == 1) {
        col = magma(dens);
    } else if (uMode == 2) {
        float s = clamp(uGain * p.x / sqrt(rhoMax), -1.0, 1.0);
        s = sign(s) * pow(abs(s), uGamma);
        col = mix(vec3(0.15, 0.3, 0.9), vec3(0.95, 0.25, 0.2), 0.5 + 0.5 * s);
        col *= 0.25 + 0.75 * abs(s);
    } else {
        float phase = atan(p.y, p.x);
        float hue = phase / (2.0 * 3.14159265) + 0.5;
        col = hsv2rgb(vec3(hue, 0.82, dens));
    }

    if (uVref > 0.0) {
        float t = clamp(abs(v) / uVref, 0.0, 1.0);
        col = mix(col, vec3(0.55, 0.55, 0.6), 0.22 * t);
    }
    FragColor = vec4(col, 1.0);
}
)";

// Probability-current overlay: one instanced 3-segment arrow per cell of a
// coarse screen grid, sampled from the current buffer, scaled/faded by |j|.
const char* kArrowVert = R"(#version 460 core
layout(std430, binding = 0) readonly buffer Cur  { vec2 jbuf[]; };
layout(std430, binding = 1) readonly buffer StatJ { uint jMaxBits; };

uniform vec2 uRes;
uniform int uN;
uniform float uLx;
uniform float uLy;
uniform float uPixPerUnit;
uniform vec2 uPanPix;
uniform int uCols;
uniform int uRows;
uniform float uArrowPx;

out float vA;

void main() {
    int inst = gl_InstanceID;
    int cx = inst % uCols;
    int cy = inst / uCols;
    vec2 scr = (vec2(cx, cy) + 0.5) / vec2(uCols, uRows) * uRes;

    float wx = (scr.x - 0.5 * uRes.x - uPanPix.x) / uPixPerUnit;
    float wy = (scr.y - 0.5 * uRes.y - uPanPix.y) / uPixPerUnit;
    int i = int(floor((wx + 0.5 * uLx) / (uLx / float(uN))));
    int k = int(floor((wy + 0.5 * uLy) / (uLy / float(uN))));
    vec2 jj = vec2(0.0);
    if (i >= 0 && k >= 0 && i < uN && k < uN) jj = jbuf[k * uN + i];

    float jmax = sqrt(max(uintBitsToFloat(jMaxBits), 1e-30));
    float mag = length(jj);
    float f = clamp(mag / jmax, 0.0, 1.0);
    float fv = sqrt(f);                 // perceptual: don't hide the weak flow
    vA = 0.30 + 0.70 * fv;
    vec2 dir = mag > 1e-20 ? jj / mag : vec2(1.0, 0.0);
    vec2 perp = vec2(-dir.y, dir.x);
    float L = uArrowPx * (0.45 + 0.55 * fv);
    vec2 a = scr - dir * L * 0.5;
    vec2 b = scr + dir * L * 0.5;
    int id = gl_VertexID;
    vec2 v = (id == 0) ? a
           : (id == 1) ? b
           : (id == 2) ? b
           : (id == 3) ? (b - dir * L * 0.32 + perp * L * 0.22)
           : (id == 4) ? b
           :             (b - dir * L * 0.32 - perp * L * 0.22);
    if (f < 0.02) v = scr;  // collapse invisible arrows to a point
    gl_Position = vec4(v / uRes * 2.0 - 1.0, 0.0, 1.0);
}
)";

const char* kArrowFrag = R"(#version 460 core
in float vA;
out vec4 FragColor;
uniform vec3 uColor;
void main() { FragColor = vec4(uColor, vA * 0.9); }
)";

} // namespace

TdseSim::~TdseSim() {
    GLuint bufs[] = {m_psi,   m_tmp,   m_spec,  m_vprop,      m_kprop, m_twiddle, m_potential,
                     m_cap,   m_vhartree, m_stat, m_currentBuf, m_statJ, m_kdisp,  m_statK};
    glDeleteBuffers(14, bufs);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
    if (m_arrowVao) glDeleteVertexArrays(1, &m_arrowVao);
}

void TdseSim::Configure(const fw::Deck& deck) {
    const int n = deck.GetInt("grid.n", 256);
    if (!IsPow2(n) || n < 64 || n > 2048) {
        throw std::runtime_error("grid.n must be a power of two in [64, 2048]");
    }
    m_grid.n = n;
    m_grid.lx = deck.GetDouble("grid.lx", 40.0);
    m_grid.ly = deck.GetDouble("grid.ly", m_grid.lx);
    m_dt = deck.GetDouble("time.dt", 0.002);
    m_substepsPerFrame = std::max(1, deck.GetInt("time.substeps_per_frame", 10));
    m_transmissionX = deck.GetDouble("output.transmission_x", 0.0);
    m_title = deck.GetString("title", "2D TDSE (GPU)");
    for (const auto& f : deck.GetStringArray("output.fields")) {
        if (f == "current") m_writeCurrent = true;
    }

    m_diagNames = deck.GetStringArray("output.diagnostics");
    if (m_diagNames.empty()) {
        m_diagNames = {"norm",    "energy",  "kinetic",       "potential_energy", "x_mean",
                       "y_mean",  "x_var",   "y_var",          "px_mean",          "py_mean",
                       "transmission"};
    }

    m_vCpu = BuildPotential(m_grid, deck);
    m_initial = BuildInitial(m_grid, deck);

    // Uniform magnetic field, symmetric gauge: H = -1/2 grad^2 - (B/2) L_z +
    // (B^2/8) r^2 + V. The diamagnetic term folds into the static potential;
    // the L_z term is a per-step rotation of psi (see RunStepMagnetic).
    m_B = deck.GetDouble("magnetic.B", 0.0);
    m_hasMagnetic = std::abs(m_B) > 1e-12;
    if (m_hasMagnetic) {
        const double c = m_B * m_B / 8.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const double x = m_grid.x(i), y = m_grid.y(j);
                m_vCpu[static_cast<size_t>(j) * n + i] += static_cast<float>(c * (x * x + y * y));
            }
        }
    }

    for (const auto& d : deck.GetTables("drive")) {
        if (m_drives.size() == 4) break;
        DriveTerm t;
        const std::string type = d.GetString("type", "tilt");
        if (type == "tilt") {
            t.type = 0;
            t.a = {static_cast<float>(d.GetDouble("amplitude", 1.0)),
                   static_cast<float>(d.GetDouble("omega", 1.0)),
                   static_cast<float>(d.GetDouble("phase", 0.0)),
                   static_cast<float>(d.GetDouble("ramp", 0.0))};
            t.b = {static_cast<float>(d.GetDouble("dir_x", 1.0)),
                   static_cast<float>(d.GetDouble("dir_y", 0.0)), 0.0f, 0.0f};
        } else if (type == "gate") {
            t.type = 1;
            t.a = {static_cast<float>(d.GetDouble("amplitude", 1.0)),
                   static_cast<float>(d.GetDouble("sigma", 1.0)),
                   static_cast<float>(d.GetDouble("t_on", 0.0)),
                   static_cast<float>(d.GetDouble("t_ramp", 0.5))};
            t.b = {static_cast<float>(d.GetDouble("x0", 0.0)),
                   static_cast<float>(d.GetDouble("y0", 0.0)), 0.0f, 0.0f};
        } else {
            continue;
        }
        m_drives.push_back(t);
    }
    m_hasDrives = !m_drives.empty();

    m_hasMeanfield = deck.GetBool("meanfield.enabled", false);
    m_mfCoupling = deck.GetDouble("meanfield.coupling", 1.0);

    // Momentum-view window: a few times the spread expected from the initial
    // momentum, packet width, and the deepest/highest part of the potential.
    {
        const double k0 = std::hypot(deck.GetDouble("initial.kx", 0.0),
                                     deck.GetDouble("initial.ky", 0.0));
        const double sigma = std::max(0.2, deck.GetDouble("initial.sigma", 1.5));
        m_kMax = kPi * n / m_grid.lx;
        m_kView = std::clamp(1.6 * k0 + 5.0 / sigma + 3.0, 3.0, 0.9 * m_kMax);
    }

    m_fft = fw::ComputeShader::FromSource(kernels::Fft1D(n));
    m_transpose = fw::ComputeShader::FromSource(kernels::Transpose(n));
    m_cmul = fw::ComputeShader::FromSource(kernels::CMul(n));
    m_reduceMax = fw::ComputeShader::FromSource(kernels::ReduceMax(n));
    m_currentProg = fw::ComputeShader::FromSource(kernels::Current(n));
    m_fftshift = fw::ComputeShader::FromSource(kernels::FftShift(n));
    m_buildVprop = fw::ComputeShader::FromSource(kernels::BuildVprop(n));
    m_shearPhase = fw::ComputeShader::FromSource(kernels::ShearPhase(n));
    m_mfRho = fw::ComputeShader::FromSource(kernels::MeanFieldRho(n));
    m_poissonMul = fw::ComputeShader::FromSource(kernels::PoissonMul(n));
    m_extractReal = fw::ComputeShader::FromSource(kernels::ExtractReal(n));
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    m_arrows = fw::Shader::FromSource(kArrowVert, kArrowFrag);
    glGenVertexArrays(1, &m_vao);
    glGenVertexArrays(1, &m_arrowVao);

    CreateBuffers();
    BuildPropagators(deck);
    UploadInitial();
}

void TdseSim::CreateBuffers() {
    const int n = m_grid.n;
    const GLsizeiptr cplx = static_cast<GLsizeiptr>(n) * n * 2 * sizeof(float);

    auto mk = [](GLuint& b, GLsizeiptr bytes) {
        if (b) glDeleteBuffers(1, &b);
        glCreateBuffers(1, &b);
        glNamedBufferData(b, bytes, nullptr, GL_DYNAMIC_DRAW);
    };
    mk(m_psi, cplx);
    mk(m_tmp, cplx);
    mk(m_spec, cplx);
    mk(m_vprop, cplx);
    mk(m_kprop, cplx);
    mk(m_twiddle, static_cast<GLsizeiptr>(n) * 2 * sizeof(float));
    mk(m_potential, static_cast<GLsizeiptr>(n) * n * sizeof(float));
    mk(m_cap, static_cast<GLsizeiptr>(n) * n * sizeof(float));
    mk(m_vhartree, static_cast<GLsizeiptr>(n) * n * sizeof(float));
    {
        const std::vector<float> zeros(static_cast<size_t>(n) * n, 0.0f);
        glNamedBufferSubData(m_vhartree, 0, static_cast<GLsizeiptr>(zeros.size() * sizeof(float)),
                             zeros.data());
    }
    mk(m_stat, sizeof(uint32_t));
    mk(m_currentBuf, cplx);
    mk(m_statJ, sizeof(uint32_t));
    mk(m_kdisp, cplx);
    mk(m_statK, sizeof(uint32_t));

    glNamedBufferSubData(m_potential, 0, static_cast<GLsizeiptr>(n) * n * sizeof(float), m_vCpu.data());
}

void TdseSim::BuildPropagators(const fw::Deck& deck) {
    const int n = m_grid.n;

    // Static potential V0 and absorbing rate W live in their own buffers; the
    // half-kick multiplier Vprop is (re)built on the GPU by BuildVprop.
    const std::vector<float> w = BuildCap(m_grid, deck);
    glNamedBufferSubData(m_cap, 0, static_cast<GLsizeiptr>(w.size() * sizeof(float)), w.data());
    RebuildVprop(0.0);

    // Kprop in the transposed layout the k-space multiply sees: index = kxIdx*n + kyIdx.
    std::vector<float> kprop(static_cast<size_t>(n) * n * 2);
    for (int a = 0; a < n; ++a) {          // kx index (transposed row)
        const double kx = m_grid.kx(a);
        for (int b = 0; b < n; ++b) {      // ky index
            const double ky = m_grid.ky(b);
            const double ph = -0.5 * (kx * kx + ky * ky) * m_dt;
            const size_t k = static_cast<size_t>(a) * n + b;
            kprop[2 * k + 0] = static_cast<float>(std::cos(ph));
            kprop[2 * k + 1] = static_cast<float>(std::sin(ph));
        }
    }
    glNamedBufferSubData(m_kprop, 0, static_cast<GLsizeiptr>(kprop.size() * sizeof(float)), kprop.data());

    std::vector<float> tw(static_cast<size_t>(n) * 2);
    for (int m = 0; m < n; ++m) {
        const double ang = 2.0 * kPi * m / n;
        tw[2 * m + 0] = static_cast<float>(std::cos(ang));
        tw[2 * m + 1] = static_cast<float>(-std::sin(ang));
    }
    glNamedBufferSubData(m_twiddle, 0, static_cast<GLsizeiptr>(tw.size() * sizeof(float)), tw.data());
}

void TdseSim::UploadInitial() {
    glNamedBufferSubData(m_psi, 0,
                         static_cast<GLsizeiptr>(m_initial.size() * sizeof(std::complex<float>)),
                         m_initial.data());
    m_stepsDone = 0;
    m_time = 0.0;
}

void TdseSim::Reset() {
    UploadInitial();
    RebuildVprop(0.0);
}

void TdseSim::RebuildVprop(double t) {
    const int n = m_grid.n;
    int types[4] = {0, 0, 0, 0};
    glm::vec4 as[4] = {}, bs[4] = {};
    for (size_t d = 0; d < m_drives.size(); ++d) {
        types[d] = m_drives[d].type;
        as[d] = m_drives[d].a;
        bs[d] = m_drives[d].b;
    }
    m_buildVprop.Use();
    m_buildVprop.SetFloat("uDt", static_cast<float>(m_dt));
    m_buildVprop.SetFloat("uTime", static_cast<float>(t));
    m_buildVprop.SetFloat("uX0", static_cast<float>(-0.5 * m_grid.lx));
    m_buildVprop.SetFloat("uDx", static_cast<float>(m_grid.dx()));
    m_buildVprop.SetFloat("uY0", static_cast<float>(-0.5 * m_grid.ly));
    m_buildVprop.SetFloat("uDy", static_cast<float>(m_grid.dy()));
    m_buildVprop.SetInt("uDriveCount", static_cast<int>(m_drives.size()));
    m_buildVprop.SetIntArray("uDriveType", types, 4);
    m_buildVprop.SetVec4Array("uDriveA", as, 4);
    m_buildVprop.SetVec4Array("uDriveB", bs, 4);
    fw::ComputeShader::BindBuffer(0, m_potential);
    fw::ComputeShader::BindBuffer(1, m_cap);
    fw::ComputeShader::BindBuffer(2, m_vprop);
    fw::ComputeShader::BindBuffer(3, m_vhartree);
    m_buildVprop.Dispatch(static_cast<GLuint>(n / 16), static_cast<GLuint>(n / 16), 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::ComputeMeanField() {
    const int n = m_grid.n;
    const GLuint groups = static_cast<GLuint>(n / 16);

    // rho = |psi|^2 -> m_spec (complex, imag 0)
    m_mfRho.Use();
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_spec);
    m_mfRho.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    Fft(m_spec, false);
    TransposeBuf(m_spec, m_tmp);
    Fft(m_tmp, false);  // m_tmp = rho_hat [kxIdx][kyIdx]

    m_poissonMul.Use();
    m_poissonMul.SetFloat("uCoupling", static_cast<float>(m_mfCoupling));
    m_poissonMul.SetFloat("uKxScale", static_cast<float>(2.0 * kPi / m_grid.lx));
    m_poissonMul.SetFloat("uKyScale", static_cast<float>(2.0 * kPi / m_grid.ly));
    fw::ComputeShader::BindBuffer(0, m_tmp);
    m_poissonMul.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    Fft(m_tmp, true);
    TransposeBuf(m_tmp, m_spec);
    Fft(m_spec, true);  // m_spec = V_H (real part)

    m_extractReal.Use();
    fw::ComputeShader::BindBuffer(0, m_spec);
    fw::ComputeShader::BindBuffer(1, m_vhartree);
    m_extractReal.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::Fft(GLuint buffer, bool inverse) {
    const int n = m_grid.n;
    m_fft.Use();
    m_fft.SetInt("uSign", inverse ? 1 : -1);
    m_fft.SetFloat("uScale", inverse ? 1.0f / n : 1.0f);
    fw::ComputeShader::BindBuffer(0, buffer);
    fw::ComputeShader::BindBuffer(1, m_twiddle);
    m_fft.Dispatch(static_cast<GLuint>(n), 1, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::TransposeBuf(GLuint src, GLuint dst) {
    const GLuint groups = static_cast<GLuint>(m_grid.n / 16);
    m_transpose.Use();
    fw::ComputeShader::BindBuffer(0, src);
    fw::ComputeShader::BindBuffer(1, dst);
    m_transpose.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::CMulBuf(GLuint dst, GLuint by) {
    const GLuint groups = static_cast<GLuint>(m_grid.n / 16);
    m_cmul.Use();
    fw::ComputeShader::BindBuffer(0, dst);
    fw::ComputeShader::BindBuffer(1, by);
    m_cmul.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::RunStep(double t0) {
    // Time-dependent potential: both half-kicks of this step use V at the
    // midpoint t0 + dt/2 (2nd-order accurate). The Hartree potential is also
    // refreshed here from the current density.
    if (m_hasMeanfield) ComputeMeanField();
    if (m_hasDrives || m_hasMeanfield) RebuildVprop(t0 + 0.5 * m_dt);
    CMulBuf(m_psi, m_vprop);      // half potential kick
    Fft(m_psi, false);            // FFT over x   -> psi[y][kx]
    TransposeBuf(m_psi, m_tmp);   // tmp[kx][y]
    Fft(m_tmp, false);            // FFT over y   -> tmp[kx][ky]
    CMulBuf(m_tmp, m_kprop);      // kinetic phase
    Fft(m_tmp, true);             // IFFT over ky -> tmp[kx][y]
    TransposeBuf(m_tmp, m_psi);   // psi[y][kx]
    Fft(m_psi, true);             // IFFT over kx -> psi[y][x]
    CMulBuf(m_psi, m_vprop);      // half potential kick
}

void TdseSim::ShearApply(GLuint buf, double amount, double origin, double step, double kscale) {
    const int n = m_grid.n;
    m_shearPhase.Use();
    m_shearPhase.SetFloat("uAmount", static_cast<float>(amount));
    m_shearPhase.SetFloat("uCoordOrigin", static_cast<float>(origin));
    m_shearPhase.SetFloat("uCoordStep", static_cast<float>(step));
    m_shearPhase.SetFloat("uKScale", static_cast<float>(kscale));
    fw::ComputeShader::BindBuffer(0, buf);
    m_shearPhase.Dispatch(static_cast<GLuint>(n / 16), static_cast<GLuint>(n / 16), 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

// exp(i alpha L_z): rotate psi(r, phi) -> psi(r, phi + alpha), via the exact
// unitary 3-shear decomposition R(-alpha) = Sx(t) Sy(s) Sx(t) with
// t = tan(alpha/2), s = -sin(alpha), each shear a Fourier phase shift.
void TdseSim::Rotate(double alpha) {
    const double twoPi = 2.0 * kPi;
    const double t = std::tan(0.5 * alpha);
    const double s = -std::sin(alpha);
    const double dx = m_grid.dx(), dy = m_grid.dy();
    const double ox = -0.5 * m_grid.lx, oy = -0.5 * m_grid.ly;
    const double ksx = twoPi / m_grid.lx, ksy = twoPi / m_grid.ly;

    auto shearX = [&](double a) {           // psi(x - a*y, y)
        Fft(m_psi, false);
        ShearApply(m_psi, a, oy, dy, ksx);
        Fft(m_psi, true);
    };
    auto shearY = [&](double a) {           // psi(x, y - a*x)
        TransposeBuf(m_psi, m_tmp);
        Fft(m_tmp, false);
        ShearApply(m_tmp, a, ox, dx, ksy);
        Fft(m_tmp, true);
        TransposeBuf(m_tmp, m_psi);
    };
    shearX(t);
    shearY(s);
    shearX(t);
}

void TdseSim::RunStepMagnetic(double t0) {
    if (m_hasMeanfield) ComputeMeanField();
    if (m_hasDrives || m_hasMeanfield) RebuildVprop(t0 + 0.5 * m_dt);
    const double half = -m_B * m_dt / 4.0;  // exp(-i M dt/2), M = -(B/2) L_z
    CMulBuf(m_psi, m_vprop);   // half U kick (V incl. diamagnetic term)
    Rotate(half);              // half L_z rotation
    Fft(m_psi, false);
    TransposeBuf(m_psi, m_tmp);
    Fft(m_tmp, false);
    CMulBuf(m_tmp, m_kprop);
    Fft(m_tmp, true);
    TransposeBuf(m_tmp, m_psi);
    Fft(m_psi, true);
    Rotate(half);
    CMulBuf(m_psi, m_vprop);   // half U kick
}

void TdseSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        if (m_hasMagnetic) RunStepMagnetic(m_time);
        else RunStep(m_time);
        m_time += m_dt;
        ++m_stepsDone;
    }
}

void TdseSim::ComputeSpectrumInto(GLuint dst) {
    const GLsizeiptr bytes = static_cast<GLsizeiptr>(m_grid.n) * m_grid.n * 2 * sizeof(float);
    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
    glCopyNamedBufferSubData(m_psi, m_spec, 0, 0, bytes);
    Fft(m_spec, false);
    TransposeBuf(m_spec, dst);
    Fft(dst, false);  // dst = psi-hat, layout [kxIdx][kyIdx]
}

void TdseSim::ComputeCurrent() {
    const GLuint groups = static_cast<GLuint>(m_grid.n / 16);
    m_currentProg.Use();
    m_currentProg.SetFloat("uDx", static_cast<float>(m_grid.dx()));
    m_currentProg.SetFloat("uDy", static_cast<float>(m_grid.dy()));
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_currentBuf);
    m_currentProg.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
}

void TdseSim::Readback() const {
    const int n = m_grid.n;
    m_scratch.resize(static_cast<size_t>(n) * n);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetNamedBufferSubData(m_psi, 0,
                            static_cast<GLsizeiptr>(m_scratch.size() * sizeof(std::complex<float>)),
                            m_scratch.data());
}

void TdseSim::Snapshot(fw::OutputWriter& writer) {
    const int n = m_grid.n;
    Readback();

    writer.WriteField("psi", m_scratch.data(), fw::NpyDtype::C8, n, n);
    if (writer.FramesWritten() == 0) {
        writer.WriteField("potential", m_vCpu.data(), fw::NpyDtype::F4, n, n);
    }

    const double dx = m_grid.dx();
    const double dy = m_grid.dy();
    const double cell = dx * dy;

    // Real-space observables from |psi|^2.
    double norm = 0.0, xm = 0.0, ym = 0.0, x2 = 0.0, y2 = 0.0, trans = 0.0, vexp = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const std::complex<float> c = m_scratch[static_cast<size_t>(j) * n + i];
            const double d = double(c.real()) * c.real() + double(c.imag()) * c.imag();
            norm += d;
            xm += m_grid.x(i) * d;
            ym += m_grid.y(j) * d;
            x2 += m_grid.x(i) * m_grid.x(i) * d;
            y2 += m_grid.y(j) * m_grid.y(j) * d;
            if (m_grid.x(i) > m_transmissionX) trans += d;
            vexp += m_vCpu[static_cast<size_t>(j) * n + i] * d;
        }
    }
    const double N2 = norm * cell;
    const double invN = N2 > 0.0 ? 1.0 / N2 : 0.0;

    // <L_z> = Integral Im[conj(psi) (x d/dy - y d/dx) psi]  (central diff, periodic).
    double lz = 0.0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const std::complex<double> p = {m_scratch[static_cast<size_t>(j) * n + i].real(),
                                            m_scratch[static_cast<size_t>(j) * n + i].imag()};
            const auto at = [&](int ii, int jj) {
                const auto& z = m_scratch[static_cast<size_t>((jj + n) % n) * n + ((ii + n) % n)];
                return std::complex<double>(z.real(), z.imag());
            };
            const std::complex<double> ddx = (at(i + 1, j) - at(i - 1, j)) / (2.0 * dx);
            const std::complex<double> ddy = (at(i, j + 1) - at(i, j - 1)) / (2.0 * dy);
            lz += std::imag(std::conj(p) * (m_grid.x(i) * ddy - m_grid.y(j) * ddx));
        }
    }
    const double Lz = lz * cell * invN;

    // Autocorrelation A(t) = <psi(0)|psi(t)> -- its |.| shows revivals, its
    // time-Fourier transform is the energy spectrum weighted by |<n|psi0>|^2.
    double aRe = 0.0, aIm = 0.0;
    for (size_t k = 0; k < m_scratch.size(); ++k) {
        const std::complex<float> p0 = m_initial[k];
        const std::complex<float> p = m_scratch[k];
        aRe += double(p0.real()) * p.real() + double(p0.imag()) * p.imag();
        aIm += double(p0.real()) * p.imag() - double(p0.imag()) * p.real();
    }
    aRe *= cell;
    aIm *= cell;

    // Kinetic energy and momentum, spectrally exact: 2D FFT of psi (into scratch
    // buffers that Step() also uses transiently) then sum against k.
    double T = 0.0, px = 0.0, py = 0.0;
    {
        ComputeSpectrumInto(m_tmp);  // m_tmp = psi-hat, layout [kxIdx][kyIdx]

        std::vector<std::complex<float>> hat(static_cast<size_t>(n) * n);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(m_tmp, 0, static_cast<GLsizeiptr>(hat.size() * sizeof(std::complex<float>)),
                                hat.data());
        double sh = 0.0, tsum = 0.0, pxs = 0.0, pys = 0.0;
        for (int a = 0; a < n; ++a) {
            const double kx = m_grid.kx(a);
            for (int b = 0; b < n; ++b) {
                const double ky = m_grid.ky(b);
                const std::complex<float> c = hat[static_cast<size_t>(a) * n + b];
                const double d = double(c.real()) * c.real() + double(c.imag()) * c.imag();
                sh += d;
                tsum += 0.5 * (kx * kx + ky * ky) * d;
                pxs += kx * d;
                pys += ky * d;
            }
        }
        const double invSh = sh > 0.0 ? 1.0 / sh : 0.0;
        T = tsum * invSh;
        px = pxs * invSh;
        py = pys * invSh;
    }

    auto put = [&](const char* name, double value) { writer.WriteScalar(name, value); };
    const double xMean = xm * cell * invN;
    const double yMean = ym * cell * invN;
    put("norm", N2);
    put("x_mean", xMean);
    put("y_mean", yMean);
    put("x_var", x2 * cell * invN - xMean * xMean);
    put("y_var", y2 * cell * invN - yMean * yMean);
    put("px_mean", px);
    put("py_mean", py);
    put("transmission", trans * cell * invN);
    const double V = vexp * cell * invN;
    const double magLz = -0.5 * m_B * Lz;  // the -(B/2) L_z term of H
    put("kinetic", T);
    put("potential_energy", V);        // includes the (B^2/8) r^2 diamagnetic term
    put("Lz", Lz);
    put("energy", T + V + magLz);
    put("autocorr_re", aRe);
    put("autocorr_im", aIm);
    put("autocorr_abs", std::sqrt(aRe * aRe + aIm * aIm));

    if (m_writeCurrent) {
        ComputeCurrent();
        m_scratch.resize(static_cast<size_t>(n) * n);
        glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
        glGetNamedBufferSubData(m_currentBuf, 0,
                                static_cast<GLsizeiptr>(m_scratch.size() * sizeof(std::complex<float>)),
                                m_scratch.data());
        // stored as complex64: real = Jx, imag = Jy
        writer.WriteField("current", m_scratch.data(), fw::NpyDtype::C8, n, n);
    }
}

fw::SimInfo TdseSim::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_grid.n;
    info.gridNy = m_grid.n;
    info.lx = m_grid.lx;
    info.ly = m_grid.ly;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = m_writeCurrent ? std::vector<std::string>{"psi", "potential", "current"}
                                      : std::vector<std::string>{"psi", "potential"};
    info.diagnostics = m_diagNames;
    return info;
}

void TdseSim::Render(int fbWidth, int fbHeight) {
    const int n = m_grid.n;
    const GLuint groups = static_cast<GLuint>(n / 16);
    const uint32_t zero = 0;

    // Pick what fills the screen: real-space psi, or (K) the momentum density.
    GLuint srcBuf = m_psi;
    GLuint statBuf = m_stat;
    double boxX = m_grid.lx, boxY = m_grid.ly;
    float vref = 0.0f;
    if (m_showMomentum) {
        ComputeSpectrumInto(m_tmp);
        m_fftshift.Use();
        fw::ComputeShader::BindBuffer(0, m_tmp);
        fw::ComputeShader::BindBuffer(1, m_kdisp);
        m_fftshift.Dispatch(groups, groups, 1);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);
        srcBuf = m_kdisp;
        statBuf = m_statK;
        boxX = 2.0 * 3.14159265358979323846 * n / m_grid.lx;  // full kx range
        boxY = 2.0 * 3.14159265358979323846 * n / m_grid.ly;
    } else {
        for (float v : m_vCpu) vref = std::max(vref, std::abs(v));
    }

    const float pixPerUnit = static_cast<float>(std::min(fbWidth, fbHeight)) /
                             static_cast<float>(std::max(boxX, boxY)) * m_zoom;

    glNamedBufferSubData(statBuf, 0, sizeof(zero), &zero);
    m_reduceMax.Use();
    fw::ComputeShader::BindBuffer(0, srcBuf);
    fw::ComputeShader::BindBuffer(1, statBuf);
    m_reduceMax.Dispatch(groups, groups, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDisable(GL_DEPTH_TEST);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2(fbWidth, fbHeight));
    m_view.SetInt("uN", n);
    m_view.SetFloat("uLx", static_cast<float>(boxX));
    m_view.SetFloat("uLy", static_cast<float>(boxY));
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetInt("uMode", m_mode);
    m_view.SetFloat("uGain", m_gain);
    m_view.SetFloat("uGamma", m_gamma);
    m_view.SetFloat("uVref", vref);

    fw::ComputeShader::BindBuffer(0, srcBuf);
    fw::ComputeShader::BindBuffer(1, m_potential);
    fw::ComputeShader::BindBuffer(2, statBuf);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    if (m_showCurrent && !m_showMomentum) {
        ComputeCurrent();
        const uint32_t z = 0;
        glNamedBufferSubData(m_statJ, 0, sizeof(z), &z);
        m_reduceMax.Use();
        fw::ComputeShader::BindBuffer(0, m_currentBuf);
        fw::ComputeShader::BindBuffer(1, m_statJ);
        m_reduceMax.Dispatch(static_cast<GLuint>(n / 16), static_cast<GLuint>(n / 16), 1);
        fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

        const int cols = std::max(8, fbWidth / 34);
        const int rows = std::max(6, fbHeight / 34);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        m_arrows.Use();
        m_arrows.SetVec2("uRes", glm::vec2(fbWidth, fbHeight));
        m_arrows.SetInt("uN", n);
        m_arrows.SetFloat("uLx", static_cast<float>(m_grid.lx));
        m_arrows.SetFloat("uLy", static_cast<float>(m_grid.ly));
        m_arrows.SetFloat("uPixPerUnit", pixPerUnit);
        m_arrows.SetVec2("uPanPix", m_panPix);
        m_arrows.SetInt("uCols", cols);
        m_arrows.SetInt("uRows", rows);
        m_arrows.SetFloat("uArrowPx", 30.0f);
        m_arrows.SetVec3("uColor", glm::vec3(0.85f, 0.98f, 1.0f));
        glLineWidth(1.6f);
        fw::ComputeShader::BindBuffer(0, m_currentBuf);
        fw::ComputeShader::BindBuffer(1, m_statJ);
        glBindVertexArray(m_arrowVao);
        glDrawArraysInstanced(GL_LINES, 0, 6, cols * rows);
        glBindVertexArray(0);
        glDisable(GL_BLEND);
    }

    glEnable(GL_DEPTH_TEST);
}

void TdseSim::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) {
        m_panPix.x += static_cast<float>(in.dx);
        m_panPix.y -= static_cast<float>(in.dy); // screen y-down -> world y-up
    }
    if (in.scrollDelta != 0.0) {
        m_zoom *= std::exp(0.12f * static_cast<float>(in.scrollDelta));
        m_zoom = std::clamp(m_zoom, 0.2f, 80.0f);
    }
}

void TdseSim::OnKey(int key, int action) {
    (void)action;
    switch (key) {
        case 'M': m_mode = (m_mode + 1) % 3; break;
        case '[': m_gain = std::max(0.15f, m_gain * 0.85f); break;
        case ']': m_gain = std::min(12.0f, m_gain * 1.18f); break;
        case '-': m_gamma = std::max(0.2f, m_gamma - 0.05f); break; // lift tails
        case '=': m_gamma = std::min(1.4f, m_gamma + 0.05f); break; // sharpen to the core
        case '0': m_zoom = 1.0f; m_panPix = {0.0f, 0.0f}; m_gain = 1.15f; m_gamma = 0.5f; break;
        case 'J': m_showCurrent = !m_showCurrent; break;
        case 'K':
            m_showMomentum = !m_showMomentum;
            m_zoom = m_showMomentum ? static_cast<float>(m_kMax / m_kView) : 1.0f;
            m_panPix = {0.0f, 0.0f};
            break;
        default: break;
    }
}

// ---------------------------------------------------------------------------
// Imaginary-time relaxation (Phase G1)
// ---------------------------------------------------------------------------

namespace {

// A few real solid-harmonic factors to seed states of different symmetry.
double SeedPoly(int k, double x, double y) {
    switch (k % 6) {
        case 0: return 1.0;
        case 1: return x;
        case 2: return y;
        case 3: return x * y;
        case 4: return x * x - y * y;
        default: return x * (x * x - 3.0 * y * y);
    }
}

} // namespace

TdseSim::RelaxResult TdseSim::Relax(const RelaxOptions& opt) {
    const int n = m_grid.n;
    const size_t nn = static_cast<size_t>(n) * n;
    const int K = std::clamp(opt.states, 1, 12);
    const double dtau = opt.dtau;
    const double cell = m_grid.dx() * m_grid.dy();

    // Effective potential = static V0 (+ optional extra, e.g. V_Hartree).
    std::vector<float> veff = m_vCpu;
    if (opt.extraPotential && opt.extraPotential->size() == nn) {
        for (size_t i = 0; i < nn; ++i) veff[i] += (*opt.extraPotential)[i];
    }

    // Real imaginary-time propagators: exp(-V dtau/2) and exp(-k^2 dtau/2)
    // (the kinetic one in the transposed [kxIdx][kyIdx] layout).
    std::vector<float> vpr(nn * 2, 0.0f), kpr(nn * 2, 0.0f);
    for (size_t i = 0; i < nn; ++i) vpr[2 * i] = static_cast<float>(std::exp(-veff[i] * dtau * 0.5));
    for (int a = 0; a < n; ++a) {
        const double kx = m_grid.kx(a);
        for (int b = 0; b < n; ++b) {
            const double ky = m_grid.ky(b);
            kpr[2 * (static_cast<size_t>(a) * n + b)] =
                static_cast<float>(std::exp(-0.5 * (kx * kx + ky * ky) * dtau));
        }
    }
    GLuint vpropR = 0, kpropR = 0;
    glCreateBuffers(1, &vpropR);
    glCreateBuffers(1, &kpropR);
    glNamedBufferData(vpropR, static_cast<GLsizeiptr>(vpr.size() * sizeof(float)), vpr.data(), GL_STATIC_DRAW);
    glNamedBufferData(kpropR, static_cast<GLsizeiptr>(kpr.size() * sizeof(float)), kpr.data(), GL_STATIC_DRAW);

    std::vector<GLuint> sb(K, 0);
    for (int k = 0; k < K; ++k) {
        glCreateBuffers(1, &sb[k]);
        glNamedBufferData(sb[k], static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), nullptr,
                          GL_DYNAMIC_DRAW);
    }

    // Seed: warm start if given, else the deck's initial state modulated by a
    // symmetry polynomial + a little noise.
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> rnd(-1.0f, 1.0f);
    std::vector<std::vector<std::complex<float>>> cpu(K, std::vector<std::complex<float>>(nn));
    for (int k = 0; k < K; ++k) {
        if (opt.warmStart && static_cast<int>(opt.warmStart->size()) >= K) {
            cpu[k] = (*opt.warmStart)[k];
            continue;
        }
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const size_t idx = static_cast<size_t>(j) * n + i;
                const double p = SeedPoly(k, m_grid.x(i), m_grid.y(j));
                const std::complex<float> base = m_initial[idx];
                cpu[k][idx] = base * static_cast<float>(p) +
                              base * std::complex<float>(0.02f * rnd(rng), 0.02f * rnd(rng));
            }
        }
    }

    auto orthonormalize = [&]() {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < k; ++j) {
                std::complex<double> ov = 0.0;
                for (size_t i = 0; i < nn; ++i)
                    ov += std::conj(std::complex<double>(cpu[j][i])) * std::complex<double>(cpu[k][i]);
                const std::complex<float> c(static_cast<float>(ov.real() * cell),
                                            static_cast<float>(ov.imag() * cell));
                for (size_t i = 0; i < nn; ++i) cpu[k][i] -= c * cpu[j][i];
            }
            double nrm = 0.0;
            for (size_t i = 0; i < nn; ++i) nrm += std::norm(cpu[k][i]);
            const float s = nrm > 0.0 ? static_cast<float>(1.0 / std::sqrt(nrm * cell)) : 1.0f;
            for (size_t i = 0; i < nn; ++i) cpu[k][i] *= s;
        }
    };
    orthonormalize();
    for (int k = 0; k < K; ++k)
        glNamedBufferSubData(sb[k], 0, static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), cpu[k].data());

    auto propReal = [&](GLuint buf) {
        CMulBuf(buf, vpropR);
        Fft(buf, false);
        TransposeBuf(buf, m_tmp);
        Fft(m_tmp, false);
        CMulBuf(m_tmp, kpropR);
        Fft(m_tmp, true);
        TransposeBuf(m_tmp, buf);
        Fft(buf, true);
        CMulBuf(buf, vpropR);
    };

    const int oe = std::max(1, opt.orthoEvery);
    for (int step = 0; step < opt.steps; ++step) {
        for (int k = 0; k < K; ++k) propReal(sb[k]);
        if ((step + 1) % oe == 0 || step == opt.steps - 1) {
            glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
            for (int k = 0; k < K; ++k)
                glGetNamedBufferSubData(sb[k], 0,
                                        static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)),
                                        cpu[k].data());
            orthonormalize();
            for (int k = 0; k < K; ++k)
                glNamedBufferSubData(sb[k], 0,
                                     static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)),
                                     cpu[k].data());
        }
        if (!opt.quiet && (step % 200 == 0 || step == opt.steps - 1)) {
            std::printf("\rrelax %d/%d   ", step + 1, opt.steps);
            std::fflush(stdout);
        }
    }
    if (!opt.quiet) std::printf("\n");

    // Energies: E_k = <T> + <V>  (FD kinetic, periodic).
    RelaxResult res;
    res.states.resize(K);
    res.energies.resize(K);
    const double dx = m_grid.dx(), dy = m_grid.dy();
    for (int k = 0; k < K; ++k) {
        double T = 0.0, V = 0.0;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                const std::complex<double> p(cpu[k][static_cast<size_t>(j) * n + i]);
                const auto at = [&](int ii, int jj) {
                    return std::complex<double>(cpu[k][static_cast<size_t>((jj + n) % n) * n + ((ii + n) % n)]);
                };
                const std::complex<double> gx = (at(i + 1, j) - at(i, j)) / dx;
                const std::complex<double> gy = (at(i, j + 1) - at(i, j)) / dy;
                T += 0.5 * (std::norm(gx) + std::norm(gy));
                V += veff[static_cast<size_t>(j) * n + i] * std::norm(p);
            }
        }
        res.energies[k] = (T + V) * cell;
        res.states[k] = std::move(cpu[k]);
    }

    glDeleteBuffers(1, &vpropR);
    glDeleteBuffers(1, &kpropR);
    glDeleteBuffers(K, sb.data());

    // Restore m_psi to the deck's initial state (relax clobbered nothing shared,
    // but be tidy in case the caller keeps using the sim).
    UploadInitial();
    return res;
}

std::vector<float> TdseSim::PoissonSolve(const std::vector<float>& rho, double coupling) {
    const int n = m_grid.n;
    const size_t nn = static_cast<size_t>(n) * n;

    // rho -> zero mean, upload as complex.
    double mean = 0.0;
    for (float r : rho) mean += r;
    mean /= static_cast<double>(nn);
    std::vector<std::complex<float>> buf(nn);
    for (size_t i = 0; i < nn; ++i) buf[i] = std::complex<float>(static_cast<float>(rho[i] - mean), 0.0f);
    glNamedBufferSubData(m_spec, 0, static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), buf.data());

    Fft(m_spec, false);
    TransposeBuf(m_spec, m_tmp);
    Fft(m_tmp, false);  // m_tmp = rho-hat, layout [kxIdx][kyIdx]

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    std::vector<std::complex<float>> hat(nn);
    glGetNamedBufferSubData(m_tmp, 0, static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), hat.data());

    // V_hat(k) = coupling * rho_hat(k) / k^2,  V_hat(0) = 0.
    for (int a = 0; a < n; ++a) {
        const double kx = m_grid.kx(a);
        for (int b = 0; b < n; ++b) {
            const double ky = m_grid.ky(b);
            const double k2 = kx * kx + ky * ky;
            const size_t idx = static_cast<size_t>(a) * n + b;
            hat[idx] = k2 > 1e-12 ? hat[idx] * static_cast<float>(coupling / k2)
                                  : std::complex<float>(0.0f, 0.0f);
        }
    }
    glNamedBufferSubData(m_tmp, 0, static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), hat.data());

    Fft(m_tmp, true);
    TransposeBuf(m_tmp, m_spec);
    Fft(m_spec, true);  // m_spec = V_H (real part)

    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
    glGetNamedBufferSubData(m_spec, 0, static_cast<GLsizeiptr>(nn * sizeof(std::complex<float>)), buf.data());
    std::vector<float> vh(nn);
    for (size_t i = 0; i < nn; ++i) vh[i] = buf[i].real();
    return vh;
}

int TdseSim::RunScf(const fw::Deck& deck, const RelaxOptions& relaxOpt, int electrons,
                    const std::string& outDir) {
    const int n = m_grid.n;
    const size_t nn = static_cast<size_t>(n) * n;
    const int K = std::clamp(relaxOpt.states, 1, 12);
    const int N = (electrons <= 0 || electrons > K) ? K : electrons;
    const double coupling = deck.GetDouble("meanfield.coupling", 1.0);
    const double tol = deck.GetDouble("scf.tol", 2e-3);
    const int maxIter = deck.GetInt("scf.max_iter", 30);
    const double mix = deck.GetDouble("scf.mix", 0.35);

    std::vector<float> vH(nn, 0.0f);
    RelaxResult r;

    for (int it = 0; it < maxIter; ++it) {
        RelaxOptions ro = relaxOpt;
        ro.states = K;
        ro.extraPotential = &vH;
        ro.warmStart = it == 0 ? nullptr : &r.states;
        ro.steps = it == 0 ? relaxOpt.steps : std::max(300, relaxOpt.steps / 3);
        ro.orthoEvery = it == 0 ? 1 : 3;
        ro.quiet = true;
        r = Relax(ro);

        std::vector<float> rho(nn, 0.0f);
        for (int k = 0; k < N; ++k)
            for (size_t i = 0; i < nn; ++i) rho[i] += std::norm(r.states[k][i]);

        const std::vector<float> vHnew = PoissonSolve(rho, coupling);
        double resid = 0.0, scale = 1e-9;
        for (size_t i = 0; i < nn; ++i) {
            resid = std::max(resid, std::abs(static_cast<double>(vHnew[i]) - vH[i]));
            scale = std::max(scale, std::abs(static_cast<double>(vHnew[i])));
        }
        resid /= scale;
        for (size_t i = 0; i < nn; ++i)
            vH[i] = static_cast<float>((1.0 - mix) * vH[i] + mix * vHnew[i]);

        std::printf("scf %2d  E:", it);
        for (int k = 0; k < K; ++k) std::printf(" %.4f", r.energies[k]);
        std::printf("   residual %.2e\n", resid);
        if (it >= 3 && resid < tol) break;
    }

    // V_total = V_ext + V_H (the SCF-converged Hartree field).
    std::vector<float> vTot(nn);
    for (size_t i = 0; i < nn; ++i) vTot[i] = m_vCpu[i] + vH[i];

    fw::SimInfo info;
    info.title = "Poisson-Schrodinger (self-consistent)";
    info.gridNx = n;
    info.gridNy = n;
    info.lx = m_grid.lx;
    info.ly = m_grid.ly;
    info.frameFields = {"psi", "potential", "v_ext", "v_hartree"};
    info.diagnostics = {"energy", "occupation"};
    fw::OutputWriter w(outDir, info, deck);
    for (int k = 0; k < K; ++k) {
        w.BeginFrame(static_cast<double>(k), static_cast<long>(k));
        w.WriteField("psi", r.states[k].data(), fw::NpyDtype::C8, n, n);
        if (k == 0) {
            w.WriteField("potential", vTot.data(), fw::NpyDtype::F4, n, n);
            w.WriteField("v_ext", m_vCpu.data(), fw::NpyDtype::F4, n, n);
            w.WriteField("v_hartree", vH.data(), fw::NpyDtype::F4, n, n);
        }
        w.WriteScalar("energy", r.energies[k]);
        w.WriteScalar("occupation", k < N ? 1.0 : 0.0);
        w.EndFrame();
    }
    w.Finish();
    std::printf("converged energies:");
    for (double e : r.energies) std::printf(" %.4f", e);
    std::printf("\n");
    return 0;
}

} // namespace tdse
