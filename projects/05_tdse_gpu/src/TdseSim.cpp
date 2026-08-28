#include "TdseSim.hpp"

#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    GLuint bufs[] = {m_psi,       m_tmp,  m_spec,       m_vprop, m_kprop,
                     m_twiddle,   m_potential, m_stat,  m_currentBuf, m_statJ};
    glDeleteBuffers(10, bufs);
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

    m_fft = fw::ComputeShader::FromSource(kernels::Fft1D(n));
    m_transpose = fw::ComputeShader::FromSource(kernels::Transpose(n));
    m_cmul = fw::ComputeShader::FromSource(kernels::CMul(n));
    m_reduceMax = fw::ComputeShader::FromSource(kernels::ReduceMax(n));
    m_currentProg = fw::ComputeShader::FromSource(kernels::Current(n));
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
    mk(m_stat, sizeof(uint32_t));
    mk(m_currentBuf, cplx);
    mk(m_statJ, sizeof(uint32_t));

    glNamedBufferSubData(m_potential, 0, static_cast<GLsizeiptr>(n) * n * sizeof(float), m_vCpu.data());
}

void TdseSim::BuildPropagators(const fw::Deck& deck) {
    const int n = m_grid.n;
    const std::vector<float> w = BuildCap(m_grid, deck);

    // Vprop = exp(-i V dt/2) * exp(-W dt/2)   (per half kick)
    std::vector<float> vprop(static_cast<size_t>(n) * n * 2);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const size_t k = static_cast<size_t>(j) * n + i;
            const double ph = -m_vCpu[k] * m_dt * 0.5;
            const double damp = std::exp(-static_cast<double>(w[k]) * m_dt * 0.5);
            vprop[2 * k + 0] = static_cast<float>(damp * std::cos(ph));
            vprop[2 * k + 1] = static_cast<float>(damp * std::sin(ph));
        }
    }
    glNamedBufferSubData(m_vprop, 0, static_cast<GLsizeiptr>(vprop.size() * sizeof(float)), vprop.data());

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
}

void TdseSim::Reset() { UploadInitial(); }

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

void TdseSim::RunStep() {
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

void TdseSim::Step(int substeps) {
    for (int s = 0; s < substeps; ++s) {
        RunStep();
        ++m_stepsDone;
    }
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

    // Kinetic energy and momentum, spectrally exact: 2D FFT of psi (into scratch
    // buffers that Step() also uses transiently) then sum against k.
    double T = 0.0, px = 0.0, py = 0.0;
    {
        const GLsizeiptr bytes = static_cast<GLsizeiptr>(n) * n * 2 * sizeof(float);
        glCopyNamedBufferSubData(m_psi, m_spec, 0, 0, bytes);
        Fft(m_spec, false);
        TransposeBuf(m_spec, m_tmp);
        Fft(m_tmp, false); // m_tmp = psi-hat, layout [kxIdx][kyIdx]

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
    put("kinetic", T);
    put("potential_energy", V);
    put("energy", T + V);

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
    const float base = static_cast<float>(std::min(fbWidth, fbHeight)) /
                       static_cast<float>(std::max(m_grid.lx, m_grid.ly));
    const float pixPerUnit = base * m_zoom;

    float vref = 0.0f;
    for (float v : m_vCpu) vref = std::max(vref, std::abs(v));

    // Peak |psi|^2 this frame -> the view auto-scales brightness to it.
    const uint32_t zero = 0;
    glNamedBufferSubData(m_stat, 0, sizeof(zero), &zero);
    m_reduceMax.Use();
    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_stat);
    m_reduceMax.Dispatch(static_cast<GLuint>(n / 16), static_cast<GLuint>(n / 16), 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT);

    glDisable(GL_DEPTH_TEST);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2(fbWidth, fbHeight));
    m_view.SetInt("uN", n);
    m_view.SetFloat("uLx", static_cast<float>(m_grid.lx));
    m_view.SetFloat("uLy", static_cast<float>(m_grid.ly));
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetInt("uMode", m_mode);
    m_view.SetFloat("uGain", m_gain);
    m_view.SetFloat("uGamma", m_gamma);
    m_view.SetFloat("uVref", vref);

    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_potential);
    fw::ComputeShader::BindBuffer(2, m_stat);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    if (m_showCurrent) {
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
        m_zoom = std::clamp(m_zoom, 0.2f, 20.0f);
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
        default: break;
    }
}

} // namespace tdse
