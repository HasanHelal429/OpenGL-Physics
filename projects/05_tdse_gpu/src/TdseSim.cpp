#include "TdseSim.hpp"

#include "kernels.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cmath>
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

layout(std430, binding = 0) readonly buffer Psi { vec2 psi[]; };
layout(std430, binding = 1) readonly buffer Pot { float pot[]; };

uniform vec2 uRes;
uniform int uN;
uniform float uLx;
uniform float uLy;
uniform float uPixPerUnit;
uniform vec2 uPanPix;
uniform int uMode;
uniform float uExposure;
uniform float uVref;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
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

    vec3 col;
    if (uMode == 1) {
        float g = 1.0 - exp(-uExposure * mag2 * float(uN) * float(uN) * dx * dy);
        col = vec3(g);
    } else if (uMode == 2) {
        float r = p.x;
        float s = tanh(uExposure * r * float(uN) * sqrt(dx * dy));
        col = mix(vec3(0.15, 0.3, 0.9), vec3(0.95, 0.25, 0.2), 0.5 + 0.5 * s);
        col *= 0.4 + 0.6 * abs(s);
    } else {
        float phase = atan(p.y, p.x);
        float hue = phase / (2.0 * 3.14159265) + 0.5;
        float val = 1.0 - exp(-uExposure * mag2 * float(uN) * float(uN) * dx * dy);
        col = hsv2rgb(vec3(hue, 0.85, val));
    }

    if (uVref > 0.0) {
        float t = clamp(abs(v) / uVref, 0.0, 1.0);
        col = mix(col, vec3(0.55, 0.55, 0.6), 0.22 * t);
    }
    FragColor = vec4(col, 1.0);
}
)";

} // namespace

TdseSim::~TdseSim() {
    GLuint bufs[] = {m_psi, m_tmp, m_spec, m_vprop, m_kprop, m_twiddle, m_potential};
    glDeleteBuffers(7, bufs);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
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
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    glGenVertexArrays(1, &m_vao);

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
    info.frameFields = {"psi", "potential"};
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

    glDisable(GL_DEPTH_TEST);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2(fbWidth, fbHeight));
    m_view.SetInt("uN", n);
    m_view.SetFloat("uLx", static_cast<float>(m_grid.lx));
    m_view.SetFloat("uLy", static_cast<float>(m_grid.ly));
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetInt("uMode", m_mode);
    m_view.SetFloat("uExposure", m_exposure);
    m_view.SetFloat("uVref", vref);

    fw::ComputeShader::BindBuffer(0, m_psi);
    fw::ComputeShader::BindBuffer(1, m_potential);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);
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
        case '[': m_exposure = std::max(0.5f, m_exposure * 0.8f); break;
        case ']': m_exposure = std::min(80.0f, m_exposure * 1.25f); break;
        case '0': m_zoom = 1.0f; m_panPix = {0.0f, 0.0f}; break;
        default: break;
    }
}

} // namespace tdse
