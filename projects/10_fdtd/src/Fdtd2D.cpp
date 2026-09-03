#include "Fdtd2D.hpp"

#include "kernels_fdtd.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

namespace fdtd {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Fdtd2D::~Fdtd2D() {
    const GLuint bufs[] = {m_bEz, m_bHx, m_bHy, m_bCa, m_bCb, m_bEzPrev, m_bSrc};
    for (GLuint b : bufs)
        if (b) glDeleteBuffers(1, &b);
}

double Source::operator()(double t) const {
    if (kind == Gaussian) {
        const double x = (t - t0) / tau;
        return amplitude * std::exp(-x * x);
    }
    // ramped sine
    const double period = 1.0 / f0;
    const double ramp = t >= rampCycles * period
                            ? 1.0
                            : 0.5 * (1.0 - std::cos(kPi * t / (rampCycles * period)));
    return amplitude * ramp * std::sin(2.0 * kPi * f0 * t);
}

void Fdtd2D::Configure(const fw::Deck& deck) {
    m_nx = deck.GetInt("grid.nx", 200);
    m_ny = deck.GetInt("grid.ny", m_nx);
    m_dx = deck.GetDouble("grid.dx", 1.0);
    m_dy = deck.GetDouble("grid.dy", m_dx);
    m_c = deck.GetDouble("physics.c", 1.0);
    m_courant = deck.GetDouble("grid.courant", 0.5);
    m_title = deck.GetString("title", "2D FDTD (TMz)");
    m_boundary = deck.GetString("boundary.type", "mur");
    if (deck.GetString("solver.backend", "cpu") == "gpu") m_useGpu = true;

    // Square-cell CFL; the general form carries the aspect ratio.
    m_dt = m_courant / (m_c * std::sqrt(1.0 / (m_dx * m_dx) + 1.0 / (m_dy * m_dy)));

    m_totalSteps = deck.GetInt("time.steps", 2000);
    m_substepsPerFrame = deck.GetInt("time.substeps_per_frame", 4);

    const std::size_t n = static_cast<std::size_t>(m_nx) * m_ny;
    m_ez.assign(n, 0.0);
    m_hx.assign(n, 0.0);
    m_hy.assign(n, 0.0);
    m_ezPrev.assign(n, 0.0);
    m_hxPrev.assign(n, 0.0);
    m_hyPrev.assign(n, 0.0);
    m_energy = 0.0;

    // E update coefficients in ca/cb form (Phase 1: vacuum, so ca = 1).
    m_ca.assign(n, 1.0);
    m_cb.assign(n, m_dt);   // dt / eps, eps = 1
    m_muInv = 1.0;

    m_sources.clear();
    for (const auto& s : deck.GetTables("source")) {
        Source src;
        const std::string k = s.GetString("kind", "gaussian");
        src.kind = (k == "sine") ? Source::Sine : Source::Gaussian;
        // Position given in physical coords or fractional; accept x/y in cells.
        src.i = std::clamp(static_cast<int>(std::lround(
                    s.GetDouble("x", 0.5 * (m_nx - 1) * m_dx) / m_dx)), 1, m_nx - 2);
        src.j = std::clamp(static_cast<int>(std::lround(
                    s.GetDouble("y", 0.5 * (m_ny - 1) * m_dy) / m_dy)), 1, m_ny - 2);
        src.amplitude = s.GetDouble("amplitude", 1.0);
        src.f0 = s.GetDouble("f0", 0.05);
        src.bandwidth = s.GetDouble("bandwidth", 0.0);
        src.rampCycles = s.GetDouble("ramp_cycles", 3.0);
        if (src.kind == Source::Gaussian) {
            // tau from bandwidth if given, else ~half a period at f0.
            src.tau = src.bandwidth > 0.0 ? 1.0 / (2.0 * kPi * src.bandwidth)
                                          : 0.5 / src.f0;
            src.t0 = s.GetDouble("t0", 4.0 * src.tau);
        }
        m_sources.push_back(src);
    }

    m_probes.clear();
    {
        // probe.points is a flat [x0, y0, x1, y1, ...] array in physical coords.
        const auto pts = deck.GetDoubleArray("probe.points");
        for (std::size_t k = 0; k + 1 < pts.size(); k += 2) {
            const int pi = std::clamp(static_cast<int>(std::lround(pts[k] / m_dx)), 0, m_nx - 1);
            const int pj = std::clamp(static_cast<int>(std::lround(pts[k + 1] / m_dy)), 0, m_ny - 1);
            m_probes.emplace_back(pi, pj);
        }
    }

    m_time = 0.0;
    m_step = 0;
    std::printf("[fdtd] %dx%d  dx=%.3g  dt=%.4g  courant=%.3f  boundary=%s  "
                "steps=%ld\n",
                m_nx, m_ny, m_dx, m_dt, m_courant, m_boundary.c_str(), m_totalSteps);
}

void Fdtd2D::Reset() {
    std::fill(m_ez.begin(), m_ez.end(), 0.0);
    std::fill(m_hx.begin(), m_hx.end(), 0.0);
    std::fill(m_hy.begin(), m_hy.end(), 0.0);
    std::fill(m_ezPrev.begin(), m_ezPrev.end(), 0.0);
    std::fill(m_hxPrev.begin(), m_hxPrev.end(), 0.0);
    std::fill(m_hyPrev.begin(), m_hyPrev.end(), 0.0);
    m_energy = 0.0;
    m_time = 0.0;
    m_step = 0;
    if (m_gpuInit) UploadToGpu();
}

void Fdtd2D::UpdateH() {
    const double cx = m_dt * m_muInv / m_dx;
    const double cy = m_dt * m_muInv / m_dy;
    for (int j = 0; j < m_ny - 1; ++j) {
        for (int i = 0; i < m_nx; ++i) {
            m_hx[idx(i, j)] -= cy * (m_ez[idx(i, j + 1)] - m_ez[idx(i, j)]);
        }
    }
    for (int j = 0; j < m_ny; ++j) {
        for (int i = 0; i < m_nx - 1; ++i) {
            m_hy[idx(i, j)] += cx * (m_ez[idx(i + 1, j)] - m_ez[idx(i, j)]);
        }
    }
}

void Fdtd2D::UpdateE() {
    for (int j = 1; j < m_ny - 1; ++j) {
        for (int i = 1; i < m_nx - 1; ++i) {
            const std::size_t p = idx(i, j);
            const double curl = (m_hy[p] - m_hy[idx(i - 1, j)]) / m_dx -
                                (m_hx[p] - m_hx[idx(i, j - 1)]) / m_dy;
            m_ez[p] = m_ca[p] * m_ez[p] + m_cb[p] * curl;
        }
    }
}

// First-order Mur absorbing condition on the four edges. Uses the pre-update
// field (m_ezPrev) for "old" values and the just-updated interior for "new".
void Fdtd2D::ApplyMur() {
    const double coef = (m_c * m_dt - m_dx) / (m_c * m_dt + m_dx);
    const double coefy = (m_c * m_dt - m_dy) / (m_c * m_dt + m_dy);
    for (int j = 0; j < m_ny; ++j) {
        m_ez[idx(0, j)] = m_ezPrev[idx(1, j)] +
            coef * (m_ez[idx(1, j)] - m_ezPrev[idx(0, j)]);
        m_ez[idx(m_nx - 1, j)] = m_ezPrev[idx(m_nx - 2, j)] +
            coef * (m_ez[idx(m_nx - 2, j)] - m_ezPrev[idx(m_nx - 1, j)]);
    }
    for (int i = 0; i < m_nx; ++i) {
        m_ez[idx(i, 0)] = m_ezPrev[idx(i, 1)] +
            coefy * (m_ez[idx(i, 1)] - m_ezPrev[idx(i, 0)]);
        m_ez[idx(i, m_ny - 1)] = m_ezPrev[idx(i, m_ny - 2)] +
            coefy * (m_ez[idx(i, m_ny - 2)] - m_ezPrev[idx(i, m_ny - 1)]);
    }
}

void Fdtd2D::InjectSources() {
    for (const Source& s : m_sources)
        m_ez[idx(s.i, s.j)] += m_cb[idx(s.i, s.j)] * s(m_time);
}

void Fdtd2D::Step(int substeps) {
    if (m_useGpu) {
        if (!m_gpuInit) InitGpu();
        StepGpu(substeps);
        return;
    }
    for (int s = 0; s < substeps; ++s) {
        m_hxPrev = m_hx;             // H(n-1/2)
        m_hyPrev = m_hy;
        UpdateH();                   // -> H(n+1/2), using E(n)

        // Exact conserved discrete energy, evaluated now (E(n), H(n±1/2)).
        double u = 0.0;
        for (std::size_t k = 0; k < m_ez.size(); ++k)
            u += m_ez[k] * m_ez[k] + m_hx[k] * m_hxPrev[k] + m_hy[k] * m_hyPrev[k];
        m_energy = 0.5 * u * m_dx * m_dy;

        if (m_boundary == "mur") m_ezPrev = m_ez;
        UpdateE();                   // -> E(n+1), using H(n+1/2)
        InjectSources();
        if (m_boundary == "mur") ApplyMur();
        // "pec": edge E_z cells are never written by UpdateE, stay 0.
        m_time += m_dt;
        ++m_step;
    }
}

double Fdtd2D::NaiveEnergy() const {
    double u = 0.0;
    for (std::size_t k = 0; k < m_ez.size(); ++k)
        u += m_ez[k] * m_ez[k] + m_hx[k] * m_hx[k] + m_hy[k] * m_hy[k];
    return 0.5 * u * m_dx * m_dy;
}

// -------------------------------------------------------------- GPU backend

namespace {
GLuint MakeBuf(const void* data, GLsizeiptr bytes) {
    GLuint b = 0;
    glCreateBuffers(1, &b);
    glNamedBufferData(b, bytes, data, GL_DYNAMIC_DRAW);
    return b;
}
}

void Fdtd2D::InitGpu() {
    m_kH = fw::ComputeShader::FromSource(kernels::UpdateH());
    m_kE = fw::ComputeShader::FromSource(kernels::UpdateE());
    m_kCopyPrev = fw::ComputeShader::FromSource(kernels::CopyPrev());
    m_kInject = fw::ComputeShader::FromSource(kernels::Inject());
    m_kMur = fw::ComputeShader::FromSource(kernels::Mur());

    const int n = m_nx * m_ny;
    m_scratch.assign(n, 0.0f);
    std::vector<float> caF(m_ca.begin(), m_ca.end());
    std::vector<float> cbF(m_cb.begin(), m_cb.end());

    m_bEz = MakeBuf(nullptr, n * sizeof(float));
    m_bHx = MakeBuf(nullptr, n * sizeof(float));
    m_bHy = MakeBuf(nullptr, n * sizeof(float));
    m_bEzPrev = MakeBuf(nullptr, n * sizeof(float));
    m_bCa = MakeBuf(caF.data(), n * sizeof(float));
    m_bCb = MakeBuf(cbF.data(), n * sizeof(float));
    m_bSrc = MakeBuf(nullptr, std::max<std::size_t>(1, m_sources.size()) * sizeof(glm::vec2));
    m_srcStage.resize(std::max<std::size_t>(1, m_sources.size()));

    m_gpuInit = true;
    UploadToGpu();
}

void Fdtd2D::UploadToGpu() {
    const int n = m_nx * m_ny;
    auto up = [&](GLuint b, const std::vector<double>& src) {
        for (int k = 0; k < n; ++k) m_scratch[k] = static_cast<float>(src[k]);
        glNamedBufferSubData(b, 0, n * sizeof(float), m_scratch.data());
    };
    up(m_bEz, m_ez);
    up(m_bHx, m_hx);
    up(m_bHy, m_hy);
    up(m_bEzPrev, m_ezPrev);
    m_cpuStale = false;
}

void Fdtd2D::StepGpu(int substeps) {
    const int n = m_nx * m_ny;
    const GLuint gx = (m_nx + 15) / 16, gy = (m_ny + 15) / 16;
    const bool mur = m_boundary == "mur";
    const double invDx = 1.0 / m_dx, invDy = 1.0 / m_dy;
    const double coefX = (m_c * m_dt - m_dx) / (m_c * m_dt + m_dx);
    const double coefY = (m_c * m_dt - m_dy) / (m_c * m_dt + m_dy);

    // Uniforms and buffer bindings that never change across substeps.
    m_kH.Use();
    m_kH.SetInt("uNx", m_nx); m_kH.SetInt("uNy", m_ny);
    m_kH.SetFloat("uCx", static_cast<float>(m_dt * m_muInv / m_dx));
    m_kH.SetFloat("uCy", static_cast<float>(m_dt * m_muInv / m_dy));
    m_kE.Use();
    m_kE.SetInt("uNx", m_nx); m_kE.SetInt("uNy", m_ny);
    m_kE.SetFloat("uInvDx", static_cast<float>(invDx));
    m_kE.SetFloat("uInvDy", static_cast<float>(invDy));
    m_kCopyPrev.Use();
    m_kCopyPrev.SetInt("uN", n);
    m_kMur.Use();
    m_kMur.SetInt("uNx", m_nx); m_kMur.SetInt("uNy", m_ny);
    m_kMur.SetFloat("uCoefX", static_cast<float>(coefX));
    m_kMur.SetFloat("uCoefY", static_cast<float>(coefY));

    fw::ComputeShader::BindBuffer(0, m_bEz);
    fw::ComputeShader::BindBuffer(1, m_bHx);
    fw::ComputeShader::BindBuffer(2, m_bHy);
    fw::ComputeShader::BindBuffer(3, m_bCa);
    fw::ComputeShader::BindBuffer(4, m_bCb);
    fw::ComputeShader::BindBuffer(5, m_bEzPrev);
    fw::ComputeShader::BindBuffer(6, m_bSrc);

    for (int s = 0; s < substeps; ++s) {
        m_kH.Use();
        m_kH.Dispatch(gx, gy);
        fw::ComputeShader::Barrier();

        if (mur) {
            m_kCopyPrev.Use();
            m_kCopyPrev.Dispatch((n + 255) / 256);
            fw::ComputeShader::Barrier();
        }

        m_kE.Use();
        m_kE.Dispatch(gx, gy);
        fw::ComputeShader::Barrier();

        // sources
        if (!m_sources.empty()) {
            for (std::size_t k = 0; k < m_sources.size(); ++k) {
                const Source& src = m_sources[k];
                m_srcStage[k] = glm::vec2(
                    static_cast<float>(idx(src.i, src.j)),
                    static_cast<float>(src(m_time)));
            }
            glNamedBufferSubData(m_bSrc, 0,
                                 m_srcStage.size() * sizeof(glm::vec2),
                                 m_srcStage.data());
            m_kInject.Use();
            m_kInject.SetInt("uCount", static_cast<int>(m_sources.size()));
            m_kInject.Dispatch(
                static_cast<GLuint>((m_sources.size() + 63) / 64));
            fw::ComputeShader::Barrier();
        }

        if (mur) {
            m_kMur.Use();
            m_kMur.Dispatch(gx, gy);
            fw::ComputeShader::Barrier();
        }

        m_time += m_dt;
        ++m_step;
    }
    m_cpuStale = true;
}

void Fdtd2D::SyncFromGpu() {
    if (!m_gpuInit || !m_cpuStale) return;
    const int n = m_nx * m_ny;
    auto down = [&](GLuint b, std::vector<double>& dst) {
        glGetNamedBufferSubData(b, 0, n * sizeof(float), m_scratch.data());
        for (int k = 0; k < n; ++k) dst[k] = m_scratch[k];
    };
    down(m_bEz, m_ez);
    down(m_bHx, m_hx);
    down(m_bHy, m_hy);
    m_energy = NaiveEnergy();   // GPU path: prev-H not tracked, use the naive sum
    m_cpuStale = false;
}

void Fdtd2D::Snapshot(fw::OutputWriter& writer) {
    if (m_useGpu) SyncFromGpu();
    writer.WriteField("Ez", m_ez.data(), fw::NpyDtype::F8, m_ny, m_nx);
    writer.WriteScalar("energy", TotalEnergy());
    double ezmax = 0.0;
    for (double v : m_ez) ezmax = std::max(ezmax, std::abs(v));
    writer.WriteScalar("Ez_max", ezmax);
    for (std::size_t k = 0; k < m_probes.size(); ++k) {
        char name[24];
        std::snprintf(name, sizeof(name), "probe%zu_Ez", k);
        writer.WriteScalar(name, m_ez[idx(m_probes[k].first, m_probes[k].second)]);
    }
}

fw::SimInfo Fdtd2D::Info() const {
    fw::SimInfo info;
    info.title = m_title;
    info.gridNx = m_nx;
    info.gridNy = m_ny;
    info.lx = (m_nx - 1) * m_dx;
    info.ly = (m_ny - 1) * m_dy;
    info.dt = m_dt;
    info.substepsPerFrame = m_substepsPerFrame;
    info.frameFields = {"Ez"};
    info.diagnostics = {"energy", "Ez_max"};
    for (std::size_t k = 0; k < m_probes.size(); ++k) {
        char name[24];
        std::snprintf(name, sizeof(name), "probe%zu_Ez", k);
        info.diagnostics.emplace_back(name);
    }
    return info;
}

} // namespace fdtd
