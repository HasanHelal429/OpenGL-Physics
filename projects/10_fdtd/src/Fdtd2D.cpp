#include "Fdtd2D.hpp"

#include "kernels_fdtd.hpp"

#include "framework/Deck.hpp"
#include "framework/OutputWriter.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

namespace fdtd {

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Fdtd2D::~Fdtd2D() {
    const GLuint bufs[] = {m_bEz, m_bHx, m_bHy, m_bCa, m_bCb, m_bEzPrev, m_bSrc,
                           m_fieldBuf, m_matBuf};
    for (GLuint b : bufs)
        if (b) glDeleteBuffers(1, &b);
    if (m_vao) glDeleteVertexArrays(1, &m_vao);
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
    m_outputH = deck.GetBool("output.h_fields", false);

    const std::size_t n = static_cast<std::size_t>(m_nx) * m_ny;
    m_ez.assign(n, 0.0);
    m_hx.assign(n, 0.0);
    m_hy.assign(n, 0.0);
    m_ezPrev.assign(n, 0.0);
    m_hxPrev.assign(n, 0.0);
    m_hyPrev.assign(n, 0.0);
    m_energy = 0.0;

    // Materials -> E update coefficients ca/cb, per-cell 1/mu, PEC mask.
    m_mat.Build(m_nx, m_ny, m_dx, m_dy, deck);
    m_ca.assign(n, 1.0);
    m_cb.assign(n, m_dt);
    m_muInvCell.assign(n, 1.0);
    for (std::size_t k = 0; k < n; ++k) {
        const double eps = m_mat.epsR[k];
        const double sg = m_mat.sigma[k];
        const double denom = 1.0 + sg * m_dt / (2.0 * eps);
        m_ca[k] = (1.0 - sg * m_dt / (2.0 * eps)) / denom;
        m_cb[k] = (m_dt / eps) / denom;
        m_muInvCell[k] = 1.0 / m_mat.muR[k];
    }
    m_hasPec = m_mat.AnyPec();

    // CPML: profiles (trivial unless boundary.type = "cpml") + auxiliary fields.
    m_pmlCells = deck.GetInt("boundary.pml_cells", 10);
    const int pml = (m_boundary == "cpml") ? m_pmlCells : 0;
    const double gradeM = deck.GetDouble("boundary.pml_grade", 3.0);
    const double kappaMax = deck.GetDouble("boundary.pml_kappa", 11.0);
    const double alphaMax = deck.GetDouble("boundary.pml_alpha", 0.15);
    const double r0 = deck.GetDouble("boundary.pml_r0", 1e-8);
    m_cpmlX.Build(m_nx, m_dx, pml, m_dt, gradeM, kappaMax, alphaMax, r0);
    m_cpmlY.Build(m_ny, m_dy, pml, m_dt, gradeM, kappaMax, alphaMax, r0);
    m_psiEzx.assign(n, 0.0);
    m_psiEzy.assign(n, 0.0);
    m_psiHxy.assign(n, 0.0);
    m_psiHyx.assign(n, 0.0);
    if (m_useGpu && (m_boundary == "cpml" || m_mat.AnyDielectric() ||
                     m_mat.AnyPec())) {
        std::fprintf(stderr, "[fdtd] GPU path is plain-TMz + Mur only so far "
                             "-- using CPU for this deck\n");
        m_useGpu = false;
    }

    m_sources.clear();
    m_tfsf = Tfsf{};
    for (const auto& s : deck.GetTables("source")) {
        Source src;
        const std::string k = s.GetString("kind", "gaussian");
        src.kind = (k == "sine") ? Source::Sine
                 : (k == "tfsf") ? Source::Tfsf : Source::Gaussian;
        src.i = std::clamp(static_cast<int>(std::lround(
                    s.GetDouble("x", 0.5 * (m_nx - 1) * m_dx) / m_dx)), 1, m_nx - 2);
        src.j = std::clamp(static_cast<int>(std::lround(
                    s.GetDouble("y", 0.5 * (m_ny - 1) * m_dy) / m_dy)), 1, m_ny - 2);
        src.amplitude = s.GetDouble("amplitude", 1.0);
        src.f0 = s.GetDouble("f0", 0.05);
        src.bandwidth = s.GetDouble("bandwidth", 0.0);
        src.rampCycles = s.GetDouble("ramp_cycles", 3.0);
        src.angleDeg = s.GetDouble("angle_deg", 90.0);
        const std::string shape = s.GetString("waveform", "sine");  // tfsf: shape
        const bool gaussianShape =
            src.kind == Source::Gaussian || (src.kind == Source::Tfsf && shape == "gaussian");
        if (gaussianShape) {
            src.tau = src.bandwidth > 0.0 ? 1.0 / (2.0 * kPi * src.bandwidth)
                                          : 0.5 / src.f0;
            src.t0 = s.GetDouble("t0", 4.0 * src.tau);
        }

        if (src.kind == Source::Tfsf) {
            const int pml = (m_boundary == "cpml") ? m_pmlCells : 0;
            const int margin = deck.GetInt("tfsf.margin", 8);
            const int c = pml + margin;
            m_tfsf.active = true;
            m_tfsf.i0 = c;               m_tfsf.i1 = m_nx - 1 - c;
            m_tfsf.j0 = c;               m_tfsf.j1 = m_ny - 1 - c;
            const double th = src.angleDeg * kPi / 180.0;
            m_tfsf.kx = std::cos(th);
            m_tfsf.ky = std::sin(th);
            m_tfsf.refX = (m_tfsf.kx >= 0 ? m_tfsf.i0 : m_tfsf.i1) * m_dx;
            m_tfsf.refY = (m_tfsf.ky >= 0 ? m_tfsf.j0 : m_tfsf.j1) * m_dy;
            m_tfsf.wave = src;
            m_tfsf.wave.kind = gaussianShape ? Source::Gaussian : Source::Sine;

            // 1D auxiliary incident grid: reach = the box's projection onto
            // the beam direction (all corners relative to the ref point).
            double span = 0.0;
            for (int cx : {m_tfsf.i0, m_tfsf.i1})
                for (int cy : {m_tfsf.j0, m_tfsf.j1}) {
                    const double xi = (cx * m_dx - m_tfsf.refX) * m_tfsf.kx +
                                      (cy * m_dy - m_tfsf.refY) * m_tfsf.ky;
                    span = std::max(span, xi);
                }
            m_inc1d.Configure(span, m_dx, m_dt, 2.0 * kPi * src.f0, th);
        } else {
            m_sources.push_back(src);
        }
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

    if (m_useGpu && m_tfsf.active) {
        std::fprintf(stderr, "[fdtd] TFSF not on the GPU path -- using CPU\n");
        m_useGpu = false;
    }

    m_time = 0.0;
    m_step = 0;
    std::printf("[fdtd] %dx%d  dx=%.3g  dt=%.4g  courant=%.3f  boundary=%s  "
                "steps=%ld%s\n",
                m_nx, m_ny, m_dx, m_dt, m_courant, m_boundary.c_str(),
                m_totalSteps, m_tfsf.active ? "  [TFSF]" : "");
}

void Fdtd2D::Reset() {
    std::fill(m_ez.begin(), m_ez.end(), 0.0);
    std::fill(m_hx.begin(), m_hx.end(), 0.0);
    std::fill(m_hy.begin(), m_hy.end(), 0.0);
    std::fill(m_ezPrev.begin(), m_ezPrev.end(), 0.0);
    std::fill(m_hxPrev.begin(), m_hxPrev.end(), 0.0);
    std::fill(m_hyPrev.begin(), m_hyPrev.end(), 0.0);
    std::fill(m_psiEzx.begin(), m_psiEzx.end(), 0.0);
    std::fill(m_psiEzy.begin(), m_psiEzy.end(), 0.0);
    std::fill(m_psiHxy.begin(), m_psiHxy.end(), 0.0);
    std::fill(m_psiHyx.begin(), m_psiHyx.end(), 0.0);
    if (m_tfsf.active) m_inc1d.ResetState();
    m_energy = 0.0;
    m_time = 0.0;
    m_step = 0;
    if (m_gpuInit) UploadToGpu();
}

// CPML-form updates: away from the PML the profile coefficients are trivial
// (b = 1, a = 0, kappa = 1), so psi stays zero and these reduce exactly to the
// plain Yee update -- no interior branch.
void Fdtd2D::UpdateH() {
    for (int j = 0; j < m_ny - 1; ++j) {
        for (int i = 0; i < m_nx; ++i) {
            const std::size_t p = idx(i, j);
            const double dEzdy = (m_ez[idx(i, j + 1)] - m_ez[p]) / m_dy;
            m_psiHxy[p] = m_cpmlY.bH[j] * m_psiHxy[p] + m_cpmlY.aH[j] * dEzdy;
            m_hx[p] -= m_dt * m_muInvCell[p] * (dEzdy / m_cpmlY.kH[j] + m_psiHxy[p]);
        }
    }
    for (int j = 0; j < m_ny; ++j) {
        for (int i = 0; i < m_nx - 1; ++i) {
            const std::size_t p = idx(i, j);
            const double dEzdx = (m_ez[idx(i + 1, j)] - m_ez[p]) / m_dx;
            m_psiHyx[p] = m_cpmlX.bH[i] * m_psiHyx[p] + m_cpmlX.aH[i] * dEzdx;
            m_hy[p] += m_dt * m_muInvCell[p] * (dEzdx / m_cpmlX.kH[i] + m_psiHyx[p]);
        }
    }
}

void Fdtd2D::UpdateE() {
    for (int j = 1; j < m_ny - 1; ++j) {
        for (int i = 1; i < m_nx - 1; ++i) {
            const std::size_t p = idx(i, j);
            const double dHydx = (m_hy[p] - m_hy[idx(i - 1, j)]) / m_dx;
            const double dHxdy = (m_hx[p] - m_hx[idx(i, j - 1)]) / m_dy;
            m_psiEzx[p] = m_cpmlX.bE[i] * m_psiEzx[p] + m_cpmlX.aE[i] * dHydx;
            m_psiEzy[p] = m_cpmlY.bE[j] * m_psiEzy[p] + m_cpmlY.aE[j] * dHxdy;
            const double curl = dHydx / m_cpmlX.kE[i] - dHxdy / m_cpmlY.kE[j] +
                                m_psiEzx[p] - m_psiEzy[p];
            m_ez[p] = m_ca[p] * m_ez[p] + m_cb[p] * curl;
        }
    }
}

void Fdtd2D::EnforcePec() {
    if (!m_hasPec) return;
    for (std::size_t k = 0; k < m_ez.size(); ++k)
        if (m_mat.pec[k]) m_ez[k] = 0.0;
}

double Fdtd2D::IncidentEz(double x, double y) const {
    const double xi = (x - m_tfsf.refX) * m_tfsf.kx + (y - m_tfsf.refY) * m_tfsf.ky;
    return m_inc1d.Ez(xi);
}

void Fdtd2D::IncidentH(double x, double y, double& hx, double& hy) const {
    const double xi = (x - m_tfsf.refX) * m_tfsf.kx + (y - m_tfsf.refY) * m_tfsf.ky;
    // The 1D grid's Hy ~ -Ez for a +xi-propagating wave; the physical
    // transverse H that pairs with the outgoing Ez is +Ez, so negate.
    const double hmag = -m_inc1d.Hmag(xi);
    // H_inc = (1/eta) (k_hat x Ez zhat) = (ky, -kx) * H_transverse,  eta = 1
    hx = m_tfsf.ky * hmag;
    hy = -m_tfsf.kx * hmag;
}

// TF/SF corrections (Taflove ch. 5), one-sided incident-field consistency
// terms added on the scattered side of the box contour.
void Fdtd2D::TfsfCorrectH() {
    if (!m_tfsf.active) return;
    const Tfsf& b = m_tfsf;
    const double cx = m_dt / m_dx, cy = m_dt / m_dy;
    for (int j = b.j0; j <= b.j1; ++j) {
        const double y = j * m_dy;
        m_hy[idx(b.i0 - 1, j)] -= cx * IncidentEz(b.i0 * m_dx, y);
        m_hy[idx(b.i1, j)]     += cx * IncidentEz(b.i1 * m_dx, y);
    }
    for (int i = b.i0; i <= b.i1; ++i) {
        const double x = i * m_dx;
        m_hx[idx(i, b.j0 - 1)] += cy * IncidentEz(x, b.j0 * m_dy);
        m_hx[idx(i, b.j1)]     -= cy * IncidentEz(x, b.j1 * m_dy);
    }
}

void Fdtd2D::TfsfCorrectE() {
    if (!m_tfsf.active) return;
    const Tfsf& b = m_tfsf;
    const double cx = m_dt / m_dx, cy = m_dt / m_dy;
    double hix, hiy;
    for (int j = b.j0; j <= b.j1; ++j) {
        const double y = j * m_dy;
        IncidentH((b.i0 - 0.5) * m_dx, y, hix, hiy);
        m_ez[idx(b.i0, j)] -= cx * hiy;
        IncidentH((b.i1 + 0.5) * m_dx, y, hix, hiy);
        m_ez[idx(b.i1, j)] += cx * hiy;
    }
    for (int i = b.i0; i <= b.i1; ++i) {
        const double x = i * m_dx;
        IncidentH(x, (b.j0 - 0.5) * m_dy, hix, hiy);
        m_ez[idx(i, b.j0)] += cy * hix;
        IncidentH(x, (b.j1 + 0.5) * m_dy, hix, hiy);
        m_ez[idx(i, b.j1)] -= cy * hix;
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
        if (m_tfsf.active) m_inc1d.StepH();   // 1D Hy -> n+1/2 (from 1D Ez(n))
        m_hxPrev = m_hx;             // H(n-1/2)
        m_hyPrev = m_hy;
        UpdateH();                   // -> H(n+1/2), using E(n)
        TfsfCorrectH();              // uses 1D Ez(n)

        // Exact conserved discrete energy, evaluated now (E(n), H(n±1/2)).
        double u = 0.0;
        for (std::size_t k = 0; k < m_ez.size(); ++k)
            u += m_ez[k] * m_ez[k] + m_hx[k] * m_hxPrev[k] + m_hy[k] * m_hyPrev[k];
        m_energy = 0.5 * u * m_dx * m_dy;

        if (m_boundary == "mur") m_ezPrev = m_ez;
        UpdateE();                   // -> E(n+1), using H(n+1/2)
        TfsfCorrectE();              // uses 1D Hy(n+1/2)
        InjectSources();
        if (m_boundary == "mur") ApplyMur();
        EnforcePec();
        if (m_tfsf.active) m_inc1d.StepE(m_tfsf.wave);   // 1D Ez -> n+1
        // "pec"/"cpml": edge E_z cells are never written by UpdateE, stay 0.
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
    // The GPU path only runs for vacuum decks (mu_r = 1 everywhere).
    m_kH.SetFloat("uCx", static_cast<float>(m_dt / m_dx));
    m_kH.SetFloat("uCy", static_cast<float>(m_dt / m_dy));
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
    if (m_outputH) {
        writer.WriteField("Hx", m_hx.data(), fw::NpyDtype::F8, m_ny, m_nx);
        writer.WriteField("Hy", m_hy.data(), fw::NpyDtype::F8, m_ny, m_nx);
    }
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
    info.frameFields = m_outputH ? std::vector<std::string>{"Ez", "Hx", "Hy"}
                                 : std::vector<std::string>{"Ez"};
    info.diagnostics = {"energy", "Ez_max"};
    for (std::size_t k = 0; k < m_probes.size(); ++k) {
        char name[24];
        std::snprintf(name, sizeof(name), "probe%zu_Ez", k);
        info.diagnostics.emplace_back(name);
    }
    return info;
}

// ------------------------------------------------------------- interactive

namespace {

const char* kViewVert = R"(#version 460 core
void main() {
    vec2 v[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    gl_Position = vec4(v[gl_VertexID], 0.0, 1.0);
}
)";

const char* kViewFrag = R"(#version 460 core
out vec4 FragColor;
layout(std430, binding = 0) readonly buffer FieldBuf { float field[]; };
layout(std430, binding = 1) readonly buffer MatBuf { float mat[]; };  // 0 vac, 1 diel, 2 pec
uniform vec2 uRes; uniform int uNx; uniform int uNy;
uniform float uLx; uniform float uLy; uniform float uPixPerUnit; uniform vec2 uPanPix;
uniform float uScale; uniform int uSigned; uniform float uGain; uniform float uGamma;

vec3 magma(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0=vec3(-0.002136,-0.000750,-0.005386), c1=vec3(0.251723,0.677631,2.494027);
    const vec3 c2=vec3(8.353717,-3.577720,0.311613), c3=vec3(-27.668733,14.264731,-13.649213);
    const vec3 c4=vec3(52.176140,-27.943606,12.944169), c5=vec3(-50.768525,29.046583,4.234153);
    const vec3 c6=vec3(18.655705,-11.489774,-5.601962);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), 0.0, 1.0);
}
vec3 diverging(float s) {
    vec3 lo=vec3(0.19,0.31,0.75), mid=vec3(0.98,0.98,0.98), hi=vec3(0.79,0.16,0.16);
    return s < 0.0 ? mix(mid, lo, clamp(-s,0.0,1.0)) : mix(mid, hi, clamp(s,0.0,1.0));
}
void main() {
    float wx = (gl_FragCoord.x - 0.5*uRes.x - uPanPix.x) / uPixPerUnit;
    float wy = (gl_FragCoord.y - 0.5*uRes.y - uPanPix.y) / uPixPerUnit;
    int i = int(floor((wx + 0.5*uLx) / (uLx/float(uNx))));
    int j = int(floor((wy + 0.5*uLy) / (uLy/float(uNy))));
    if (i<0||j<0||i>=uNx||j>=uNy) { FragColor = vec4(0.03,0.03,0.045,1.0); return; }
    int p = j*uNx + i;
    float v = field[p];
    vec3 col;
    if (uSigned == 1) {
        float s = sign(v) * pow(clamp(abs(v/uScale)*uGain,0.0,1.0), uGamma);
        col = diverging(clamp(s,-1.0,1.0));
    } else {
        col = magma(pow(clamp(v/uScale*uGain,0.0,1.0), uGamma));
    }
    float m = mat[p];
    if (m > 1.5)      col = vec3(0.12);                 // PEC: dark solid
    else if (m > 0.5) col = mix(col, vec3(0.35,0.55,0.5), 0.28);  // dielectric tint
    FragColor = vec4(col, 1.0);
}
)";

} // namespace

void Fdtd2D::EnsureRenderResources() {
    if (m_renderReady) return;
    m_view = fw::Shader::FromSource(kViewVert, kViewFrag);
    glGenVertexArrays(1, &m_vao);
    const int n = m_nx * m_ny;
    glCreateBuffers(1, &m_fieldBuf);
    glCreateBuffers(1, &m_matBuf);
    glNamedBufferData(m_fieldBuf, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    m_fieldScratch.assign(n, 0.0f);
    m_matScratch.assign(n, 0.0f);
    for (int k = 0; k < n; ++k)
        m_matScratch[k] = m_mat.pec[k] ? 2.0f : (m_mat.epsR[k] != 1.0 ? 1.0f : 0.0f);
    glNamedBufferData(m_matBuf, n * sizeof(float), m_matScratch.data(), GL_STATIC_DRAW);
    m_text = std::make_unique<fw::TextRenderer>();
    m_font = fw::Font::FromFile("C:\\Windows\\Fonts\\consola.ttf", 16.0f);
    m_renderReady = true;
}

void Fdtd2D::RepackField() {
    const int n = m_nx * m_ny;
    for (int k = 0; k < n; ++k) {
        const double h2 = m_hx[k] * m_hx[k] + m_hy[k] * m_hy[k];
        switch (m_viewMode) {
        case 1:  m_fieldScratch[k] = static_cast<float>(
                     0.5 * (m_mat.epsR[k] * m_ez[k] * m_ez[k] + h2)); break;
        case 2:  m_fieldScratch[k] = static_cast<float>(
                     std::abs(m_ez[k]) * std::sqrt(h2)); break;
        default: m_fieldScratch[k] = static_cast<float>(m_ez[k]); break;
        }
    }
    glNamedBufferSubData(m_fieldBuf, 0, n * sizeof(float), m_fieldScratch.data());
}

void Fdtd2D::Render(int fbWidth, int fbHeight) {
    EnsureRenderResources();
    if (m_useGpu) SyncFromGpu();
    RepackField();

    float scale = 0.0f;
    if (m_viewMode == 0)
        for (float v : m_fieldScratch) scale = std::max(scale, std::abs(v));
    else
        for (float v : m_fieldScratch) scale = std::max(scale, v);
    if (scale < 1e-12f) scale = 1.0f;

    const double lx = (m_nx - 1) * m_dx, ly = (m_ny - 1) * m_dy;
    const float pixPerUnit =
        static_cast<float>(std::min(fbWidth / lx, fbHeight / ly)) * m_zoom;

    glDisable(GL_DEPTH_TEST);
    glClearColor(0.03f, 0.03f, 0.045f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    m_view.Use();
    m_view.SetVec2("uRes", glm::vec2((float)fbWidth, (float)fbHeight));
    m_view.SetInt("uNx", m_nx); m_view.SetInt("uNy", m_ny);
    m_view.SetFloat("uLx", (float)lx); m_view.SetFloat("uLy", (float)ly);
    m_view.SetFloat("uPixPerUnit", pixPerUnit);
    m_view.SetVec2("uPanPix", m_panPix);
    m_view.SetFloat("uScale", scale);
    m_view.SetInt("uSigned", m_viewMode == 0 ? 1 : 0);
    m_view.SetFloat("uGain", m_gain);
    m_view.SetFloat("uGamma", m_gamma);
    fw::ComputeShader::BindBuffer(0, m_fieldBuf);
    fw::ComputeShader::BindBuffer(1, m_matBuf);
    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    static const char* kNames[3] = {"Ez", "E-energy", "|S| (Poynting)"};
    char line[160];
    std::snprintf(line, sizeof(line),
                  "field: %s (M)   t = %.1f   step %ld   [%s, %s]",
                  kNames[m_viewMode], m_time, m_step, m_boundary.c_str(),
                  m_tfsf.active ? "TFSF" : "soft src");
    m_text->SetViewport(fbWidth, fbHeight);
    m_text->Draw(m_font, line, glm::vec2(13.0f, 25.0f), glm::vec4(0, 0, 0, 0.55f));
    m_text->Draw(m_font, line, glm::vec2(12.0f, 24.0f), glm::vec4(1, 1, 0.85f, 0.95f));
    glEnable(GL_DEPTH_TEST);
}

void Fdtd2D::OnViewInput(const fw::ViewInput& in) {
    if (in.dragging) { m_panPix.x += (float)in.dx; m_panPix.y -= (float)in.dy; }
    if (in.scrollDelta != 0.0) {
        m_zoom *= std::exp(0.12f * (float)in.scrollDelta);
        m_zoom = std::clamp(m_zoom, 0.25f, 40.0f);
    }
}

void Fdtd2D::OnKey(int key, int action) {
    (void)action;
    switch (key) {
    case 'M': m_viewMode = (m_viewMode + 1) % 3; break;
    case '[': m_gain = std::max(0.1f, m_gain * 0.8f); break;
    case ']': m_gain = std::min(20.0f, m_gain * 1.25f); break;
    case '-': m_gamma = std::max(0.2f, m_gamma - 0.05f); break;
    case '=': m_gamma = std::min(2.0f, m_gamma + 0.05f); break;
    case '0': m_zoom = 1.0f; m_panPix = {0, 0}; m_gain = 1.0f; m_gamma = 0.7f; break;
    default: break;
    }
}

} // namespace fdtd
