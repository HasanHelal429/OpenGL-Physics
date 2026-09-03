#include "BiotSavart.hpp"

#include "kernels_biot_savart.hpp"

#include "framework/Deck.hpp"

#include <cmath>
#include <cstdio>

namespace mag {

namespace {
constexpr double kPi = 3.14159265358979323846;

// An orthonormal (u, v) spanning the plane perpendicular to `n`.
void Basis(const glm::dvec3& n, glm::dvec3& u, glm::dvec3& v) {
    const glm::dvec3 e =
        std::abs(n.z) < 0.9 ? glm::dvec3(0, 0, 1) : glm::dvec3(1, 0, 0);
    u = glm::normalize(glm::cross(n, e));
    v = glm::cross(n, u);
}

void AppendPolyline(std::vector<WireSegment>& out, const std::vector<glm::dvec3>& pts,
                    bool closed, double current) {
    const std::size_t n = pts.size();
    if (n < 2) return;
    const std::size_t last = closed ? n : n - 1;
    for (std::size_t k = 0; k < last; ++k) {
        const glm::dvec3& a = pts[k];
        const glm::dvec3& b = pts[(k + 1) % n];
        WireSegment s;
        s.mid = 0.5 * (a + b);
        s.dl = current * (b - a);
        out.push_back(s);
    }
}

std::vector<glm::dvec3> Circle(const glm::dvec3& c, const glm::dvec3& u,
                               const glm::dvec3& v, double R, int segs) {
    std::vector<glm::dvec3> p(segs);
    for (int k = 0; k < segs; ++k) {
        const double th = 2.0 * kPi * k / segs;
        p[k] = c + R * (std::cos(th) * u + std::sin(th) * v);
    }
    return p;
}
} // namespace

std::vector<CoilSpec> ParseCoils(const fw::Deck& deck) {
    std::vector<CoilSpec> coils;
    for (const auto& t : deck.GetTables("coil")) {
        CoilSpec c;
        c.shape = t.GetString("shape", "loop");
        const auto ctr = t.GetDoubleArray("center");
        if (ctr.size() == 3) c.center = {ctr[0], ctr[1], ctr[2]};
        const auto ax = t.GetDoubleArray("axis");
        if (ax.size() == 3) c.axis = {ax[0], ax[1], ax[2]};
        if (glm::length(c.axis) < 1e-12) c.axis = {0, 0, 1};
        c.axis = glm::normalize(c.axis);
        c.radius = t.GetDouble("radius", 1.0);
        c.current = t.GetDouble("current", 1.0);
        c.turns = t.GetInt("turns", 1);
        c.length = t.GetDouble("length", 0.0);
        c.spacing = t.GetDouble("spacing", 0.0);
        c.segmentsPerTurn = t.GetInt("segments", 128);
        coils.push_back(c);
    }
    return coils;
}

std::vector<WireSegment> BuildSegments(const std::vector<CoilSpec>& coils) {
    std::vector<WireSegment> segs;
    for (const CoilSpec& c : coils) {
        glm::dvec3 u, v;
        Basis(c.axis, u, v);

        if (c.shape == "helmholtz") {
            const double s = c.spacing > 0.0 ? c.spacing : c.radius;
            for (int side = -1; side <= 1; side += 2) {
                const glm::dvec3 cc = c.center + (0.5 * s * side) * c.axis;
                AppendPolyline(segs, Circle(cc, u, v, c.radius, c.segmentsPerTurn),
                               true, c.current);
            }
        } else if (c.shape == "solenoid") {
            const int total = std::max(1, c.turns) * c.segmentsPerTurn;
            const double L = c.length > 0.0 ? c.length : 2.0 * c.radius;
            std::vector<glm::dvec3> helix(total + 1);
            for (int k = 0; k <= total; ++k) {
                const double th = 2.0 * kPi * k / c.segmentsPerTurn;
                const double ax = -0.5 * L + L * double(k) / total;
                helix[k] = c.center + ax * c.axis +
                           c.radius * (std::cos(th) * u + std::sin(th) * v);
            }
            AppendPolyline(segs, helix, false, c.current);
        } else { // loop
            AppendPolyline(segs, Circle(c.center, u, v, c.radius, c.segmentsPerTurn),
                           true, c.current * std::max(1, c.turns));
        }
    }
    return segs;
}

glm::dvec3 BiotSavartAt(const std::vector<WireSegment>& segs, const glm::dvec3& r,
                        double mu0) {
    glm::dvec3 acc(0.0);
    for (const WireSegment& s : segs) {
        const glm::dvec3 R = r - s.mid;
        const double r2 = glm::dot(R, R);
        if (r2 < 1e-20) continue;
        acc += glm::cross(s.dl, R) / (r2 * std::sqrt(r2));
    }
    return (mu0 / (4.0 * kPi)) * acc;
}

// --------------------------------------------------------------------- GPU

void BiotSavartField::Init(const Grid& g) {
    if (m_ready) return;
    m_prog = fw::ComputeShader::FromSource(kernels::BiotSavart());
    m_cells = g.count();
    glCreateBuffers(1, &m_outBuf);
    glNamedBufferData(m_outBuf,
                      static_cast<GLsizeiptr>(m_cells * 4 * sizeof(float)),
                      nullptr, GL_DYNAMIC_DRAW);
    m_ready = true;
}

void BiotSavartField::Evaluate(const Grid& g, const SlicePlane& plane, double mu0,
                               const std::vector<WireSegment>& segs,
                               std::vector<double>& bx, std::vector<double>& by,
                               std::vector<double>& bz) {
    const int nseg = static_cast<int>(segs.size());
    if (nseg > m_segCap) {
        if (m_segBuf) glDeleteBuffers(1, &m_segBuf);
        glCreateBuffers(1, &m_segBuf);
        glNamedBufferData(m_segBuf,
                          static_cast<GLsizeiptr>(nseg * 8 * sizeof(float)),
                          nullptr, GL_DYNAMIC_DRAW);
        m_segCap = nseg;
    }
    std::vector<float> packed(static_cast<std::size_t>(nseg) * 8);
    for (int s = 0; s < nseg; ++s) {
        packed[s * 8 + 0] = static_cast<float>(segs[s].mid.x);
        packed[s * 8 + 1] = static_cast<float>(segs[s].mid.y);
        packed[s * 8 + 2] = static_cast<float>(segs[s].mid.z);
        packed[s * 8 + 4] = static_cast<float>(segs[s].dl.x);
        packed[s * 8 + 5] = static_cast<float>(segs[s].dl.y);
        packed[s * 8 + 6] = static_cast<float>(segs[s].dl.z);
    }
    glNamedBufferSubData(m_segBuf, 0,
                         static_cast<GLsizeiptr>(packed.size() * sizeof(float)),
                         packed.data());

    m_prog.Use();
    m_prog.SetInt("uNx", g.nx);
    m_prog.SetInt("uNy", g.ny);
    m_prog.SetFloat("uX0", static_cast<float>(-0.5 * g.lx));
    m_prog.SetFloat("uDx", static_cast<float>(g.dx()));
    m_prog.SetFloat("uY0", static_cast<float>(-0.5 * g.ly));
    m_prog.SetFloat("uDy", static_cast<float>(g.dy()));
    m_prog.SetInt("uNseg", nseg);
    m_prog.SetInt("uPlane", static_cast<int>(plane.kind));
    m_prog.SetFloat("uOffset", static_cast<float>(plane.offset));
    m_prog.SetFloat("uK", static_cast<float>(mu0 / (4.0 * kPi)));

    fw::ComputeShader::BindBuffer(0, m_segBuf);
    fw::ComputeShader::BindBuffer(1, m_outBuf);
    m_prog.Dispatch((g.nx + 15) / 16, (g.ny + 15) / 16, 1);
    fw::ComputeShader::Barrier(GL_SHADER_STORAGE_BARRIER_BIT |
                               GL_BUFFER_UPDATE_BARRIER_BIT);

    std::vector<float> out(static_cast<std::size_t>(m_cells) * 4);
    glGetNamedBufferSubData(m_outBuf, 0,
                            static_cast<GLsizeiptr>(out.size() * sizeof(float)),
                            out.data());

    bx.assign(m_cells, 0.0);
    by.assign(m_cells, 0.0);
    bz.assign(m_cells, 0.0);
    for (int k = 0; k < m_cells; ++k) {
        const glm::dvec3 w(out[k * 4 + 0], out[k * 4 + 1], out[k * 4 + 2]);
        const glm::dvec2 ip = plane.InPlane(w);
        bx[k] = ip.x;
        by[k] = ip.y;
        // out-of-plane component
        switch (plane.kind) {
        case SlicePlane::XZ: bz[k] = w.y; break;
        case SlicePlane::YZ: bz[k] = w.x; break;
        default:             bz[k] = w.z; break;
        }
    }
}

} // namespace mag
