#pragma once

#include "LwFields.hpp"      // PathSample

#include <glm/glm.hpp>

#include <cmath>
#include <vector>

namespace lw {

// Fixed-step ring buffer of a charge's history (t, r, v, a), with cubic
// Catmull-Rom lookup in t. Spans at least the domain's light-crossing time
// so every retarded-time query for an in-domain observer lands inside it.
// Startup is handled by Prefill(): lay down an assumed past (uniform drift or
// a steady circular orbit) so forces are defined from t = 0.
class TrajectoryBuffer {
public:
    void Init(double dt, std::size_t cap, double tStart) {
        m_dt = dt;
        m_cap = cap < 4 ? 4 : cap;
        m_buf.assign(m_cap, Node{});
        m_n = 0;
        m_size = 0;
        m_tBase = tStart;
    }

    double Dt() const { return m_dt; }
    std::size_t Span() const { return m_size; }
    double TLatest() const { return m_tBase + (m_n - 1) * m_dt; }
    double TOldest() const { return m_tBase + (m_n - m_size) * m_dt; }

    void Push(const glm::dvec3& r, const glm::dvec3& v, const glm::dvec3& a) {
        m_buf[m_n % m_cap] = Node{r, v, a};
        ++m_n;
        if (m_size < m_cap) ++m_size;
    }

    // Overwrite the whole span with a steady past: sample the callable at
    // t = tEnd - (cap-1-k) dt for k = 0..cap-1, so the newest sample sits at
    // tEnd. Used once, before the first dynamics step.
    template <class Fn>
    void Prefill(double tEnd, Fn&& f) {
        m_tBase = tEnd - static_cast<double>(m_cap - 1) * m_dt;
        for (std::size_t k = 0; k < m_cap; ++k) {
            const double t = m_tBase + static_cast<double>(k) * m_dt;
            const PathSample s = f(t);
            m_buf[k % m_cap] = Node{s.r, s.v, s.a};
        }
        m_n = static_cast<long>(m_cap);
        m_size = m_cap;
    }

    PathSample Sample(double t) const {
        const double fk = (t - m_tBase) / m_dt;
        long k = static_cast<long>(std::floor(fk));
        const double lo = m_n - static_cast<long>(m_size);
        const double hi = m_n - 1;
        // Catmull-Rom needs k-1..k+2; clamp the stencil into the live range.
        if (k < lo + 1) k = static_cast<long>(lo) + 1;
        if (k > hi - 2) k = static_cast<long>(hi) - 2;
        if (k < lo) k = static_cast<long>(lo);          // span < 4: degenerate
        const double u = fk - static_cast<double>(k);
        const Node& p0 = at(k - 1);
        const Node& p1 = at(k);
        const Node& p2 = at(k + 1);
        const Node& p3 = at(k + 2);
        PathSample s;
        s.r = cr(p0.r, p1.r, p2.r, p3.r, u);
        s.v = cr(p0.v, p1.v, p2.v, p3.v, u);
        s.a = cr(p0.a, p1.a, p2.a, p3.a, u);
        return s;
    }

    PathFn PathFn_() const {
        return [this](double t) { return Sample(t); };
    }

private:
    struct Node { glm::dvec3 r{0.0}, v{0.0}, a{0.0}; };

    const Node& at(long k) const {
        long lo = m_n - static_cast<long>(m_size);
        if (k < lo) k = lo;
        if (k > m_n - 1) k = m_n - 1;
        return m_buf[static_cast<std::size_t>(k) % m_cap];
    }

    static glm::dvec3 cr(const glm::dvec3& p0, const glm::dvec3& p1,
                         const glm::dvec3& p2, const glm::dvec3& p3, double u) {
        const double u2 = u * u, u3 = u2 * u;
        return 0.5 * ((2.0 * p1) + (-p0 + p2) * u +
                      (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2 +
                      (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3);
    }

    double m_dt = 0.05;
    std::size_t m_cap = 4;
    std::vector<Node> m_buf;
    long m_n = 0;              // total pushes (sample k has time tBase + k dt)
    std::size_t m_size = 0;
    double m_tBase = 0.0;
};

} // namespace lw
