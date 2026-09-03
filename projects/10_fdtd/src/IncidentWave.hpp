#pragma once

#include <cmath>
#include <vector>

namespace fdtd {

// A 1D auxiliary FDTD grid that carries the TFSF incident plane wave along
// its propagation direction. Using a 1D FDTD (rather than the analytic
// e^{i(k.r - w t)}) means the incident field has the *same* numerical
// dispersion as the 2D grid, so the TF/SF contour cancellation stays clean
// no matter how large the box or how oblique the incidence -- the reason
// production FDTD codes always do it this way.
//
// The 1D grid runs a plain 1D scheme  dEz/dt = dHy/dxi,  dHy/dt = dEz/dxi
// (c = eps = mu = 1), hard-sourced at xi = 0, with a 1st-order Mur at the far
// end. Query IncidentEz / IncidentHmag at any distance xi along the beam.
class IncidentWave1D {
public:
    // `span` is the max |xi| the 2D box reaches. `dx`, `dt` are the 2D grid's.
    // `omega` and `theta` (the incidence angle) fix the 1D spacing `dxi` so
    // the 1D numerical phase velocity matches the 2D grid's at this angle --
    // the condition for a clean TF/SF cancellation (Taflove sec. 5.6).
    void Configure(double span, double dx, double dt, double omega, double theta) {
        m_dt = dt;
        m_dxi = DispersionMatchedDxi(dx, dt, omega, theta);
        m_origin = 80;
        const int reach = static_cast<int>(std::ceil(span / m_dxi)) + 80;
        m_n = m_origin + reach;
        m_ez.assign(m_n, 0.0);
        m_hy.assign(m_n, 0.0);
        m_ezOld = 0.0;
        m_time = 0.0;
    }

    // Solve the 2D FDTD dispersion relation for the numerical |k| at (omega,
    // theta), then the 1D relation for the dxi that reproduces the same |k|.
    static double DispersionMatchedDxi(double dx, double dt, double omega,
                                       double theta) {
        const double S = std::sin(0.5 * omega * dt) / dt;   // c = 1
        const double ct = std::cos(theta), st = std::sin(theta);
        // 2D: [S]^2 = [sin(k ct dx/2)/dx]^2 + [sin(k st dx/2)/dx]^2, solve k
        double k = omega;                                   // vacuum guess
        for (int it = 0; it < 60; ++it) {
            const double a = std::sin(0.5 * k * ct * dx) / dx;
            const double b = std::sin(0.5 * k * st * dx) / dx;
            const double f = a * a + b * b - S * S;
            const double da = 0.5 * ct * std::cos(0.5 * k * ct * dx) / dx;
            const double db = 0.5 * st * std::cos(0.5 * k * st * dx) / dx;
            const double df = 2.0 * (a * da + b * db);
            if (std::abs(df) < 1e-30) break;
            const double dk = f / df;
            k -= dk;
            if (std::abs(dk) < 1e-14) break;
        }
        // 1D: S = sin(k dxi/2)/dxi, solve dxi
        double dxi = dx;
        for (int it = 0; it < 60; ++it) {
            const double f = std::sin(0.5 * k * dxi) / dxi - S;
            const double df = (0.5 * k * std::cos(0.5 * k * dxi) * dxi -
                               std::sin(0.5 * k * dxi)) / (dxi * dxi);
            if (std::abs(df) < 1e-30) break;
            const double d = f / df;
            dxi -= d;
            if (std::abs(d) < 1e-14) break;
        }
        return dxi;
    }

    // Leapfrog, split to interleave with the 2D grid: StepH (Hy: n-1/2 ->
    // n+1/2 from Ez(n)) is called before the 2D H update, StepE (Ez: n ->
    // n+1 from Hy(n+1/2)) after the 2D E update.
    void StepH() {
        const double c = m_dt / m_dxi;
        for (int m = 0; m < m_n - 1; ++m)
            m_hy[m] += c * (m_ez[m + 1] - m_ez[m]);
    }
    template <class Waveform>
    void StepE(const Waveform& src) {
        const double c = m_dt / m_dxi;
        const double ezOldEnd = m_ez[m_n - 1];
        for (int m = 1; m < m_n; ++m)
            m_ez[m] += c * (m_hy[m] - m_hy[m - 1]);
        m_ez[0] = src(m_time + m_dt);
        const double coef = (c - 1.0) / (c + 1.0);
        m_ez[m_n - 1] = m_ezOld + coef * (m_ez[m_n - 2] - ezOldEnd);
        m_ezOld = m_ez[m_n - 2];
        m_time += m_dt;
    }
    void ResetState() {
        std::fill(m_ez.begin(), m_ez.end(), 0.0);
        std::fill(m_hy.begin(), m_hy.end(), 0.0);
        m_ezOld = 0.0;
        m_time = 0.0;
    }

    // Incident Ez at signed distance xi from the box reference point.
    double Ez(double xi) const { return SampleE(xi); }
    // Incident transverse-H magnitude at xi (H is at half-cells).
    double Hmag(double xi) const {
        const double f = (xi + m_origin * m_dxi) / m_dxi - 0.5;
        int m = static_cast<int>(std::floor(f));
        if (m < 0) m = 0;
        if (m > m_n - 2) m = m_n - 2;
        const double w = f - m;
        return (1.0 - w) * m_hy[m] + w * m_hy[m + 1];
    }

private:
    double SampleE(double xi) const {
        const double f = (xi + m_origin * m_dxi) / m_dxi;
        int m = static_cast<int>(std::floor(f));
        if (m < 0) m = 0;
        if (m > m_n - 2) m = m_n - 2;
        const double w = f - m;
        return (1.0 - w) * m_ez[m] + w * m_ez[m + 1];
    }

    int m_n = 0, m_origin = 0;
    double m_dxi = 1.0, m_dt = 1.0, m_time = 0.0, m_ezOld = 0.0;
    std::vector<double> m_ez, m_hy;
};

} // namespace fdtd
