#include "TdseField.hpp"

#include "framework/Deck.hpp"

#include <algorithm>
#include <cmath>

namespace tdse {

namespace {
constexpr double kPi = 3.14159265358979323846;

double Hermite(int nn, double xi) {
    // Physicists' Hermite polynomials via recurrence.
    double hm1 = 1.0;      // H_0
    if (nn == 0) return hm1;
    double h = 2.0 * xi;   // H_1
    for (int k = 2; k <= nn; ++k) {
        double hn = 2.0 * xi * h - 2.0 * (k - 1) * hm1;
        hm1 = h;
        h = hn;
    }
    return h;
}
} // namespace

std::vector<float> BuildPotential(const Grid& g, const fw::Deck& deck) {
    const int n = g.n;
    std::vector<float> v(static_cast<size_t>(n) * n, 0.0f);

    const auto wells = deck.GetTables("potential");
    for (const auto& p : wells) {
        const std::string type = p.GetString("type", "free");
        for (int j = 0; j < n; ++j) {
            const double y = g.y(j);
            for (int i = 0; i < n; ++i) {
                const double x = g.x(i);
                double add = 0.0;
                if (type == "free") {
                    add = 0.0;
                } else if (type == "harmonic") {
                    const double w = p.GetDouble("omega", 1.0);
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double y0 = p.GetDouble("y0", 0.0);
                    add = 0.5 * w * w * ((x - x0) * (x - x0) + (y - y0) * (y - y0));
                } else if (type == "barrier") {
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double width = p.GetDouble("width", 0.5);
                    const double height = p.GetDouble("height", 10.0);
                    if (std::abs(x - x0) <= 0.5 * width) add = height;
                } else if (type == "double_slit") {
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double thick = p.GetDouble("thickness", 0.6);
                    const double sep = p.GetDouble("slit_sep", 3.0);
                    const double sw = p.GetDouble("slit_width", 0.8);
                    const double height = p.GetDouble("height", 30.0);
                    const bool inWall = std::abs(x - x0) <= 0.5 * thick;
                    const bool inSlit = (std::abs(y - 0.5 * sep) <= 0.5 * sw) ||
                                        (std::abs(y + 0.5 * sep) <= 0.5 * sw);
                    if (inWall && !inSlit) add = height;
                } else if (type == "well") {
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double y0 = p.GetDouble("y0", 0.0);
                    const double radius = p.GetDouble("radius", 3.0);
                    const double depth = p.GetDouble("depth", 5.0);
                    if ((x - x0) * (x - x0) + (y - y0) * (y - y0) <= radius * radius) add = -depth;
                } else if (type == "coulomb") {
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double y0 = p.GetDouble("y0", 0.0);
                    const double strength = p.GetDouble("strength", 1.0);
                    const double soft = p.GetDouble("softening", 0.5);
                    const double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0) + soft * soft;
                    add = -strength / std::sqrt(r2);
                }
                v[static_cast<size_t>(j) * n + i] += static_cast<float>(add);
            }
        }
    }
    return v;
}

std::vector<float> BuildCap(const Grid& g, const fw::Deck& deck) {
    const int n = g.n;
    std::vector<float> w(static_cast<size_t>(n) * n, 0.0f);

    const std::string type = deck.GetString("boundary.type", "cap");
    if (type == "periodic" || type == "none") return w;

    const double width = deck.GetDouble("boundary.cap_width", 4.0);
    const double strength = deck.GetDouble("boundary.cap_strength", 3.0);
    const double xEdge = 0.5 * g.lx - width;
    const double yEdge = 0.5 * g.ly - width;

    for (int j = 0; j < n; ++j) {
        const double y = g.y(j);
        const double dyIn = std::max(0.0, std::abs(y) - yEdge) / std::max(width, 1e-9);
        for (int i = 0; i < n; ++i) {
            const double x = g.x(i);
            const double dxIn = std::max(0.0, std::abs(x) - xEdge) / std::max(width, 1e-9);
            const double ramp = dxIn * dxIn + dyIn * dyIn; // quadratic in each axis
            w[static_cast<size_t>(j) * n + i] = static_cast<float>(strength * ramp);
        }
    }
    return w;
}

std::vector<std::complex<float>> BuildInitial(const Grid& g, const fw::Deck& deck) {
    const int n = g.n;
    std::vector<std::complex<double>> psi(static_cast<size_t>(n) * n, {0.0, 0.0});

    const std::string type = deck.GetString("initial.type", "gaussian");

    if (type == "hermite_gauss") {
        const int nx = deck.GetInt("initial.nx", 0);
        const int ny = deck.GetInt("initial.ny", 0);
        const double omega = deck.GetDouble("initial.omega", 1.0);
        const double x0 = deck.GetDouble("initial.x0", 0.0);
        const double y0 = deck.GetDouble("initial.y0", 0.0);
        const double a = std::sqrt(omega); // sqrt(m*omega/hbar), hbar=m=1
        for (int j = 0; j < n; ++j) {
            const double y = g.y(j) - y0;
            const double fy = std::exp(-0.5 * a * a * y * y) * Hermite(ny, a * y);
            for (int i = 0; i < n; ++i) {
                const double x = g.x(i) - x0;
                const double fx = std::exp(-0.5 * a * a * x * x) * Hermite(nx, a * x);
                psi[static_cast<size_t>(j) * n + i] = {fx * fy, 0.0};
            }
        }
    } else { // gaussian wavepacket
        const double x0 = deck.GetDouble("initial.x0", 0.0);
        const double y0 = deck.GetDouble("initial.y0", 0.0);
        const double sigma = deck.GetDouble("initial.sigma", 1.5);
        const double kx = deck.GetDouble("initial.kx", 0.0);
        const double ky = deck.GetDouble("initial.ky", 0.0);
        const double s2 = 2.0 * sigma * sigma;
        for (int j = 0; j < n; ++j) {
            const double y = g.y(j);
            for (int i = 0; i < n; ++i) {
                const double x = g.x(i);
                const double r2 = (x - x0) * (x - x0) + (y - y0) * (y - y0);
                const double env = std::exp(-r2 / s2);
                const double ph = kx * x + ky * y;
                psi[static_cast<size_t>(j) * n + i] = {env * std::cos(ph), env * std::sin(ph)};
            }
        }
    }

    // Normalize: sum |psi|^2 dx dy = 1.
    double norm2 = 0.0;
    for (const auto& c : psi) norm2 += std::norm(c);
    norm2 *= g.dx() * g.dy();
    const double inv = norm2 > 0.0 ? 1.0 / std::sqrt(norm2) : 1.0;

    std::vector<std::complex<float>> out(psi.size());
    for (size_t k = 0; k < psi.size(); ++k) {
        out[k] = std::complex<float>(static_cast<float>(psi[k].real() * inv),
                                     static_cast<float>(psi[k].imag() * inv));
    }
    return out;
}

} // namespace tdse
