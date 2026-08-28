#include "TdseField.hpp"

#include "framework/Deck.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <string>

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

// Add one initial-state component (read from `d` at `pre`+key) into the psi
// accumulator. The component is L2-normalized first, then scaled by
// weight * e^{i phase} -- so `weight` is the true amplitude coefficient c_n.
// `pre` is "initial." for a plain [initial] table, "" for a [[initial.component]].
void AddComponent(std::vector<std::complex<double>>& psi, const Grid& g,
                  const fw::Deck& d, const std::string& pre) {
    auto D = [&](const char* k, double def) { return d.GetDouble(pre + k, def); };
    auto I = [&](const char* k, int def) { return d.GetInt(pre + k, def); };
    const std::string type = d.GetString(pre + "type", "gaussian");
    const double phase = D("phase", 0.0);
    const std::complex<double> cw =
        D("weight", 1.0) * std::complex<double>(std::cos(phase), std::sin(phase));

    const int n = g.n;
    std::vector<std::complex<double>> c(static_cast<size_t>(n) * n, {0.0, 0.0});

    if (type == "hermite_gauss") {
        const int nx = I("nx", 0), ny = I("ny", 0);
        const double a = std::sqrt(D("omega", 1.0)); // sqrt(m omega / hbar)
        const double x0 = D("x0", 0.0), y0 = D("y0", 0.0);
        for (int j = 0; j < n; ++j) {
            const double y = g.y(j) - y0;
            const double fy = std::exp(-0.5 * a * a * y * y) * Hermite(ny, a * y);
            for (int i = 0; i < n; ++i) {
                const double x = g.x(i) - x0;
                const double fx = std::exp(-0.5 * a * a * x * x) * Hermite(nx, a * x);
                c[static_cast<size_t>(j) * n + i] = fx * fy;
            }
        }
    } else { // gaussian wavepacket
        const double x0 = D("x0", 0.0), y0 = D("y0", 0.0);
        const double sigma = D("sigma", 1.5);
        const double sx = D("sigma_x", sigma), sy = D("sigma_y", sigma);
        const double kx = D("kx", 0.0), ky = D("ky", 0.0);
        for (int j = 0; j < n; ++j) {
            const double y = g.y(j);
            for (int i = 0; i < n; ++i) {
                const double x = g.x(i);
                const double env = std::exp(-(x - x0) * (x - x0) / (2.0 * sx * sx) -
                                            (y - y0) * (y - y0) / (2.0 * sy * sy));
                const double ph = kx * x + ky * y;
                c[static_cast<size_t>(j) * n + i] =
                    env * std::complex<double>(std::cos(ph), std::sin(ph));
            }
        }
    }

    double nrm2 = 0.0;
    for (const auto& z : c) nrm2 += std::norm(z);
    nrm2 *= g.dx() * g.dy();
    const std::complex<double> scale = nrm2 > 0.0 ? cw / std::sqrt(nrm2) : cw;
    for (size_t k = 0; k < c.size(); ++k) psi[k] += scale * c[k];
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
                    const double wx = p.GetDouble("omega_x", w);
                    const double wy = p.GetDouble("omega_y", w);
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double y0 = p.GetDouble("y0", 0.0);
                    add = 0.5 * (wx * wx * (x - x0) * (x - x0) + wy * wy * (y - y0) * (y - y0));
                } else if (type == "box") {
                    const double x0 = p.GetDouble("x0", 0.0);
                    const double y0 = p.GetDouble("y0", 0.0);
                    const double hx = p.GetDouble("half_x", 5.0);
                    const double hy = p.GetDouble("half_y", 5.0);
                    const double wall = p.GetDouble("wall", 5000.0);
                    if (std::abs(x - x0) > hx || std::abs(y - y0) > hy) add = wall;
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

    if (deck.GetString("initial.type", "gaussian") == "superposition") {
        for (const auto& comp : deck.GetTables("initial.component")) {
            AddComponent(psi, g, comp, "");
        }
    } else {
        AddComponent(psi, g, deck, "initial.");
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
